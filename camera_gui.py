"""
camera_gui.py - 图形界面版：实时画面 + 人脸识别 + 和 AI 对话

和 agent.py 控制台版的关系：
  · agent.py 一行没删，菜单 1/2/3/4 照旧能用
  · 这个是【额外】的新入口，专门做成一个窗口，不弹黑框

大白话版运行说明：
  1. 双击 启动摄像头窗口.bat   （或者命令行 run_camera_gui.bat）
  2. 窗口左边是实时画面，右边是聊天
  3. 想说话就在右下角输入框打字，回车发送
  4. 下面一排按钮：分析画面 / 现在有谁 / 存图 / 脸库名单 / 存报告

命令行参数（可选，都给默认值，不填也能跑）：
  python camera_gui.py                  # 用内置摄像头
  python camera_gui.py 1                # 用 1 号摄像头
  python camera_gui.py 我的视频.mp4      # 用视频文件当画面来源
  python camera_gui.py http://192.168.1.5:8080/video   # 手机摄像头
"""
import os
import sys
import queue
import threading
import time
from datetime import datetime

# ---- 所有输出都走 UTF-8，避免中文在窗口/日志里变乱码 ----
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk

import agent
import yolo_tools
import face_tools
from chat_agent import AgentSession, make_live_tools

# ==================== 配置区（想改就改这里） ====================
VIDEO_W, VIDEO_H = 640, 480     # 画面显示尺寸（屏幕只有 1366 宽，别做太大）
FACE_EVERY_N = 3                # 每几帧做一次人脸识别（数字越大越省 CPU、框越飘）
SHOW_OBJECTS = False            # 是否在画面上同时跑 YOLO 物体检测（吃 CPU，默认关）
# 物体检测用哪个模型、跑多大分辨率 —— 这两项直接决定准度和速度，可在窗口里随时换
# 实测（2848x1600 图，3 张合计）：yolov8n@640=55 个目标，yolov8s@960=85 个（+55%）
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
# 可选的模型清单（下拉框里出现哪些）——先确认文件真的存在，不存在的不显示
YOLO_CHOICES = [m for m in ("yolov8n.pt", "yolov8s.pt", "yolo11s.pt") if os.path.exists(m)]
UI_TICK_MS = 40                 # 界面刷新间隔（约 25 帧/秒）
# ===============================================================


class CameraThread(threading.Thread):
    """后台抓帧线程：抓画面 → 找人脸 → 认人 → 画框 → 丢给界面

    为什么要开线程？
      以前用 cv2.imshow，窗口必须由主线程画，摄像头只好霸占主线程，
      结果就是「你一打字画面就卡住」。
      现在画面变成一张图片塞进窗口，摄像头就能安心跑在后台了。
    """

    def __init__(self, source, tracker, out_queue, show_objects=SHOW_OBJECTS,
                 yolo_model=YOLO_MODEL, yolo_imgsz=YOLO_IMGSZ):
        super().__init__(daemon=True)
        self.source = source
        self.tracker = tracker
        self.queue = out_queue
        self.show_objects = show_objects
        self.yolo_model = yolo_model        # 可在界面里随时换
        self.yolo_imgsz = yolo_imgsz        # 同上
        self.running = True
        self.frame_no = 0
        self.fps = 0.0
        self.detect_ms = 0.0                # 这一步花了多久，显示出来好调参
        self._last_fps_t = time.time()
        self._fps_count = 0
        self._lock = threading.Lock()
        self._latest_raw = None      # 供「分析当前画面」拿最新原图
        self._latest_annotated = None
        self._face_done_at = 0.0     # 上一次检测完成的时间
        self._frame_pushed_at = 0.0  # 上一次把画面推给界面的时间
        self._yolo = None
        self._yolo_name = None
        self.stats = {"faces": 0, "known": len(tracker.library),
                      "new": [], "repeated": [], "pending": []}

    # ---------- 物体检测模型（按需加载，换模型时自动重载）----------
    def _get_yolo(self):
        if self._yolo is None or self._yolo_name != self.yolo_model:
            import yolo_tools
            self._yolo = yolo_tools.get_model(self.yolo_model)
            self._yolo_name = self.yolo_model
        return self._yolo

    # ---------- 给外部用 ----------
    def latest_frame(self):
        with self._lock:
            return None if self._latest_raw is None else self._latest_raw.copy()

    def latest_annotated(self):
        with self._lock:
            return None if self._latest_annotated is None else self._latest_annotated.copy()

    def stop(self):
        self.running = False

    # ---------- 主循环 ----------
    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.queue.put({"type": "fatal",
                            "text": "打不开摄像头（%s）。可能被别的程序占着。" % self.source})
            return
        self.queue.put({"type": "info", "text": "摄像头已打开，正在识别…"})

        while self.running:
            ok, frame = cap.read()
            if not ok:
                # 读到视频结尾（用视频文件时）就停下
                self.queue.put({"type": "info", "text": "画面读取结束。"})
                break
            self.frame_no += 1

            with self._lock:
                self._latest_raw = frame.copy()

            annotated = frame
            if self.frame_no % FACE_EVERY_N == 0:
                t_det = time.time()
                try:
                    info = self.tracker.process(frame, draw=True)
                    self.stats["faces"] = info["faces"]
                    self.stats["known"] = len(self.tracker.library)
                    self.stats["new"] = info["new"]
                    self.stats["repeated"] = info["repeated"]
                    self.stats["pending"] = info.get("pending", [])
                    for nf in info["new"]:
                        self.queue.put({"type": "new_face", "info": nf})
                except Exception as exc:
                    self.queue.put({"type": "warn", "text": "人脸识别出错：%s" % exc})
                self.detect_ms = (time.time() - t_det) * 1000
                annotated = frame

                if self.show_objects:
                    try:
                        model = self._get_yolo()
                        # 直接按 yolo_imgsz 推理，不再先缩小图像
                        # （实测：先缩一半会丢掉约 40% 的目标，这是之前准度差的主因之一）
                        for r in model.predict(source=annotated, imgsz=self.yolo_imgsz,
                                               save=False, verbose=False):
                            for box in r.boxes:
                                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                                name = model.names[int(box.cls[0])]
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 1)
                                cv2.putText(annotated, "%s %.2f" % (name, float(box.conf[0])),
                                            (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255, 128, 0), 1)
                    except Exception as exc:
                        self.queue.put({"type": "warn", "text": "物体检测出错：%s" % exc})

                self._face_done_at = time.time()   # 标记：这一轮的识别做完了

            # 画面上只写英文（cv2.putText 不支持中文，中文都放状态栏）
            hud = "faces:%d pending:%d known:%d %.0fms fps:%.0f" % (
                self.stats["faces"], len(self.stats.get("pending", [])),
                self.stats["known"], self.detect_ms, self.fps)
            cv2.putText(annotated, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            with self._lock:
                self._latest_annotated = annotated.copy()

            # 缩放成界面要的尺寸
            try:
                show = cv2.resize(annotated, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
            except Exception:
                show = annotated
            rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

            # 只在「画面真的更新过」时才推给界面。
            # 为什么？识别一张脸要一两百毫秒，摄像头却每秒能读 30 帧 ——
            # 不加这道闸，队列会被同一张画面的副本撑爆，界面越用越卡。
            if self._face_done_at > self._frame_pushed_at:
                self._frame_pushed_at = time.time()
                self.queue.put({"type": "frame", "image": rgb, "stats": dict(self.stats)})
            else:
                # 这一轮的检测还没做完，先把摄像头缓冲排空，别攒着旧画面
                cap.grab()
                time.sleep(0.005)
                continue

            # 算 FPS
            self._fps_count += 1
            now = time.time()
            if now - self._last_fps_t >= 1.0:
                self.fps = self._fps_count / (now - self._last_fps_t)
                self._fps_count = 0
                self._last_fps_t = now

        cap.release()
        self.queue.put({"type": "info", "text": "画面线程已停止。"})


class App:
    """窗口本体"""

    def __init__(self, root, source, threshold=None, show_objects=SHOW_OBJECTS,
                 yolo_model=YOLO_MODEL, yolo_imgsz=YOLO_IMGSZ):
        self.root = root
        self.source = source
        self.queue = queue.Queue()
        self.camera = None
        self.photo = None            # 必须留引用，否则图片会被回收、画面变白
        self.yolo_model = yolo_model
        self.yolo_imgsz = yolo_imgsz

        root.title("华小牛 · 摄像头 + 人脸识别 + 对话")
        root.geometry("1270x760")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---------- 左：画面 ----------
        left = ttk.Frame(root, padding=6)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        ttk.Label(left, text="实时画面（红=新登记  绿=认识的人  黄=本场临时  橙=投票中  灰=太小）",
                  font=("Microsoft YaHei", 9, "bold")).pack(anchor=tk.W)
        self.video = tk.Label(left, width=VIDEO_W, height=VIDEO_H,
                              background="#111111", borderwidth=1, relief="solid")
        self.video.pack()
        self.status = ttk.Label(left, text="正在启动…", font=("Consolas", 10))
        self.status.pack(anchor=tk.W, pady=(6, 0))
        self.hint = ttk.Label(left, text="", foreground="#0a7", font=("Microsoft YaHei", 9))
        self.hint.pack(anchor=tk.W)

        # ---------- 物体识别调节区（准度 vs 速度，自己现场找平衡）----------
        tune = ttk.LabelFrame(left, text="物体识别（YOLO）调参 · 改了立刻生效", padding=6)
        tune.pack(fill=tk.X, pady=(6, 0))
        self.var_objects = tk.BooleanVar(value=bool(show_objects))
        ttk.Checkbutton(tune, text="开启物体识别（关掉能省一半 CPU）",
                        variable=self.var_objects,
                        command=self.on_toggle_objects).pack(anchor=tk.W)

        row = ttk.Frame(tune); row.pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(row, text="模型：").pack(side=tk.LEFT)
        self.var_model = tk.StringVar(value=self.yolo_model)
        cb = ttk.Combobox(row, textvariable=self.var_model, width=12,
                          values=YOLO_CHOICES or [self.yolo_model], state="readonly")
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda e: self.on_tune_change())

        ttk.Label(row, text="  分辨率：").pack(side=tk.LEFT)
        self.var_imgsz = tk.IntVar(value=self.yolo_imgsz)
        sp = ttk.Spinbox(row, from_=320, to=1600, increment=160, width=7,
                         textvariable=self.var_imgsz, command=self.on_tune_change)
        sp.pack(side=tk.LEFT)
        sp.bind("<Return>", lambda e: self.on_tune_change())
        sp.bind("<FocusOut>", lambda e: self.on_tune_change())
        ttk.Label(tune, text="实测：yolov8n@640 = 55 个目标，yolov8s@960 = 85 个（+55%），但更吃 CPU",
                  font=("Microsoft YaHei", 8), foreground="#666").pack(anchor=tk.W, pady=(4, 0))
        self.save_path = os.path.join("output_results", "yolo_choice.json")
        self._load_tune()

        # ---------- 右：聊天 ----------
        right = ttk.Frame(root, padding=6)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text="和 AI 对话", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.chat = scrolledtext.ScrolledText(right, wrap=tk.WORD, width=52,
                                              font=("Microsoft YaHei", 10), state=tk.DISABLED)
        self.chat.pack(fill=tk.BOTH, expand=True, pady=(2, 6))
        for tag, color in (("you", "#1565c0"), ("ai", "#2e7d32"),
                           ("sys", "#8a6d00"), ("alert", "#c62828")):
            self.chat.tag_config(tag, foreground=color)

        # 按钮排
        btns = ttk.Frame(right)
        btns.pack(fill=tk.X, pady=(0, 4))
        self.btn_ai = ttk.Button(btns, text="🤖 分析当前画面", command=self.on_analyze)
        self.btn_who = ttk.Button(btns, text="👥 现在有谁", command=self.on_who)
        self.btn_save = ttk.Button(btns, text="📷 存图", command=self.on_save)
        self.btn_lib = ttk.Button(btns, text="📖 脸库名单", command=self.on_library)
        self.btn_rep = ttk.Button(btns, text="📝 存报告", command=self.on_report)
        for b in (self.btn_ai, self.btn_who, self.btn_save, self.btn_lib, self.btn_rep):
            b.pack(side=tk.LEFT, padx=2)

        # 输入区
        entry_row = ttk.Frame(right)
        entry_row.pack(fill=tk.X)
        self.entry = ttk.Entry(entry_row, font=("Microsoft YaHei", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda e: self.on_send())
        self.btn_send = ttk.Button(entry_row, text="发送", command=self.on_send, width=6)
        self.btn_send.pack(side=tk.LEFT, padx=(4, 0))

        # ---------- 会话 ----------
        self.session = AgentSession(tool_map=make_live_tools(self.get_frame))
        self.busy = False

        # 启动摄像头线程
        self.tracker = face_tools.FaceTracker(threshold=threshold).load_library()
        self.threshold = self.tracker.threshold

        self.say("sys", "窗口已就绪。脸库里目前有 %d 个人，对话历史 %d 条。"
                 % (len(self.tracker.library), self.session.remember_line()))
        self.say("sys", "输入框打字回车即可提问；左边画面里的人脸框会自动标注。")
        if not face_tools.models_ready():
            self.say("alert", "⚠️ 找不到人脸模型（models/ 里的两个 onnx），人脸识别用不了！")

        self.start_camera()
        self.root.after(UI_TICK_MS, self.pump)

    # ==================== 画面 ====================
    def _load_tune(self):
        """读回上次的调参选择，免得每次重开都要重设

        注意：存进来的值不能无条件信。分辨率要是被人存成 1600，
        下次启动会莫名其妙变得极慢，所以超出合理范围的一律拒绝。
        """
        try:
            import json
            with open(self.save_path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("model") in (YOLO_CHOICES or [self.yolo_model]):
                self.yolo_model = d["model"]
                self.var_model.set(d["model"])
            sz = d.get("imgsz")
            if isinstance(sz, int) and 320 <= sz <= 1600:
                self.yolo_imgsz = sz
                self.var_imgsz.set(sz)
            if isinstance(d.get("objects"), bool):
                self.var_objects.set(d["objects"])
        except Exception:
            pass

    def _save_tune(self):
        try:
            import json
            os.makedirs("output_results", exist_ok=True)
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump({"model": self.yolo_model, "imgsz": self.yolo_imgsz,
                           "objects": bool(self.var_objects.get())},
                          f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def on_toggle_objects(self):
        on = bool(self.var_objects.get())
        if self.camera:
            self.camera.show_objects = on
        self.say("sys", "物体识别已%s" % ("开启（会比较吃 CPU）" if on else "关闭（省 CPU，人脸更流畅）"))
        self._save_tune()

    def on_tune_change(self):
        """模型/分辨率一改，立刻生效（模型第一次用会花几秒加载）"""
        try:
            self.yolo_imgsz = int(self.var_imgsz.get())
        except Exception:
            self.yolo_imgsz = YOLO_IMGSZ
        self.yolo_model = self.var_model.get()
        if self.camera:
            self.camera.yolo_model = self.yolo_model
            self.camera.yolo_imgsz = self.yolo_imgsz
        self.say("sys", "物体识别改为 %s @ %d（第一次用这个模型要加载几秒）"
                 % (self.yolo_model, self.yolo_imgsz))
        self._save_tune()

    def start_camera(self):
        self.camera = CameraThread(self.source, self.tracker, self.queue,
                                   show_objects=bool(self.var_objects.get()),
                                   yolo_model=self.yolo_model,
                                   yolo_imgsz=self.yolo_imgsz)
        self.camera.start()

    def get_frame(self):
        """给「看当前画面」这类工具用：拿最新的一帧原图"""
        return None if self.camera is None else self.camera.latest_frame()

    def pump(self):
        """定时把后台线程的消息搬进界面（tkinter 只能在主线程里动）"""
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg.get("type")

                if kind == "frame":
                    img = Image.fromarray(msg["image"])
                    self.photo = ImageTk.PhotoImage(img)
                    self.video.configure(image=self.photo)
                    st = msg["stats"]
                    pending = len(st.get("pending", []))
                    cam = self.camera
                    self.status.configure(
                        text="画面 %d 张脸 | 投票中 %d | 脸库 %d 人 | 阈值 %.2f | 识别 %.0fms | %.1f fps"
                             % (st["faces"], pending, st["known"], self.threshold,
                                cam.detect_ms if cam else 0, cam.fps if cam else 0))

                elif kind == "new_face":
                    info = msg["info"]
                    self.say("alert", "🆕 发现新面孔：%s（相似度 %.3f）%s"
                             % (info["id"], info["score"],
                                "，已抓拍 " + info["snapshot"] if info.get("snapshot") else ""))
                    self.hint.configure(text="刚刚登记了 %s" % info["id"])

                elif kind in ("info", "warn"):
                    self.say("sys", ("⚠️ " if kind == "warn" else "") + msg.get("text", ""))

                elif kind == "fatal":
                    self.say("alert", msg.get("text", "出错了"))
                    self.hint.configure(text="摄像头不可用")

                elif kind == "agent_done":
                    self.busy = False
                    self.btn_send.configure(state=tk.NORMAL)
                    self.btn_ai.configure(state=tk.NORMAL)
                    self.say("ai", msg["reply"])
                    for step in msg.get("trace", []):
                        if step.get("type") == "tool_call":
                            self.say("sys", "   ↳ 调用工具 %s(%s)" % (step["tool"], step.get("args", "")))
                        elif step.get("type") == "tool_result":
                            self.say("sys", "   ↳ %s 返回 %s 字节，耗时 %sms"
                                     % (step["tool"], step.get("size"), step.get("ms")))
                        elif step.get("type") == "error":
                            self.say("alert", "   ↳ 出错：%s" % step.get("text"))
                        elif step.get("type") == "done":
                            self.say("sys", "   ↳ 本轮共用 %.1f 秒" % step.get("seconds", 0))

                elif kind == "local_done":
                    self.busy = False
                    self.say("ai", msg["reply"])

        except queue.Empty:
            pass
        self.root.after(UI_TICK_MS, self.pump)

    # ==================== 聊天 ====================
    def say(self, tag, text):
        self.chat.configure(state=tk.NORMAL)
        stamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"you": "你", "ai": "华小牛", "sys": "", "alert": ""}.get(tag, "")
        if tag == "sys":
            self.chat.insert(tk.END, "[%s] %s\n" % (stamp, text), tag)
        else:
            self.chat.insert(tk.END, "[%s] %s> %s\n" % (stamp, prefix, text), tag)
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def camera_context(self):
        """把摄像头此刻的情况告诉 AI，它回答才有的放矢"""
        st = self.camera.stats if self.camera else {"faces": 0}
        people = []
        for pid, _ in (self.camera.tracker.session.items() if self.camera else []):
            p = next((x for x in self.tracker.library.people if x["id"] == pid), None)
            people.append("%s(第%d次见)" % (pid, p.get("times_seen", 0)) if p else pid)
        return "画面中%d人；在场：%s；脸库共%d人；判定阈值%.2f" % (
            len(people), "、".join(people) if people else "暂无",
            len(self.tracker.library), self.threshold)

    def on_send(self):
        text = self.entry.get().strip()
        if not text:
            return
        if self.busy:
            self.say("sys", "上一句还在处理，稍等一下…")
            return
        self.entry.delete(0, tk.END)
        self.say("you", text)
        self.ask_agent(text)

    def on_analyze(self):
        self.say("you", "（点了「分析当前画面」）")
        self.ask_agent("请看看我摄像头当前的画面里有什么，并结合人脸识别结果说一句。")

    def ask_agent(self, text):
        if self.busy:
            return
        self.busy = True
        self.btn_send.configure(state=tk.DISABLED)
        self.btn_ai.configure(state=tk.DISABLED)
        self.say("sys", "正在思考…（画面继续刷新）")

        ctx = self.camera_context()
        step = {"type": "tool_call", "tool": "(准备中)", "args": ""}

        def work():
            reply, trace = self.session.ask(text, extra_context=ctx)
            self.queue.put({"type": "agent_done", "reply": reply, "trace": trace})
        threading.Thread(target=work, daemon=True).start()

    # ==================== 按钮 ====================
    def on_who(self):
        """不用 AI，本地立刻报一遍，零花费零延迟"""
        session = self.camera.tracker.session if self.camera else {}
        if not session:
            self.say("ai", "现在画面里没有人。")
            return
        parts = []
        for pid in session:
            p = next((x for x in self.tracker.library.people if x["id"] == pid), None)
            parts.append("%s（第 %d 次见）" % (pid, p.get("times_seen", 0)) if p else pid)
        self.say("ai", "当前画面里 %d 个人：%s" % (len(parts), "、".join(parts)))

    def on_save(self):
        frame = self.camera.latest_annotated() if self.camera else None
        if frame is None:
            self.say("alert", "还没有画面可存。")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("output_results", "frame_manual_%s.jpg" % stamp)
        os.makedirs("output_results", exist_ok=True)
        cv2.imwrite(path, frame)
        self.say("sys", "📁 已存图：" + path)

    def on_library(self):
        lib = self.tracker.library
        if not len(lib):
            self.say("ai", "脸库还是空的——还没遇到过任何人。")
            return
        lines = ["脸库里现在有 %d 个人（见面次数=走开又回来的次数）：" % len(lib)]
        for p in lib.people:
            n_emb = len(face_tools.embeddings_of(p))
            lines.append("  · %s：见面 %d 次，累计被看到 %d 帧，存了 %d 份特征，首次 %s%s"
                         % (p["id"], p.get("times_seen", 0), p.get("frames_seen", 0),
                            n_emb, p.get("first_seen", "?"),
                            "，抓拍 " + os.path.basename(p["snapshot"]) if p.get("snapshot") else ""))
        self.say("ai", "\n".join(lines))

    def on_report(self):
        self.say("you", "（点了「存报告」）")
        self.ask_agent("请把当前摄像头的情况（画面里有什么、在场的人是谁、脸库有多少人）"
                       "整理成一份简短的 Markdown 报告，调用 write_report 存下来。")

    # ==================== 关闭 ====================
    def on_close(self):
        try:
            if self.camera:
                self.camera.stop()
            if len(self.tracker.library):
                self.tracker.library.save()
            agent.save_memory(self.session.messages)
            print("已保存脸库和对话记录，窗口关闭。")
        finally:
            self.root.destroy()


def pick_source(argv):
    """从命令行第一个参数猜画面来源：数字=摄像头编号，其他=文件/网址"""
    if len(argv) < 2:
        return 0
    a = argv[1].strip()
    if a.isdigit():
        return int(a)
    return a


def main():
    source = pick_source(sys.argv)
    threshold = None
    if len(sys.argv) > 2:
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            threshold = None

    root = tk.Tk()
    App(root, source, threshold=threshold)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
