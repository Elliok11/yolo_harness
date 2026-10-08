"""
label_gui.py - 打标签窗口（给物体检测训练准备数据）

大白话用法：
  1. 双击 启动打标签.bat   （或者 python label_gui.py）
  2. 右边先加几个类别，比如「水杯」「安全帽」
  3. 左边点一张图，中间用鼠标【拖一个框】圈住目标
  4. 按 D / A 翻下一张 / 上一张，框会自动存盘
  5. 标够了再用 train_model.py 去训练

鼠标和键盘：
  · 在空白处按住拖  = 画一个新框（用当前选中的类别）
  · 点框内部        = 选中它（变红），再拖 = 移动
  · 拖框的角        = 改大小
  · Delete          = 删掉选中的框
  · A / D           = 上一张 / 下一张
  · 数字键 1~9      = 快速切换当前类别
  · Ctrl+S          = 手动存一次（其实画完就自动存了）

重要：这个程序只新增功能，不改动你原有的摄像头、识别、对话那套东西。
标签存在 labels/ 目录，离你原来的 output_results/ 远远的。
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, ImageTk

import label_data as ld

# 颜色表：每个类别一个颜色，方便一眼区分
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
           "#e6beff", "#9a6324", "#800000", "#808000", "#000075"]
SELECTED = "#ff0000"       # 选中的框用红色
MIN_BOX_PX = 4             # 太小的框当误操作丢掉


class LabelApp:
    def __init__(self, root):
        self.root = root
        root.title("打标签工具 · 给 YOLO 训练准备数据")
        root.geometry("1280x800")

        # ---------- 状态 ----------
        self.classes = ld.load_classes()
        self.items = []                  # [(显示名, 路径)]
        self.idx = 0                     # 当前第几张
        self.boxes = []                  # 当前图的框（比例坐标）
        self.selected = -1               # 选中第几个框
        self.pil_img = None              # 原图
        self.photo = None                # 缩放后给画布用的
        self.scale = 1.0                 # 缩放比
        self.off_x = self.off_y = 0      # 图片在画布里的左上角位置
        self.cur_class = 0               # 当前类别编号
        self.drag = None                 # 正在拖拽的信息
        self.show_boxes = True           # 是否显示已画的框（空格切换）

        self._build_ui()
        self.reload_sources()
        self._bind_keys()
        if not self.classes:
            self.say("还没有类别。先在右边输入名字，点【加类别】。\n"
                     "比如：水杯、安全帽、我的猫……")
        else:
            self.load_current()
        # 刚建窗口时画布还没布局，尺寸是 1x1，图片会被缩成一个点。
        # 等窗口真正显示出来再重画一次，保证第一眼就看到图。
        self.root.after(120, self.redraw)

    # ==================== 界面 ====================
    def _build_ui(self):
        # 顶栏
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        self.lbl_pos = ttk.Label(top, text="", font=("Microsoft YaHei", 11, "bold"))
        self.lbl_pos.pack(side=tk.LEFT)
        self.lbl_hint = ttk.Label(top, text="", foreground="#0a7")
        self.lbl_hint.pack(side=tk.LEFT, padx=12)
        ttk.Button(top, text="🔄 重新扫描图片", command=self.reload_sources).pack(side=tk.RIGHT)

        # 工具条
        bar = ttk.Frame(self.root, padding=(6, 0))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="◀ 上一张 (A)", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(bar, text="下一张 (D) ▶", command=self.next_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="💾 存盘 (Ctrl+S)", command=self.save_now).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="🤖 自动预标注", command=self.auto_label).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="🗑 清空本图框", command=self.clear_boxes).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="⌫ 删除选中 (Del)", command=self.delete_selected).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="📊 看统计", command=self.show_stats).pack(side=tk.LEFT, padx=3)
        self.var_show = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="显示框(空格)", variable=self.var_show,
                        command=self.redraw).pack(side=tk.LEFT, padx=8)

        # 主体：左列表 / 中画布 / 右类别
        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, padding=6)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="图片列表", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.lst = tk.Listbox(left, width=34, font=("Consolas", 9), exportselection=False)
        self.lst.pack(fill=tk.BOTH, expand=True)
        self.lst.bind("<<ListboxSelect>>", self.on_pick_image)
        self.lbl_count = ttk.Label(left, text="")
        self.lbl_count.pack(anchor=tk.W, pady=(4, 0))
        ttk.Button(left, text="📁 从摄像头抓一张进来", command=self.grab_from_live).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(left, text="➕ 添加外部图片…", command=self.add_external).pack(fill=tk.X, pady=(3, 0))

        mid = ttk.Frame(body, padding=6)
        mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(mid, background="#202020", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_move)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        right = ttk.Frame(body, padding=6)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="类别", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.lst_cls = tk.Listbox(right, width=22, height=14, font=("Microsoft YaHei", 10),
                                  exportselection=False)
        self.lst_cls.pack(fill=tk.Y)
        self.lst_cls.bind("<<ListboxSelect>>", self.on_pick_class)

        row = ttk.Frame(right); row.pack(fill=tk.X, pady=(6, 0))
        self.ent_cls = ttk.Entry(row, width=14, font=("Microsoft YaHei", 10))
        self.ent_cls.pack(side=tk.LEFT)
        self.ent_cls.bind("<Return>", lambda e: self.add_class())
        ttk.Button(row, text="加类别", command=self.add_class, width=7).pack(side=tk.LEFT, padx=2)
        ttk.Button(right, text="🖊 重命名选中类别", command=self.rename_class).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(right, text="❌ 删除选中类别", command=self.del_class).pack(fill=tk.X, pady=(3, 0))
        ttk.Label(right, text="提示：数字键 1~9 快速切类别",
                  font=("Microsoft YaHei", 8), foreground="#666").pack(anchor=tk.W, pady=(8, 0))

        # 底栏信息
        self.lbl_msg = ttk.Label(self.root, text="", font=("Consolas", 9), anchor=tk.W)
        self.lbl_msg.pack(side=tk.BOTTOM, fill=tk.X)
        self._refresh_class_list()

    def _bind_keys(self):
        r = self.root
        r.bind("<Key-a>", lambda e: self.prev_image())
        r.bind("<Key-A>", lambda e: self.prev_image())
        r.bind("<Key-d>", lambda e: self.next_image())
        r.bind("<Key-D>", lambda e: self.next_image())
        r.bind("<Delete>", lambda e: self.delete_selected())
        r.bind("<Control-s>", lambda e: self.save_now())
        r.bind("<space>", lambda e: self.toggle_boxes())
        for i in range(1, 10):
            r.bind("<Key-%d>" % i, lambda e, n=i - 1: self.set_class(n))

    # ==================== 基础动作 ====================
    def say(self, text):
        self.lbl_msg.configure(text=text)

    def toggle_boxes(self):
        self.var_show.set(not self.var_show.get())
        self.redraw()

    def reload_sources(self):
        self.items = ld.list_sources()
        self.lst.delete(0, tk.END)
        for name, _ in self.items:
            self.lst.insert(tk.END, name)
        self.lbl_count.configure(text="共 %d 张图" % len(self.items))
        if self.items:
            if self.idx >= len(self.items):
                self.idx = 0
            self.lst.selection_clear(0, tk.END)
            self.lst.selection_set(self.idx)
            self.lst.see(self.idx)
            self.load_current()
        else:
            self.say("没找到任何图片。把图片放进 images/ 或 label_images/ 再点【重新扫描】。")

    def _refresh_class_list(self):
        self.lst_cls.delete(0, tk.END)
        for i, c in enumerate(self.classes):
            self.lst_cls.insert(tk.END, "%d. %s" % (i + 1, c))
        if self.classes:
            if self.cur_class >= len(self.classes):
                self.cur_class = 0
            self.lst_cls.selection_clear(0, tk.END)
            self.lst_cls.selection_set(self.cur_class)
            self.lst_cls.itemconfig(self.cur_class, background="#cfe8ff")

    # ==================== 图片加载 ====================
    def load_current(self, keep_boxes=False):
        if not self.items:
            return
        name, path = self.items[self.idx]
        try:
            self.pil_img = Image.open(path).convert("RGB")
        except Exception as exc:
            self.say("这张图打不开：%s（%s）" % (name, exc))
            self.pil_img = None
            return
        if not keep_boxes:
            self.boxes = ld.load_boxes(path)
            self.selected = -1
        self.lbl_pos.configure(text="[%d/%d] %s   %dx%d"
                               % (self.idx + 1, len(self.items), name,
                                  self.pil_img.width, self.pil_img.height))
        self.redraw()
        self.say("已载入，%d 个框。拖鼠标画框，D 翻下一张（自动存盘）。" % len(self.boxes))

    def current_path(self):
        return self.items[self.idx][1] if self.items else None

    def next_image(self):
        if not self.items:
            return
        self.save_now(silent=True)
        self.idx = (self.idx + 1) % len(self.items)
        self.lst.selection_clear(0, tk.END); self.lst.selection_set(self.idx); self.lst.see(self.idx)
        self.load_current()

    def prev_image(self):
        if not self.items:
            return
        self.save_now(silent=True)
        self.idx = (self.idx - 1) % len(self.items)
        self.lst.selection_clear(0, tk.END); self.lst.selection_set(self.idx); self.lst.see(self.idx)
        self.load_current()

    def on_pick_image(self, event=None):
        sel = self.lst.curselection()
        if not sel or sel[0] == self.idx:
            return
        self.save_now(silent=True)
        self.idx = sel[0]
        self.load_current()

    # ==================== 画布 ====================
    def redraw(self):
        c = self.canvas
        c.delete("all")
        if self.pil_img is None:
            return
        cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 10 or ch < 10:
            return
        iw, ih = self.pil_img.size
        self.scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * self.scale)), max(1, int(ih * self.scale))
        self.off_x, self.off_y = (cw - nw) // 2, (ch - nh) // 2

        resample = Image.LANCZOS if self.scale < 1 else Image.NEAREST
        self.photo = ImageTk.PhotoImage(self.pil_img.resize((nw, nh), resample))
        c.create_image(self.off_x, self.off_y, anchor=tk.NW, image=self.photo)

        if not self.var_show.get():
            return
        for i, b in enumerate(self.boxes):
            x1, y1, x2, y2 = ld.to_pixels(b, iw, ih)
            X1 = self.off_x + x1 * self.scale
            Y1 = self.off_y + y1 * self.scale
            X2 = self.off_x + x2 * self.scale
            Y2 = self.off_y + y2 * self.scale
            color = SELECTED if i == self.selected else PALETTE[b["cls"] % len(PALETTE)]
            width = 3 if i == self.selected else 2
            c.create_rectangle(X1, Y1, X2, Y2, outline=color, width=width)
            label = self.classes[b["cls"]] if b["cls"] < len(self.classes) else "?%d" % b["cls"]
            c.create_text(X1 + 3, Y1 + 9, anchor=tk.W, text=label, fill=color,
                          font=("Microsoft YaHei", 10, "bold"))
            if i == self.selected:
                for (hx, hy) in ((X1, Y1), (X2, Y1), (X1, Y2), (X2, Y2)):
                    c.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4, fill=color, outline="white")

    def canvas_to_image(self, cx, cy):
        """画布坐标 -> 原图坐标"""
        return ((cx - self.off_x) / self.scale, (cy - self.off_y) / self.scale)

    def hit_test(self, cx, cy):
        """看鼠标点到了哪个框。返回 (编号, 部位)，部位是 'move' 或角 'nw/ne/sw/se'"""
        if self.pil_img is None:
            return -1, None
        iw, ih = self.pil_img.size
        tol = 8
        for i in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = ld.to_pixels(self.boxes[i], iw, ih)
            X1 = self.off_x + x1 * self.scale; Y1 = self.off_y + y1 * self.scale
            X2 = self.off_x + x2 * self.scale; Y2 = self.off_y + y2 * self.scale
            corners = {"nw": (X1, Y1), "ne": (X2, Y1), "sw": (X1, Y2), "se": (X2, Y2)}
            for name, (hx, hy) in corners.items():
                if abs(cx - hx) <= tol and abs(cy - hy) <= tol:
                    return i, name
            if min(X1, X2) <= cx <= max(X1, X2) and min(Y1, Y2) <= cy <= max(Y1, Y2):
                return i, "move"
        return -1, None

    # 鼠标交互
    def on_press(self, ev):
        if self.pil_img is None:
            return
        if not self.classes:
            messagebox.showwarning("还没有类别", "请先在右边加一个类别，比如「水杯」。")
            return
        i, part = self.hit_test(ev.x, ev.y)
        ix, iy = self.canvas_to_image(ev.x, ev.y)
        if i >= 0:
            self.selected = i
            self.drag = {"mode": part, "index": i, "start": (ix, iy),
                         "orig": dict(self.boxes[i])}
        else:
            self.selected = len(self.boxes)
            self.boxes.append({"cls": self.cur_class, "x": ix / self.pil_img.width,
                               "y": iy / self.pil_img.height, "w": 0.0, "h": 0.0})
            self.drag = {"mode": "new", "index": self.selected,
                         "start": (ix, iy), "orig": None}
        self.redraw()

    def on_drag(self, ev):
        if not self.drag or self.pil_img is None:
            return
        ix, iy = self.canvas_to_image(ev.x, ev.y)
        sx, sy = self.drag["start"]
        i = self.drag["index"]
        mode = self.drag["mode"]
        iw, ih = self.pil_img.size

        if mode in ("new",):
            x1, y1 = sx, sy
            self.boxes[i].update(ld.to_ratio(x1, y1, ix, iy, iw, ih))
        elif mode == "move":
            o = self.drag["orig"]
            dx, dy = (ix - sx) / iw, (iy - sy) / ih
            self.boxes[i].update({"x": min(max(o["x"] + dx, 0), 1),
                                  "y": min(max(o["y"] + dy, 0), 1),
                                  "w": o["w"], "h": o["h"]})
        else:
            # 拖某个角：把对角固定，改这一角
            o = self.drag["orig"]
            ox1, oy1, ox2, oy2 = ld.to_pixels(o, iw, ih)
            if mode == "nw": ox1, oy1 = ix, iy
            elif mode == "ne": ox2, oy1 = ix, iy
            elif mode == "sw": ox1, oy2 = ix, iy
            elif mode == "se": ox2, oy2 = ix, iy
            self.boxes[i].update(ld.to_ratio(ox1, oy1, ox2, oy2, iw, ih))
        self.redraw()

    def on_release(self, ev):
        if not self.drag:
            return
        i = self.drag["index"]
        mode = self.drag["mode"]
        # 太小的框当误点，丢掉
        if i < len(self.boxes):
            b = self.boxes[i]
            if b["w"] * self.pil_img.width < MIN_BOX_PX or b["h"] * self.pil_img.height < MIN_BOX_PX:
                self.boxes.pop(i)
                self.selected = -1
                self.drag = None
                self.redraw()
                self.say("框太小，已忽略（想画框请按住拖出一段距离）。")
                return
        self.drag = None
        self.save_now(silent=True)
        self.say("已保存，本图现在 %d 个框。" % len(self.boxes))

    def on_move(self, ev):
        if self.pil_img is None:
            return
        i, part = self.hit_test(ev.x, ev.y)
        cursor = "crosshair"
        if i >= 0:
            cursor = {"move": "fleur", "nw": "size_nw_se", "se": "size_nw_se",
                      "ne": "size_ne_sw", "sw": "size_ne_sw"}.get(part, "crosshair")
        self.canvas.configure(cursor=cursor)

    # ==================== 框操作 ====================
    def delete_selected(self):
        if 0 <= self.selected < len(self.boxes):
            self.boxes.pop(self.selected)
            self.selected = -1
            self.save_now(silent=True)
            self.redraw()
            self.say("已删除选中的框。")

    def clear_boxes(self):
        if not self.boxes:
            return
        if messagebox.askyesno("确认", "清空这张图的全部 %d 个框？" % len(self.boxes)):
            self.boxes = []
            self.selected = -1
            self.save_now(silent=True)
            self.redraw()

    def save_now(self, silent=False):
        p = self.current_path()
        if not p:
            return
        n = ld.save_boxes(p, self.boxes)
        if not silent:
            self.say("已存盘：%s（%d 个框）" % (ld.label_path_for(p), n))

    # ==================== 类别操作 ====================
    def on_pick_class(self, event=None):
        sel = self.lst_cls.curselection()
        if sel:
            self.set_class(sel[0])

    def set_class(self, n):
        if 0 <= n < len(self.classes):
            self.cur_class = n
            self.lst_cls.selection_clear(0, tk.END)
            self.lst_cls.selection_set(n)
            for i in range(len(self.classes)):
                self.lst_cls.itemconfig(i, background="#cfe8ff" if i == n else "white")
            self.say("当前类别：%s（画新框会用它）" % self.classes[n])
            # 如果正选中某个框，顺手把它改成这个类别
            if 0 <= self.selected < len(self.boxes):
                self.boxes[self.selected]["cls"] = n
                self.save_now(silent=True)
                self.redraw()

    def add_class(self):
        name = self.ent_cls.get().strip()
        if not name:
            return
        if name in self.classes:
            self.say("已经有这个类别了：%s" % name)
            return
        self.classes = ld.add_class(self.classes, name)
        self.ent_cls.delete(0, tk.END)
        self._refresh_class_list()
        self.set_class(len(self.classes) - 1)
        self.say("已加类别：%s" % name)

    def rename_class(self):
        sel = self.lst_cls.curselection()
        if not sel:
            self.say("先在上面点一个类别再改名。")
            return
        i = sel[0]
        new = simpledialog.askstring("改名", "新名字：", initialvalue=self.classes[i])
        if not new:
            return
        self.classes[i] = new.strip()
        ld.save_classes(self.classes)
        self._refresh_class_list()
        self.redraw()
        self.say("类别 %d 改名为：%s" % (i + 1, self.classes[i]))

    def del_class(self):
        sel = self.lst_cls.curselection()
        if not sel:
            self.say("先点一个类别再删。")
            return
        i = sel[0]
        name = self.classes[i]
        if not messagebox.askyesno(
                "确认删除",
                "删掉类别「%s」？\n\n注意：所有标成它的框都会被删掉，\n"
                "比它编号大的类别会自动往前挪，已有标签不会错位。" % name):
            return
        self.classes, _ = ld.remove_class(self.classes, name)
        self._refresh_class_list()
        self.load_current()
        self.say("已删除类别：%s" % name)

    # ==================== 自动预标注 ====================
    def auto_label(self):
        """用现成的 YOLO 先把它能认的都画出来，你只改错的，不用从零画"""
        p = self.current_path()
        if not p or self.pil_img is None:
            return
        try:
            import yolo_tools
        except Exception as exc:
            self.say("加载 YOLO 失败：%s" % exc)
            return
        model = yolo_tools.get_model()
        self.say("正在预标注…（第一次会慢一点）")
        self.root.update_idletasks()
        try:
            res = model.predict(source=p, save=False, verbose=False)
        except Exception as exc:
            self.say("预标注失败：%s" % exc)
            return

        # 把 COCO 的类别名并进我们的类别表（没有就自动加）
        added = 0
        iw, ih = self.pil_img.size
        for r in res:
            for box in r.boxes:
                name = model.names[int(box.cls[0])]
                if name not in self.classes:
                    self.classes = ld.add_class(self.classes, name)
                    added += 1
                ci = self.classes.index(name)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                b = ld.to_ratio(x1, y1, x2, y2, iw, ih)
                b["cls"] = ci
                self.boxes.append(b)
        self._refresh_class_list()
        self.save_now(silent=True)
        self.redraw()
        self.say("预标注完成：本图现在 %d 个框%s。把错的拖一拖、删一删就行。"
                 % (len(self.boxes), ("，新增 %d 个类别" % added) if added else ""))

    # ==================== 其它 ====================
    def show_stats(self):
        messagebox.showinfo("标注统计", ld.class_counts_text(self.classes) +
                            "\n\n标签目录：" + os.path.abspath(ld.LABEL_ROOT))

    def grab_from_live(self):
        """把摄像头最近一帧抓进来当训练素材（需要摄像头程序在跑过）"""
        live = os.path.join("output_results", "frame_live.jpg")
        if not os.path.exists(live):
            self.say("没有找到 %s。先用摄像头程序跑一下（菜单第4项或图形界面版），它会把当前画面存在那儿。" % live)
            return
        os.makedirs("label_images", exist_ok=True)
        import shutil
        from datetime import datetime
        dst = os.path.join("label_images", "live_%s.jpg" % datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.copy2(live, dst)
        self.reload_sources()
        for i, (_, p) in enumerate(self.items):
            if os.path.abspath(p) == os.path.abspath(dst):
                self.idx = i
                self.lst.selection_clear(0, tk.END); self.lst.selection_set(i); self.lst.see(i)
                self.load_current()
                break
        self.say("已把摄像头那一帧存成 %s，可以直接标了。" % dst)

    def add_external(self):
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(
            title="选图片（会复制进 label_images/）",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("全部", "*.*")])
        if not paths:
            return
        os.makedirs("label_images", exist_ok=True)
        import shutil
        for src in paths:
            dst = os.path.join("label_images", os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
        self.reload_sources()
        self.say("已加入 %d 张图片到 label_images/。" % len(paths))


def main():
    root = tk.Tk()
    LabelApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
