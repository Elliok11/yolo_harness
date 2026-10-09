"""
web_server.py - 给桌面应用用的本地 Web 服务

它把项目已有的能力（摄像头、人脸识别、AI 对话）暴露成 HTTP 接口，
这样 Electron / Tauri / 浏览器都能拿来当界面。

为什么不用 FastAPI / Flask？
  因为它们都要额外 pip 安装，而我这台机器网络很差（装 n8n 时踩了一堆坑）。
  而 Python 标准库的 http.server 已经能做这件事：
    · MJPEG 视频流 —— 用 multipart/x-mixed-replace 直接推给 <img> 标签
    · JSON 接口   —— 自己写几十行路由就够
  结论：零额外依赖。

接口一览：
  GET  /              界面（index.html）
  GET  /video         摄像头画面，MJPEG 流（浏览器 <img src="/video"> 就能放）
  GET  /stats         当前状态：人脸数、脸库人数、物体列表……
  POST /chat          发一句话给 AI，返回回答
  POST /snapshot      保存当前画面
  POST /shutdown      关掉服务

用法：
  python web_server.py                # 默认 127.0.0.1:8765
  python web_server.py --port 9000
  python web_server.py --camera 1
  python web_server.py --no-camera    # 只做 AI 对话，不开摄像头（没摄像头时调试用）
"""
import os
import sys
import io
import json
import time
import argparse
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

# 界面文件放在同目录的 web/ 下
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class CameraWorker(threading.Thread):
    """后台抓帧线程：读摄像头 → 跑人脸识别 → 保存最新画面

    和 camera_gui.py 里那套是一个思路：摄像头不能占着主线程，
    否则 Web 服务就没法响应别的请求了。
    """

    def __init__(self, source=0, enable_face=True, enable_objects=False):
        super().__init__(daemon=True)
        self.source = source
        self.enable_face = enable_face
        self.enable_objects = enable_objects
        self.running = True
        self.lock = threading.Lock()
        self.jpeg = None            # 最新一帧的 JPEG 字节（给视频流用）
        self.stats = {"faces": 0, "known": 0, "pending": 0, "fps": 0.0,
                      "objects": [], "camera_ok": False, "message": "正在启动…"}
        self.tracker = None
        self._new_faces = []

    def run(self):
        if self.enable_face:
            try:
                import face_tools
                self.tracker = face_tools.FaceTracker()
                self.tracker.load_library()
                self.tracker.reset_session()
                self.stats["known"] = len(self.tracker.library)
            except Exception as exc:
                self.stats["message"] = "人脸模块加载失败：%s" % exc
                self.tracker = None

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.stats["message"] = "打不开摄像头（编号 %s）" % self.source
            self.stats["camera_ok"] = False
            return
        self.stats["camera_ok"] = True
        self.stats["message"] = "摄像头已就绪"

        model = None
        if self.enable_objects:
            try:
                import yolo_tools
                model = yolo_tools.get_model()
            except Exception:
                model = None

        fps_t0 = time.time()
        fps_n = 0
        frame_no = 0

        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            frame_no += 1

            # 人脸识别（每 3 帧一次，省 CPU）
            if self.tracker and frame_no % 3 == 0:
                try:
                    info = self.tracker.process(frame, draw=True)
                    self.stats["faces"] = info["faces"]
                    self.stats["known"] = len(self.tracker.library)
                    self.stats["pending"] = len(info.get("pending", []))
                    for nf in info["new"]:
                        self._new_faces.append(nf)
                except Exception:
                    pass

            # 物体识别（可选，吃 CPU）
            if model is not None and frame_no % 3 == 0:
                try:
                    objs = []
                    for r in model.predict(source=frame, imgsz=640, verbose=False):
                        for b in r.boxes:
                            name = model.names[int(b.cls[0])]
                            objs.append({"name": name, "conf": round(float(b.conf[0]), 2)})
                    self.stats["objects"] = objs[:20]
                except Exception:
                    pass

            # 编码成 JPEG 给视频流
            ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok2:
                with self.lock:
                    self.jpeg = buf.tobytes()

            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                self.stats["fps"] = round(fps_n / (now - fps_t0), 1)
                fps_n = 0
                fps_t0 = now

        cap.release()

    def latest_jpeg(self):
        with self.lock:
            return self.jpeg

    def pop_new_faces(self):
        f, self._new_faces = self._new_faces, []
        return f

    def stop(self):
        self.running = False


class Handler(BaseHTTPRequestHandler):
    """HTTP 请求处理"""

    worker = None          # 由 main() 注入
    agent_session = None   # AI 会话
    server_version = "YoloHarnessWeb/1.0"

    # 关掉每帧都刷屏的访问日志
    def log_message(self, fmt, *args):
        pass

    # ---------- 工具 ----------
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception:
            self.send_error(404, "Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # 必须禁用缓存：否则改了界面文件，浏览器还拿旧的，
        # 会出现「代码明明改了但页面没变」这种让人抓狂的情况（我刚踩过）
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    # ---------- 路由 ----------
    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self._file(os.path.join(WEB_DIR, "app.js"), "application/javascript; charset=utf-8")
            return
        if path == "/style.css":
            self._file(os.path.join(WEB_DIR, "style.css"), "text/css; charset=utf-8")
            return

        if path == "/stats":
            # 没有摄像头线程时也要返回一份完整的默认值，
            # 否则前端拿到 null 会显示成 "null"，很难看
            st = dict(self.worker.stats) if self.worker else {
                "faces": 0, "known": 0, "pending": 0, "fps": 0.0,
                "objects": [], "camera_ok": False, "message": "摄像头未启动（--no-camera）",
            }
            if self.worker:
                nf = self.worker.pop_new_faces()
                if nf:
                    st["new_faces"] = nf
            if self.agent_session is not None:
                st["provider"] = self.agent_session.get("provider", "")
                st["model"] = self.agent_session.get("model", "")
            self._json(st)
            return

        if path == "/video":
            self._stream_video()
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            payload = {}

        if path == "/chat":
            msg = (payload.get("message") or "").strip()
            if not msg:
                self._json({"error": "消息为空"}, 400)
                return
            sess = self.agent_session
            if not sess or sess.get("session") is None:
                self._json({"error": "AI 会话没初始化（检查 agent.py 的配置）"}, 500)
                return
            try:
                reply, _trace = sess["session"].ask(msg)
                self._json({"reply": reply})
            except Exception as exc:
                self._json({"error": str(exc)[:300]}, 500)
            return

        if path == "/snapshot":
            jpg = self.worker.latest_jpeg() if self.worker else None
            if not jpg:
                self._json({"error": "还没有画面"}, 400)
                return
            from datetime import datetime
            os.makedirs("output_results", exist_ok=True)
            out = os.path.join("output_results",
                               "web_%s.jpg" % datetime.now().strftime("%Y%m%d_%H%M%S"))
            with open(out, "wb") as f:
                f.write(jpg)
            self._json({"saved": out})
            return

        if path == "/shutdown":
            self._json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        self.send_error(404, "Not Found")

    def _stream_video(self):
        """MJPEG 视频流：一次连接，持续推帧

        multipart/x-mixed-replace 这个格式很老但非常好用：
        浏览器把它当"会自己变的图片"，所以只要一个 <img src="/video"> 就能看到实时画面，
        不用 WebSocket、不用轮询、不用写一行前端流处理代码。
        """
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                jpg = self.worker.latest_jpeg() if self.worker else None
                if jpg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)      # 约 20fps，够了
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass                       # 用户关了页面，正常现象


def main():
    ap = argparse.ArgumentParser(description="桌面应用的本地 Web 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--camera", default="0", help="摄像头编号，或视频文件路径")
    ap.add_argument("--no-camera", action="store_true", help="不开摄像头（只做 AI 对话）")
    ap.add_argument("--objects", action="store_true", help="同时跑物体识别（吃 CPU）")
    ap.add_argument("--provider", default=None, help="模型后端 local/cloud/n8n")
    ap.add_argument("--with-memory", action="store_true",
                    help="带上命令行那边的对话记忆（默认不带，因为会显著变慢）")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    cam = args.camera
    if isinstance(cam, str) and cam.isdigit():
        cam = int(cam)

    # ---- 启动 AI 会话 ----
    #
    # 关于「要不要带上之前的对话记忆」——这是个性能问题，不是小事：
    #   agent_memory.json 里会累积大量【工具调用的原始结果】
    #   （比如整份图片列表、整份识别结果），实测 61 条消息就有 3.5 万字、约 1.7 万 tokens。
    #   在 CPU 上让 1.5B 的小模型啃这么长的 prompt，一次问答要 149 秒；
    #   而只问一句话只要 1.1 秒 —— 差 130 倍。
    #
    #   Web 界面的使用场景是「看着摄像头随问随答」，拖着一堆过时历史毫无意义，
    #   所以默认开一个干净会话。想要延续命令行那边记忆的话，加 --with-memory。
    session_holder = {"session": None, "provider": "", "model": ""}
    try:
        if args.provider:
            sys.argv = ["agent.py", "--provider", args.provider]
        import agent
        import chat_agent
        session_holder["session"] = chat_agent.AgentSession(
            load_from_memory=args.with_memory)
        session_holder["provider"] = agent.PROVIDER
        session_holder["model"] = agent.MODEL_NAME
        n_hist = session_holder["session"].history_count
        print("🤖 AI 就绪：%s / %s（带 %d 条历史）"
              % (agent.PROVIDER, agent.MODEL_NAME, n_hist))
        if n_hist > 10:
            print("   ⚠️ 历史较长，CPU 上每次问答会明显变慢。")
            print("      想要更快：去掉 --with-memory，或删掉 output_results/agent_memory.json")
    except Exception as exc:
        print("⚠️ AI 会话初始化失败：%s" % exc)
        print("   （界面还能用，只是聊天功能不可用）")

    # ---- 启动摄像头线程 ----
    worker = None
    if not args.no_camera:
        worker = CameraWorker(cam, enable_face=True, enable_objects=args.objects)
        worker.start()
        print("📷 摄像头线程已启动（编号 %s）" % cam)
    else:
        print("📷 已跳过摄像头（--no-camera）")

    Handler.worker = worker
    Handler.agent_session = session_holder

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://%s:%d" % (args.host, args.port)
    print()
    print("=" * 58)
    print("  🌐 服务已启动：%s" % url)
    print("  按 Ctrl+C 停止")
    print("=" * 58)

    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止…")
    finally:
        if worker:
            worker.stop()
        httpd.server_close()
        print("已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
