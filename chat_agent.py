"""
chat_agent.py - 图形界面和 AI 之间的那层「接线板」

为什么不直接在界面文件里调 agent？
  因为一次提问可能触发多轮工具调用（AI 先看画面、再看图片、再写报告……），
  界面需要知道这中间发生了什么，才能把过程展示给你看。
  所以这里把它包一层：问一句 → 返回最终回答 + 中间过程。

控制台版的 agent.py 完全不受影响，这个文件只是额外多一条路。
"""
import threading
import time

import agent
import yolo_tools


class AgentSession:
    """一段连续的对话。记住上下文，并且知道怎么调用工具。"""

    def __init__(self, tool_map=None, load_from_memory=True):
        self.tool_map = dict(agent.TOOL_MAP)
        if tool_map:
            self.tool_map.update(tool_map)
        self.lock = threading.Lock()          # 防止两句话同时问，把上下文搅乱
        self.messages = agent.load_memory() if load_from_memory else \
            [{"role": "system", "content": agent.SYSTEM_PROMPT}]
        # 每次提问的中间过程都记在这，界面可以取走展示
        self.trace = []

    # ---------------- 只读信息 ----------------
    @property
    def history_count(self):
        return len(self.messages) - 1

    def remember_line(self):
        """给界面显示：这次启动读回来了多少条历史"""
        return self.history_count

    # ---------------- 提问 ----------------
    def ask(self, text, extra_context=None):
        """问一句，返回 (最终回答, 中间过程列表)

        整个调用是阻塞的，界面应该把它放到后台线程里跑，避免卡住窗口。
        """
        with self.lock:
            self.trace = []
            # 摄像头状态【必须作为 system 消息】插入，不能拼在用户消息里。
            #
            # 为什么？我第一版是拼在用户消息里的（"问题\n\n【当前摄像头状态】…"），
            # 测试时用全新会话能work，但真实使用中会失效 ——
            # 因为对话历史里已经积累了模型自己说过的「我无法查看实时画面」。
            # 模型最信的是它自己刚说过的话，一条夹在用户消息里的描述压不过它。
            # 这就是工程日志【坑 1】记过的那个模式（AI 会跟自己的旧话保持一致）。
            #
            # 放在 system 角色里，权重高得多，而且明确下指令，才压得住。
            if extra_context:
                self.messages.append({
                    "role": "system",
                    "content": (
                        "【摄像头实时数据 —— 已经取好了，你直接看就行】\n"
                        "%s\n\n"
                        "注意：上面这些数据是程序【刚刚】从摄像头实时取到的，"
                        "已经在你手里了，不需要你去抓取或等待。\n"
                        "回答要求：\n"
                        "  1. 直接根据上面的数据回答，不要说你无法查看实时画面\n"
                        "  2. 不要说『请稍等』『我这就去抓取』这类话 —— 数据已经在了\n"
                        "  3. 用户问画面里有什么，就把上面的人脸和物体信息用自然语言说出来"
                        % extra_context),
                })
            self.messages.append({"role": "user", "content": text})
            t0 = time.time()
            try:
                reply = agent.run_agent_turn(self.messages, tool_map=self.tool_map)
            except Exception as exc:
                reply = "（调用 AI 失败：%s）" % exc
                self.trace.append({"type": "error", "text": str(exc)})
            took = time.time() - t0
            self.trace.append({"type": "done", "seconds": round(took, 1)})
            agent.save_memory(self.messages)
            return reply, list(self.trace)

    def add_note(self, text):
        """把一句系统提示塞进上下文（比如『刚发现新面孔 人C』）"""
        self.messages.append({"role": "user", "content": "[系统提示] " + text})


# ---------------- 给界面用的现成工具 ----------------
def make_live_tools(live_frame_getter):
    """造几个「面向当前画面」的工具，交给 AI 用

    live_frame_getter 是个函数，调用它返回当前帧（numpy 数组），
    有了它，AI 每次想看画面时拿到的都是【此刻】的画面，而不是启动时那张。

    注意：这里不能直接调 yolo_tools.detect_objects()，
    因为它会把参数当成「images 文件夹里的文件名」去拼路径，
    而我们的实时画面在 output_results/ 下，所以要在本文件里自己识别。
    """
    import os
    import json
    import cv2

    def analyze_live_frame():
        """看摄像头【此刻】的画面"""
        frame = live_frame_getter()
        if frame is None:
            return json.dumps({"error": "摄像头没有画面（可能没开或刚断开）"},
                              ensure_ascii=False)

        # 把此刻这一帧落盘，方便 AI / 你事后查看
        path = agent.LIVE_FRAME
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, frame)

        # 直接对这张图跑 YOLO（不经过 detect_objects，避免路径拼接问题）
        model = yolo_tools.get_model()
        predicts = model.predict(source=path, save=False, verbose=False)
        img_h, img_w = frame.shape[:2]
        detections = []
        for result in predicts:
            for box in result.boxes:
                x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                detections.append({
                    "object": model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [x1, y1, x2, y2],
                    "size": [round(x2 - x1, 1), round(y2 - y1, 1)],
                    "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                })
        summary = {}
        for d in detections:
            summary[d["object"]] = summary.get(d["object"], 0) + 1

        return json.dumps({
            "image": path,
            "image_size": [img_w, img_h],
            "total_objects": len(detections),
            "summary": summary,
            "details": detections,
            "hint": "这是刚刚抓取的最新画面",
        }, ensure_ascii=False)

    return {"analyze_live_frame": analyze_live_frame}
