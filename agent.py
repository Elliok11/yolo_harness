"""
agent.py - Agent Harness 核心 + 三种运行模式合体
DeepSeek（大脑）+ 工具箱（手脚）+ 循环 = Agent

现在这个文件一个顶三个：
  1. 单次识别       —— 对应 main.py 菜单第 1 项
  2. 文件夹监控     —— 对应 main.py 菜单第 2 项
  3. 摄像头监控     —— 对应 main.py 菜单第 3 项
   +  聊天式 AI 助手 —— 用大白话指挥，AI 自己决定调哪个工具

两种用法：
  · 敲数字 1 / 2 / 3 / 0（或输入「菜单」）→ 走菜单流程，和 main.py 完全一致
  · 敲别的话，比如「images 里有几个人」→ 交给 AI 处理

安全说明：菜单第 2、3 项原本是永不返回的死循环，这里统一收在"限量执行"
函数里 —— 跑够次数/秒数就自动交还控制权，绝不把你卡在程序里。
"""
import os
import json
import sys
import time
from datetime import datetime

# Windows 控制台默认可能用 GBK，会把中文显示成「浣犳槸…」这种乱码。
# 下面这 3 行把输入输出强制成 UTF-8，保证你看到的中文是正常的。
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from openai import OpenAI
import cv2
import yolo_tools

load_dotenv()

# ==================== 大模型 Provider 配置 ====================
# 支持三种后端，用环境变量 LLM_PROVIDER 或命令行 --provider 切换：
#
#   local  —— 本地 Ollama（默认）。完全离线、不花钱、数据不出本机。
#             前提：装好 Ollama 并拉过模型，例如 ollama pull qwen2.5:1.5b
#   cloud  —— DeepSeek 云端 API。更强，但需要联网和 API Key。
#   n8n    —— 经 n8n 编排平台中转（任务一.2 要求的"通过 API 接入应用"）。
#             请求路径变成：agent.py → n8n Webhook → n8n HTTP 节点 → 本地 Ollama
#
# 实测（本机 qwen2.5:1.5b）：普通对话 2.2 秒，带工具调用 5.1 秒，支持 function calling。
#
# 也可以单独覆盖任意一项：
#   LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
# 命令行覆盖（优先级最高）：
#   python agent.py --provider cloud
#   python agent.py --provider local --model qwen2.5:7b
_PROVIDERS = {
    "local": {"base_url": "http://127.0.0.1:11434/v1", "api_key": "ollama",
              "model": "qwen2.5:1.5b", "label": "本地 Ollama"},
    "cloud": {"base_url": "https://api.deepseek.com",
              "api_key": os.getenv("DEEPSEEK_API_KEY") or "",
              "model": "deepseek-chat", "label": "DeepSeek 云端"},
    "n8n": {"base_url": os.getenv("N8N_BASE_URL") or "http://127.0.0.1:5678",
            "api_key": os.getenv("N8N_API_KEY") or "",
            "model": os.getenv("OLLAMA_MODEL") or "qwen2.5:1.5b",
            "label": "n8n 中转",
            "webhook": os.getenv("N8N_WEBHOOK_PATH") or "agent"},
}

# 命令行 --provider / --model
_cli = sys.argv[1:]
def _arg(name):
    if name in _cli:
        i = _cli.index(name)
        if i + 1 < len(_cli):
            return _cli[i + 1]
    return None

PROVIDER = (_arg("--provider") or os.getenv("LLM_PROVIDER") or "local").strip().lower()
if PROVIDER not in _PROVIDERS:
    print("⚠️ 不认识的 provider：%s，回退到 local" % PROVIDER)
    PROVIDER = "local"
_cfg = _PROVIDERS[PROVIDER]

BASE_URL = _arg("--base-url") or os.getenv("LLM_BASE_URL") or _cfg["base_url"]
API_KEY = os.getenv("LLM_API_KEY") or _cfg["api_key"]
MODEL_NAME = _arg("--model") or os.getenv("LLM_MODEL") or _cfg["model"]
PROVIDER_LABEL = _cfg["label"] + ("（自定义地址）" if BASE_URL != _cfg["base_url"] else "")


class _N8nClient:
    """把 n8n 的 Webhook 包装成「看起来像 OpenAI 客户端」

    为什么要做包装？
      这样 run_agent_turn 里那行 client.chat.completions.create(...) 一个字都不用改，
      三种后端共用同一段代码 —— 换后端只是换一个对象。

    代价（重要）：
      n8n 那条工作流是【纯文本】接口，接收 {"message": "..."}，返回模型回复。
      它没有工具调用的能力，所以 n8n 模式下：
        · 可以正常聊天问答
        · 不能调用 list_images / detect_objects 等工具
      这不是偷懒，而是这条路的真实边界 —— 详细讨论见 docs/工程日志.md
    """

    def __init__(self, base, webhook_path, model, timeout=300):
        self.webhook_path = webhook_path
        self.webhook_url = base.rstrip("/") + "/webhook/" + webhook_path.lstrip("/")
        self.model = model
        self.timeout = timeout
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer):
            self.completions = _N8nClient._Completions(outer)

    class _Completions:
        def __init__(self, outer):
            self.outer = outer

        def create(self, model=None, messages=None, tools=None, **kwargs):
            import urllib.request
            import urllib.error

            # 把 messages 拍平成一段文本；system 提示单独抽出来，因为 n8n 那边
            # 已经写好了自己的 system 提示，不该重复塞
            parts = []
            for m in (messages or []):
                role = m.get("role")
                content = m.get("content")
                if role == "system" or not content:
                    continue
                if role == "user":
                    parts.append(str(content))
                elif role == "assistant":
                    parts.append("（我之前的回答）" + str(content))
            prompt = "\n\n".join(parts[-6:]) or "你好"

            payload = json.dumps({"message": prompt}).encode("utf-8")
            req = urllib.request.Request(
                self.outer.webhook_url, data=payload, method="POST",
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.outer.timeout) as r:
                    raw = r.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:200]
                if e.code == 404:
                    raise RuntimeError(
                        "n8n 的 Webhook 返回 404。常见原因：\n"
                        "  1) 工作流没发布（在界面上点 Publish）\n"
                        "  2) path 不对，当前用的是 '%s'\n"
                        "  查一下：python create_workflow.py" % self.outer.webhook_path)
                raise RuntimeError("n8n 返回 HTTP %d：%s" % (e.code, detail))
            except Exception as exc:
                raise RuntimeError(
                    "连不上 n8n（%s）。请确认：\n"
                    "  1) n8n 在跑（双击 启动n8n.bat）\n"
                    "  2) 工作流已发布\n"
                    "  3) 地址正确：%s" % (str(exc)[:80], self.outer.webhook_url))

            try:
                d = json.loads(raw)
            except Exception:
                raise RuntimeError("n8n 返回的不是 JSON：%s" % raw[:200])

            # 工作流返回的是 Ollama 的原始响应，回复在 message.content
            text = ""
            if isinstance(d, dict):
                text = ((d.get("message") or {}).get("content")
                        or (d.get("choices") or [{}])[0].get("message", {}).get("content")
                        or "")
            if not text:
                text = "（n8n 返回了空回复，原始内容：%s）" % raw[:150]

            class _Msg:
                def __init__(self, content):
                    self.content = content
                    self.tool_calls = None      # n8n 这条链路不支持工具调用
                    self.role = "assistant"

            class _Choice:
                def __init__(self, m):
                    self.message = m

            class _Resp:
                def __init__(self, m):
                    self.choices = [_Choice(m)]
                    self.usage = None

            return _Resp(_Msg(text))


if PROVIDER == "n8n":
    client = _N8nClient(BASE_URL, _cfg["webhook"], MODEL_NAME)
    # n8n 那条工作流一次只做一轮问答，没有工具循环，所以轮数上限压到 1
    AGENT_MAX_ROUNDS_HINT = 1
else:
    client = OpenAI(api_key=API_KEY or "not-needed", base_url=BASE_URL)
    AGENT_MAX_ROUNDS_HINT = None


def provider_info():
    """给界面/日志用的一句话说明，顺便做后端自检"""
    info = "模型后端：%s | 模型：%s | 地址：%s" % (PROVIDER_LABEL, MODEL_NAME, BASE_URL)

    if PROVIDER == "n8n":
        url = BASE_URL.rstrip("/") + "/webhook/" + _cfg["webhook"]
        info += "\n   调用链：agent.py → n8n → 本地 Ollama"
        info += "\n   Webhook：%s" % url
        info += "\n   ⚠️ 这条链路只能聊天问答，**不支持工具调用**（n8n 工作流是纯文本接口）"
        try:
            import urllib.request
            with urllib.request.urlopen(BASE_URL.rstrip("/") + "/healthz", timeout=8):
                pass
        except Exception:
            try:
                import urllib.request
                urllib.request.urlopen(BASE_URL, timeout=8)
            except Exception as exc:
                info += "\n   ⚠️ 连不上 n8n（%s）" % str(exc)[:60]
                info += "\n      请双击 启动n8n.bat，或用 python agent.py --provider local"
        return info

    if PROVIDER == "local":
        try:
            models = [m.id for m in client.models.list().data]
            if MODEL_NAME not in models:
                info += "\n⚠️ Ollama 在跑，但没有 %s 这个模型。现有：%s" % (MODEL_NAME, models)
                info += "\n   拉一个：ollama pull %s" % MODEL_NAME
                info += "\n   或换用已有的：python agent.py --model %s" % (models[0] if models else "模型名")
        except Exception as exc:
            info += "\n⚠️ 连不上本地 Ollama（%s）" % str(exc)[:60]
            info += "\n   请先启动 Ollama，或改用云端：python agent.py --provider cloud"
    return info
# ==============================================================

# 系统提示词：告诉 AI 它是谁、能干什么、以及【它有记忆】这件关键事实
SYSTEM_PROMPT = """你是一个视觉识别助手，代号"华小牛"。
你可以通过调用工具来查看和识别图片文件夹中的内容。

【重要：你有记忆，不要说反了】
你和用户的对话历史会被自动保存下来，每次启动都会完整加载给你。
所以当用户问"上次聊了什么""之前识别过什么"时，先去看你手里的对话历史，
能答就照着答，答不了就说"这段历史里没有相关记录"。
绝对不要说"我每次对话都是独立的""我读不到历史记录"这类话 —— 那是错的，
你确实能看到历史。如果历史里真的没有，就老实说没有。

【你的工具】
1. list_images         —— 看 images 文件夹里有哪些图片
2. detect_objects      —— 识别【一张】图片，返回物体、数量、以及每个物体的位置坐标(bbox/size/center)
3. detect_all_images   —— 把 images 里所有图一次性全认一遍，并保存带框标注图
4. run_folder_monitor  —— 文件夹监控，认完新图就自动结束（可指定轮数、间隔秒数）
5. run_camera_monitor  —— 摄像头实时监控（可指定看几秒、间隔、摄像头号）
6. run_camera_chat     —— 摄像头+人脸识别+对话：认出见过的人、登记陌生面孔、抓拍留证
7. list_camera_devices —— 查有哪些摄像头能用
8. write_report        —— 把分析结论存成 Markdown 报告
9. analyze_live_frame  —— 看摄像头【当前实时画面】（摄像头在跑的时候才有用）

【重要：用户在看摄像头时】
如果用户说"看看现在画面里有什么""分析当前画面"，
说明摄像头刚把画面存到了 output_results/frame_live.jpg，
你直接调用 detect_objects 识别 "frame_live.jpg" 就能看到实时画面。
人脸识别由摄像头程序本地负责，它的结果会随问题一起告诉你。

【用户提到"上次/之前"识别过的内容时】
历史里说过的就直接引用；如果历史里没有，可以用 detect_all_images 重新识别一遍，
但要讲清楚"这是我刚重新识别的"，不要假装那是上次的结果。

请一步步思考，自主决定调用哪个工具，最后用自然的中文回答用户。"""

# ==================== 运行护栏（防止 AI 越跑越上瘾） ====================
MAX_ROUNDS = 50        # 文件夹监控最多查多少轮
MAX_DURATION = 300     # 摄像头监控最多看多少秒
# 一轮提问里，最多允许 AI 连续调用工具几轮。
# 原来写死 8，遇到「把文件夹里所有图都识别一遍并总结」这种活，
# AI 可能一轮只认一张，8 轮就到了，活干一半就被打断。
# 提到 12 给它更多余地；超了也不会丢结果（工具结果都在历史里，说「继续」就能接着干）。
try:
    MAX_AGENT_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "12"))
except ValueError:
    MAX_AGENT_ROUNDS = 12
# =======================================================================

# ==================== 🥇 持久记忆：AI 的记忆不再一关就没 ====================
MEMORY_FILE = os.path.join(yolo_tools.OUTPUT_FOLDER, "agent_memory.json")
MEMORY_KEEP = 60       # 最多保留最近多少条消息（防止文件无限长大）
# ==========================================================================

# ==================== 🥉 决策留痕：AI 干了什么、花了多久 ====================
LOG_FOLDER = "logs"    # 和 harness.py / harness_camera.py 用的是同一个文件夹
DECISION_LOG = os.path.join(LOG_FOLDER, "agent_decisions.jsonl")
# 想亲眼看到「到底发了什么给 DeepSeek」，把下面这个改成 True，
# 原始请求会存到 logs/agent_last_request.json（排查记忆问题时很有用）
DEBUG_DUMP_REQUEST = False
DEBUG_REQUEST_FILE = os.path.join(LOG_FOLDER, "agent_last_request.json")
_SENSITIVE_KEYS = ("api_key", "apikey", "token", "secret", "password", "authorization")
# 这几个字段名里虽然带 token，但它们是「用量统计」，恰恰要留痕（方便算花了多少钱）
_TOKEN_STATS = ("prompt_tokens", "completion_tokens", "total_tokens")
_MAX_LOG_TEXT = 400    # 日志里长文本最多留多少字符
# ==========================================================================


def _redact(obj):
    """把敏感字段（密钥之类）挡在日志之外，顺便压掉超长文本"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k).lower()
            if key in _TOKEN_STATS:
                out[k] = v                      # 用量统计照常记录，不隐藏
            elif any(s in key for s in _SENSITIVE_KEYS):
                out[k] = "***已隐藏***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_redact(x) for x in obj]
    if isinstance(obj, str):
        return obj if len(obj) <= _MAX_LOG_TEXT else obj[:_MAX_LOG_TEXT] + "...(截断)"
    return obj


def _to_plain(obj):
    """把 openai 的 Pydantic 对象转成能存盘的普通字典/列表

    注意：绝不返回「字符串」来顶替一个对象 —— 否则字符串混进对话历史，
    存盘时就会崩（这个坑我踩过一次）。
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _to_plain(obj.model_dump(exclude_none=True))
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return _to_plain(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: _to_plain(v) for k, v in vars(obj).items()}
        except Exception:
            pass
    return {"_无法序列化的对象": type(obj).__name__}


def _drop_orphan_tool_messages(messages):
    """把「说了要调工具、却没有工具回复」的记录清干净。

    为什么必须这么做？
      DeepSeek（以及所有 OpenAI 格式的接口）有条硬规定：
        assistant 消息里只要带了 tool_calls，后面就必须紧跟对应的 tool 回复。
      一旦缺了，接口直接报 400：
        "An assistant message with 'tool_calls' must be followed by tool messages..."
      这种残缺记录怎么来的？调用工具的过程中被打断（比如 Ctrl+C、程序崩了、
      或者摄像头模式把终端占了），"要调工具"那句话留下了，回复永远没来。
      更麻烦的是它被存进记忆，之后每次启动都报错 —— 等于记忆被永久污染。

    处理办法：
      · 缺回复的 assistant 消息：把它的 tool_calls 摘掉（有正文就留正文，没正文就整条删）
      · 顺带把没有对应 tool_calls 的孤立 tool 回复也删掉
    """
    if not messages:
        return messages

    # 第一遍：找出哪些 tool_call 收到了回复；谁没收到就记下来
    answered = set()
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            answered.add(m["tool_call_id"])

    missing = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                tid = tc.get("id") if isinstance(tc, dict) else None
                if tid and tid not in answered:
                    missing.add(tid)

    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = [tc.get("id") for tc in m["tool_calls"] if isinstance(tc, dict)]
            if any(i in missing for i in ids):
                # 这条的工具调用没人回。有正文就当普通回答留着，没正文就丢掉
                body = (m.get("content") or "").strip()
                if body:
                    keep = {"role": "assistant", "content": m["content"]}
                    out.append(keep)
                continue
        if m.get("role") == "tool" and m.get("tool_call_id") not in answered:
            continue
        out.append(m)
    return out


def _prune_history(messages):
    """只保留最近 MEMORY_KEEP 条，并保证 tool 调用链完整。

    顺序很重要：先补 role → 再裁长度 → 最后清理断链。
    因为裁剪本身就是最容易把调用链切断的动作（切一半就断），所以清理必须放最后。
    """
    clean = [m for m in messages if isinstance(m, dict)]   # 防御：脏数据直接丢掉
    if not clean:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    # 防御：上游少给 role 时自己补上，否则这条会被误当垃圾丢掉
    for m in clean:
        if not m.get("role"):
            m["role"] = "assistant" if m.get("tool_calls") else "user"
    head = clean[:1]
    tail = clean[1:][-MEMORY_KEEP:]
    if not tail:
        return head
    # 裁剪可能把「要调工具」和「工具回复」切开，所以最后统一清理断链
    return _drop_orphan_tool_messages(head + tail)


def load_memory():
    """启动时把上次的对话读回来。

    顺手做一件事：如果发现记忆里有「残缺的工具调用」（会导致接口报 400），
    清理之后立刻写回文件 —— 让记忆自愈，不然那个坏记录会一直躺在盘上。
    """
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        msgs = data.get("messages") or []
        msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") in
                ("system", "user", "assistant", "tool")]
        if not msgs or msgs[0].get("role") != "system":
            return [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs[0] = {"role": "system", "content": SYSTEM_PROMPT}   # 提示词始终用最新的
        fixed = _prune_history(msgs)
        # 注意：不能用「条数变没变」来判断有没有修复。
        # 因为「说了要调工具却没回复」的那条如果带正文，我会保留正文、只摘掉工具调用，
        # 条数是不变的 —— 但内容确实变了，文件就得写回去。这个坑我踩过一次。
        if json.dumps(fixed, ensure_ascii=False) != json.dumps(msgs, ensure_ascii=False):
            print("🧹 记忆里有残缺的工具调用记录，已清理并写回")
            try:
                data["messages"] = fixed
                with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return fixed
    except FileNotFoundError:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    except Exception as exc:
        print("⚠️ 记忆文件读不出来（%s），这次当全新开始" % exc)
        return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_memory(messages):
    """每轮之后把对话存起来，下次启动接着用"""
    try:
        os.makedirs(yolo_tools.OUTPUT_FOLDER, exist_ok=True)
        data = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "messages": _prune_history(_to_plain(messages)),
        }
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print("⚠️ 记忆没存上：%s" % exc)


def log_decision(event, **fields):
    """往 logs/agent_decisions.jsonl 追加一行痕迹（一眼看出 AI 干了什么）"""
    try:
        os.makedirs(LOG_FOLDER, exist_ok=True)
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "event": event}
        rec.update(_redact(fields))
        with open(DECISION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def detect_images():
    """功能1：把 images 文件夹里的图全部识别一遍（对应 main.py 菜单第 1 项）"""
    from ultralytics import YOLO

    folder = "images"
    if not os.path.isdir(folder):
        return {"error": "images 文件夹不存在"}

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    if not files:
        return {"error": "images 文件夹里没有图片"}

    model = yolo_tools.get_model()
    results = []
    for filename in files:
        image_path = os.path.join(folder, filename)
        print("正在识别：" + filename)
        predicts = model.predict(source=image_path, save=True)
        names = []
        for result in predicts:
            for box in result.boxes:
                cls = int(box.cls[0])
                names.append(model.names[cls])
        counts = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1
        print("  检测到 %d 个目标" % len(names))
        results.append({"image": filename, "total": len(names), "summary": counts})

    print("识别完成！标注图保存在 runs 文件夹")
    return {
        "mode": "单次识别",
        "image_count": len(files),
        "results": results,
        "note": "带识别框的标注图已保存到 runs/detect/ 文件夹",
    }


def analyze_live_frame():
    """工具9：看摄像头当前的实时画面

    摄像头程序（控制台版或图形界面版）会把画面存到
    output_results/frame_live.jpg，这个工具就是去识别它。

    注意：这里不能调 yolo_tools.detect_objects()，因为它会把参数当作
    「images 文件夹里的文件名」去拼路径，而实时画面在 output_results/ 下。
    """
    path = LIVE_FRAME
    if not os.path.exists(path):
        return json.dumps({
            "error": "现在没有可用画面。摄像头没在运行，或者还没抓过帧。",
            "hint": "请先开启摄像头（菜单第 4 项，或图形界面版）。",
        }, ensure_ascii=False)

    import time as _time
    age = _time.time() - os.path.getmtime(path)
    model = yolo_tools.get_model()
    predicts = model.predict(source=path, save=False, verbose=False)
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
        "total_objects": len(detections),
        "summary": summary,
        "details": detections,
        "snapshot_age_seconds": round(age, 1),
        "hint": ("画面是 %.1f 秒前抓的" % age) if age > 5 else "画面是刚刚抓的",
    }, ensure_ascii=False)


def run_mode_2():
    """功能2：文件夹监控模式（对应 main.py 菜单第 2 项）"""
    import harness
    harness.main()


LIVE_FRAME = os.path.join("output_results", "frame_live.jpg")


def save_live_frame(frame):
    """把摄像头当前画面存成一个固定文件，这样 AI 也能「亲眼看」实时画面"""
    try:
        os.makedirs("output_results", exist_ok=True)
        cv2.imwrite(LIVE_FRAME, frame)
        return LIVE_FRAME
    except Exception:
        return None


def run_camera_chat(camera_source=0, duration=0, threshold=None):
    """摄像头 + 人脸识别 + 对话（菜单第 4 项）

    人脸识别能做的事：
      · 见过的人 → 绿框，标签写着「人A x3」（第 3 次见到）
      · 没见过的人 → 红框 + 控制台提醒 + 存一张抓拍到 output_results/new_faces/
      · 每隔一会儿落一次盘，中途 Ctrl+C 也不丢

    快捷键（焦点在摄像头窗口时按）：
      T = 进入打字模式（打完回车回到画面）
      F = 让 AI 分析当前画面
      R = 报一下现在画面里有谁
      S = 手动存一张当前画面
      Q = 退出

    duration=0 表示一直看到你按 Q；大于 0 则表示最多看多少秒。
    """
    import threading
    import queue
    from datetime import datetime as _dt

    try:
        import face_tools
    except Exception as exc:
        print("❌ 人脸模块加载失败：%s" % exc)
        return

    if not face_tools.models_ready():
        print("❌ 缺人脸模型文件，请确认 models/ 里有两个 .onnx：")
        print("   ", face_tools.DETECT_MODEL)
        print("   ", face_tools.RECOG_MODEL)
        return

    if isinstance(camera_source, str) and camera_source.strip().isdigit():
        camera_source = int(camera_source.strip())

    import harness_camera
    harness_camera.ensure_folders()

    print("=" * 62)
    print("🎥 摄像头 + 人脸识别 + 对话 已启动")
    print("   正在加载 YOLO 和人脸模型，稍等一下...")
    print("=" * 62)

    model = yolo_tools.get_model()
    tracker = face_tools.FaceTracker(threshold=threshold).load_library()

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("❌ 打不开摄像头（%s）。可能被别的程序占着。" % camera_source)
        print("   小技巧：先用菜单 3 或 list_camera_devices 试出能用的编号。")
        return

    tracker.reset_session()
    threshold_now = tracker.threshold

    print("✅ 都就绪了！")
    print("   脸库里现在有 %d 个人（想清空就删掉 %s）" % (len(tracker.library), face_tools.FACES_JSON))
    print("   快捷键：[T]打字提问  [F]让AI分析画面  [R]报现在有谁  [S]存图  [Q]退出")
    print("   （摄像头窗口要保持选中，按键才有效）")
    print("=" * 62)

    results = queue.Queue()
    typing = {"on": False, "stamp": 0.0}

    def start_typing():
        """后台起一个线程去接你的输入，这样画面不会因为等你打字而定住"""
        if typing["on"]:
            return
        typing["on"] = True
        typing["stamp"] = time.time()

        def worker():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                line = None
            results.put(line)
        threading.Thread(target=worker, daemon=True).start()

    def finish_typing(line):
        typing["on"] = False
        if line is None:
            return
        line = line.strip()
        if not line:
            return
        snap = save_live_frame(frame)
        print("你> " + line)
        if snap:
            print("   （已把当前画面存成 %s，AI 可以看它）" % snap)
        turn = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages[1:]:
            if m.get("role") in ("user", "assistant") and not m.get("tool_calls"):
                turn.append({"role": m["role"], "content": m.get("content")})
        turn.append({"role": "user", "content": line})
        try:
            reply = run_agent_turn(turn)
        except Exception as exc:
            reply = "（调用 AI 失败：%s）" % exc
            log_decision("camera_chat_error", error=str(exc))
        messages.append({"role": "user", "content": line})
        messages.append({"role": "assistant", "content": reply})
        save_memory(messages)
        print("\n华小牛> " + str(reply))
        print()
        typing["stamp"] = time.time()

    def describe_now():
        """不用 AI，本地立刻报一遍画面里有谁"""
        parts = []
        for pid, info in tracker.session.items():
            person = next((p for p in tracker.library.people if p["id"] == pid), None)
            if person:
                parts.append("%s（第%d次见）" % (pid, person.get("times_seen", 0)))
            else:
                parts.append(pid)
        print("📋 当前画面：%d 张脸%s" % (len(tracker.session),
                                     ("，" + "、".join(parts)) if parts else ""))

    messages = load_memory()
    last_info = {"new": [], "faces": 0}
    last_pending_state = None      # 用来判断「投票中/脸太小」的提示要不要重打
    start_time = time.time()
    last_save = time.time()
    frame = None

    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                print("⏱️  到时间了，自动收工")
                break

            ret, frame = cap.read()
            if not ret:
                print("⚠️  读不到摄像头画面，稍后重试")
                time.sleep(0.3)
                continue

            if not typing["on"]:
                info = tracker.process(frame)
                last_info = info
                for nf in info["new"]:
                    print("🆕 发现新面孔！编号 %s（相似度 %.3f）%s" % (
                        nf["id"], nf["score"],
                        "，抓拍：" + nf["snapshot"] if nf.get("snapshot") else ""))
                # 只在状态「变化」时提示，否则每帧都刷会淹掉有用的信息
                pend = info.get("pending", [])
                too_small = sum(1 for x in pend if x.get("reason") == "脸太小")
                voting = len(pend) - too_small
                state = (voting, too_small)
                if state != last_pending_state:
                    if voting:
                        print("   …有 %d 张新脸在投票中（连续 %d 帧确认后才登记）"
                              % (voting, face_tools.VOTE_FRAMES))
                    if too_small:
                        print("   …有 %d 张脸太小（<%d像素），不参与认人"
                              % (too_small, face_tools.MIN_FACE_SIZE))
                    last_pending_state = state
                if info["new"] and time.time() - last_save > 10:
                    tracker.library.save()
                    last_save = time.time()

            # 画一层状态信息
            hud = "faces:%d  known:%d  %s" % (
                len(tracker.session), len(tracker.library),
                "打字中..." if typing["on"] else "[T]type [F]AI [R]who [S]save [Q]quit")
            cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            try:
                cv2.imshow("YOLO Camera + Face", frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print("⚠️  当前环境弹不出窗口，已退出（人脸识别需要看画面）")
                break

            # 收到你打完的字
            try:
                line = results.get_nowait()
                print()   # 把你输入的那行和提示分开
                finish_typing(line)
                continue
            except queue.Empty:
                pass

            if key == ord('q') or key == ord('Q'):
                print("🛑 你按了 Q，收工")
                break
            elif key == ord('t') or key == ord('T'):
                start_typing()
            elif key == ord('f') or key == ord('F'):
                snap = save_live_frame(frame)
                print("🤖 让 AI 看看当前画面...（画面会定住几秒）")
                turn = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in messages[1:]:
                    if m.get("role") in ("user", "assistant") and not m.get("tool_calls"):
                        turn.append({"role": m["role"], "content": m.get("content")})
                turn.append({"role": "user", "content":
                             "这是我摄像头当前画面（已存成 %s）。请调用 detect_objects 看它，"
                             "并结合人脸识别结果告诉我画面里有什么。人脸识别结果：%s"
                             % (snap, json.dumps(
                                 {"在场人数": len(tracker.session),
                                  "脸库人数": len(tracker.library)}, ensure_ascii=False))})
                try:
                    reply = run_agent_turn(turn)
                except Exception as exc:
                    reply = "（调用 AI 失败：%s）" % exc
                print("\n华小牛> " + str(reply) + "\n")
            elif key == ord('r') or key == ord('R'):
                describe_now()
            elif key == ord('s') or key == ord('S'):
                p = save_live_frame(frame)
                stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
                manual = os.path.join("output_results", "frame_manual_%s.jpg" % stamp)
                try:
                    cv2.imwrite(manual, frame)
                    print("📁 已存图：" + manual)
                except Exception as exc:
                    print("⚠️ 存图失败：%s" % exc)

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C，收工")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if len(tracker.library):
            tracker.library.save()
        print("📊 这场直播一共见到 %d 张脸，脸库里现在有 %d 个人"
              % (len(tracker.session), len(tracker.library)))
        print("   脸库文件：%s" % face_tools.FACES_JSON)
        print("   事件日志：%s" % face_tools.FACE_LOG)
        print("   新人抓拍：%s" % face_tools.NEW_FACE_DIR)


def run_mode_3(camera_source=None):
    """功能3：摄像头监控模式（对应 main.py 菜单第 3 项）"""
    import harness_camera
    print("提示：摄像头画面窗口里按 Q 键，或 Ctrl+C，都能停下来")
    if camera_source is None or camera_source == "":
        harness_camera.main()
    elif isinstance(camera_source, str) and camera_source.isdigit():
        harness_camera.main(camera_source=int(camera_source))
    else:
        harness_camera.main(camera_source=camera_source)


def print_menu():
    """菜单：前 3 项和 main.py 一致，第 4 项是人脸+对话"""
    print("=" * 45)
    print("   🚀 YOLO Harness 启动器")
    print("=" * 45)
    print("  1. 单次识别（识别 images 文件夹里的图片）")
    print("  2. 文件夹监控模式（自动识别新图片）")
    print("  3. 摄像头监控模式（内置摄像头 / 手机摄像头）")
    print("  4. 🎥 摄像头 + 人脸识别 + 对话（新）")
    print("  0. 退出")
    print("  -----------------------------------------")
    print("  💬 直接打中文也能用，例如：images 里有几个人？")
    print("     想看菜单就输入：菜单")
    print("=" * 45)


# 工具说明书（告诉 DeepSeek 有哪些工具可用、怎么用）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_images",
            "description": "查看 images 文件夹里有哪些图片，返回图片列表和数量",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_objects",
            "description": "用 YOLOv8 识别一张图片，返回图中所有物体、数量统计和置信度",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_name": {
                        "type": "string",
                        "description": "要识别的图片文件名"
                    }
                },
                "required": ["image_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_all_images",
            "description": "把 images 文件夹里的所有图片一次性全部识别，并保存带框的标注图。用户说『全认一遍』『都看看』『批量识别』时用它",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_folder_monitor",
            "description": "启动文件夹监控：每隔几秒检查 images 文件夹，自动识别新出现的图片。会自动结束并把结果返回给你，传参决定跑多少轮",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_rounds": {
                        "type": "integer",
                        "description": "最多检查几轮，用户说『跑5次』就填5。不填默认1轮"
                    },
                    "interval": {
                        "type": "number",
                        "description": "每轮之间隔几秒，不填默认5秒"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_camera_monitor",
            "description": "启动摄像头实时监控：按固定间隔抓取画面并识别，自动保存带框图片和JSON，到时间自动结束并把结果返回给你。用户说『开摄像头看20秒』时用它",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "总共看多少秒，用户说『看30秒』就填30。不填默认10秒"
                    },
                    "interval": {
                        "type": "number",
                        "description": "每隔几秒抓一帧，不填默认5秒"
                    },
                    "camera": {
                        "type": "string",
                        "description": "摄像头编号（0是内置摄像头）或手机摄像头地址，不填默认0"
                    },
                    "show_window": {
                        "type": "boolean",
                        "description": "是否弹出实时画面窗口，不填默认true；用户说『别弹窗』就填false"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_camera_chat",
            "description": "启动「摄像头 + 人脸识别 + 对话」模式：实时识别人脸，见过的人认出来，没见过的人自动登记并抓拍。用户说『开摄像头认人』『看摄像头有没有陌生人』时用它。注意：这个模式会接管画面窗口，运行期间需要用户按 Q 才能停止",
            "parameters": {
                "type": "object",
                "properties": {
                    "camera": {
                        "type": "string",
                        "description": "摄像头编号或地址，不填默认0（内置摄像头）"
                    },
                    "duration": {
                        "type": "number",
                        "description": "最多看多少秒，不填默认0表示一直看到用户按Q"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "判定是否同一个人的相似度阈值，默认0.40。调大更严格（不容易认错人），调小更宽松"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_camera_devices",
            "description": "试探 0～4 号摄像头哪个能用，并列出可用设备。用户说『有哪些摄像头』『摄像头能打开吗』时用它",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": "把分析内容保存为 Markdown 报告文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要保存的报告内容（Markdown 格式）"
                    }
                },
                "required": ["content"]
            }
        }
    }
]

# 工具名 → 真实函数的映射表
TOOL_MAP = {
    "list_images": yolo_tools.list_images,
    "detect_objects": yolo_tools.detect_objects,
    "detect_all_images": yolo_tools.detect_all_images,
    "run_folder_monitor": yolo_tools.run_folder_monitor,
    "run_camera_monitor": yolo_tools.run_camera_monitor,
    "run_camera_chat": run_camera_chat,
    "list_camera_devices": yolo_tools.list_camera_devices,
    "write_report": yolo_tools.write_report,
    "analyze_live_frame": analyze_live_frame,
}


def run_agent():
    print("=" * 60)
    print("🤖 华小牛 已启动（%s 大脑 + YOLO 眼睛）" % PROVIDER_LABEL)
    print("   " + provider_info().replace("\n", "\n   "))
    print("   敲 1/2/3/4 用菜单，敲中文直接聊天，敲 q 退出")
    print("   换了后端想清空记忆：删掉 %s" % MEMORY_FILE)
    print("=" * 60)
    print_menu()

    # 对话历史（Agent 的"记忆"）—— 🥇 从文件里读回来，不再是白纸
    messages = load_memory()
    remembered = len(messages) - 1
    if remembered > 0:
        print("🧠 已读回上次的 %d 条对话记录（想重新开始就删掉 %s）" % (remembered, MEMORY_FILE))
    last_choice = None

    while True:
        user_input = input("\n你> ").strip()

        # ---------- 情况0：快捷键，直接复用上一次的菜单选择 ----------
        if user_input in ("", "r", "R", "重复", "再来一次"):
            if last_choice == "1":
                print("🔁 再来一遍单次识别")
                print(json.dumps(detect_images(), ensure_ascii=False, indent=2)[:300])
                continue
            if last_choice == "2":
                print("🔁 再来一遍文件夹监控")
                run_mode_2()
                continue
            if last_choice == "3":
                print("🔁 再来一遍摄像头监控")
                run_mode_3()
                continue
            continue

        # ---------- 情况0.5：主动看菜单 ----------
        if user_input in ("菜单", "menu", "帮助", "help", "?"):
            print_menu()
            continue

        if user_input.lower() == "q":
            save_memory(messages)
            print("👋 再见！（对话已存到 %s）" % MEMORY_FILE)
            break

        # ---------- 情况1：敲数字 → 走 main.py 那套菜单流程 ----------
        if user_input == "1":
            last_choice = "1"
            print(detect_images())
            continue

        if user_input == "2":
            last_choice = "2"
            run_mode_2()   # 内部是限次数版本，跑够就自动回来
            continue

        if user_input == "3":
            last_choice = "3"
            source = input(
                "摄像头地址（直接回车 = 内置摄像头，\n"
                "手机摄像头示例：http://192.168.1.5:8080/video）："
            ).strip()
            run_mode_3(source)
            continue

        if user_input == "4":
            last_choice = "4"
            source = input(
                "摄像头地址（直接回车 = 内置摄像头，\n"
                "手机摄像头示例：http://192.168.1.5:8080/video）："
            ).strip()
            secs = input("最多看多少秒（直接回车 = 一直看，按 Q 结束）：").strip()
            dur = float(secs) if secs.replace(".", "", 1).isdigit() else 0
            run_camera_chat(camera_source=source or 0, duration=dur)
            continue

        if user_input == "0":
            save_memory(messages)
            print("👋 再见！（对话已存到 %s）" % MEMORY_FILE)
            break

        # ---------- 情况2：说人话 → 交给 AI 大脑 ----------
        messages.append({"role": "user", "content": user_input})
        try:
            answer = run_agent_turn(messages)
        except KeyboardInterrupt:
            print("\n（这一轮被你打断了，之前的对话仍然记着）")
            save_memory(messages)
            break
        except Exception as exc:
            # 🥉 网络/额度/密钥出问题也要留痕，方便你事后查
            print("\n⚠️ 调用 AI 出错了：%s" % exc)
            log_decision("turn_error", error=str(exc))
            save_memory(messages)
            continue
        if answer:
            print("\n华小牛> " + answer)
        save_memory(messages)   # 🥇 每轮都存，不怕意外断电


def run_agent_turn(messages, tool_map=None):
    """AI 大脑的一轮思考：最多 8 回合，防止 AI 无限调用工具

    tool_map 可以不传（默认用本文件的 TOOL_MAP）；
    图形界面版会传自己的工具表进来，共享同一个大脑。
    """
    tools = TOOL_MAP if tool_map is None else tool_map
    t_turn = time.time()
    log_decision("turn_start", user=messages[-1].get("content", ""))

    # ★ 发请求前先体检一次调用链。
    # 一旦出现「说了要调工具却没回复」的记录，DeepSeek 会直接回 400 把整轮对话打断。
    # 与其让你看到报错，不如在这里就地清掉病根（不改动你原本的对话内容）。
    cleaned = _drop_orphan_tool_messages(messages)
    if len(cleaned) != len(messages):
        dropped = len(messages) - len(cleaned)
        messages[:] = cleaned
        log_decision("auto_fix_history", dropped=dropped,
                     note="清理了残缺的工具调用记录，避免接口报400")
        print("  ⚠️ 检测到 %d 条残缺的工具调用记录，已自动清理（避免接口报错）" % dropped)

    for round_num in range(MAX_AGENT_ROUNDS):
        t_api = time.time()
        # 可选：把这次真正发出去的请求原样存一份，方便排查「记忆到底发没发出去」
        if DEBUG_DUMP_REQUEST:
            try:
                os.makedirs(LOG_FOLDER, exist_ok=True)
                with open(DEBUG_REQUEST_FILE, "w", encoding="utf-8") as f:
                    json.dump(_redact(messages), f, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
        )
        api_ms = int((time.time() - t_api) * 1000)
        msg = response.choices[0].message

        usage = getattr(response, "usage", None)
        log_decision(
            "llm_call", round=round_num + 1, took_ms=api_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

        # 情况 1：AI 决定调用工具
        if msg.tool_calls:
            plain = _to_plain(msg)
            # 保险：万一上游返回里缺 role，历史就会变成非法结构（接口报 422）。
            # 正常 OpenAI/DeepSeek 返回都带 role="assistant"，但防御一下不亏。
            if isinstance(plain, dict) and not plain.get("role"):
                plain["role"] = "assistant"
            messages.append(plain)  # 记住 AI 的决定（转成纯字典，才能存盘）

            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name

                # 安全处理参数（参数为空时不会崩溃）
                if tool_call.function.arguments:
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}
                        log_decision("bad_arguments", tool=func_name,
                                     raw=str(tool_call.function.arguments))
                else:
                    func_args = {}

                print("  🔧 [第%d轮] AI 调用: %s(%s)" % (round_num + 1, func_name, func_args))
                log_decision("tool_call", round=round_num + 1, tool=func_name,
                             tool_call_id=tool_call.id, args=func_args)

                # 安全执行工具（AI 幻觉出不存在的工具名时不会崩溃）
                t_tool = time.time()
                if func_name in tools:
                    try:
                        result = tools[func_name](**func_args)
                    except Exception as exc:
                        result = json.dumps(
                            {"error": "工具执行出错：%s" % exc}, ensure_ascii=False
                        )
                else:
                    result = json.dumps({"error": "没有这个工具: %s" % func_name}, ensure_ascii=False)
                tool_ms = int((time.time() - t_tool) * 1000)

                print("  📋 [结果] %s..." % result[:150])
                log_decision("tool_result", tool=func_name, took_ms=tool_ms,
                             result_size=len(result), preview=result[:200])

                # 把结果喂回给 AI
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            continue  # 让 AI 看到结果后继续思考

        # 情况 2：AI 给出最终回答
        else:
            messages.append({"role": "assistant", "content": msg.content})
            log_decision("turn_end", rounds=round_num + 1,
                         took_ms=int((time.time() - t_turn) * 1000), reply=msg.content)
            return msg.content

    # 走到这里 = 工具调用轮数用满了，还没轮到 AI 说最终答复。
    # 这种情况绝不能糊弄过去，因为：
    #   · 工具已经真跑过了（可能识别了几十张图），这些结果都已经写进对话历史
    #   · 只是 AI 没来得及做收尾总结
    # 所以提示必须说清两件事：① 活干了一半 ② 直接说「继续」就能接着干。
    # 早期版本这里写的是"你可以再说一次，或者换个说法"，那会误导用户重新提问，
    # 等于把已经干完的活又做一遍。
    note = ("（我这轮连续调用工具 %d 次，达到单轮上限，先停一下。\n"
            "已经执行的结果都保留着，**你直接说「继续」我就接着往下做**，\n"
            "不用重新提问或换说法。如果想让单轮能做更多事，"
            "可以设环境变量 AGENT_MAX_ROUNDS 调大上限。）" % MAX_AGENT_ROUNDS)
    messages.append({"role": "assistant", "content": note})   # 记进历史，下一轮能接上
    log_decision("turn_end", rounds=MAX_AGENT_ROUNDS, hit_round_limit=True,
                 took_ms=int((time.time() - t_turn) * 1000), reply=note)
    return note


if __name__ == "__main__":
    run_agent()
