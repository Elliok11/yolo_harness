"""
create_workflow.py - 用 API 在 n8n 里创建「本地大模型接口」工作流

为什么要用代码建工作流，而不是在界面上点？
  1. 界面点出来的东西存在 n8n 的数据库里，别人 clone 你的仓库看不到
  2. 写成代码 + 存成 JSON，别人一条命令就能重建同样的流程（可复现）
  3. 顺便演示了「用 API 管理 n8n」这个能力

工作流结构（只有两个节点，故意保持简单）：
  [Webhook]  ──>  [HTTP Request 调 Ollama]
     ↑                    ↑
   你的程序发请求      问本地模型
   POST /webhook/agent  POST /api/chat

用法：
  python create_workflow.py            # 创建（如果已存在就更新）
  python create_workflow.py --delete   # 删掉它
"""
import os
import sys
import json
import argparse
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import dotenv_values
except ImportError:
    print("需要 python-dotenv：pip install python-dotenv")
    sys.exit(1)

ENV = dotenv_values(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
                    encoding="utf-8")
N8N_BASE = (ENV.get("N8N_BASE_URL") or "http://127.0.0.1:5678").strip()
N8N_KEY = (ENV.get("N8N_API_KEY") or "").strip()
OLLAMA_BASE = (ENV.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434/v1").strip()
# Ollama 的 OpenAI 兼容接口基址是 .../v1，去掉 /v1 得到原生接口基址
OLLAMA_NATIVE = OLLAMA_BASE[:-3] if OLLAMA_BASE.endswith("/v1") else OLLAMA_BASE
OLLAMA_MODEL = (ENV.get("OLLAMA_MODEL") or "qwen2.5:1.5b").strip()

WORKFLOW_NAME = "本地大模型接口 (Ollama)"
WEBHOOK_PATH = "agent"
SYSTEM_PROMPT = "你是一个视觉识别助手，代号华小牛。回答要简洁、口语化，用中文。"


def api(method, path, payload=None):
    """调 n8n 的 REST API"""
    url = N8N_BASE + "/api/v1" + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "X-N8N-API-KEY": N8N_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode("utf-8")
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError("HTTP %d: %s" % (e.code, detail[:500]))


def find_existing(name):
    """按名字找已存在的工作流，返回 id 或 None"""
    d = api("GET", "/workflows?limit=250")
    for w in d.get("data", []):
        if w.get("name") == name:
            return w.get("id")
    return None


def build_definition():
    """构造工作流的 JSON 定义"""

    # ---- 节点 1：Webhook（对外暴露一个网址）----
    webhook = {
        "parameters": {
            "httpMethod": "POST",
            "path": WEBHOOK_PATH,
            # When Last Node Finishes = 把最后一个节点的输出直接当 HTTP 响应返回，
            # 这样就不需要额外的 Respond to Webhook 节点，少一层出错的可能
            "responseMode": "lastNode",
            "options": {},
        },
        "id": "webhook-node",
        "name": "Webhook",
        "type": "n8n-nodes-base.webhook",
        "typeVersion": 2,
        "position": [0, 0],
        "webhookId": "agent-webhook",
    }

    # ---- 节点 2：HTTP Request 调 Ollama ----
    # 关键点：jsonBody 整体写成「求值表达式」，用 ={ ... } 包住。
    #
    # 为什么不能只在内层写 "={{ $json.body.message }}"？
    #   我第一版就是那么写的，结果 n8n 把那一整串当成【普通字符串】原样发了出去，
    #   Ollama 收到的是字面量 "={{ $json.body.message }}"，模型一脸懵，
    #   回了句「你可能是输入了错误的信息」。
    #   正确做法是让整个 body 成为一个表达式对象，n8n 才会真正求值。
    http_node = {
        "parameters": {
            "method": "POST",
            "url": OLLAMA_NATIVE + "/api/chat",
            "sendBody": True,
            "specifyBody": "json",
            "jsonBody": "={{ {\n"
                        "  model: '%s',\n"
                        "  stream: false,\n"
                        "  messages: [\n"
                        "    { role: 'system', content: '%s' },\n"
                        "    { role: 'user', content: $json.body.message }\n"
                        "  ]\n"
                        "} }}" % (OLLAMA_MODEL, SYSTEM_PROMPT.replace("'", "\\'")),
            "options": {"timeout": 300000},   # Ollama 在 CPU 上慢，给 5 分钟
        },
        "id": "ollama-node",
        "name": "问 Ollama",
        "type": "n8n-nodes-base.httpRequest",
        "typeVersion": 4.2,
        "position": [220, 0],
    }

    # 连线：Webhook 的输出 -> HTTP Request 的输入
    connections = {
        "Webhook": {
            "main": [[{"node": "问 Ollama", "type": "main", "index": 0}]]
        }
    }

    return {
        "name": WORKFLOW_NAME,
        "nodes": [webhook, http_node],
        "connections": connections,
        "settings": {"executionOrder": "v1"},
    }


def show_summary(wf, created):
    print()
    print("=" * 62)
    print("工作流%s成功" % ("创建" if created else "更新"))
    print("=" * 62)
    print("  名称   : %s" % wf.get("name"))
    print("  ID     : %s" % wf.get("id"))
    print("  节点数 : %d" % len(wf.get("nodes", [])))
    for n in wf.get("nodes", []):
        print("      · %-12s (%s)" % (n.get("name"), n.get("type")))
    print()
    print("  调用地址（POST）:")
    print("      %s/webhook/%s" % (N8N_BASE, WEBHOOK_PATH))
    print("  请求体格式:")
    print('      {"message": "你的问题"}')
    print()
    print("  打开界面看它:")
    print("      %s/workflow/%s" % (N8N_BASE, wf.get("id")))
    print()
    print("  ⚠️ 重要：要在界面上点右上角【Publish / Active】把它启用，")
    print("     否则生产地址（/webhook/）不会生效，只有测试地址能用。")


def main():
    ap = argparse.ArgumentParser(description="用 API 在 n8n 里创建工作流")
    ap.add_argument("--delete", action="store_true", help="删掉这个工作流")
    args = ap.parse_args()

    if not N8N_KEY:
        print("❌ .env 里没有 N8N_API_KEY。")
        print("   请到 n8n 界面 → Settings → n8n API → Create an API key，")
        print("   然后把那串 eyJ... 填进 .env 的 N8N_API_KEY=")
        return 1

    print("n8n 地址 : %s" % N8N_BASE)
    print("Ollama   : %s (模型 %s)" % (OLLAMA_NATIVE, OLLAMA_MODEL))

    try:
        existing = find_existing(WORKFLOW_NAME)
    except Exception as exc:
        print("❌ 连不上 n8n：%s" % exc)
        print("   请确认 n8n 在跑（双击 启动n8n.bat）")
        return 1

    if args.delete:
        if not existing:
            print("没找到名为「%s」的工作流，无需删除。" % WORKFLOW_NAME)
            return 0
        api("DELETE", "/workflows/%s" % existing)
        print("✅ 已删除工作流 %s" % existing)
        return 0

    definition = build_definition()
    if existing:
        print("已存在同名工作流（id=%s），将覆盖它的节点定义。" % existing)
        wf = api("PUT", "/workflows/%s" % existing, definition)
        created = False
    else:
        wf = api("POST", "/workflows", definition)
        created = True

    show_summary(wf, created)

    # 存一份 JSON 到仓库，方便别人重建
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "n8n_workflows", "local_llm_agent.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(definition, f, ensure_ascii=False, indent=2)
    print("  已导出定义到: %s" % os.path.relpath(out))
    print("  （这样别人 clone 仓库后，跑一次本脚本就能重建同样的工作流）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
