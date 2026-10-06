"""
agent.py - Agent Harness 核心
DeepSeek（大脑）+ 工具箱（手脚）+ 循环 = Agent
"""
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import yolo_tools

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 系统提示词：告诉 AI 它是谁、能干什么
SYSTEM_PROMPT = """你是一个视觉识别助手，代号"华小牛"。
你可以通过调用工具来查看和识别图片文件夹中的内容。

你的工作流程：
1. 需要了解有哪些图片时，调用 list_images
2. 需要识别图片内容时，调用 detect_objects
3. 用户要求保存分析报告时，调用 write_report

请一步步思考，自主决定调用哪个工具，最后用自然的中文回答用户。"""

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
    "write_report": yolo_tools.write_report,
}


def run_agent():
    print("=" * 50)
    print("🤖 华小牛 Agent 已启动（DeepSeek + YOLO Harness）")
    print("输入你的任务，输入 q 退出")
    print("=" * 50)

    # 对话历史（Agent 的"记忆"）
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("\n你> ").strip()
        if user_input.lower() == "q":
            print("👋 再见！")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # ===== Agent Loop 核心：最多 8 轮，防止 AI 无限调用工具 =====
        for round_num in range(8):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message

            # 情况 1：AI 决定调用工具
            if msg.tool_calls:
                messages.append(msg)  # 记住 AI 的决定

                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name

                    # 安全处理参数（参数为空时不会崩溃）
                    if tool_call.function.arguments:
                        func_args = json.loads(tool_call.function.arguments)
                    else:
                        func_args = {}

                    print(f"  🔧 [第{round_num+1}轮] AI 调用: {func_name}({func_args})")

                    # 安全执行工具（AI 幻觉出不存在的工具名时不会崩溃）
                    if func_name in TOOL_MAP:
                        result = TOOL_MAP[func_name](**func_args)
                    else:
                        result = json.dumps({"error": f"没有这个工具: {func_name}"}, ensure_ascii=False)

                    print(f"  📋 [结果] {result[:150]}...")

                    # 把结果喂回给 AI
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                continue  # 让 AI 看到结果后继续思考

            # 情况 2：AI 给出最终回答
            else:
                print(f"\n华小牛> {msg.content}")
                messages.append({"role": "assistant", "content": msg.content})
                break


if __name__ == "__main__":
    run_agent()
