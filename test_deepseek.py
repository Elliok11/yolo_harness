import os
from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 文件里读取 API Key
load_dotenv()

# 创建一个连接 DeepSeek 的客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"   # 指向 DeepSeek 而不是 OpenAI
)

# 发送第一条消息
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "你好！请用一句话介绍你自己。"}
    ]
)

# 打印模型的回复
print(response.choices[0].message.content)
