#  YOLO Harness — 自动化视觉识别 + 人脸识别 + 打标签训练

> 基于 YOLOv8n 的视觉识别系统。既能让 AI 自动、持续地跑，也能自己标数据训练专属模型。

---

##  这是什么

一个用**摄像头和图片做视觉识别**的项目，能力分四大块：

| 模块 | 能干什么 |
|---|---|
|  **识别** | 认图里的物体（80 类通用物体：人、车、狗、杯子…） |
|  **认人** | 人脸识别：认出熟人、登记陌生人、抓拍留证 |
|  **对话** | 接大模型，用中文指挥它干活（**默认走本地 Ollama**，也可切云端 DeepSeek） |
|  **训练** | 自己打标签 → 训练专属模型 |

### Harness 是什么

Harness 是让 AI 程序**自动运行、自动决策**的那层框架。区别在于：

| 普通 AI 程序 | Harness 系统 |
|---|---|
| 手动跑一次就结束 | 启动后自动循环 |
| 一次处理一张图 | 持续监控，自动发现新任务 |
| 需要人盯着 | 能自己决定下一步 |
| 只能识别 | 能调工具、存结果、做判断 |

本项目里 `agent.py` 的 **Agent Loop**（AI 自己决定调哪个工具）就是 harness 的核心。
另外 `harness.py` / `harness_camera.py` 里的"定时循环"是更朴素的自动化外壳 ——
两种"harness"含义不同，别混。

### 项目文档

| 文档 | 内容 |
|---|---|
| `README.md` | 你正在看的这份：怎么跑、怎么配 |
| [`docs/工程日志.md`](docs/工程日志.md) | **开发过程与 11 个真实踩坑**（现象→排查→根因→解决→教训） |
| [`docs/AI使用说明.md`](docs/AI使用说明.md) | AI 参与情况与协作方式（如实说明） |

---

##  四个入口

命令符快速开始：
```bash
d:
cd desktop/yolo_test
venv\Scripts\activate

```

### 1. 命令行主程序 `agent.py`（推荐入口）

```bash
python agent.py

```

启动后是**菜单 + 聊天**二合一：

```
  1. 单次识别（识别 images 文件夹里的图片）
  2. 文件夹监控模式（自动识别新图片）
  3. 摄像头监控模式（内置摄像头 / 手机摄像头）
  4.  摄像头 + 人脸识别 + 对话
  0. 退出
  -----------------------------------------
  💬 直接打中文也能用，例如：images 里有几个人？
```

* 敲 `1/2/3/4` → 走菜单
* 敲中文 → 交给 AI，它自己决定调哪个工具
* 输入 `菜单` → 随时调出菜单

### 2. 图形界面版 `camera_gui.py`（摄像头 + 对话）

双击 **`启动摄像头窗口.bat`**（不弹黑框）。

左边实时画面，右边聊天。**这个版本才能在开摄像头的同时打字提问** ——
因为摄像头跑在后台线程、画面是画进窗口的，不像命令行版会被 `cv2.imshow` 霸占主线程。

按钮：`分析当前画面` `现在有谁` `存图` `脸库名单` `存报告`

窗口左下角还能**实时调物体识别**：开关、换模型（yolov8n / yolov8s / yolo11s）、改分辨率，选择会被记住。

### 3. 打标签 `label_gui.py`

双击 **`启动打标签.bat`**，给训练准备数据。

* **空白处拖鼠标** = 画新框　**点框内部** = 选中后拖动　**拖角** = 改大小
* **A/D** 翻页　**Delete** 删框　**数字 1~9** 切类别　**画完自动存盘**
* ** 自动预标注** —— 先用现成模型画出它能认的，你只改错的，不用从零画

存的格式是标准 YOLO txt（`类别 中心x 中心y 宽 高`，都归一化到 0~1）。

### 4. 训练 `train_model.py`

```bash
python train_model.py --check     # 只看数据准备得怎么样，不训练
python train_model.py             # 开始训练
python train_model.py --resume    # 断了接着跑

```

六步流水线：查依赖 → 按 8:2 分训练/验证集 → 生成 `data.yaml` →
**量老模型成绩** → 训练 → **量新模型并自动对比**，最后告诉你"到底变准了没有"。

> ⚠️ **CPU 训练很慢**。参考：200 张图 × 100 轮，4 核 CPU 约 4~8 小时；有显卡则几十分钟。

### 5. n8n 工作流版 `create_workflow.py`（图形化编排）

除了自己写 Python 编排 Agent，这个项目**还用 n8n 搭了一条等价的路**，
把本地大模型包成一个 HTTP 接口：

```
你的程序 ──POST {"message":"..."}──> n8n Webhook ──> HTTP Request ──> 本地 Ollama
                                          ↑                              │
                                          └──────── 回答原路返回 ─────────┘
```

**两条命令就能重建整个工作流**（工作流定义以 JSON 存在仓库里，不依赖界面手点）：

```bash
python create_workflow.py           # 创建（已存在则更新）
python create_workflow.py --delete  # 删掉
```

**怎么调用它**（Python 示例）：

```python
import requests

r = requests.post("http://127.0.0.1:5678/webhook/agent",
                  json={"message": "用一句话介绍你自己"},
                  timeout=300)
print(r.json()["message"]["content"])
```

实测响应时间：首次 3.4 秒（模型要加载），之后稳定 **1.5 秒**。

**为什么用代码建工作流而不是在界面上点？**

| | 界面点出来 | 代码建（本项目） |
|---|---|---|
| 存在哪 | n8n 的数据库里，别人看不到 | **JSON 进 git，可 diff** |
| 别人能否复现 | ❌ | ✅ 跑一条命令就行 |

**需要的环境**：n8n 服务在跑（双击 `启动n8n.bat`），Ollama 服务在跑（双击 `启动Ollama服务.bat`），
并且 `.env` 里填好 `N8N_API_KEY`（在 n8n 界面 → Settings → n8n API 里创建）。

工作流定义文件：`n8n_workflows/local_llm_agent.json`

---

##  环境准备

### 1. 大模型后端（二选一，默认本地）

程序**默认使用本地 Ollama**，完全离线、不花钱、数据不出本机。
想用云端 DeepSeek 也可以随时切，命令加一个参数即可。

**方案 A：本地 Ollama（默认）**

```bash
# 1. 装 Ollama：https://ollama.com/download
# 2. 拉一个支持工具调用的模型
ollama pull qwen2.5:1.5b

# 3. 启动服务（Windows 用项目里的启动器，它带了必要的环境设置）
#    双击：启动Ollama服务.bat
#    或者手动：ollama serve

# 4. 什么都不用配，直接跑
python agent.py
```

> ⚠️ **如果你有 AMD 老显卡，必须关掉 Vulkan 再启动**，否则推理会崩（报 `0xc0000005`）。
> 项目里的 `启动Ollama服务.bat` 已经带上了 `set OLLAMA_VULKAN=false`，用它启动即可。
> 原因见 [工程日志 · 坑 10](docs/工程日志.md)。

Ollama 自带 OpenAI 兼容接口（`http://127.0.0.1:11434/v1`），
所以项目里用的还是标准的 `openai` 库，只是换了地址，没有引入新依赖。

本机实测（qwen2.5:1.5b，无显卡）：普通对话 2.2 秒，
**带工具调用的一轮 Agent 对话约 54 秒**。免费离线的代价就是慢。

**方案 B：云端 DeepSeek**

项目根目录建 `.env`：

```
DEEPSEEK_API_KEY=你的密钥
```

然后 `python agent.py --provider cloud`。`.env` 已在 `.gitignore` 里，不会被提交。

**切换方式汇总**

```bash
python agent.py                            # 默认：本地 Ollama + qwen2.5:1.5b
python agent.py --provider cloud           # 切云端 DeepSeek
python agent.py --provider n8n             # 经 n8n 中转（见下一节）
python agent.py --model qwen2.5:7b         # 换本地模型
python agent.py --base-url http://x/v1     # 换任意 OpenAI 兼容地址
```

也可以用环境变量：`LLM_PROVIDER` / `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY`。

启动了哪个后端一目了然，比如 n8n 模式：

```
模型后端：n8n 中转 | 模型：qwen2.5:1.5b | 地址：http://127.0.0.1:5678
   调用链：agent.py → n8n → 本地 Ollama
   Webhook：http://127.0.0.1:5678/webhook/agent
   ⚠️ 这条链路只能聊天问答，不支持工具调用（n8n 工作流是纯文本接口）
```

### 三种后端的能力对比

同一个 Agent，三种走法，能力**并不等价**：

| 后端 | 调用链 | 聊天问答 | **调用工具** | 速度 |
|---|---|---|---|---|
| `local` | 直连 Ollama | ✅ | ✅ **9 个工具全可用** | 1.5~8 秒 |
| `cloud` | 直连 DeepSeek | ✅ | ✅ 9 个工具全可用 | 1~3 秒 |
| `n8n` | 经 n8n 中转 | ✅ | ❌ **不支持** | 2~6 秒 |

**为什么 n8n 那条不支持工具调用？** 不是偷懒，是这条路的真实边界：

- 那条 n8n 工作流是**纯文本接口**：收 `{"message": "..."}`，返回模型回复；
- 工具调用需要 **多轮往返**（模型说"我要调工具" → 执行 → 结果喂回模型 → 再判断），
  而**图形化连线很难表达这种循环**；
- 你当然可以在 n8n 里把这个循环画出来，但会立刻变成一个很难维护的蜘蛛网 ——
  这正是"低代码"的适用边界。

所以本项目的定位是：
**自己能控制的、需要多轮决策的部分用代码写（`agent.py`）；
连接外部服务、需要图形化调试的部分用 n8n。**

详细讨论见 [工程日志 · 第六节](docs/工程日志.md)。

启动时程序会自检：**本地模型不存在、或 Ollama 没启动，它会直接告诉你该拉哪个模型、该启动什么**，
而不是丢一个看不懂的报错出来。

> 提示：本地小模型和云端大模型的差距是明显的。换了后端如果感觉回答变笨，那是正常的。

### 2. 装依赖

```bash
python -m venv venv
venv\Scripts\activate
pip install ultralytics openai python-dotenv pandas seaborn tqdm

```

### 3. 下载 YOLO 模型

程序会自动下载 `yolov8n.pt`。想更准可以手动下 `yolov8s.pt` / `yolo11s.pt` 放到项目根目录。

实测（3 张 2848×1600 图，合计检出目标数）：

| 模型 | imgsz | 检出数 | 高置信度 |
|---|---|---|---|
| yolov8n | 640 | 55 | 36 |
| yolov8n | 960 | 77 | 44 |
| **yolov8s** | **960** | **85（+55%）** | **46** |

### 4. 人脸模型（要用认人功能才需要）

人脸识别用的是 **OpenCV 自带的** YuNet + SFace，不用装额外库，但要下两个模型文件放进 `models/`：

```
models/face_detection_yunet_2023mar.onnx      (0.2 MB)
models/face_recognition_sface_2021dec.onnx    (37 MB)
```

下载地址（OpenCV Zoo 官方）：

```
https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
```



### 5. DeepSeek 密钥（只在用云端时才需要）

项目根目录建 `.env`：

```
DEEPSEEK_API_KEY=你的密钥
```

`.env` 已在 `.gitignore` 里，没提交。

---

##  人脸识别怎么工作

| 步骤 | 用什么 | 说明 |
|---|---|---|
| 找脸在哪 | YuNet | 检出人脸框和 5 个关键点 |
| 认这是谁 | SFace | 把脸变成 128 维特征向量 |
| 判断同一个人 | 余弦相似度 | 超过阈值（默认 0.40）算同一人 |

**实际调参时要注意的（都是实测结论）：**

* **脸太小认不准** —— 小于 60 像素的脸不参与比对，画面上显示灰框 `too small`。
  宁可说"太远"，也不要瞎认。
* **新面孔要连续 3 帧确认才登记**（多帧投票），避免单帧误判污染脸库。
* **"见面次数" = 走开又回来的次数**，不是被看到的帧数。离开超过 5 秒再出现才算新的一次。
* **阈值别调太低** —— 实测 0.30 时不同人之间会误认 13%，0.363 以上才收敛到 0。默认 0.40。

可调环境变量：

| 变量 | 默认 | 作用 |
|---|---|---|
| `FACE_THRESHOLD` | 0.40 | 判定同一人的严格度 |
| `FACE_MIN_SIZE` | 60 | 小于此像素数的脸不认 |
| `FACE_MIN_SCORE` | 0.5 | 人脸检测置信度下限 |
| `FACE_VOTE_FRAMES` | 3 | 新面孔要连续几帧确认 |
| `FACE_SESSION_GAP` | 5 | 隔几秒算"离开过" |
| `YOLO_MODEL` | yolov8n.pt | 物体识别用哪个模型 |
| `YOLO_IMGSZ` | 640 | 识别分辨率（调大更准更慢） |

---

##  目录结构

```
yolo_test/
│
├── 【入口程序】
│   ├── agent.py                        # 主程序：菜单 + AI 聊天 + Agent Loop（三种模型后端）
│   ├── camera_gui.py                   # 图形界面：摄像头 + 人脸识别 + 对话
│   ├── label_gui.py                    # 打标签窗口
│   ├── create_workflow.py              # 用 API 在 n8n 里创建/更新工作流
│   ├── train_model.py                  # 训练程序
│   ├── detect.py                       # 最早的批量识别脚本
│   ├── harness.py                      # 文件夹监控（定时循环）
│   ├── harness_camera.py               # 摄像头监控（定时循环）
│   └── main.py                         # 最早的菜单启动器
│
├── 【模块】
│   ├── chat_agent.py                   # 界面与 AI 之间的接线层
│   ├── face_tools.py                   # 人脸识别（检测/认人/脸库）
│   ├── label_data.py                   # 打标签的数据层（YOLO 格式读写）
│   └── yolo_tools.py                   # AI 可调用的 9 个工具
│
├── 【双击启动】（不用记命令）
│   ├── 启动Ollama服务.bat               # 启动本地大模型（已含必要环境设置）
│   ├── 启动n8n.bat                      # 启动 n8n 编排平台
│   ├── 启动摄像头窗口.bat / _调试.bat    # 图形界面版
│   └── 启动打标签.bat / _调试.bat        # 打标签工具
│
├── 【文档】
│   ├── README.md                       # 本文件
│   └── docs/
│       ├── 工程日志.md                  # 开发过程与 11 个真实踩坑
│       └── AI使用说明.md                # AI 参与情况（如实说明）
│
├── 【进仓库的配置】
│   └── n8n_workflows/local_llm_agent.json   # n8n 工作流定义（可复现）
│
└── 【不进仓库、需要自己准备的】
    ├── venv/            # Python 虚拟环境（1156 MB，别人自己建）
    ├── models/          # 人脸模型 onnx（37 MB，需自行下载）
    ├── images/          # 图片素材（含 coco/ 归档目录）
    ├── labels/          # 打标签结果
    ├── output_results/  # 识别输出、脸库、对话记忆
    └── logs/            # 运行日志、决策留痕
```

**仓库里只跟踪 24 个文件、约 246 KB**，全是代码和文档。
图片、模型、虚拟环境、运行产物一律不进 git（`.gitignore` 已排除）。

> 📌 目前 `images/` 是空的，`images/coco/` 里有 15 张 COCO 样例图。
> `detect.py` 和菜单第 1 项扫的是 `images/` **根目录**，
> 想直接跑批量识别的话，把图放进 `images/` 根目录即可。

---

## 🤖 AI 能调用的 9 个工具

| 工具 | 干什么 |
|---|---|
| `list_images` | 看 images 里有哪些图 |
| `detect_objects` | 认一张图（含位置坐标 bbox / center / size） |
| `detect_all_images` | 批量认完所有图 |
| `analyze_live_frame` | 看摄像头**此刻**的画面 |
| `run_folder_monitor` | 文件夹监控（限定轮数） |
| `run_camera_monitor` | 摄像头监控（限定秒数） |
| `run_camera_chat` | 摄像头 + 人脸识别 + 对话 |
| `list_camera_devices` | 查有哪些摄像头能用 |
| `write_report` | 把结论存成 Markdown 报告 |

---

##  设计上的几个坑（踩过才写进来。。。）

1. **AI 会跟自己的旧话保持一致** —— 如果记忆里存着它说"我没有记忆"，它之后会一直这么说。
   所以被污染的记忆必须清掉，光改提示词没用。
2. **工具调用链不能断** —— assistant 说了要调工具，就必须紧跟对应的 tool 回复，
   否则接口直接报 400。程序启动时会自动体检并清理残缺记录。
3. **控制台版没法和摄像头对话** —— `cv2.imshow` 必须在主线程，会挡住 `input()`。
   要边看边聊得用图形界面版。
4. **AI 怎么"看见"摄像头画面** —— 摄像头模式会把当前帧存成 `output_results/frame_live.jpg`，
   AI 通过看这个文件来获取实时画面。
5. **别用 `git add -A`** —— 项目里可能有上万张图片素材，会被整批提交进仓库。
   `images/`、`dataset/` 都已在 `.gitignore` 里排除。

---

##  更新日志

* **最新**：接入人脸识别（YuNet + SFace）、图形界面版、打标签工具、训练程序、
  多帧投票、小脸门槛、记忆自愈
* 早期：接入 DeepSeek 实现 LLM 驱动的 Agent Harness（Function Calling + Agent Loop）
* 最早：YOLOv8n 自动化视觉识别 Harness（菜单 + 文件夹监控 + 摄像头监控）



## 后续展望方向
* 增加语音系统，能为视障人士辨别人脸或者物品，能够实时语音交互且提供建议
* 将该系统嵌入食堂监控，智能提醒不文明占座行为
* ……
