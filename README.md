# 🚀 YOLO Harness - 自动化视觉识别系统

> 基于 YOLOv8n 的 Harness 架构演示项目 —— 让 AI 模型自动、持续、智能地运行

---

## 📌 项目简介

本项目是一个基于 **YOLOv8n** 的自动化视觉识别 Harness 系统，展示了如何让 AI 模型从"手动调用"升级为"自主运行"。

### 什么是 Harness？

**Harness（框架/平台）** 是一个让 AI 程序能够自动运行、自动决策、自动完成任务的系统。

| 普通 AI 程序 | Harness 系统 |
|-------------|-------------|
| 手动输入命令 → 运行一次 → 结束 | 启动后自动循环运行 |
| 一次处理一张图 | 持续监控，自动发现新任务 |
| 需要人工盯着 | 自主运行，只需启动 |
| 只能识别 | 可以调用其他工具、保存结果、做决策 |

---

## 🎯 核心功能

本项目提供三种运行模式：

### 1️⃣ 单次识别模式（detect.py）

识别指定文件夹中的所有图片，适合批量处理。

```bash
python detect.py

```


特点：

遍历 images/ 文件夹
识别所有图片
输出识别结果到控制台
完成后自动结束


### 2️⃣ 文件夹监控模式（harness.py）

持续监控文件夹，自动发现并识别新图片。

```bash
python harness.py

```
特点：

每 5 秒自动检查新图片
只处理未识别过的图片（不重复）
结果保存为 JSON 文件
自动记录运行日志
按 Ctrl+C 停止

输出示例：

[2026-10-01 18:16:43] 模型加载完成！
--- 第 1 次循环 ---
📷 发现 3 张新图片
正在识别：日常场景图 (4).png
✅ 识别完成，检测到 17 个目标，结果已保存


### 3️⃣ 摄像头实时监控模式（harness_camera.py）
连接摄像头，实时捕获画面并识别。

```bash
python harness_camera.py

```

特点：

实时显示摄像头画面
每 5 秒自动截取一帧并识别
保存带识别框的图片
同时保存 JSON 数据
按 Q 键 或 Ctrl+C 停止
输出文件：

output_results/frame_*.jpg —— 带识别框的截图
output_results/result_*.json —— 识别数据
🏗️ 项目结构
yolo_test/
├── harness.py              # 文件夹监控 Harness
├── harness_camera.py       # 摄像头监控 Harness
├── detect.py               # 单次识别脚本
├── yolov8n.pt              # YOLO 模型文件
├── images/                 # 输入图片文件夹
├── output_results/         # 识别结果输出
│   ├── result_*.json       # JSON 数据
│   └── frame_*.jpg         # 带框图片（摄像头模式）
├── logs/                   # 运行日志
│   └── harness_*.log
└── venv/                   # Python 虚拟环境


🛠️ 技术栈

Python 3.11.5

Ultralytics YOLOv8

YOLOv8n 模型

OpenCV（摄像头视频流处理）

📦 安装与运行
环境准备




# 进入项目目录

```bash
cd D:\desktop\yolo

```


