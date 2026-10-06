"""
yolo_tools.py - 工具箱
把 YOLO 的能力包装成标准工具，供 AI 大脑（DeepSeek）调用
"""
import os
import json
from datetime import datetime
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "output_results"

# 模型只加载一次，重复使用（省时间）
_model = None
def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def list_images():
    """工具1：查看 images 文件夹里有哪些图片"""
    if not os.path.exists(IMAGE_FOLDER):
        return json.dumps({"error": "images 文件夹不存在"}, ensure_ascii=False)
    files = [f for f in os.listdir(IMAGE_FOLDER)
             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    return json.dumps({"count": len(files), "images": files}, ensure_ascii=False)


def detect_objects(image_name):
    """工具2：用 YOLO 识别一张图片，返回物体和数量统计"""
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    if not os.path.exists(image_path):
        return json.dumps({"error": f"找不到图片 {image_name}，请先调用 list_images 查看可用图片"}, ensure_ascii=False)

    model = get_model()
    results = model.predict(source=image_path, save=False, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({"object": model.names[cls], "confidence": round(conf, 3)})

    # 统计每种物体出现几次
    summary = {}
    for d in detections:
        summary[d["object"]] = summary.get(d["object"], 0) + 1

    return json.dumps({
        "image": image_name,
        "total_objects": len(detections),
        "summary": summary,
        "details": detections
    }, ensure_ascii=False)


def write_report(content):
    """工具3：把 AI 的分析结论保存成报告文件"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUT_FOLDER, f"report_{timestamp}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return json.dumps({"success": True, "saved_to": filepath}, ensure_ascii=False)


# ============ 单独运行本文件时，测试工具是否正常 ============
if __name__ == "__main__":
    print("=== 测试工具1：列出图片 ===")
    print(list_images())

    print("\n=== 测试工具2：识别第一张图片 ===")
    imgs = json.loads(list_images())
    if imgs.get("images"):
        print(detect_objects(imgs["images"][0]))
    else:
        print("images 文件夹是空的，先放几张测试图！")
