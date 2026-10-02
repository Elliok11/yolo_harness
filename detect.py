from ultralytics import YOLO
import os

# 加载模型
model = YOLO("yolov8n.pt")

# 设置图片文件夹
image_folder = "images"

# 遍历文件夹里的所有图片
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        print(f"正在识别：{filename}")
        
        # 执行识别
        results = model.predict(source=image_path, save=True)
        
        # 打印识别结果
        for result in results:
            boxes = result.boxes
            print(f"  检测到 {len(boxes)} 个目标")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"    - {model.names[cls]} ({conf:.2f})")

print("所有图片识别完成！")
