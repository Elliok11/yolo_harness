import cv2
import time
import os
import json
from datetime import datetime
from ultralytics import YOLO

# ================= 配置区域 =================
MODEL_PATH = "yolov8n.pt"        # 模型文件路径
CAMERA_ID = 0
                     # 摄像头编号（0 通常是主摄像头）
OUTPUT_FOLDER = "output_results"  # 结果保存文件夹
LOG_FOLDER = "logs"               # 日志文件夹
CAPTURE_INTERVAL = 5              # 每隔多少秒截取一帧
# ===========================================

def ensure_folders():
    """确保输出文件夹存在"""
    for folder in [OUTPUT_FOLDER, LOG_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"创建文件夹：{folder}")

def log_message(message):
    """记录日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"[{timestamp}] {message}"
    print(log_text)
    
    # 同时保存到日志文件
    log_file = os.path.join(LOG_FOLDER, f"harness_camera_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_text + "\n")

def save_result(frame_name, detections, frame):
    """保存识别结果：带框图片 + JSON 数据"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存带识别框的图片
    image_path = os.path.join(OUTPUT_FOLDER, f"frame_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)
    
    # 保存 JSON 数据
    json_path = os.path.join(OUTPUT_FOLDER, f"result_{timestamp}.json")
    result = {
        "frame": frame_name,
        "timestamp": datetime.now().isoformat(),
        "detections": detections
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return image_path, json_path

def main(camera_source=CAMERA_ID):
    """主循环"""
    print("=" * 50)
    print("📸 YOLO 摄像头 Harness 启动！")
    print(f"💾 结果保存到：{OUTPUT_FOLDER}")
    print(f"⏱️  捕获间隔：{CAPTURE_INTERVAL} 秒")
    print("按 Ctrl+C 停止运行")
    print("=" * 50)
    
    # 确保文件夹存在
    ensure_folders()
    
    # 加载模型
    log_message("正在加载模型...")
    model = YOLO(MODEL_PATH)
    log_message("模型加载完成！")
    
    # 打开摄像头
    log_message("正在打开摄像头...")
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        log_message("❌ 无法打开摄像头！请检查摄像头是否正常。")
        return
    
    log_message("✅ 摄像头已打开！")
    
    # 进入主循环
    frame_count = 0
    captured_count = 0
    
    try:
        while True:
            frame_count += 1
            
            # 从摄像头读取一帧
            ret, frame = cap.read()
            
            if not ret:
                log_message("⚠️  无法读取摄像头画面！")
                time.sleep(1)
                continue
            
            # 每隔 CAPTURE_INTERVAL 秒捕获一次
            if frame_count % (CAPTURE_INTERVAL * 10) == 0:  # 假设 30fps，每 5 秒约 150 帧
                captured_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_name = f"frame_{timestamp}"
                
                log_message(f"--- 第 {captured_count} 次捕获 ---")
                log_message(f"正在识别：{frame_name}")
                
                # 执行识别
                results = model.predict(source=frame, save=False, verbose=False)
                
                # 提取结果并画框
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        
                        detections.append({
                            "class": class_name,
                            "confidence": round(conf, 3),
                            "bbox": box.xyxy[0].tolist()
                        })
                        
                        # 在图上画框和标签
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 保存结果
                image_path, json_path = save_result(frame_name, detections, frame)
                log_message(f"✅ 识别完成，检测到 {len(detections)} 个目标")
                log_message(f"📁 图片：{os.path.basename(image_path)}")
                log_message(f"📄 JSON: {os.path.basename(json_path)}")
            
            # 显示实时画面（按 Q 键退出）
            cv2.imshow('YOLO Camera Harness', frame)
            
            # 等待 1ms（必须，否则画面会卡）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log_message("🛑 用户按 Q 键，Harness 停止运行")
                break
            
    except KeyboardInterrupt:
        log_message("🛑 用户中断，Harness 停止运行")
    
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("\n感谢使用，再见！👋")

if __name__ == "__main__":
    main()
