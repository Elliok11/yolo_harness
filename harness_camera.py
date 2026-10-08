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

def unique_stamp():
    """生成一个「绝不重名」的时间戳，格式 20261008_165033_123456

    为什么需要它？
      原来只用秒级时间戳（%Y%m%d_%H%M%S），同一秒内保存两次就会同名，
      后写的把先写的覆盖掉（图片和 JSON 都会丢）。
      现在：① 加微秒降低撞车概率 ② 万一还撞上就加序号 _1 _2，直到不重名。
    """
    base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stamp = base
    n = 1
    while (os.path.exists(os.path.join(OUTPUT_FOLDER, f"frame_{stamp}.jpg"))
           or os.path.exists(os.path.join(OUTPUT_FOLDER, f"result_{stamp}.json"))):
        stamp = "%s_%d" % (base, n)
        n += 1
    return stamp

def save_result(frame_name, detections, frame):
    """保存识别结果：带框图片 + JSON 数据（文件名保证唯一，不会互相覆盖）"""
    timestamp = unique_stamp()
    
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
    last_capture = time.time()     # 上次捕获的时刻（按真实时间算，不数帧数）
    fps_t0 = time.time()           # 用于统计真实帧率
    fps_frames = 0
    real_fps = 0.0

    try:
        while True:
            frame_count += 1
            fps_frames += 1

            # 从摄像头读取一帧
            ret, frame = cap.read()

            if not ret:
                log_message("⚠️  无法读取摄像头画面！")
                time.sleep(1)
                continue

            elapsed = time.time() - last_capture

            # 每隔 CAPTURE_INTERVAL 秒捕获一次。
            #
            # 这里以前写的是 frame_count % (CAPTURE_INTERVAL * 10) == 0，
            # 注释说"假设 30fps，每 5 秒约 150 帧" —— 但那是个想当然的假设：
            #   · 摄像头实际 30fps 时：50 帧 = 约 1.7 秒就抓一次（不是 5 秒），
            #     检测负载和产生的文件都是设定的 3 倍
            #   · 摄像头实际 15fps 时：又变成约 3.3 秒
            #   · 笔记本摄像头常自动降帧，实际值还会乱飘
            # 所以改成看「真实过了多少秒」，你设 5 秒就一定是 5 秒。
            if elapsed >= CAPTURE_INTERVAL:
                last_capture = time.time()
                captured_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_name = f"frame_{timestamp}"

                log_message(f"--- 第 {captured_count} 次捕获（距上次 {elapsed:.1f} 秒）---")
                if real_fps:
                    log_message(f"📹 摄像头实际帧率约 {real_fps:.1f} fps")
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

            # 每秒统计一次真实帧率（顺便让捕获间隔的判断心里有数）
            if time.time() - fps_t0 >= 1.0:
                real_fps = fps_frames / (time.time() - fps_t0)
                fps_frames = 0
                fps_t0 = time.time()

    except KeyboardInterrupt:
        log_message("🛑 用户中断，Harness 停止运行")
    
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("\n感谢使用，再见！👋")

if __name__ == "__main__":
    main()
