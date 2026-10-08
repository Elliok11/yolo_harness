import time
import os
import json
from datetime import datetime
from ultralytics import YOLO

# ================= 配置区域 =================
MODEL_PATH = "yolov8n.pt"        # 模型文件路径
IMAGE_FOLDER = "images"           # 输入图片文件夹
OUTPUT_FOLDER = "output_results"  # 结果保存文件夹
LOG_FOLDER = "logs"               # 日志文件夹
INTERVAL = 5                      # 每隔多少秒识别一次
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
    log_file = os.path.join(LOG_FOLDER, f"harness_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_text + "\n")

def unique_stamp():
    """生成一个「绝不重名」的时间戳，格式 20261008_165033_123456

    为什么需要它？
      原来只用秒级时间戳（%Y%m%d_%H%M%S），文件夹监控一次发现多张新图时，
      这几张图会在同一秒内接连保存，文件名完全一样 —— 后写的把先写的覆盖掉。
      实测：同一秒保存 3 张，磁盘上只剩 1 个文件，前两张的结果永久丢失。
    现在做了两层保险：
      1. 加上微秒（撞车概率降到几乎为零）
      2. 万一还是撞上（同名文件已存在），就加序号 _1 _2 直到不重名
    """
    base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stamp = base
    n = 1
    while os.path.exists(os.path.join(OUTPUT_FOLDER, f"result_{stamp}.json")):
        stamp = "%s_%d" % (base, n)
        n += 1
    return stamp


def save_result(image_name, detections):
    """保存识别结果到 JSON 文件（文件名保证唯一，不会互相覆盖）"""
    timestamp = unique_stamp()
    result_file = os.path.join(OUTPUT_FOLDER, f"result_{timestamp}.json")

    result = {
        "image": image_name,
        "timestamp": datetime.now().isoformat(),
        "detections": detections
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result_file

def main():
    """主循环"""
    print("=" * 50)
    print("🚀 YOLO Harness 启动！")
    print(f"📁 图片文件夹：{IMAGE_FOLDER}")
    print(f"💾 结果保存到：{OUTPUT_FOLDER}")
    print(f"⏱️  识别间隔：{INTERVAL} 秒")
    print("按 Ctrl+C 停止运行")
    print("=" * 50)
    
    # 确保文件夹存在
    ensure_folders()
    
    # 加载模型
    log_message("正在加载模型...")
    model = YOLO(MODEL_PATH)
    log_message("模型加载完成！")
    
    # 记录已处理过的图片，避免重复
    processed_images = set()
    
    # 进入主循环
    iteration = 0
    try:
        while True:
            iteration += 1
            log_message(f"--- 第 {iteration} 次循环 ---")
            
            # 获取文件夹中的所有图片
            image_files = [f for f in os.listdir(IMAGE_FOLDER) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            
            if not image_files:
                log_message("⚠️  图片文件夹为空，等待中...")
            else:
                # 找出新图片（没处理过的）
                new_images = [f for f in image_files if f not in processed_images]
                
                if new_images:
                    log_message(f"📷 发现 {len(new_images)} 张新图片")
                    
                    for img_name in new_images:
                        img_path = os.path.join(IMAGE_FOLDER, img_name)
                        log_message(f"正在识别：{img_name}")
                        
                        # 执行识别
                        results = model.predict(source=img_path, save=False, verbose=False)
                        
                        # 提取结果
                        detections = []
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                detections.append({
                                    "class": model.names[cls],
                                    "confidence": round(conf, 3)
                                })
                        
                        # 保存结果
                        result_path = save_result(img_name, detections)
                        log_message(f"✅ 识别完成，检测到 {len(detections)} 个目标，结果已保存")
                        
                        # 标记为已处理
                        processed_images.add(img_name)
                else:
                    log_message("ℹ️  没有新图片，等待中...")
            
            # 等待下一次循环
            time.sleep(INTERVAL)
            
    except KeyboardInterrupt:
        log_message("🛑 用户中断，Harness 停止运行")
        print("\n感谢使用，再见！👋")

if __name__ == "__main__":
    main()
