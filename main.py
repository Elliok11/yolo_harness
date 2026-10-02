import os

def run_detect():
    """单次识别模式"""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    image_folder = "images"

    files = [f for f in os.listdir(image_folder)
             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    if not files:
        print("⚠️  images 文件夹里没有图片！")
        return

    for filename in files:
        image_path = os.path.join(image_folder, filename)
        print(f"正在识别：{filename}")
        results = model.predict(source=image_path, save=True)
        for result in results:
            print(f"  检测到 {len(result.boxes)} 个目标")

    print("✅ 识别完成！标注图片保存在 runs 文件夹")

def main():
    try:
        while True:
            print("=" * 45)
            print("   🚀 YOLO Harness 启动器")
            print("=" * 45)
            print("  1. 单次识别（识别 images 文件夹里的图片）")
            print("  2. 文件夹监控模式（自动识别新图片）")
            print("  3. 摄像头监控模式（内置摄像头 / 手机摄像头）")
            print("  0. 退出")
            print("=" * 45)

            choice = input("请输入数字并回车：").strip()

            if choice == "1":
                run_detect()
            elif choice == "2":
                import harness
                harness.main()
            elif choice == "3":
                import harness_camera
                source = input(
                    "摄像头地址（直接回车 = 内置摄像头，\n"
                    "手机摄像头示例：http://192.168.1.5:8080/video）："
                ).strip()
                if source == "":
                    harness_camera.main()
                elif source.isdigit():
                    harness_camera.main(camera_source=int(source))
                else:
                    harness_camera.main(camera_source=source)
            elif choice == "0":
                print("👋 再见！")
                break
            else:
                print("⚠️  无效输入，请重新选择")

    except KeyboardInterrupt:
        print("\n👋 程序已终止")

if __name__ == "__main__":
    main()
