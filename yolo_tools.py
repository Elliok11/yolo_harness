"""
yolo_tools.py - 工具箱
把 YOLO 的能力包装成标准工具，供 AI 大脑（DeepSeek）调用
"""
import os
import json
import time
from datetime import datetime
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "output_results"

# 模型只加载一次，重复使用（省时间）
# 支持换模型：想用更大更准的 yolov8s.pt / yolo11s.pt，直接传名字进来，
# 每个模型各自缓存一份，换回来时不用重新加载。
_MODELS = {}
def get_model(model_path=None):
    """取 YOLO 模型。不传就用默认的 yolov8n.pt"""
    name = model_path or MODEL_PATH
    if name not in _MODELS:
        _MODELS[name] = YOLO(name)
    return _MODELS[name]


def list_images():
    """工具1：查看 images 文件夹里有哪些图片"""
    if not os.path.exists(IMAGE_FOLDER):
        return json.dumps({"error": "images 文件夹不存在"}, ensure_ascii=False)
    files = [f for f in os.listdir(IMAGE_FOLDER)
             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    return json.dumps({"count": len(files), "images": files}, ensure_ascii=False)


def detect_objects(image_name):
    """工具2：用 YOLO 识别一张图片，返回物体、数量统计和【位置坐标】

    坐标说明（给 AI 看的）：
      bbox   = [左上x, 左上y, 右下x, 右下y]，单位是像素
      size   = [宽, 高]，单位是像素
      center = [中心x, 中心y]
    这样 AI 才能说出「人在画面偏左/偏上」这类有用的判断，
    而不是只知道「有 3 个人」。
    """
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    if not os.path.exists(image_path):
        return json.dumps({"error": f"找不到图片 {image_name}，请先调用 list_images 查看可用图片"}, ensure_ascii=False)

    import cv2

    # 先拿到图片自身的尺寸，方便 AI 判断物体在画面的哪个位置
    img_w = img_h = 0
    _probe = cv2.imread(image_path)
    if _probe is not None:
        img_h, img_w = _probe.shape[:2]

    model = get_model()
    results = model.predict(source=image_path, save=False, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append({
                "object": model.names[cls],
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "size": [round(x2 - x1, 1), round(y2 - y1, 1)],
                "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
            })

    # 统计每种物体出现几次
    summary = {}
    for d in detections:
        summary[d["object"]] = summary.get(d["object"], 0) + 1

    return json.dumps({
        "image": image_name,
        "image_size": [img_w, img_h],
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


# ==================================================================
#  以下是「三种运行模式」的限量版工具
#  原来 harness.py / harness_camera.py 是永不返回的死循环，
#  直接给 AI 调用会把聊天程序卡死；这里的每个函数都保证「跑够就还」。
# ==================================================================

MAX_ROUNDS = 50        # 文件夹监控最多查多少轮（护栏）
MAX_DURATION = 300     # 摄像头监控最多看多少秒（护栏）


def _clamp_rounds(max_rounds):
    """把轮数收进合法范围，返回 (最终轮数, 是否被夹过)"""
    try:
        max_rounds = int(max_rounds)
    except (TypeError, ValueError):
        max_rounds = 1
    if max_rounds < 1:
        max_rounds = 1
    capped = max_rounds > MAX_ROUNDS
    return (MAX_ROUNDS if capped else max_rounds), capped


def _clamp_duration(duration):
    """把秒数收进合法范围，返回 (最终秒数, 是否被夹过)"""
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 10.0
    if duration < 0.5:
        duration = 0.5          # 至少给半秒，否则一帧都抓不到
    capped = duration > MAX_DURATION
    return (float(MAX_DURATION) if capped else duration), capped


def _capture_plan(duration, interval):
    """按真实时间算出该抓哪几帧，而不是按帧数瞎猜。

    返回 [第0.0秒, 第interval秒, ...] 的时间点列表，保证：
      · duration=0.5 这种很短的情况也至少抓 1 帧
      · 既不会抓超时，也不会漏抓
    """
    if interval <= 0:
        interval = 1.0
    plan = []
    t = 0.0
    while t < duration:
        plan.append(round(t, 3))
        t += interval
    if not plan:
        plan = [0.0]
    return plan


def detect_all_images():
    """工具4：把 images 文件夹里的图全部识别一遍（对应 main.py 菜单第 1 项）"""
    if not os.path.isdir(IMAGE_FOLDER):
        return json.dumps({"error": "images 文件夹不存在"}, ensure_ascii=False)

    files = [f for f in os.listdir(IMAGE_FOLDER)
             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    if not files:
        return json.dumps({"error": "images 文件夹里没有图片"}, ensure_ascii=False)

    model = get_model()
    results = []
    for filename in files:
        image_path = os.path.join(IMAGE_FOLDER, filename)
        predicts = model.predict(source=image_path, save=True, verbose=False)
        names = []
        for result in predicts:
            for box in result.boxes:
                names.append(model.names[int(box.cls[0])])
        counts = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1
        results.append({"image": filename, "total": len(names), "summary": counts})

    return json.dumps({
        "mode": "单次识别（全部图片）",
        "image_count": len(files),
        "results": results,
        "note": "带识别框的标注图已保存到 runs/detect/ 文件夹",
    }, ensure_ascii=False)


def run_folder_monitor(max_rounds=1, interval=5):
    """工具5：文件夹监控的「限量版」——查够 max_rounds 轮就自动返回

    对应 main.py 菜单第 2 项，但不会无限循环。
    """
    import harness  # 复用现成的发动机，不复制代码

    max_rounds, capped = _clamp_rounds(max_rounds)
    try:
        interval = float(interval)
    except (TypeError, ValueError):
        interval = 5.0
    harness.ensure_folders()
    model = get_model()

    processed = set()
    found = []          # 每一轮发现的「新图片」
    total_objects = 0
    seen = []

    for round_no in range(1, max_rounds + 1):
        if not os.path.isdir(IMAGE_FOLDER):
            break
        files = [f for f in os.listdir(IMAGE_FOLDER)
                 if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        new_ones = [f for f in files if f not in processed]
        print("--- 第 %d/%d 轮：发现 %d 张新图片 ---" % (round_no, max_rounds, len(new_ones)))
        harness.log_message("--- 第 %d 次循环（限量版）---" % round_no)

        if new_ones:
            for img_name in new_ones:
                img_path = os.path.join(IMAGE_FOLDER, img_name)
                print("正在识别：" + img_name)
                predicts = model.predict(source=img_path, save=False, verbose=False)
                detections = []
                for result in predicts:
                    for box in result.boxes:
                        detections.append({
                            "class": model.names[int(box.cls[0])],
                            "confidence": round(float(box.conf[0]), 3),
                        })
                path = harness.save_result(img_name, detections)
                total_objects += len(detections)
                seen.append({"image": img_name, "objects": len(detections)})
                harness.log_message("✅ 识别完成，检测到 %d 个目标，结果已保存" % len(detections))
                processed.add(img_name)
            found.append({"round": round_no, "new_images": new_ones})

        if round_no < max_rounds:
            time.sleep(interval)

    return json.dumps({
        "mode": "文件夹监控（限量版）",
        "rounds_run": max_rounds,
        "interval_seconds": interval,
        "new_images_total": sum(len(x["new_images"]) for x in found),
        "rounds_with_new_images": found,
        "detected": seen,
        "total_objects": total_objects,
        "note": ("本来设的轮数超过上限，已自动夹到 %d 轮；" % MAX_ROUNDS if capped else "")
                + "结果 JSON 已保存到 output_results/",
    }, ensure_ascii=False)


def _probe_camera(src, timeout=6):
    """试探某个摄像头源能不能打开（单次尝试，不会卡住）"""
    import cv2
    cap = None
    try:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                return {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
            time.sleep(0.05)
        return None
    except Exception:
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def list_camera_devices():
    """工具6：逐个试探 0～4 号摄像头，看哪个能用（你手机当摄像头也能接）"""
    import harness_camera  # noqa: F401  （确认摄像头依赖在位）

    available, unavailable = [], []
    for idx in range(5):
        info = _probe_camera(idx)
        if info:
            available.append({"device": idx, "resolution": "%dx%d" % (info["width"], info["height"])})
        else:
            unavailable.append(idx)

    hint = "没有可用摄像头：请检查摄像头是否被别的程序占用，或用手机 IP 摄像头。"
    if available:
        hint = "可用摄像头有 %d 个，想实时识别可以直接说：开摄像头看 10 秒" % len(available)
    return json.dumps({
        "available": available,
        "unavailable": unavailable,
        "hint": hint,
    }, ensure_ascii=False)


def run_camera_monitor(duration=10, interval=5, camera=0, show_window=True):
    """工具7：摄像头监控的「限量版」——看够 duration 秒就自动返回

    对应 main.py 菜单第 3 项，但不会无限循环。
    这里按「真实时间」计算，不像原版按帧数猜时间。
    """
    import cv2
    import harness_camera

    duration, capped = _clamp_duration(duration)
    try:
        interval = float(interval)
    except (TypeError, ValueError):
        interval = 5.0
    if interval <= 0:
        interval = 1.0
    plan = _capture_plan(duration, interval)   # 该在第几秒抓帧，按真实时间算

    if isinstance(camera, str) and camera.strip().isdigit():
        camera = int(camera.strip())

    harness_camera.ensure_folders()
    model = get_model()
    harness_camera.log_message("正在加载模型...")
    harness_camera.log_message("模型加载完成！（限量版：最多看 %g 秒）" % duration)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        return json.dumps({
            "error": "打不开摄像头（编号或地址：%s）" % camera,
            "hint": "可能被别的程序占用。可以先调用 list_camera_devices 看哪个能用。",
        }, ensure_ascii=False)

    frames_saved = []
    total_objects = 0
    per_class = {}
    start = time.time()
    plan_idx = 0
    frame_no = 0
    last_show = 0.0
    stopped_early = False

    try:
        while plan_idx < len(plan):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            now = time.time()
            elapsed = now - start
            frame_no += 1

            # 还没到下一个抓帧时间点，就只刷新画面，别白费 CPU
            if elapsed + 1e-9 < plan[plan_idx]:
                if show_window and now - last_show >= 0.03:
                    last_show = now
                    try:
                        cv2.imshow('YOLO Camera Harness', frame)
                        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                            harness_camera.log_message("🛑 用户按 Q 键，提前停止")
                            stopped_early = True
                            break
                    except cv2.error:
                        show_window = False
                else:
                    time.sleep(0.005)
                continue

            # 到点了 → 抓这一帧做识别
            plan_idx += 1
            detections = []
            for result in model.predict(source=frame, save=False, verbose=False):
                for box in result.boxes:
                    name = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    detections.append({
                        "class": name,
                        "confidence": round(conf, 3),
                        "bbox": box.xyxy[0].tolist(),
                    })
                    per_class[name] = per_class.get(name, 0) + 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "%s %.2f" % (name, conf), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            total_objects += len(detections)
            fname = "frame_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path, json_path = harness_camera.save_result(fname, detections, frame)
            frames_saved.append({
                "image": os.path.basename(img_path),
                "json": os.path.basename(json_path),
                "objects": len(detections),
            })
            harness_camera.log_message("✅ 识别完成，检测到 %d 个目标" % len(detections))

            if show_window:
                try:
                    cv2.imshow('YOLO Camera Harness', frame)
                    last_show = time.time()
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        harness_camera.log_message("🛑 用户按 Q 键，提前停止")
                        stopped_early = True
                        break
                except cv2.error:
                    show_window = False
                    print("（当前环境没法弹窗，已自动切换成静默模式）")
    finally:
        cap.release()
        if show_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return json.dumps({
        "mode": "摄像头监控（限量版）",
        "camera": camera,
        "duration_seconds": duration,
        "interval_seconds": interval,
        "planned_captures": len(plan),
        "captures": len(frames_saved),
        "stopped_early": stopped_early,
        "total_objects": total_objects,
        "per_class": per_class,
        "saved": frames_saved,
        "note": ("看的时间超过上限，已自动夹到 %d 秒；" % MAX_DURATION if capped else "")
                + "带框图片和 JSON 都在 output_results/ 里",
    }, ensure_ascii=False)


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
