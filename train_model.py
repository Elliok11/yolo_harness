"""
train_model.py - 用你自己标的标签训练一个专属检测模型

它一共干这 6 件事：
  1. 检查依赖和环境          （缺什么会告诉你）
  2. 把 labels/ 里标好的数据 按 8:2 分成 训练集 / 验证集
  3. 生成 data.yaml          （YOLO 认的数据清单）
  4. 先量一下【老模型】在你的验证集上多少分（这是对比的基准）
  5. 训练【新模型】（从 yolov8n.pt 微调，CPU 上会比较慢）
  6. 训练完再量一次，告诉你【到底变准了还是变差了】

为什么要先量老模型？
  因为「新模型 mAP 0.6」这种数字本身说明不了什么。只有跟旧模型同台比，
  你才知道这顿折腾值不值。所以对比是必须的一步，不是我多事。

常用命令：
  python train_model.py --check              只看数据准备得怎么样，不训练
  python train_model.py                      用默认参数训练（50轮，CPU保守设置）
  python train_model.py --epochs 100         多训一会儿
  python train_model.py --resume             上次训到一半断了，接着跑
"""
import os
import sys
import json
import time
import random
import shutil
import argparse
from datetime import datetime

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ==================== 配置 ====================
DATASET_ROOT = "dataset"          # 整理好的训练数据放这
TRAIN_RUNS = "train_runs"         # 训练过程文件放这（权重、曲线、日志）
BASE_MODEL = "yolov8n.pt"         # 从哪个模型开始微调
OUTPUT_MODEL = "yolov8n_custom.pt"  # 训练好的模型复制到项目根目录，方便直接用
SEED = 42
# ==============================================


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


# ==================== 1. 依赖检查 ====================
def check_requirements():
    """缺的包列出来，并给出安装命令"""
    missing = []
    for mod, pkg in [("torch", "torch"), ("ultralytics", "ultralytics"),
                     ("cv2", "opencv-python"), ("yaml", "pyyaml"),
                     ("pandas", "pandas"), ("tqdm", "tqdm")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("❌ 缺这些包，训练跑不了：")
        print("   " + " ".join(missing))
        print("\n装一下（用阿里云源，国内快）：")
        print("   .\\venv\\Scripts\\python.exe -m pip install %s "
              "-i https://mirrors.aliyun.com/pypi/simple/" % " ".join(missing))
        return False
    return True


def device_info():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return "cuda", "显卡 %s（会快很多）" % name
    return "cpu", "CPU（%d 核，会比较慢）" % (os.cpu_count() or 1)


# ==================== 2+3. 准备数据集 ====================
def prepare_dataset(val_ratio=0.2, clean=True):
    """把标注好的图片和标签整理成 YOLO 要的目录结构

    返回 (图片总数, 训练数, 验证数, 类别表)
    """
    import label_data as ld

    classes = ld.load_classes()
    if not classes:
        print("❌ 还没有任何类别。先用打标签工具加类别并标注。")
        return 0, 0, 0, []

    # 找出所有「有对应标签文件」的图片
    pairs = []
    for name, path in ld.list_sources():
        lp = ld.label_path_for(path)
        if os.path.exists(lp) and ld.read_label_file(lp):
            pairs.append((path, lp))

    if not pairs:
        print("❌ 一张标好的图都没有。先用打标签工具画框。")
        return 0, 0, 0, classes

    if len(pairs) < 10:
        print("⚠️  只有 %d 张标注图，太少了 —— 训出来基本没用。" % len(pairs))
        print("    建议至少 50 张，最好 150 张以上。要不要继续你自己决定。")

    # 清掉旧的，重新来（避免上次的残留混进来）
    if clean and os.path.isdir(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT, ignore_errors=True)

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(DATASET_ROOT, sub), exist_ok=True)

    # 固定随机种子再打乱，这样每次分的训练/验证集一样，两次训练才可比
    rnd = random.Random(SEED)
    rnd.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) >= 5 else 0
    val_set = pairs[:n_val]
    train_set = pairs[n_val:]

    def copy_into(items, split):
        for img, lab in items:
            base = os.path.splitext(os.path.basename(img))[0]
            ext = os.path.splitext(img)[1]
            shutil.copy2(img, os.path.join(DATASET_ROOT, "images", split, base + ext))
            shutil.copy2(lab, os.path.join(DATASET_ROOT, "labels", split, base + ".txt"))

    copy_into(train_set, "train")
    copy_into(val_set, "val")

    # data.yaml：YOLO 靠它找数据、认类别
    yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# 由 train_model.py 自动生成，别手改（重跑会覆盖）\n")
        f.write("path: %s\n" % os.path.abspath(DATASET_ROOT).replace("\\", "/"))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: %d\n" % len(classes))
        f.write("names:\n")
        for c in classes:
            f.write("  - %s\n" % c)

    return len(pairs), len(train_set), len(val_set), classes


def dataset_report():
    """--check 模式：把数据情况打印清楚就退出"""
    import label_data as ld
    classes = ld.load_classes()
    print("=" * 62)
    print("数据检查（只检查，不训练）")
    print("=" * 62)
    if not classes:
        print("类别表是空的。先用打标签工具加类别。")
        return 1
    print("类别（%d 个）:" % len(classes))
    for i, c in enumerate(classes):
        print("   %d. %s" % (i + 1, c))

    print()
    print(ld.class_counts_text(classes))

    print()
    counts = ld.dataset_stats(classes)["per_class"]
    few = [classes[i] for i, n in counts.items() if n < 10]
    if few:
        print("⚠️  这些类别样本太少（<10 个框）：%s" % "、".join(few))
        print("    样本太少的类别基本学不会，建议多标几张。")

    n, tr, va, _ = prepare_dataset()
    print()
    if n:
        print("已整理成数据集：")
        print("   总图 %d 张 -> 训练 %d 张 / 验证 %d 张" % (n, tr, va))
        print("   目录: %s" % os.path.abspath(DATASET_ROOT))
        print("   清单: %s" % os.path.join(DATASET_ROOT, "data.yaml"))
        print()
        print("没问题的话，去掉 --check 直接跑训练：")
        print("   python train_model.py")
    return 0


# ==================== 4+6. 评估 ====================
def evaluate(weights, yaml_path, imgsz=640, tag="模型"):
    """在一个模型上跑验证集，返回关键指标。失败返回 None"""
    from ultralytics import YOLO
    if not os.path.exists(weights):
        log("跳过评估：找不到 %s" % weights)
        return None
    try:
        m = YOLO(weights)
        res = m.val(data=yaml_path, imgsz=imgsz, verbose=False, plots=False)
        box = res.box
        out = {
            "weights": weights,
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
        }
        log("%s 评估完成：mAP50=%.3f  mAP50-95=%.3f  精确率=%.3f  召回率=%.3f"
            % (tag, out["mAP50"], out["mAP50-95"], out["precision"], out["recall"]))
        return out
    except Exception as exc:
        log("评估 %s 时出错：%s" % (tag, exc))
        return None


def compare_report(old, new):
    """把新旧两个模型的成绩做成一张人话表格"""
    print()
    print("=" * 62)
    print("成绩对比（在你自己的验证集上）")
    print("=" * 62)
    if not old or not new:
        print("有一边没测出来，没法对比。单个结果：")
        print("   老模型:", old)
        print("   新模型:", new)
        return None

    rows = [("mAP50", "mAP50", True),
            ("mAP50-95", "mAP50-95", True),
            ("precision", "精确率", True),
            ("recall", "召回率", True)]
    print("   %-10s %10s %10s %12s" % ("指标", "老模型", "新模型", "变化"))
    print("   " + "-" * 48)
    deltas = {}
    for key, cn, higher_better in rows:
        o, n = old[key], new[key]
        d = n - o
        deltas[key] = d
        arrow = "↑" if d > 0.0005 else ("↓" if d < -0.0005 else "=")
        mark = ""
        if abs(d) > 0.0005:
            good = (d > 0) == higher_better
            mark = "✅" if good else "⚠️"
        print("   %-10s %10.3f %10.3f %+10.3f %s%s" % (cn, o, n, d, arrow, mark))

    print()
    better = deltas["mAP50"] > 0.005
    worse = deltas["mAP50"] < -0.005
    if better:
        print("   >>> 结论：新模型确实更准了（mAP50 提升 %.3f）。建议换用新模型。" % deltas["mAP50"])
    elif worse:
        print("   >>> 结论：新模型反而变差了（mAP50 下降 %.3f）。" % abs(deltas["mAP50"]))
        print("       常见原因：标注太少（<50张）、框画得不紧、或者某几个类别样本太少。")
        print("       建议：别急着换，先去多标几十张再来。")
    else:
        print("   >>> 结论：两个模型差不多（mAP50 变化 %.3f）。" % deltas["mAP50"])
        print("       说明当前数据量下，预训练模型已经到顶了。要多标数据才有提升空间。")
    return deltas


# ==================== 5. 训练 ====================
def train(args):
    if not check_requirements():
        return 1

    dev, dev_desc = device_info()
    log("训练设备：%s" % dev_desc)

    n, tr, va, classes = prepare_dataset(val_ratio=args.val_ratio)
    if not n:
        return 1
    yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    log("数据就绪：总 %d 张（训练 %d / 验证 %d），类别 %d 个" % (n, tr, va, len(classes)))

    # 时间预估（CPU 上这点很重要，免得你以为卡死了）
    if dev == "cpu":
        sec_per_img = 0.55 if args.imgsz <= 640 else 1.1
        est = tr * args.epochs * sec_per_img / max(1, args.batch) * 1.6
        log("⚠️  CPU 训练很慢，估算需要 %.1f 小时（%d 张 × %d 轮）"
            % (est / 3600, tr, args.epochs))
        log("   中途想停就按 Ctrl+C，之后加 --resume 可以接着跑。")

    # ---- 先量老模型的成绩，作为对比基准 ----
    log("先评估【老模型】%s，作为对比基准…" % args.base)
    old_metrics = evaluate(args.base, yaml_path, args.imgsz, tag="老模型")

    # ---- 开始训练 ----
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    SETTINGS["datasets_dir"] = os.path.abspath(".")   # 别让它在别处找数据

    run_name = args.name or ("exp_%s" % datetime.now().strftime("%m%d_%H%M"))
    log("开始训练（%s）。日志会实时打出来，别关窗口。" % run_name)

    if args.resume:
        ckpt = os.path.join(TRAIN_RUNS, run_name, "weights", "last.pt")
        if not os.path.exists(ckpt):
            log("❌ 找不到断点 %s，没法续训。用 --name 指定正确的训练名，或去掉 --resume。" % ckpt)
            return 1
        log("从断点续训：%s" % ckpt)
        model = YOLO(ckpt)
        model.train(resume=True)
    else:
        model = YOLO(args.base)
        model.train(
            data=yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=dev,
            workers=args.workers,
            project=TRAIN_RUNS,
            name=run_name,
            exist_ok=True,
            seed=SEED,
            patience=args.patience,   # 多少轮没进步就早停，省时间
            cache=False,              # CPU 内存紧张，别缓存图片
            plots=True,
            val=True,
            verbose=True,
        )

    best = os.path.join(TRAIN_RUNS, run_name, "weights", "best.pt")
    if not os.path.exists(best):
        log("❌ 训练结束了，但没找到 best.pt，看看上面的报错。")
        return 1
    log("训练完成，最好的权重：%s" % best)

    # ---- 新模型成绩 ----
    log("评估【新模型】…")
    new_metrics = evaluate(best, yaml_path, args.imgsz, tag="新模型")

    deltas = compare_report(old_metrics, new_metrics)

    # ---- 落盘记录 + 复制到项目根目录 ----
    out = os.path.join(TRAIN_RUNS, run_name, "report.json")
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"time": datetime.now().isoformat(timespec="seconds"),
                       "images": {"total": n, "train": tr, "val": va},
                       "classes": classes, "epochs": args.epochs,
                       "imgsz": args.imgsz, "batch": args.batch,
                       "old": old_metrics, "new": new_metrics,
                       "delta": deltas}, f, ensure_ascii=False, indent=2)
        log("报告已存：%s" % out)
    except Exception as exc:
        log("报告没存上：%s" % exc)

    if new_metrics:
        shutil.copy2(best, OUTPUT_MODEL)
        log("新模型已复制到项目根目录：%s" % OUTPUT_MODEL)
        print()
        print("=" * 62)
        print("怎么用这个新模型")
        print("=" * 62)
        print("  1) 打标签窗口的「🤖 自动预标注」会自动用它")
        print("  2) 图形界面版可在下拉框里选它")
        print("  3) 命令行临时指定：")
        print('     $env:YOLO_MODEL="%s"' % OUTPUT_MODEL)
        if deltas and deltas.get("mAP50", 0) <= 0.005:
            print()
            print("  ⚠️ 提醒：这次没比老模型更好，先别急着当主力用。")
    return 0


# ==================== 入口 ====================
def main():
    ap = argparse.ArgumentParser(
        description="用你自己标的标签训练检测模型（CPU 也能跑，就是慢）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""例子：
  python train_model.py --check
  python train_model.py
  python train_model.py --epochs 100 --imgsz 640
  python train_model.py --resume --name exp_1007_1530
""")
    ap.add_argument("--check", action="store_true", help="只检查数据，不训练")
    ap.add_argument("--epochs", type=int, default=50, help="训练多少轮（默认 50）")
    ap.add_argument("--imgsz", type=int, default=640, help="训练分辨率（默认 640，CPU 上别调大）")
    ap.add_argument("--batch", type=int, default=4, help="批大小（默认 4，CPU 越小越省内存）")
    ap.add_argument("--workers", type=int, default=2, help="读数据的线程数（默认 2）")
    ap.add_argument("--patience", type=int, default=15, help="多少轮没进步就早停（默认 15）")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    ap.add_argument("--base", default=BASE_MODEL, help="从哪个模型开始微调（默认 %s）" % BASE_MODEL)
    ap.add_argument("--name", default=None, help="这次训练的名字（也是续训时要对上的名字）")
    ap.add_argument("--resume", action="store_true", help="接着上次没跑完的继续")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.check:
        return dataset_report()

    t0 = time.time()
    try:
        code = train(args)
    except KeyboardInterrupt:
        print()
        log("你按了 Ctrl+C，训练中断了。")
        log("已训的部分还在。想接着跑，用：")
        log("   python train_model.py --resume --name %s" % (args.name or "<上面显示的 run 名>"))
        return 130
    log("总耗时 %.1f 分钟" % ((time.time() - t0) / 60))
    return code


if __name__ == "__main__":
    sys.exit(main())
