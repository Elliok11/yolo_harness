"""
label_data.py - 打标签的「数据层」

它只管三件事，不碰界面：
  1. 有哪些图可以标（扫描 images/、output_results/、new_faces/，还能自己加）
  2. 每张图上的框存哪、怎么存（标准 YOLO 格式）
  3. 有哪几个类别（classes.txt）

为什么单独拆一个文件？
  因为界面（label_gui.py）只负责画和点，数据怎么存跟界面无关。
  以后你想写个别的工具（比如统计标了多少、导出成别的格式），
  直接 import 这个文件就行，不用碰界面代码。

YOLO 标签格式（一行一个框）：
    类别编号  中心x  中心y  宽  高
    这四个数都是【0~1 的比例】，除以图片宽高得来的。
比如 0 0.5 0.5 0.2 0.3 表示：0号类别，框的中心在图片正中间，宽占 20%、高占 30%。
"""
import os
import glob
import json

# ==================== 目录约定 ====================
LABEL_ROOT = "labels"          # 标签都放这，和图片一一对应
DATASET_ROOT = "dataset"       # 训练数据集（train/val 划分）也放这
CLASSES_FILE = os.path.join(LABEL_ROOT, "classes.txt")

# 扫描素材时看这些目录
SCAN_DIRS = [
    "images",
    "output_results",
    os.path.join("output_results", "new_faces"),
    "label_images",            # 你自己丢新图进来的地方
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ==================== 类别表 ====================
def load_classes():
    """读类别表。一行一个名字，行号就是它的编号"""
    if not os.path.exists(CLASSES_FILE):
        return []
    with open(CLASSES_FILE, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def save_classes(classes):
    """写类别表"""
    os.makedirs(LABEL_ROOT, exist_ok=True)
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")


def add_class(classes, name):
    """加一个类别。重名直接返回原表，不重复加"""
    name = (name or "").strip()
    if not name or name in classes:
        return classes
    classes.append(name)
    save_classes(classes)
    return classes


def remove_class(classes, name):
    """删类别。

    注意：删了之后，其它类别里编号比它大的都要往前挪一位，
    否则已有的标签会整体错位 —— 这是最容易出 bug 的地方，所以单独处理。
    """
    if name not in classes:
        return classes, False
    idx = classes.index(name)
    classes.pop(idx)
    save_classes(classes)

    # 所有标签文件里：等于 idx 的框作废，大于 idx 的减一
    changed = 0
    for path in all_label_files():
        boxes = read_label_file(path)
        new_boxes = []
        for b in boxes:
            if b["cls"] == idx:
                continue                      # 这个类别的框直接丢掉
            if b["cls"] > idx:
                b["cls"] -= 1
            new_boxes.append(b)
        if len(new_boxes) != len(boxes):
            changed += len(boxes) - len(new_boxes)
        write_label_file(path, new_boxes)
    return classes, changed > 0


# ==================== 图片扫描 ====================
def list_sources():
    """列出所有可以标注的图片，返回 [(显示名, 完整路径), ...]"""
    items, seen = [], set()
    for d in SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            if not p.lower().endswith(IMAGE_EXTS):
                continue
            real = os.path.abspath(p).lower()
            if real in seen:
                continue
            seen.add(real)
            rel = os.path.relpath(p)
            items.append((rel, p))
    return items


# ==================== 标签读写 ====================
def label_path_for(image_path):
    """这张图的标签应该存哪：labels/文件名.txt

    用文件名（不是完整路径）来对齐，这样你在 images/ 和 output_results/ 里
    有同名图时会共用一份标签 —— 那种情况很少见，但能避免路径里的斜杠出乱子。
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(LABEL_ROOT, base + ".txt")


def all_label_files():
    if not os.path.isdir(LABEL_ROOT):
        return []
    return sorted(glob.glob(os.path.join(LABEL_ROOT, "*.txt")))


def read_label_file(path):
    """读一个标签文件，返回 [{"cls":0,"x":0.5,"y":0.5,"w":0.2,"h":0.3}, ...]"""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                continue
            boxes.append({"cls": cls, "x": x, "y": y, "w": w, "h": h})
    return boxes


def write_label_file(path, boxes):
    """写标签文件。没有框就把文件删掉（保持目录干净）"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not boxes:
        if os.path.exists(path):
            os.remove(path)
        return
    with open(path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write("%d %.6f %.6f %.6f %.6f\n"
                    % (int(b["cls"]), b["x"], b["y"], b["w"], b["h"]))


def load_boxes(image_path):
    """读这张图的框（比例坐标）"""
    return read_label_file(label_path_for(image_path))


def save_boxes(image_path, boxes):
    """存这张图的框"""
    write_label_file(label_path_for(image_path), boxes)
    return len(boxes)


# ==================== 像素 ↔ 比例 换算 ====================
def to_pixels(box, img_w, img_h):
    """比例坐标 -> 像素坐标，返回 (x1, y1, x2, y2)"""
    cx, cy = box["x"] * img_w, box["y"] * img_h
    w, h = box["w"] * img_w, box["h"] * img_h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def to_ratio(x1, y1, x2, y2, img_w, img_h):
    """像素坐标 -> 比例坐标（顺手夹在画面内，避免拖出边界）"""
    x1, x2 = sorted((max(0, min(x1, img_w)), max(0, min(x2, img_w))))
    y1, y2 = sorted((max(0, min(y1, img_h)), max(0, min(y2, img_h))))
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    return {"x": cx / img_w, "y": cy / img_h, "w": w / img_w, "h": h / img_h}


# ==================== 统计 ====================
def dataset_stats(classes=None):
    """数一数标了多少：图片数、框数、每个类别多少框"""
    classes = load_classes() if classes is None else classes
    per_class = {i: 0 for i in range(len(classes))}
    total = images_labelled = 0
    for p in all_label_files():
        boxes = read_label_file(p)
        if boxes:
            images_labelled += 1
        total += len(boxes)
        for b in boxes:
            if b["cls"] in per_class:
                per_class[b["cls"]] += 1
    return {
        "images_labelled": images_labelled,
        "total_boxes": total,
        "per_class": per_class,
        "class_names": classes,
    }


def class_counts_text(classes=None):
    st = dataset_stats(classes)
    lines = ["已标注图片 %d 张，共 %d 个框" % (st["images_labelled"], st["total_boxes"])]
    for i, name in enumerate(st["class_names"]):
        lines.append("  %-12s %d 个框" % (name, st["per_class"].get(i, 0)))
    return "\n".join(lines)
