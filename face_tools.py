"""
face_tools.py - 人脸识别工具箱（给摄像头用）

思路和大白话说明：
  1. YuNet  负责「找到脸在哪」      —— 模型 0.22 MB
  2. SFace  负责「这张脸是谁」      —— 模型 36.9 MB，把脸变成 128 个数字（特征向量）
  3. 两张脸的特征一比（余弦相似度），超过阈值就算同一个人
  4. 没见过的人自动编号：人A、人B、人C……

用它需要的东西：
  models/face_detection_yunet_2023mar.onnx
  models/face_recognition_sface_2021dec.onnx
（这两个文件已经下好放在项目 models/ 里了）

环境变量（可选，用来现场调参）：
  FACE_THRESHOLD  判定同一人的相似度阈值，默认 0.40
  FACE_MATCH_MODE 比对方式：cosine（默认）或 norml2
"""
import os
import json
import time
from datetime import datetime

import cv2
import numpy as np

# ==================== 路径与参数 ====================
MODELS_DIR = "models"
DETECT_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOG_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

FACES_JSON = os.path.join("output_results", "known_faces.json")
NEW_FACE_DIR = os.path.join("output_results", "new_faces")
FACE_LOG = os.path.join("logs", "face_events.jsonl")

# 判定同一人的阈值。官方给 SFace 的参考值是 0.363，但那是针对清晰真人照；
# 我拿你 images 里那些带口罩/偏卡通的图实测，0.25 会误认 5 对、0.363 才收敛到 1 对，
# 所以默认取 0.40 更稳。真人视频里可以适当往下调。
THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.40"))
MATCH_MODE = os.getenv("FACE_MATCH_MODE", "cosine").strip().lower()
if MATCH_MODE not in ("cosine", "norml2"):
    MATCH_MODE = "cosine"

# ---------------- 人脸质量门槛（提高准度的关键之一）----------------
# 为什么要有它？我在一张 92 人的合影上实测：脸从 150x188 一路小到 8x11 像素，
# 而 SFace 对小于 ~40 像素的脸，特征基本是噪声 —— 硬拿去比对必然张冠李戴。
# 所以宁可老实说「太远，认不准」，也不要认错人。
MIN_FACE_SIZE = int(os.getenv("FACE_MIN_SIZE", "60"))   # 脸的最短边小于这个值就不认
MIN_FACE_SCORE = float(os.getenv("FACE_MIN_SCORE", "0.5"))  # YuNet 置信度低于这个值就当没看见

# ---------------- 多帧投票（提高准度的关键之二）----------------
# 单帧判定太草率：一帧认错就永远错。改成「连续几帧都认成同一个人」才落定。
VOTE_FRAMES = int(os.getenv("FACE_VOTE_FRAMES", "3"))

# 每个人最多存几份特征（存多份、取最高分，比只存一份稳得多）
MAX_EMBEDDINGS = 3

# 每隔几秒才给同一个人沉淀一份新特征（避免 3 个名额被同一秒的近似画面占满）
EMBED_EVERY_SECONDS = float(os.getenv("FACE_EMBED_EVERY", "10"))

# 连续多少帧没见到同一张脸，才算他「离开了」（避免一闪而过就换编号）
EXPIRE_SECONDS = 3.0

# 「见过几次」怎么算：一个人从画面里消失超过这么多秒、然后又出现，才算新的一次见面。
# 否则他一直坐在镜头前，次数应该一直不变（我第一版写成每帧 +1，几分钟就涨到几百次，很离谱）。
SESSION_GAP_SECONDS = float(os.getenv("FACE_SESSION_GAP", "5"))

_DET = None
_REC = None


# ==================== 模型加载 ====================
def models_ready():
    """两个模型文件是不是都在"""
    return os.path.exists(DETECT_MODEL) and os.path.exists(RECOG_MODEL)


def _check_models():
    if not models_ready():
        raise FileNotFoundError(
            "缺少人脸模型文件：\n  %s\n  %s\n请把它们放到 models/ 文件夹里。"
            % (DETECT_MODEL, RECOG_MODEL)
        )


def get_detector():
    """YuNet 人脸检测器（只加载一次，重复用）"""
    global _DET
    if _DET is None:
        _check_models()
        _DET = cv2.FaceDetectorYN.create(
            DETECT_MODEL, "", (320, 320),
            score_threshold=0.7, nms_threshold=0.3, top_k=5000
        )
    return _DET


def get_recognizer():
    """SFace 人脸特征提取器（只加载一次，重复用）"""
    global _REC
    if _REC is None:
        _check_models()
        _REC = cv2.FaceRecognizerSF.create(RECOG_MODEL, "")
    return _REC


def detect_faces(frame, score_threshold=None):
    """找出画面里所有的脸

    返回列表，每项：
      {"bbox": [x,y,w,h], "score": 0.93, "feature": np.array(1,128),
       "landmarks": [...], "big_enough": True}
    """
    det = get_detector()
    rec = get_recognizer()
    if score_threshold is None:
        score_threshold = MIN_FACE_SCORE          # 默认放宽到 0.5，别再丢侧脸/模糊脸
    h, w = frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(frame)
    out = []
    if faces is None:
        return out
    for f in faces:
        conf = float(f[-1])
        if conf < score_threshold:
            continue
        x, y, bw, bh = [float(v) for v in f[:4]]
        aligned = rec.alignCrop(frame, f)
        try:
            feat = rec.feature(aligned).copy()
        except Exception:
            continue
        out.append({
            "bbox": [x, y, bw, bh],
            "score": round(conf, 3),
            "feature": feat,
            "landmarks": [round(float(v), 1) for v in f[4:14]],
            # 脸太小的话特征不可靠，标记出来，后面不拿它认人
            "big_enough": min(bw, bh) >= MIN_FACE_SIZE,
        })
    return out


def similarity(feat1, feat2):
    """两张脸有多像：越大越像。cosine 模式下 1.0 = 完全一样"""
    rec = get_recognizer()
    mode = (cv2.FaceRecognizerSF_FR_NORM_L2 if MATCH_MODE == "norml2"
            else cv2.FaceRecognizerSF_FR_COSINE)
    return float(rec.match(feat1, feat2, mode))


def is_same_person(feat1, feat2, threshold=None):
    """判断两张脸是不是同一个人。

    注意两种模式方向相反：
      cosine —— 越大越像（> 阈值 = 同一人）
      norml2 —— 越小越像（< 阈值 = 同一人），阈值按 1.128 换算
    """
    th = THRESHOLD if threshold is None else threshold
    s = similarity(feat1, feat2)
    if MATCH_MODE == "norml2":
        return (s < (1.128 if threshold is None else th)), s
    return (s > th), s


# ==================== 脸库：记住见过谁 ====================
def embeddings_of(person):
    """取出某个人的全部特征。

    兼容两种存法：
      · 新格式 person["embeddings"] = [vec1, vec2, ...]
      · 老格式 person["feature"]    = vec        （只有一份）
    这样旧的脸库文件不用转换也能继续用。
    """
    feats = person.get("embeddings")
    if not feats:
        one = person.get("feature")
        feats = [one] if one else []
    return [f for f in feats if f]


def next_label(n):
    """编号转名字：0->人A, 1->人B, ... 25->人Z, 26->人AA"""
    s, n = "", int(n)
    while True:
        s = chr(ord('A') + n % 26) + s
        n = n // 26 - 1
        if n < 0:
            break
    return "人" + s


def append_event(event, **fields):
    """往 logs/face_events.jsonl 记一行（谁什么时候出现、是不是新面孔）"""
    try:
        os.makedirs(os.path.dirname(FACE_LOG), exist_ok=True)
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "event": event}
        rec.update(fields)
        with open(FACE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


class FaceLibrary:
    """认识过的所有人。可以存盘、下次接着认。"""

    def __init__(self, path=FACES_JSON):
        self.path = path
        self.people = []
        # 每个人的「上次出现时刻」，只放内存不存盘（datetime 没法直接写进 JSON）
        self._presence_ts = {}

    # ---------- 存 / 读 ----------
    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.people = data.get("people", [])
        except FileNotFoundError:
            self.people = []
        except Exception as exc:
            print("⚠️ 脸库读不出来（%s），这次当空库" % exc)
            self.people = []
        return self

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            # 撇掉临时字段：_last_emb_ts 这类只用于内存计算，写进文件只会碍事
            people = []
            for p in self.people:
                people.append({k: v for k, v in p.items() if not k.startswith("_")})
            data = {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "threshold": THRESHOLD,
                "match_mode": MATCH_MODE,
                "people": people,
            }
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print("⚠️ 脸库没存上：%s" % exc)

    # ---------- 认人 ----------
    def match(self, feature, threshold=None):
        """拿一张脸去脸库里找。

        每个人可以存【多份】特征（不同角度/光线各存一份），
        比对时取"和任何一份最像"的那个分数 —— 这比只存一份稳得多。
        返回 (找到的人 或 None, 那个人的最高相似度)
        """
        best_person, best_score = None, -999.0
        for person in self.people:
            feats = embeddings_of(person)
            if not feats:
                continue
            top, hit = -999.0, False
            for stored in feats:
                arr = np.asarray(stored, dtype=np.float32).reshape(1, -1)
                if arr.size != 128:
                    continue
                same, score = is_same_person(feature, arr, threshold)
                if score > top:
                    top = score
                if same:
                    hit = True
            if top > best_score:
                best_score = top
            if hit:
                return person, top          # 多份特征里只要有任意一份对上，就算同一个人
        return best_person, best_score

    def update_embedding(self, person, feature):
        """把这张脸补进他的特征集（最多存 MAX_EMBEDDINGS 份）

        存多份的意义：同一个人换个角度/换种光线，总有一份能对上。
        """
        feats = embeddings_of(person)
        if len(feats) >= MAX_EMBEDDINGS:
            return person
        new = np.asarray(feature, dtype=np.float32).reshape(-1)
        for s in feats:
            arr = np.asarray(s, dtype=np.float32).reshape(-1)
            if arr.size == new.size:
                # 已经很像了就不用再存（相似度 > 0.75 认为重复）
                if similarity(feature, arr.reshape(1, -1)) > 0.75:
                    return person
        feats.append(new.tolist())
        # 兼容老格式：只留 feature 字段，读的时候由 embeddings_of 统一处理
        person["feature"] = feats[0]
        person["embeddings"] = feats
        return person

    def add(self, feature, snapshot_path=None, note=None):
        """登记一个新人"""
        label = next_label(len(self.people))
        vec = np.asarray(feature, dtype=np.float32).reshape(-1).tolist()
        person = {
            "id": label,
            "feature": vec,             # 老字段，保持兼容
            "embeddings": [vec],        # 新的多特征
            "first_seen": datetime.now().isoformat(timespec="seconds"),
            "last_seen": datetime.now().isoformat(timespec="seconds"),
            "times_seen": 1,
            "snapshot": snapshot_path,
            "note": note,
        }
        self.people.append(person)
        return person

    def touch(self, person):
        """这个人【重新出现】了：见面的次数 +1

        注意：这里只在「上次出现已经隔了一段空档」时才 +1。
        刚开始我写成了每帧 +1，结果同一张脸在画面里坐几分钟就变成「见过 900 次」，
        这种数字毫无意义。现在改成了：人走开一会儿再回来，才算新的一次见面。
        """
        now_dt = datetime.now()
        # 上次出现时间：优先用内存里的（准确），没有就从存盘的字符串恢复
        last_ts = self._presence_ts.get(person["id"])
        if last_ts is None:
            try:
                last_ts = datetime.fromisoformat(person.get("last_seen", "")).timestamp()
            except Exception:
                last_ts = None
        gap = (now_dt.timestamp() - last_ts) if last_ts else 9999
        if gap >= SESSION_GAP_SECONDS:
            person["times_seen"] = int(person.get("times_seen") or 0) + 1
        self._presence_ts[person["id"]] = now_dt.timestamp()

        person["last_seen"] = now_dt.isoformat(timespec="seconds")
        # 纯粹用来统计的「被看到多少帧」，不参与展示
        person["frames_seen"] = int(person.get("frames_seen") or 0) + 1
        return person

    def __len__(self):
        return len(self.people)

    def summary(self):
        return [
            {"id": p["id"], "times_seen": p.get("times_seen", 0),
             "first_seen": p.get("first_seen"), "snapshot": p.get("snapshot")}
            for p in self.people
        ]


def save_new_face_snapshot(frame, box, label):
    """把新面孔的抓拍存下来留证，返回文件路径"""
    try:
        os.makedirs(NEW_FACE_DIR, exist_ok=True)
        x, y, w, h = [int(v) for v in box]
        pad = int(max(w, h) * 0.35)      # 稍微扩一圈，别只留一张脸
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        path = os.path.join(NEW_FACE_DIR, "newface_%s_%s.jpg" % (
            label, datetime.now().strftime("%Y%m%d_%H%M%S")))
        cv2.imwrite(path, crop)
        return path
    except Exception as exc:
        print("⚠️ 抓拍没存上：%s" % exc)
        return None


# ==================== 认人追踪器（摄像头主循环用这个） ====================
class FaceTracker:
    """把「检测 → 认人 → 画框 → 提示」这条流水线打包起来

    用法：
        t = FaceTracker()
        t.load_library()                  # 把以前认识的人读回来
        info = t.process(frame)           # 每帧调一次，它会就地画框
        print(info["new"], info["repeated"])
    """

    def __init__(self, library_path=FACES_JSON, threshold=None, autosave=True):
        self.library = FaceLibrary(library_path)
        self.threshold = THRESHOLD if threshold is None else float(threshold)
        self.autosave = autosave
        self.counter = 0
        # 下面这些由 reset_session 统一初始化，这里先调一次，免得没开直播就 process 时崩
        self.session = {}        # 本场直播里的临时身份
        self.votes = {}          # 候选人的得票数（多帧投票用）
        self.candidates = {}     # 还在观察、尚未登记的新脸
        self._next_candidate = 0
        self.next_temp = 0

    # ---------- 准备 ----------
    def load_library(self):
        self.library.load()
        return self

    def reset_session(self):
        """新开一场直播，临时身份清零（脸库不动）"""
        self.session = {}
        self.votes = {}
        self.candidates = {}
        self._next_candidate = 0
        self.next_temp = 0

    # ---------- 主流程 ----------
    def process(self, frame, draw=True):
        """处理一帧。返回这一帧的统计信息

        多帧投票是怎么回事（这是提高准度的关键）：
          一张新脸不会第 1 帧就直接写进脸库，而是先进「候选区」，
          连续 FACE_VOTE_FRAMES 帧都稳定地认不出同一个人，才正式登记。
          这样偶发的一两帧误判就不会污染脸库。
        """
        faces = detect_faces(frame)
        now = time.time()
        self.counter += 1

        new_this_frame, repeat_this_frame, pending_this_frame = [], [], []

        for face in faces:
            # ---- 脸太小：不认人，只画个灰框提醒你「离远/太小，认不准」----
            if not face.get("big_enough", True):
                if draw:
                    self._draw(frame, face,
                               "too small %.2f" % face["score"], (128, 128, 128))
                pending_this_frame.append({"reason": "脸太小", "score": face["score"]})
                continue

            # ---- 先问脸库 ----
            person, score = self.library.match(face["feature"], self.threshold)

            if person is not None:
                # 认出来了：老面孔（见面临界判断都在 _presence 里，不在这里每帧 +1）
                self._presence(person, face["feature"], now)
                pid = person["id"]
                self._remember(pid, face["feature"], now)
                self._clear_votes_for(pid)
                tag = "%s x%d" % (pid, person["times_seen"])
                color = (0, 255, 0)                     # 绿框 = 认识的人
                repeat_this_frame.append({"id": pid, "score": round(score, 3)})

            else:
                # 没认出：可能是本场刚编过号的临时面孔
                pid, how = self._match_session(face["feature"], now)
                if how == "session":
                    self._remember(pid, face["feature"], now)
                    tag = "%s" % pid
                    color = (0, 255, 255)               # 黄框 = 本场直播里的临时面孔
                    repeat_this_frame.append({"id": pid, "score": round(score, 3)})
                else:
                    # 全新的脸 → 进候选区投票，票够了才登记
                    cid = self._candidate_for(face, now)
                    self.votes[cid] = self.votes.get(cid, 0) + 1
                    need = max(1, VOTE_FRAMES)
                    if self.votes[cid] < need:
                        tag = "new? %d/%d" % (self.votes[cid], need)
                        color = (0, 165, 255)           # 橙框 = 新面孔观察中
                        pending_this_frame.append({"id": cid, "votes": self.votes[cid],
                                                   "need": need, "score": face["score"]})
                    else:
                        # 票够了，正式登记
                        snap = (save_new_face_snapshot(frame, face["bbox"], "new")
                                if self.autosave else None)
                        person = self.library.add(face["feature"], snap,
                                                  note="摄像头自动登记（%d帧投票通过）" % need)
                        pid = person["id"]
                        self._remember(pid, face["feature"], now)
                        self._clear_votes_for(cid)
                        tag = "%s 新!" % pid
                        color = (0, 0, 255)             # 红框 = 新登记的人
                        rec = {"id": pid, "score": round(score, 3),
                               "snapshot": os.path.basename(snap) if snap else None,
                               "bbox": [round(v, 1) for v in face["bbox"]]}
                        new_this_frame.append(rec)
                        append_event("new_face", id=pid, score=round(score, 3),
                                     snapshot=rec["snapshot"], bbox=rec["bbox"],
                                     votes=need)
                        if self.autosave:
                            self.library.save()

            if draw:
                self._draw(frame, face, tag, color)

        # 清掉长时间没出现的临时身份和过期候选
        for tid in [k for k, v in self.session.items() if now - v["last_seen"] > EXPIRE_SECONDS]:
            self.session.pop(tid, None)
        for cid in [k for k, v in self.candidates.items() if now - v["last_seen"] > EXPIRE_SECONDS]:
            self.candidates.pop(cid, None)
            self.votes.pop(cid, None)

        return {
            "faces": len(faces),
            "new": new_this_frame,
            "repeated": repeat_this_frame,
            "pending": pending_this_frame,       # 还在投票、或脸太小的
            "library_size": len(self.library),
        }

    # ---------- 内部工具 ----------
    def _candidate_for(self, face, now):
        """给一张「没认出来的新脸」分配候选编号（同一张脸连续出现会复用同一个号）"""
        best_cid, best_score = None, -999.0
        for cid, info in self.candidates.items():
            for feat in info["features"]:
                s = similarity(face["feature"], feat)
                if s > best_score:
                    best_score, best_cid = s, cid
        if best_cid is not None and best_score > self.threshold:
            info = self.candidates[best_cid]
            info["last_seen"] = now
            if len(info["features"]) < MAX_EMBEDDINGS:
                info["features"].append(face["feature"])
            return best_cid
        self._next_candidate += 1
        cid = "cand%d" % self._next_candidate
        self.candidates[cid] = {"features": [face["feature"]], "last_seen": now}
        return cid

    def _clear_votes_for(self, key):
        self.votes.pop(key, None)
        self.candidates.pop(key, None)

    def _match_session(self, feature, now):
        """在「本场直播临时身份」里找。找到返回 (编号, 'session')，否则 (候选号, 'candidate')"""
        best_tid, best_score = None, -999.0
        for tid, info in self.session.items():
            for feat in info["features"]:
                s = similarity(feature, feat)
                if s > best_score:
                    best_score, best_tid = s, tid
        if best_tid is not None and best_score > self.threshold:
            return best_tid, "session"
        return None, "candidate"

    def _presence(self, person, feature, now):
        """这个人这一帧被看到了：该加次的加次，该沉淀特征的沉淀。

        为什么要单独这一步？
          因为原来每帧都无条件 touch() + update_embedding()：
            · 次数几分钟涨到几百，数字失去意义
            · 3 个特征名额被同一秒的近似画面占满，多角度特征根本存不进来
          现在改成：
            · 只有「离开过又回来」才把见面次数 +1
            · 每 EMBED_EVERY_SECONDS 秒才沉淀一份特征，让名额用在真正不同的画面上
        """
        self.library.touch(person)          # 内部自带空档判断，只有重新出现才 +1

        last_emb = person.get("_last_emb_ts")
        if last_emb is None or (now - last_emb) >= EMBED_EVERY_SECONDS:
            before = len(embeddings_of(person))
            self.library.update_embedding(person, feature)
            if len(embeddings_of(person)) > before:
                person["_last_emb_ts"] = now
        return person

    def _remember(self, pid, feature, now):
        info = self.session.get(pid)
        if info is None:
            self.session[pid] = {"features": [feature], "last_seen": now}
            return
        info["last_seen"] = now
        if len(info["features"]) < MAX_EMBEDDINGS:
            info["features"].append(feature)

    @staticmethod
    def _draw(frame, face, tag, color):
        x, y, w, h = [int(v) for v in face["bbox"]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = "%s %.2f" % (tag, face["score"])
        ty = y - 8 if y - 8 > 12 else y + h + 18
        cv2.putText(frame, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

