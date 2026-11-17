# ============================================================
# 1. Environment, paths, global configuration
# ============================================================
import os, sys, json, math, random, time, copy, gc, csv, re, string
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
IN_KAGGLE = Path("/kaggle").exists()
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)


def _is_edge_ai_project(root):
    d = Path(root) / "data"
    return all((d / f).is_file() for f in ("intent_train.csv", "intent_test.csv", "sample_submission.csv")) \
        and (d / "train" / "images").is_dir() and (d / "train" / "labels").is_dir() and (d / "test" / "images").is_dir()


_candidates = [Path(os.environ["EDGE_AI_ROOT"])] if os.environ.get("EDGE_AI_ROOT") else []
_candidates += [Path.cwd(), Path.cwd() / "edge_ai", *Path.cwd().parents,
                Path("/content/drive/MyDrive/edge_ai"), Path("/content/edge_ai"),
                Path("/kaggle/working/edge_ai"), Path("/kaggle/input/edge-ai")]
ROOT = next((c.resolve() for c in _candidates if _is_edge_ai_project(c)), None)
if ROOT is None:
    raise FileNotFoundError("Project folder with data/ not found. Set EDGE_AI_ROOT or place the notebook next to data/.")
DATA = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CHECKPOINTS_DIR = ROOT / "checkpoints"
INTENT_DIR = OUTPUT_DIR / "intent_v4"
STUDENT_DIR = CHECKPOINTS_DIR / "student_v4"
for d in (OUTPUT_DIR, CHECKPOINTS_DIR, INTENT_DIR, STUDENT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SMOKE_TEST = os.environ.get("EDGE_AI_SMOKE", "0") == "1"   # tiny subsets, 1 epoch, CPU allowed: for checking the code only
HAS_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 0
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3 if HAS_CUDA else 0.0
BIG_GPU = GPU_GB >= 12
DATA_WORKERS = 0 if os.name == "nt" else max(0, min(8, (os.cpu_count() or 2) - 1))


def _by_gpu(small, mid, big, huge):
    """pick a value by GPU memory: <8 GB, 8-14 GB, 14-22 GB, >=22 GB"""
    return small if GPU_GB < 8 else mid if GPU_GB < 14 else big if GPU_GB < 22 else huge


_student_batch = _by_gpu(8, 16, 32, 48)

CFG = {
    "seed": 42,
    "val_fraction": 0.2,                       # fixed image-level split shared by teacher and student
    # ---- intent ----
    "intent_epochs": 60, "intent_patience": 12, "intent_batch": 64, "intent_final_fit": True,
    "intent_final_epochs": 20,                 # fixed: grouped val saturates at epoch 1, so best_epoch is not informative
    "intent_swap_frac": 0.35,                  # in-scope question with its object swapped for a non-target noun -> OUT_OF_SCOPE
    "intent_eda_frac": 1.0,                    # one random word insert/delete copy per question (label kept)
    # ---- teacher (Ultralytics, full fine-tune, used offline only) ----
    "teacher_weights": ["yolo26m.pt"],         # add "yolo26x.pt" for a 2-teacher logit ensemble (needs more VRAM)
    "teacher_imgsz": 960, "teacher_epochs": 100, "teacher_patience": 25,
    "teacher_batch": _by_gpu(2, 4, 8, 16),            # yolo26m @960 with multi-scale up to 1.35x
    "teacher_multi_scale": 0.35,                     # 624..1296 px: covers the 640/768 sizes the student feeds it during KD
    "teacher_init_from_legacy": True,                # start from the merged LoRA teacher (val mAP50 0.9947) instead of COCO weights
    # ---- student (<=200k deployable params, 640x480 input, strides 8/16/32) ----
    "student_epochs": 200 if BIG_GPU else 160, "student_patience": 40,
    "student_batch": _student_batch,
    "student_lr": round(1.5e-3 * (_student_batch / 16) ** 0.5, 6), "student_min_lr": 1e-5,
    "student_warmup_epochs": 3, "student_weight_decay": 5e-4,
    "ema_decay": 0.9995,
    "train_sizes": [(512, 384), (640, 480), (768, 576)],   # multi-scale (W,H), all divisible by 32
    "val_size": (640, 480),
    "mosaic_prob": 0.30, "zoom_prob": 0.70, "zoom_range": (0.6, 1.5), "flip_prob": 0.5, "color_prob": 0.5,
    "cache_images": True,
    # ---- distillation ----
    "kd_feat_weight": 5.0, "kd_feat_tau": 1.0, "kd_logit_weight": 1.0, "kd_logit_T": 1.0,
    "kd_warmup_epochs": 1, "kd_ramp_epochs": 5,
    # ---- export ----
    "tta_hflip": False, "tta_sizes": [(704, 528)],   # no TTA / no ensemble: ONE pass at 704x528 (val mAP50 0.9837 vs 0.9829 @640)
    "nms_iou": 0.55,                                  # class-aware NMS IoU at inference (0.55 > 0.6 on val)
    "max_det": 100,                                   # competition limit: at most 100 boxes per image
}
if SMOKE_TEST:
    CFG.update(intent_epochs=2, intent_patience=2, teacher_epochs=1, student_epochs=1, student_batch=4,
               train_sizes=[(640, 480)], cache_images=False)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if HAS_CUDA:
        torch.cuda.manual_seed_all(seed)


seed_everything(CFG["seed"])
torch.backends.cudnn.benchmark = True


def load_ckpt(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


NC = 7
CLASS_NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]
NATIVE_W, NATIVE_H = 320, 240


def detector_train_val_paths(seed=None):
    """Fixed image-level split used by every detector stage (teacher validation, student validation)."""
    seed = CFG["seed"] if seed is None else seed
    paths = sorted((DATA / "train" / "images").glob("*.png"))
    random.Random(seed).shuffle(paths)
    n_val = int(round(CFG["val_fraction"] * len(paths)))
    train_paths, val_paths = paths[n_val:], paths[:n_val]
    if SMOKE_TEST:
        train_paths, val_paths = train_paths[:48], val_paths[:16]
    return train_paths, val_paths


def read_label_rows(path):
    rows = []
    label_path = DATA / "train" / "labels" / f"{Path(path).stem}.txt"
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) == 5:
                c, cx, cy, w, h = int(v[0]), *map(float, v[1:])
                if 0 <= c < NC and w > 0 and h > 0:
                    rows.append([c, cx, cy, w, h])
    return rows


print(f"ROOT={ROOT}\ndevice={DEVICE} gpu={GPU_GB:.1f} GB smoke={SMOKE_TEST}")
print(f"teacher batch={CFG['teacher_batch']} student batch={CFG['student_batch']} lr={CFG['student_lr']} epochs={CFG['student_epochs']} workers={DATA_WORKERS}")


# ============================================================
# 2. Intent TextCNN v4 (word + char CNN, scope/action heads, 199,852 params)
#    - validation groups = object-agnostic skeletons (no sentence frame shared by train and val)
#    - augmentation mined from TRAIN only:
#        * object swap: COUNT/EXIST/LOCATION question with its target object replaced by a non-target noun
#          taken from the train OUT_OF_SCOPE questions -> OUT_OF_SCOPE (in-scope intents always name one of the 7 classes)
#        * EDA: random insertion / deletion of one word (never inside the object phrase), label kept
#    - final fit on ALL labelled questions (+ augmentation) for a fixed number of epochs
# ============================================================
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

INTENT_NAMES = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
INTENT_TO_ID = {n: i for i, n in enumerate(INTENT_NAMES)}
WORD_VOCAB_SIZE, WORD_DIM = 512, 96
CHAR_VOCAB_SIZE, CHAR_DIM = 40, 24
WORD_FILTERS, CHAR_FILTERS = 64, 32
FUSION_DIM, MAX_WORDS, MAX_CHARS = 150, 32, 128
INTENT_PARAMS_EXPECTED = 199_852

OBJECT_ALIASES = [
    ("xe oto do choi", "toy_car_obj"), ("o to do choi", "toy_car_obj"), ("oto do choi", "toy_car_obj"),
    ("xe do choi", "toy_car_obj"), ("dien thoai di dong", "phone_obj"), ("smartphone", "phone_obj"),
    ("dien thoai", "phone_obj"), ("phone", "phone_obj"), ("chuot may tinh", "mouse_obj"), ("chuot", "mouse_obj"),
    ("quyen sach", "book_obj"), ("cuon sach", "book_obj"), ("sach", "book_obj"), ("binh nuoc", "bottle_obj"),
    ("chai nuoc", "bottle_obj"), ("chai nhua", "bottle_obj"), ("chai", "bottle_obj"), ("coc uong nuoc", "cup_obj"),
    ("coc", "cup_obj"), ("tach", "cup_obj"), ("ly", "cup_obj"), ("thung giay", "box_obj"), ("hop giay", "box_obj"),
    ("hop", "box_obj"),
]
POLITE_PREFIXES = ("cho minh hoi", "cho toi hoi", "ban oi", "xin hoi", "xin cho hoi", "hay cho toi hoi", "hay cho minh hoi")


def clean_intent_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_intent_text(text):
    text = clean_intent_text(text)
    for phrase, canonical in sorted(OBJECT_ALIASES, key=lambda it: len(it[0]), reverse=True):
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", canonical, text)
    return re.sub(r"\s+", " ", text).strip()


def intent_template_key(text):
    text = canonical_intent_text(text)
    for phrase in POLITE_PREFIXES:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)
    text = re.sub(r"\b(trong|buc|khung|hinh|anh|canh|nay|kia|day)\b", " IMG ", text)
    text = re.sub(r"\b\d+\b", " NUM ", text)
    return re.sub(r"\s+", " ", text).strip()


_ALIAS_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a, _ in sorted(OBJECT_ALIASES, key=lambda it: len(it[0]), reverse=True)) + r")\b")


def mine_non_target_nouns(frame):
    """Non-target object nouns from OUT_OF_SCOPE questions of `frame` ("X nam o dau ...", "co thay X khong")."""
    nouns = []
    for q in frame[frame.intent == "OUT_OF_SCOPE"].question.map(clean_intent_text):
        q = re.sub(r"^(cho minh hoi|cho toi hoi|ban oi|xin hoi)\s+", "", q)
        for pat in (r"^(.+?) nam o dau", r"co thay (.+?) khong"):
            m = re.search(pat, q)
            if m and not _ALIAS_RE.search(m.group(1)) and len(m.group(1).split()) <= 4 and m.group(1) not in nouns:
                nouns.append(m.group(1))
    return nouns


def intent_skeleton_key(text, nouns):
    key = re.sub(r"\w+_obj", "OBJ", intent_template_key(text))
    if nouns:
        key = re.sub(r"\b(" + "|".join(re.escape(n) for n in sorted(nouns, key=len, reverse=True)) + r")\b", "NOBJ", key)
    return key


def augment_intent_frame(frame, seed):
    rng = random.Random(seed)
    nouns = mine_non_target_nouns(frame)
    common = [w for w, _ in Counter(w for q in frame.question.map(clean_intent_text) for w in q.split()).most_common(300)]

    def eda(q):
        toks = q.split()
        if rng.random() < 0.5 and len(toks) > 3:
            spans = [m.span() for m in _ALIAS_RE.finditer(q)]
            i = rng.randrange(len(toks)); pos = len(" ".join(toks[:i])) + (1 if i else 0)
            if not any(a <= pos < b for a, b in spans):
                toks.pop(i)
        else:
            toks.insert(rng.randrange(len(toks) + 1), rng.choice(common))
        return " ".join(toks)

    rows = []
    for q, y in zip(frame.question, frame.intent):
        cq = clean_intent_text(q)
        if nouns and y in ("COUNT", "EXIST", "LOCATION") and _ALIAS_RE.search(cq) and rng.random() < CFG["intent_swap_frac"]:
            rows.append((_ALIAS_RE.sub(lambda m: rng.choice(nouns), cq, count=1), "OUT_OF_SCOPE"))
        if rng.random() < CFG["intent_eda_frac"]:
            rows.append((eda(cq), y))
    aug = pd.DataFrame(rows, columns=["question", "intent"])
    print(f"[intent] augmentation: {len(nouns)} non-target nouns, +{len(aug)} rows {dict(Counter(aug.intent))}")
    return pd.concat([frame[["question", "intent"]], aug], ignore_index=True)


def make_intent_vocab(texts):
    counts = Counter(tok for t in texts for tok in canonical_intent_text(t).split())
    words = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(WORD_VOCAB_SIZE - 2)]
    words += [f"<extra_{i}>" for i in range(WORD_VOCAB_SIZE - len(words))]
    return {w: i for i, w in enumerate(words)}


CHAR_ALPHABET = string.ascii_lowercase + string.digits + "_ "
CHAR_TO_ID = {"<pad>": 0, "<unk>": 1}
CHAR_TO_ID.update({c: i + 2 for i, c in enumerate(CHAR_ALPHABET)})
assert len(CHAR_TO_ID) == CHAR_VOCAB_SIZE


def encode_words(text, vocab):
    ids = [vocab.get(t, 1) for t in canonical_intent_text(text).split()[:MAX_WORDS]]
    return ids + [0] * (MAX_WORDS - len(ids))


def encode_chars(text):
    ids = [CHAR_TO_ID.get(c, 1) for c in clean_intent_text(text)[:MAX_CHARS]]
    return ids + [0] * (MAX_CHARS - len(ids))


class IntentDataset(Dataset):
    def __init__(self, frame, vocab):
        self.words = torch.tensor([encode_words(q, vocab) for q in frame.question], dtype=torch.long)
        self.chars = torch.tensor([encode_chars(q) for q in frame.question], dtype=torch.long)
        self.labels = torch.tensor([INTENT_TO_ID[y] for y in frame.intent], dtype=torch.long)
        self.scope = (self.labels != INTENT_TO_ID["OUT_OF_SCOPE"]).long()
        self.operation = self.labels.clone()
        self.operation[self.labels == INTENT_TO_ID["OUT_OF_SCOPE"]] = -100

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.words[i], self.chars[i], self.labels[i], self.scope[i], self.operation[i]


class IntentTextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embedding = nn.Embedding(WORD_VOCAB_SIZE, WORD_DIM, padding_idx=0)
        self.char_embedding = nn.Embedding(CHAR_VOCAB_SIZE, CHAR_DIM, padding_idx=0)
        self.word_convs = nn.ModuleList([nn.Conv1d(WORD_DIM, WORD_FILTERS, k) for k in (2, 3, 4, 5)])
        self.char_convs = nn.ModuleList([nn.Conv1d(CHAR_DIM, CHAR_FILTERS, k) for k in (3, 4, 5)])
        self.fusion = nn.Linear(WORD_FILTERS * 4 + CHAR_FILTERS * 3, FUSION_DIM)
        self.norm = nn.LayerNorm(FUSION_DIM)
        self.dropout = nn.Dropout(0.4)
        self.scope_head = nn.Linear(FUSION_DIM, 2)
        self.operation_head = nn.Linear(FUSION_DIM, 4)

    @staticmethod
    def _masked_pool(conv, emb, mask):
        width = conv.kernel_size[0]
        values = F.relu(conv(emb.transpose(1, 2)))
        valid = mask.unfold(1, width, 1).any(dim=-1)
        return values.masked_fill(~valid.unsqueeze(1), -1e4).max(dim=-1).values

    def forward(self, words, chars):
        wm, cm = words.ne(0), chars.ne(0)
        pooled = [self._masked_pool(c, self.word_embedding(words), wm) for c in self.word_convs]
        pooled += [self._masked_pool(c, self.char_embedding(chars), cm) for c in self.char_convs]
        feats = self.dropout(self.norm(F.relu(self.fusion(torch.cat(pooled, 1)))))
        return self.scope_head(feats), self.operation_head(feats)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def intent_predict_ids(model, loader, device):
    model.eval(); preds, gold = [], []
    with torch.no_grad():
        for words, chars, labels, _, _ in loader:
            s, o = model(words.to(device), chars.to(device))
            sp, op = s.argmax(1).cpu(), o.argmax(1).cpu()
            preds.extend(torch.where(sp == 0, INTENT_TO_ID["OUT_OF_SCOPE"], op).tolist()); gold.extend(labels.tolist())
    return preds, gold


def _fit_intent(train_frame, vocab, epochs, device, val_loader=None, patience_limit=None, tag=""):
    model = IntentTextCNN().to(device)
    assert count_params(model) == INTENT_PARAMS_EXPECTED, count_params(model)
    loader = DataLoader(IntentDataset(train_frame, vocab), batch_size=CFG["intent_batch"], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-5)
    best_f1, best_epoch, best_state, patience, history = -1.0, 0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for words, chars, labels, scope, operation in loader:
            words, chars, scope, operation = words.to(device), chars.to(device), scope.to(device), operation.to(device)
            s, o = model(words, chars)
            loss = F.cross_entropy(s, scope)
            valid = operation.ne(-100)
            if valid.any():
                loss = loss + F.cross_entropy(o[valid], operation[valid])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            total += float(loss) * len(words)
        sched.step()
        row = {"epoch": epoch, "train_loss": total / len(train_frame)}
        if val_loader is not None:
            preds, gold = intent_predict_ids(model, val_loader, device)
            row["val_accuracy"] = accuracy_score(gold, preds); row["val_macro_f1"] = f1_score(gold, preds, average="macro")
            if row["val_macro_f1"] > best_f1 + 1e-6:
                best_f1, best_epoch, patience = row["val_macro_f1"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
        history.append(row)
        print(f"[intent{tag}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if patience_limit and patience >= patience_limit:
            break
    if best_state is None:
        best_state, best_epoch = copy.deepcopy(model.state_dict()), epochs
    return best_state, best_epoch, best_f1, history


def train_intent_v4():
    device = DEVICE
    best_path = INTENT_DIR / "best.pt"
    if best_path.exists() and not SMOKE_TEST:
        st = load_ckpt(best_path)
        print("Intent v4 already trained:", best_path, "val_acc=", st.get("val_accuracy"), "fit=", st.get("fit_mode"))
        return best_path
    frame = pd.read_csv(DATA / "intent_train.csv")
    if SMOKE_TEST:
        frame = frame.sample(400, random_state=0).reset_index(drop=True)
    _nouns_all = mine_non_target_nouns(frame)
    groups = frame.question.map(lambda q: intent_skeleton_key(q, _nouns_all)).to_numpy()
    tr_idx, va_idx = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG["seed"]).split(frame.question, frame.intent, groups))
    train_frame, val_frame = frame.iloc[tr_idx].reset_index(drop=True), frame.iloc[va_idx].reset_index(drop=True)
    assert not (set(groups[tr_idx]) & set(groups[va_idx])), "template leakage"
    train_frame = augment_intent_frame(train_frame, CFG["seed"])     # nouns mined from the train fold only
    vocab = make_intent_vocab(train_frame.question)
    val_loader = DataLoader(IntentDataset(val_frame, vocab), batch_size=256, shuffle=False)
    print(f"[intent] train={len(train_frame)} val={len(val_frame)} templates={len(set(groups))}")
    state, best_epoch, best_f1, history = _fit_intent(train_frame, vocab, CFG["intent_epochs"], device, val_loader, CFG["intent_patience"], tag="/val")
    model = IntentTextCNN().to(device); model.load_state_dict(state)
    preds, gold = intent_predict_ids(model, val_loader, device)
    val_acc = accuracy_score(gold, preds)
    print(classification_report(gold, preds, labels=list(range(5)), target_names=INTENT_NAMES, zero_division=0))
    torch.save({"model": state, "word_vocab": vocab, "char_vocab": CHAR_TO_ID, "intent_names": INTENT_NAMES,
                "params": INTENT_PARAMS_EXPECTED, "epoch": best_epoch, "val_accuracy": val_acc, "val_macro_f1": best_f1,
                "fit_mode": "grouped_val"}, INTENT_DIR / "val_run_best.pt")
    pd.DataFrame(history).to_csv(INTENT_DIR / "history_val_run.csv", index=False)
    if CFG["intent_final_fit"]:
        # Refit on every labelled question (+ augmentation) with the vocabulary of the full set, fixed epoch count.
        final_epochs = CFG["intent_final_epochs"]
        seed_everything(CFG["seed"])
        frame_all = augment_intent_frame(frame, CFG["seed"])
        vocab_all = make_intent_vocab(frame_all.question)
        state_all, _, _, history_all = _fit_intent(frame_all, vocab_all, final_epochs, device, tag="/final")
        torch.save({"model": state_all, "word_vocab": vocab_all, "char_vocab": CHAR_TO_ID, "intent_names": INTENT_NAMES,
                    "params": INTENT_PARAMS_EXPECTED, "epoch": final_epochs, "val_accuracy": val_acc, "val_macro_f1": best_f1,
                    "fit_mode": "final_fit_all_data_aug", "val_run_best_epoch": best_epoch}, best_path)
        pd.DataFrame(history_all).to_csv(INTENT_DIR / "history_final_fit.csv", index=False)
    else:
        torch.save(load_ckpt(INTENT_DIR / "val_run_best.pt"), best_path)
    print(f"saved {best_path} (grouped-val acc={val_acc:.4f}, macro_f1={best_f1:.4f} @ epoch {best_epoch})")
    return best_path


intent_checkpoint = train_intent_v4()


# ============================================================
# 3. Teacher: full fine-tune of YOLO26 (pretrained allowed for the teacher only)
#    - imgsz 960 + multi-scale so the teacher is strongest on small objects
#    - best checkpoint selected by mAP@0.5 (the leaderboard metric)
#    - LoRA helpers are kept only so the older LoRA teacher checkpoint can still be loaded for KD
# ============================================================
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

TEACHER_SEED = CFG["seed"]


class LoRAConv2d(nn.Module):
    """Compatibility shim: lets checkpoints/teacher_yolo (LoRA teacher) unpickle and merge."""
    def __init__(self, base_conv, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base_conv
        self.rank, self.alpha = int(rank), float(alpha)
        self.scale = self.alpha / self.rank
        self.lora_A = nn.Conv2d(base_conv.in_channels, self.rank, kernel_size=base_conv.kernel_size, stride=base_conv.stride,
                                padding=base_conv.padding, dilation=base_conv.dilation, bias=False, padding_mode=base_conv.padding_mode)
        self.lora_B = nn.Conv2d(self.rank, base_conv.out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    # pass-throughs so Ultralytics fuse()/export code can treat the wrapper like a Conv2d
    weight = property(lambda self: self.base.weight)
    bias = property(lambda self: self.base.bias)
    stride = property(lambda self: self.base.stride)
    padding = property(lambda self: self.base.padding)
    dilation = property(lambda self: self.base.dilation)
    groups = property(lambda self: self.base.groups)
    kernel_size = property(lambda self: self.base.kernel_size)
    in_channels = property(lambda self: self.base.in_channels)
    out_channels = property(lambda self: self.base.out_channels)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))

    @torch.no_grad()
    def merged_base(self):
        delta = torch.einsum("or,rihw->oihw", self.lora_B.weight[:, :, 0, 0], self.lora_A.weight)
        self.base.weight.add_(self.scale * delta.to(self.base.weight.dtype))
        return self.base


def merge_yolo_lora(model):
    merged = 0

    def visit(parent):
        nonlocal merged
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRAConv2d):
                setattr(parent, name, child.merged_base()); merged += 1
            else:
                visit(child)
    visit(model)
    return merged


class MAP50Trainer(DetectionTrainer):
    """Select best.pt by mAP@0.5 instead of the default 0.1*mAP50 + 0.9*mAP50-95 fitness."""
    def validate(self):
        previous_best = self.best_fitness
        metrics, fitness = super().validate()
        if metrics is None:
            return metrics, fitness
        fitness = float(metrics["metrics/mAP50(B)"])
        self.best_fitness = fitness if previous_best is None else max(previous_best, fitness)
        return metrics, fitness


def make_teacher_data_yaml():
    train_paths, val_paths = detector_train_val_paths()
    train_list, val_list = OUTPUT_DIR / "teacher_train_images.txt", OUTPUT_DIR / "teacher_val_images.txt"
    train_list.write_text("".join(f"{p.resolve()}\n" for p in train_paths), encoding="utf-8")
    val_list.write_text("".join(f"{p.resolve()}\n" for p in val_paths), encoding="utf-8")
    data_yaml = OUTPUT_DIR / "teacher_data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(DATA.resolve()), "train": str(train_list.resolve()),
                                         "val": str(val_list.resolve()), "names": {i: n for i, n in enumerate(CLASS_NAMES)}},
                                        sort_keys=False), encoding="utf-8")
    return data_yaml


LEGACY_LORA_TEACHER = CHECKPOINTS_DIR / "teacher_yolo" / "weights" / "best.pt"


def merged_legacy_teacher_init(weights_name):
    """Merge the LoRA teacher into plain convolutions and save an Ultralytics checkpoint usable as training init.
    Only applies to the same base architecture (yolo26m)."""
    if not (CFG["teacher_init_from_legacy"] and LEGACY_LORA_TEACHER.exists() and Path(weights_name).stem == "yolo26m"):
        return None
    out = OUTPUT_DIR / "teacher_init_yolo26m_from_lora_merged.pt"
    if not out.exists():
        y = YOLO(str(LEGACY_LORA_TEACHER))
        merged = merge_yolo_lora(y.model)
        y.model.float()
        y.save(str(out))                      # Model.save() stores the merged in-memory model and drops the EMA
        print(f"merged {merged} LoRA layers of the legacy teacher -> {out}")
    return out


def train_teacher(weights_name):
    """Full fine-tune `weights_name` (e.g. yolo26m.pt). Resumable. Returns the best checkpoint path."""
    stem = Path(weights_name).stem
    run_dir = CHECKPOINTS_DIR / f"teacher_{stem}_full"
    best_path, last_path = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if best_path.exists() and status.get("completed"):
        print(f"Teacher {stem}: reusing completed run {best_path} (mAP50={status.get('best_map50')})")
        return best_path
    if SMOKE_TEST:
        print("SMOKE_TEST: skipping teacher training.")
        return best_path if best_path.exists() else None
    data_yaml = make_teacher_data_yaml()
    init = merged_legacy_teacher_init(weights_name) or (ROOT / weights_name if (ROOT / weights_name).exists() else weights_name)
    print(f"Teacher {stem}: init weights = {init}")
    common = dict(trainer=MAP50Trainer, device=0 if HAS_CUDA else "cpu", workers=DATA_WORKERS)
    resumed = False
    if last_path.exists():
        try:
            print(f"Teacher {stem}: resuming from {last_path}")
            YOLO(str(last_path)).train(resume=True, **common)
            resumed = True
        except AssertionError as exc:          # "nothing to resume": the run already reached its last epoch
            print(f"Teacher {stem}: resume not possible ({exc}); treating the run as finished")
            resumed = best_path.exists()
    if not resumed:
        model = YOLO(str(init))
        model.train(
            data=str(data_yaml), imgsz=CFG["teacher_imgsz"], epochs=CFG["teacher_epochs"], batch=CFG["teacher_batch"],
            pretrained=True, optimizer="MuSGD", lr0=4e-4, lrf=0.05, momentum=0.948, weight_decay=2.7e-4,
            warmup_epochs=3.0, nbs=64, cos_lr=True, patience=CFG["teacher_patience"], save_period=5,
            multi_scale=CFG["teacher_multi_scale"],
            scale=0.6, mosaic=1.0, close_mosaic=15, mixup=0.05, copy_paste=0.0, degrees=0.0, translate=0.1,
            fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, erasing=0.0,
            project=str(CHECKPOINTS_DIR), name=run_dir.name, exist_ok=True, seed=TEACHER_SEED, deterministic=False,
            amp=True, plots=False, val=True, **common,
        )
    results_csv = run_dir / "results.csv"
    best_map50 = None
    if results_csv.exists():
        frame = pd.read_csv(results_csv); frame.columns = [c.strip() for c in frame.columns]
        best_map50 = float(frame["metrics/mAP50(B)"].max())
    status_path.write_text(json.dumps({"completed": True, "weights": weights_name, "imgsz": CFG["teacher_imgsz"],
                                       "best_map50": best_map50, "best": str(best_path)}, indent=2))
    print(f"Teacher {stem} done: best mAP50={best_map50} -> {best_path}")
    return best_path


teacher_checkpoints = []
for _w in CFG["teacher_weights"]:
    _ckpt = train_teacher(_w)
    if _ckpt is not None and Path(_ckpt).exists():
        teacher_checkpoints.append(Path(_ckpt))
# Fallback so the student can start while a new teacher is still training elsewhere: the earlier LoRA teacher (val mAP50 0.9947).
_legacy = CHECKPOINTS_DIR / "teacher_yolo" / "weights" / "best.pt"
if not teacher_checkpoints and _legacy.exists():
    print("No full-finetuned teacher yet; falling back to the legacy LoRA teacher:", _legacy)
    teacher_checkpoints = [_legacy]
if not teacher_checkpoints:
    raise FileNotFoundError("No teacher checkpoint available for distillation.")
print("Teachers for KD:", [str(p) for p in teacher_checkpoints])


# ============================================================
# 4. Student V4: 640x480 input, predictions at strides 8/16/32 (grids 60x80 / 30x40 / 15x20)
#    - RepDW blocks (dw5x5 + dw3x3 + identity, fused to ONE dw5x5 at deploy)
#    - shared head across levels (per-level bias), REG_MAX=16, IoU-aware branch
#    - deployable parameter count (after fusing BN / rep branches) must be <= 200,000
# ============================================================
STRIDES = (8, 16, 32)
REG_MAX = 16
# Overlapping size ranges (max box side in px at the 640x480 training/val resolution): each level
# sees every class, and objects near a boundary are learned by two levels.
LEVEL_RANGES = ((0.0, 160.0), (96.0, 320.0), (192.0, float("inf")))
CENTER_RADIUS = 1.5
OUT_CH = NC + 4 * REG_MAX + 1          # [cls logits | 4*REG_MAX distance logits | IoU logit]
PARAM_LIMIT = 200_000


def _fold_bn(weight, bn):
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    return weight * scale.view(-1, 1, 1, 1), bn.bias - bn.running_mean * scale


@torch.no_grad()
def fuse_conv_bn(conv, bn):
    w, b = _fold_bn(conv.weight, bn)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True).to(conv.weight.device)
    fused.weight.copy_(w)
    fused.bias.copy_(b + (conv.bias if conv.bias is not None else 0.0))
    return fused


class RepDW(nn.Module):
    """Depthwise 5x5 + 3x3 + identity (train) == one depthwise 5x5 with bias (deploy)."""
    def __init__(self, c, stride=1):
        super().__init__()
        self.c, self.stride = c, stride
        self.dw5 = nn.Conv2d(c, c, 5, stride, 2, groups=c, bias=False); self.bn5 = nn.BatchNorm2d(c)
        self.dw3 = nn.Conv2d(c, c, 3, stride, 1, groups=c, bias=False); self.bn3 = nn.BatchNorm2d(c)
        self.bn_id = nn.BatchNorm2d(c) if stride == 1 else None

    def forward(self, x):
        y = self.bn5(self.dw5(x)) + self.bn3(self.dw3(x))
        if self.bn_id is not None:
            y = y + self.bn_id(x)
        return y

    @torch.no_grad()
    def fuse(self):
        k5, b5 = _fold_bn(self.dw5.weight, self.bn5)
        k3, b3 = _fold_bn(self.dw3.weight, self.bn3)
        k, b = k5 + F.pad(k3, [1, 1, 1, 1]), b5 + b3
        if self.bn_id is not None:
            kid = torch.zeros_like(k5); kid[:, 0, 2, 2] = 1.0
            kid, bid = _fold_bn(kid, self.bn_id)
            k, b = k + kid, b + bid
        conv = nn.Conv2d(self.c, self.c, 5, self.stride, 2, groups=self.c, bias=True).to(k.device)
        conv.weight.copy_(k); conv.bias.copy_(b)
        return conv


class ConvBNAct(nn.Module):
    def __init__(self, ci, co, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(co)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class DSBlock(nn.Module):
    """RepDW(5x5) -> SiLU -> 1x1 -> BN, residual (1x1+BN projection when shape changes), SiLU."""
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.dw = RepDW(ci, stride)
        self.pw = nn.Conv2d(ci, co, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(co)
        self.skip = nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co)) if (ci != co or stride != 1) else nn.Identity()

    def forward(self, x):
        y = F.silu(self.dw(x))
        y = self.pw_bn(self.pw(y))
        return F.silu(y + self.skip(x))


def ds_stack(c, n):
    return nn.Sequential(*[DSBlock(c, c) for _ in range(n)])


class SharedHead(nn.Module):
    """One tower shared by all levels (saves ~30k params vs per-level towers); per-level output biases."""
    def __init__(self, w, levels=3):
        super().__init__()
        self.tower = ds_stack(w, 2)
        self.cls_block = DSBlock(w, w)
        self.reg_block = DSBlock(w, w)
        self.cls_out = nn.Conv2d(w, NC, 1)
        self.reg_out = nn.Conv2d(w, 4 * REG_MAX + 1, 1)
        self.level_cls_bias = nn.Parameter(torch.zeros(levels, NC))
        self.level_reg_bias = nn.Parameter(torch.zeros(levels, 4 * REG_MAX + 1))
        bins = torch.arange(REG_MAX, dtype=torch.float32)
        nn.init.normal_(self.cls_out.weight, std=0.01); nn.init.normal_(self.reg_out.weight, std=0.01)
        with torch.no_grad():
            self.cls_out.bias.fill_(-4.6)
            self.reg_out.bias[:4 * REG_MAX].copy_(-torch.abs(bins - 3.0).repeat(4))
            self.reg_out.bias[-1] = -2.0

    def forward(self, feats):
        outs = []
        for i, f in enumerate(feats):
            t = self.tower(f)
            cls = self.cls_out(self.cls_block(t)) + self.level_cls_bias[i].view(1, -1, 1, 1)
            reg = self.reg_out(self.reg_block(t)) + self.level_reg_bias[i].view(1, -1, 1, 1)
            outs.append(torch.cat([cls, reg], 1))
        return outs


class StudentV4(nn.Module):
    def __init__(self, w=80, c1=40, c2=56, c3=72, c3b=88, c4=104):
        super().__init__()
        self.stem = ConvBNAct(3, 24, 3, 2)                                            # 320x240
        self.s1 = nn.Sequential(DSBlock(24, c1, 2), DSBlock(c1, c1))                  # 160x120
        self.s2 = nn.Sequential(DSBlock(c1, c2, 2), DSBlock(c2, c2), DSBlock(c2, c2))  # 80x60  -> P3
        self.s3 = nn.Sequential(DSBlock(c2, c3, 2), DSBlock(c3, c3), DSBlock(c3, c3b))  # 40x30 -> P4
        self.s4 = nn.Sequential(DSBlock(c3b, c4, 2), DSBlock(c4, c4))                 # 20x15  -> P5
        self.l5, self.l4, self.l3 = ConvBNAct(c4, w, 1), ConvBNAct(c3b, w, 1), ConvBNAct(c2, w, 1)
        self.f5, self.f4, self.f3 = ds_stack(w, 1), ds_stack(w, 2), ds_stack(w, 3)
        self.head = SharedHead(w)
        self.neck_width = w

    def forward(self, x, return_feats=False):
        x = self.stem(x)
        c3 = self.s2(self.s1(x))
        c4 = self.s3(c3)
        c5 = self.s4(c4)
        p5 = self.f5(self.l5(c5))
        p4 = self.f4(self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.f3(self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        feats = [p3, p4, p5]
        outs = self.head(feats)
        return (outs, feats) if return_feats else outs


@torch.no_grad()
def to_deploy(model):
    """Return a fused copy: RepDW -> single dw5x5, Conv+BN -> Conv with bias. Same outputs in eval mode."""
    deploy = copy.deepcopy(model).eval()

    def visit(module):
        for name, child in list(module.named_children()):
            if isinstance(child, RepDW):
                setattr(module, name, child.fuse())
            elif isinstance(child, ConvBNAct):
                child.conv = fuse_conv_bn(child.conv, child.bn); child.bn = nn.Identity()
            elif isinstance(child, DSBlock):
                child.dw = child.dw.fuse()
                child.pw = fuse_conv_bn(child.pw, child.pw_bn); child.pw_bn = nn.Identity()
                if isinstance(child.skip, nn.Sequential):
                    child.skip = fuse_conv_bn(child.skip[0], child.skip[1])
            else:
                visit(child)
    visit(deploy)
    return deploy


def count_deploy_params(model):
    return sum(p.numel() for p in to_deploy(model).parameters())


def build_deploy_model():
    """Empty StudentV4 in its deployable (fused) form: every conv is a plain Conv2d with bias, no BatchNorm and no
    re-param branches. Nothing is computed here; the weights come from checkpoints/student_v4/deploy.pt."""
    model = StudentV4().eval()

    def plain(conv):
        return nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                         conv.dilation, conv.groups, bias=True)

    def visit(module):
        for name, child in list(module.named_children()):
            if isinstance(child, ConvBNAct):
                child.conv = plain(child.conv); child.bn = nn.Identity()
            elif isinstance(child, DSBlock):
                child.dw = plain(child.dw.dw5)                       # RepDW -> one depthwise 5x5 with bias
                child.pw = plain(child.pw); child.pw_bn = nn.Identity()
                if isinstance(child.skip, nn.Sequential):
                    child.skip = plain(child.skip[0])
            elif isinstance(child, RepDW):
                setattr(module, name, plain(child.dw5))
            else:
                visit(child)
    visit(model)
    return model


def expected_ltrb(raw, stride):
    """Distance logits -> expected ltrb in input pixels. raw: [B, OUT_CH, H, W]."""
    b, _, h, w = raw.shape
    dist = raw[:, NC:NC + 4 * REG_MAX].float().reshape(b, 4, REG_MAX, h, w)
    bins = torch.arange(REG_MAX, device=raw.device, dtype=torch.float32).view(1, 1, REG_MAX, 1, 1)
    return (dist.softmax(2) * bins).sum(2) * stride


_student = StudentV4()
_train_params = sum(p.numel() for p in _student.parameters())
_deploy_params = count_deploy_params(_student)
with torch.no_grad():
    _x = torch.randn(1, 3, 480, 640)
    _student.eval()
    _a = _student(_x); _b = to_deploy(_student)(_x)
    _max_diff = max(float((u - v).abs().max()) for u, v in zip(_a, _b))
print(f"StudentV4 params: train-time={_train_params:,}  deployable(fused)={_deploy_params:,}  limit={PARAM_LIMIT:,}")
print(f"fused-vs-unfused max abs diff: {_max_diff:.2e}; grids @640x480: {[tuple(o.shape[-2:]) for o in _a]}")
assert _deploy_params <= PARAM_LIMIT, "deployable parameter count exceeds the competition limit"
assert _max_diff < 1e-3
_skeleton = build_deploy_model()
assert {k: v.shape for k, v in _skeleton.state_dict().items()} == {k: v.shape for k, v in to_deploy(_student).state_dict().items()}
print(f"deploy skeleton (for deploy.pt): {sum(p.numel() for p in _skeleton.parameters()):,} params, same layout as the fused model")
del _student, _x, _a, _b, _skeleton


# ============================================================
# 5. Data pipeline (native 320x240 augmentation, GPU upsampling), target assignment,
#    detection loss (VariFocal + IoU branch + GIoU + DFL), decoding and mAP@0.5 evaluation
# ============================================================
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import batched_nms

_IMAGE_CACHE = {}


def load_native_image(path):
    """uint8 HxWx3 RGB at native 320x240 (cached in RAM when CFG['cache_images'])."""
    key = str(path)
    arr = _IMAGE_CACHE.get(key)
    if arr is None:
        img = Image.open(path).convert("RGB")
        if img.size != (NATIVE_W, NATIVE_H):
            img = img.resize((NATIVE_W, NATIVE_H), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        if CFG["cache_images"]:
            _IMAGE_CACHE[key] = arr
    return arr


def preload_images(paths, tag=""):
    if not CFG["cache_images"]:
        return
    t0 = time.time()
    for p in paths:
        load_native_image(p)
    print(f"cached {len(paths)} {tag} images in RAM ({time.time() - t0:.0f}s, ~{len(paths) * NATIVE_W * NATIVE_H * 3 / 1e9:.2f} GB)")


def transform_rows(rows, sx, sy, dx, dy, out_w, out_h, keep_visible=0.4):
    """rows normalized in the source image; the source is scaled by (sx, sy) [px] and shifted by (dx, dy) [px]
    into an out_w x out_h canvas. Returns normalized rows in the canvas, clipped, dropping mostly-hidden boxes."""
    out = []
    for c, cx, cy, w, h in rows:
        x1, y1 = cx * sx - w * sx / 2 + dx, cy * sy - h * sy / 2 + dy
        x2, y2 = cx * sx + w * sx / 2 + dx, cy * sy + h * sy / 2 + dy
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        ix1, iy1, ix2, iy2 = max(0.0, x1), max(0.0, y1), min(float(out_w), x2), min(float(out_h), y2)
        if ix2 - ix1 < 2 or iy2 - iy1 < 2:
            continue
        if (ix2 - ix1) * (iy2 - iy1) / area < keep_visible:
            continue
        out.append([c, (ix1 + ix2) / (2 * out_w), (iy1 + iy2) / (2 * out_h), (ix2 - ix1) / out_w, (iy2 - iy1) / out_h])
    return out


def _resize(arr, w, h):
    return np.asarray(Image.fromarray(arr).resize((max(1, int(w)), max(1, int(h))), Image.Resampling.BILINEAR), dtype=np.uint8)


def zoom_aug(arr, rows, scale):
    """scale > 1: zoom in (random crop); scale < 1: zoom out (random placement on a gray canvas)."""
    sw, sh = max(8, round(NATIVE_W * scale)), max(8, round(NATIVE_H * scale))
    scaled = _resize(arr, sw, sh)
    canvas = np.full((NATIVE_H, NATIVE_W, 3), 114, dtype=np.uint8)
    if scale >= 1.0:
        left, top = random.randint(0, sw - NATIVE_W), random.randint(0, sh - NATIVE_H)
        canvas[:] = scaled[top:top + NATIVE_H, left:left + NATIVE_W]
        dx, dy = -left, -top
    else:
        dx, dy = random.randint(0, NATIVE_W - sw), random.randint(0, NATIVE_H - sh)
        canvas[dy:dy + sh, dx:dx + sw] = scaled
    return canvas, transform_rows(rows, sw, sh, dx, dy, NATIVE_W, NATIVE_H)


def mosaic_aug(items):
    """items: 4 x (arr, rows). 2x2 mosaic of native images, scaled 0.5-0.85, then a 320x240 crop."""
    big = np.zeros((NATIVE_H * 2, NATIVE_W * 2, 3), dtype=np.uint8)
    rows_big = []
    for k, (arr, rows) in enumerate(items):
        ox, oy = (k % 2) * NATIVE_W, (k // 2) * NATIVE_H
        big[oy:oy + NATIVE_H, ox:ox + NATIVE_W] = arr
        rows_big += transform_rows(rows, NATIVE_W, NATIVE_H, ox, oy, NATIVE_W * 2, NATIVE_H * 2, keep_visible=0.0)
    s = random.uniform(0.5, 0.85)
    sw, sh = round(NATIVE_W * 2 * s), round(NATIVE_H * 2 * s)
    scaled = _resize(big, sw, sh)
    canvas = np.full((NATIVE_H, NATIVE_W, 3), 114, dtype=np.uint8)
    left, top = random.randint(0, max(0, sw - NATIVE_W)), random.randint(0, max(0, sh - NATIVE_H))
    crop = scaled[top:top + NATIVE_H, left:left + NATIVE_W]
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    return canvas, transform_rows(rows_big, sw, sh, -left, -top, NATIVE_W, NATIVE_H)


def color_aug(arr):
    img = Image.fromarray(arr)
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.8:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))
    return np.asarray(img, dtype=np.uint8)


class DetDataset(Dataset):
    """Returns native-resolution tensors [3,240,320] and normalized labels [n,5]; upsampling happens on the GPU."""
    def __init__(self, paths, augment):
        self.paths = list(paths)
        self.augment = augment
        self.rows = [read_label_rows(p) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def _item(self, i):
        return load_native_image(self.paths[i]), [r[:] for r in self.rows[i]]

    def __getitem__(self, i):
        arr, rows = self._item(i)
        if self.augment:
            if random.random() < CFG["mosaic_prob"]:
                others = [self._item(random.randrange(len(self.paths))) for _ in range(3)]
                arr, rows = mosaic_aug([(arr, rows)] + others)
            elif random.random() < CFG["zoom_prob"]:
                arr, rows = zoom_aug(arr, rows, random.uniform(*CFG["zoom_range"]))
            if random.random() < CFG["flip_prob"]:
                arr = np.ascontiguousarray(arr[:, ::-1])
                rows = [[c, 1.0 - cx, cy, w, h] for c, cx, cy, w, h in rows]
            if random.random() < CFG["color_prob"]:
                arr = color_aug(arr)
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)
        return tensor, torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)


def det_collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images), list(labels)


def upsample_batch(images, size):
    w, h = size
    return F.interpolate(images, size=(h, w), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------- targets
@torch.no_grad()
def build_targets(labels, feat_shapes, width, height, device):
    """Per level: cls [B,H,W] (-1 = negative), ltrb [B,4,H,W] in stride units.
    Overlapping level ranges + center sampling + smallest-area wins; every GT gets >= 1 positive."""
    targets = []
    grids = []
    for (h, w), stride in zip(feat_shapes, STRIDES):
        px = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
        py = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
        grids.append((px, py))
        targets.append({"cls": torch.full((len(labels), h, w), -1, dtype=torch.long, device=device),
                        "ltrb": torch.zeros((len(labels), 4, h, w), device=device)})
    for b, rows in enumerate(labels):
        if rows.numel() == 0:
            continue
        rows = rows.to(device)
        n = rows.shape[0]
        cx, cy, bw, bh = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        max_side = torch.maximum(bw, bh)
        area = bw * bh
        assigned = torch.zeros(n, dtype=torch.bool, device=device)
        level_masks = []
        for li, ((px, py), stride, (lo, hi)) in enumerate(zip(grids, STRIDES, LEVEL_RANGES)):
            l = px[None, None, :] - x1[:, None, None]
            t = py[None, :, None] - y1[:, None, None]
            r = x2[:, None, None] - px[None, None, :]
            d = y2[:, None, None] - py[None, :, None]
            inside = (l > 0) & (t > 0) & (r > 0) & (d > 0)
            rad = CENTER_RADIUS * stride
            center = ((px[None, None, :] - cx[:, None, None]).abs() <= rad) & ((py[None, :, None] - cy[:, None, None]).abs() <= rad)
            elig = (max_side > lo) & (max_side <= hi)
            m = inside & center & elig[:, None, None]
            assigned |= m.flatten(1).any(1)
            level_masks.append((m, l, t, r, d))
        # fallback: a GT without any positive point gets the nearest grid point on its first eligible level
        for gi in torch.nonzero(~assigned).flatten().tolist():
            for li, ((px, py), stride, (lo, hi)) in enumerate(zip(grids, STRIDES, LEVEL_RANGES)):
                if lo < float(max_side[gi]) <= hi or li == len(STRIDES) - 1:
                    ix = int(((cx[gi] / stride) - 0.5).round().clamp(0, len(px) - 1))
                    iy = int(((cy[gi] / stride) - 0.5).round().clamp(0, len(py) - 1))
                    level_masks[li][0][gi, iy, ix] = True
                    break
        for li, (m, l, t, r, d) in enumerate(level_masks):
            if not m.any():
                continue
            stride = STRIDES[li]
            area_map = area[:, None, None].expand_as(m).clone()
            area_map[~m] = float("inf")
            amin, idx = area_map.min(0)
            pos = torch.isfinite(amin)
            if not pos.any():
                continue
            ys, xs = torch.nonzero(pos, as_tuple=True)
            sel = idx[ys, xs]
            targets[li]["cls"][b, ys, xs] = rows[sel, 0].long()
            ltrb = torch.stack([l[sel, 0, xs], t[sel, ys, 0], r[sel, 0, xs], d[sel, ys, 0]], 0)   # [4,P]
            targets[li]["ltrb"][b, :, ys, xs] = ltrb.clamp_min(0.0) / stride
    return targets


def giou_ltrb(pred, target):
    """pred/target: [P,4] ltrb (same units). Returns (giou, iou)."""
    pl, pt, pr, pb = pred.unbind(-1); tl, tt, tr, tb = target.unbind(-1)
    inter = (torch.minimum(pl, tl) + torch.minimum(pr, tr)).clamp_min(0) * (torch.minimum(pt, tt) + torch.minimum(pb, tb)).clamp_min(0)
    area_p, area_t = (pl + pr) * (pt + pb), (tl + tr) * (tt + tb)
    union = (area_p + area_t - inter).clamp_min(1e-6)
    iou = inter / union
    enclose = ((torch.maximum(pl, tl) + torch.maximum(pr, tr)) * (torch.maximum(pt, tt) + torch.maximum(pb, tb))).clamp_min(1e-6)
    return iou - (enclose - union) / enclose, iou


def detection_loss(outputs, targets, return_parts=False):
    device = outputs[0].device
    vfl = torch.zeros((), device=device); iou_l = torch.zeros((), device=device)
    giou_l = torch.zeros((), device=device); dfl_l = torch.zeros((), device=device)
    n_pos = 0; w_sum = torch.zeros((), device=device)
    bins = torch.arange(REG_MAX, device=device, dtype=torch.float32)
    for raw, tg, stride in zip(outputs, targets, STRIDES):
        b, _, h, w = raw.shape
        cls_logit = raw[:, :NC].float()
        iou_logit = raw[:, -1].float()
        pos = tg["cls"] >= 0
        q = torch.zeros_like(cls_logit)
        if pos.any():
            n_pos += int(pos.sum())
            dist = raw[:, NC:NC + 4 * REG_MAX].float().reshape(b, 4, REG_MAX, h, w).permute(0, 3, 4, 1, 2)[pos]  # [P,4,R]
            pred = (dist.softmax(-1) * bins).sum(-1)                       # [P,4] stride units
            truth = tg["ltrb"].permute(0, 2, 3, 1)[pos]                    # [P,4]
            giou, iou = giou_ltrb(pred, truth)
            iou_d = iou.detach().clamp(0.0, 1.0)
            bi, yi, xi = torch.nonzero(pos, as_tuple=True)
            q[bi, tg["cls"][pos], yi, xi] = iou_d
            iou_l = iou_l + F.binary_cross_entropy_with_logits(iou_logit[pos], iou_d, reduction="sum")
            wq = iou_d.clamp_min(0.1)
            giou_l = giou_l + ((1.0 - giou) * wq).sum()
            tr = truth.clamp(0.0, REG_MAX - 1.001)
            lo = tr.floor().long(); hi = (lo + 1).clamp(max=REG_MAX - 1)
            w_hi = tr - lo.float(); w_lo = 1.0 - w_hi
            logp = dist.log_softmax(-1)
            dfl_each = -(logp.gather(-1, lo.unsqueeze(-1)).squeeze(-1) * w_lo + logp.gather(-1, hi.unsqueeze(-1)).squeeze(-1) * w_hi).mean(-1)
            dfl_l = dfl_l + (dfl_each * wq).sum()
            w_sum = w_sum + wq.sum()
        # VariFocal loss: positives weighted by their IoU target, negatives by 0.75 * p^2
        p = cls_logit.sigmoid()
        bce = F.binary_cross_entropy_with_logits(cls_logit, q, reduction="none")
        weight = torch.where(q > 0, q, 0.75 * p.detach().pow(2))
        vfl = vfl + (bce * weight).sum()
    norm = max(n_pos, 1)
    w_sum = w_sum.clamp_min(1e-6)
    parts = {"vfl": vfl / norm, "iou": 0.5 * iou_l / norm, "giou": 2.0 * giou_l / w_sum, "dfl": 0.25 * dfl_l / w_sum}
    total = sum(parts.values())
    return (total, parts) if return_parts else total


# ---------------------------------------------------------------- decode / eval
@torch.no_grad()
def decode_outputs(outputs, width, height, conf=0.001, pre_nms=3000, nms_iou=0.6, max_det=300, multi_label=True):
    """Returns per image (boxes[N,4] in input px, scores[N], classes[N])."""
    results = []
    batch = outputs[0].shape[0]
    for bi in range(batch):
        boxes_all, scores_all, classes_all = [], [], []
        for raw_level, stride in zip(outputs, STRIDES):
            raw = raw_level[bi:bi + 1].float()
            _, _, h, w = raw.shape
            scores = raw[0, :NC].sigmoid() * raw[0, -1].sigmoid()[None]          # [NC,h,w]
            ltrb = expected_ltrb(raw, stride)[0]                                 # [4,h,w]
            yy, xx = torch.meshgrid(torch.arange(h, device=raw.device, dtype=torch.float32),
                                    torch.arange(w, device=raw.device, dtype=torch.float32), indexing="ij")
            px, py = (xx + 0.5) * stride, (yy + 0.5) * stride
            boxes = torch.stack((px - ltrb[0], py - ltrb[1], px + ltrb[2], py + ltrb[3]), -1)   # [h,w,4]
            if multi_label:
                c, y, x = torch.nonzero(scores >= conf, as_tuple=True)
            else:
                s_max, c_max = scores.max(0)
                y, x = torch.nonzero(s_max >= conf, as_tuple=True); c = c_max[y, x]
            if c.numel() == 0:
                continue
            boxes_all.append(boxes[y, x]); scores_all.append(scores[c, y, x]); classes_all.append(c)
        if not scores_all:
            results.append((torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)))
            continue
        boxes, scores, classes = torch.cat(boxes_all), torch.cat(scores_all), torch.cat(classes_all)
        if scores.numel() > pre_nms:
            top = scores.topk(pre_nms).indices
            boxes, scores, classes = boxes[top], scores[top], classes[top]
        boxes[:, 0::2].clamp_(0, width); boxes[:, 1::2].clamp_(0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
        keep = batched_nms(boxes, scores, classes, nms_iou)[:max_det] if scores.numel() else torch.empty((0,), dtype=torch.long)
        results.append((boxes[keep].cpu(), scores[keep].cpu(), classes[keep].cpu()))
    return results


def merge_detections(parts, nms_iou=0.6, max_det=300):
    """Merge TTA detection sets (lists of (boxes, scores, classes)) with class-aware NMS."""
    boxes = torch.cat([p[0] for p in parts]); scores = torch.cat([p[1] for p in parts]); classes = torch.cat([p[2] for p in parts])
    if scores.numel() == 0:
        return boxes, scores, classes
    keep = batched_nms(boxes, scores, classes, nms_iou)[:max_det]
    return boxes[keep], scores[keep], classes[keep]


@torch.no_grad()
def predict_native(model, images_native, device, sizes=None, hflip=False, conf=0.001, max_det=None):
    """images_native: [B,3,240,320] in [0,1]. Returns per-image detections in NATIVE pixel coordinates,
    capped to the top-`max_det` boxes by score (default CFG["max_det"] = 100)."""
    sizes = sizes or [CFG["val_size"]]
    max_det = max_det or CFG["max_det"]
    model.eval()
    per_image = [[] for _ in range(images_native.shape[0])]
    for (w, h) in sizes:
        x = upsample_batch(images_native.to(device, non_blocking=True), (w, h))
        variants = [(x, False)] + ([(x.flip(-1), True)] if hflip else [])
        for xv, flipped in variants:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outs = model(xv)
            dets = decode_outputs(outs, w, h, conf=conf, nms_iou=CFG["nms_iou"])
            for i, (bx, sc, cl) in enumerate(dets):
                bx = bx.clone()
                if flipped:
                    bx[:, [0, 2]] = w - bx[:, [2, 0]]
                bx[:, 0::2] *= NATIVE_W / w; bx[:, 1::2] *= NATIVE_H / h
                per_image[i].append((bx, sc, cl))
    final = []
    for p in per_image:
        bx, sc, cl = merge_detections(p, nms_iou=CFG["nms_iou"], max_det=max_det) if len(p) > 1 else p[0]
        order = sc.argsort(descending=True)[:max_det]          # final hard cap, highest scores first
        final.append((bx[order], sc[order], cl[order]))
    return final


def _iou_one(box, boxes):
    lt, rb = torch.maximum(box[:2], boxes[:, :2]), torch.minimum(box[2:], boxes[:, 2:])
    inter = (rb - lt).clamp_min(0).prod(1)
    return inter / ((box[2:] - box[:2]).clamp_min(0).prod() + (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1) - inter).clamp_min(1e-7)


def map50_from_detections(all_dets, all_gts):
    """all_dets: list per image of (boxes, scores, classes) native px; all_gts: list per image of [n,5] normalized rows."""
    preds = [[] for _ in range(NC)]; gts = [[] for _ in range(NC)]
    for img_id, (det, rows) in enumerate(zip(all_dets, all_gts)):
        per_cls = [[] for _ in range(NC)]
        for c, cx, cy, w, h in rows.tolist():
            per_cls[int(c)].append(torch.tensor([(cx - w / 2) * NATIVE_W, (cy - h / 2) * NATIVE_H, (cx + w / 2) * NATIVE_W, (cy + h / 2) * NATIVE_H]))
        for c in range(NC):
            gts[c].append(torch.stack(per_cls[c]) if per_cls[c] else torch.zeros((0, 4)))
        for box, score, c in zip(*det):
            preds[int(c)].append((float(score), img_id, box))
    aps = {}
    for c in range(NC):
        total = sum(len(g) for g in gts[c])
        if total == 0:
            aps[CLASS_NAMES[c]] = 0.0; continue
        ranked = sorted(preds[c], key=lambda it: -it[0])
        matched = [torch.zeros(len(g), dtype=torch.bool) for g in gts[c]]
        tp = np.zeros(len(ranked)); fp = np.zeros(len(ranked))
        for k, (_, img_id, box) in enumerate(ranked):
            g = gts[c][img_id]
            if len(g) == 0:
                fp[k] = 1; continue
            o = _iou_one(box, g); j = int(o.argmax())
            if float(o[j]) >= 0.5 and not matched[img_id][j]:
                tp[k] = 1; matched[img_id][j] = True
            else:
                fp[k] = 1
        if not ranked:
            aps[CLASS_NAMES[c]] = 0.0; continue
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec, prec = ctp / total, ctp / np.maximum(ctp + cfp, 1e-12)
        aps[CLASS_NAMES[c]] = float(np.mean([prec[rec >= r].max() if np.any(rec >= r) else 0.0 for r in np.linspace(0, 1, 101)]))
    return float(np.mean(list(aps.values()))), aps


@torch.no_grad()
def evaluate_map50(model, loader, device, sizes=None, hflip=False):
    dets, gts = [], []
    for images, labels in loader:
        dets += predict_native(model, images, device, sizes=sizes, hflip=hflip)
        gts += labels
    return map50_from_detections(dets, gts)


# ============================================================
# 6. Distillation: online teacher on the SAME augmented batch
#    - feature KD (channel-wise distribution, CWD) on the 3 neck levels: teacher@640 (P3/P4/P5) and
#      student@640 (strides 8/16/32) have identical grids, so no resampling is needed
#    - logit KD: teacher one2many class logits vs student class logits at every anchor (soft labels)
#    - multi-scale batches keep the alignment (both networks are fully convolutional)
# ============================================================


class TeacherKD:
    """Frozen Ultralytics teacher(s). feats from the first teacher, class logits averaged over all."""
    def __init__(self, checkpoints, device):
        self.device = device
        self.models, self.feat_layers = [], []
        for ckpt in checkpoints:
            model = YOLO(str(ckpt)).model
            merged = merge_yolo_lora(model)
            model = model.float().to(device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            head = model.model[-1]
            if int(head.nc) != NC:
                raise ValueError(f"teacher {ckpt} has nc={head.nc}, expected {NC}")
            self.models.append(model)
            self.feat_layers.append(list(head.f))
            print(f"teacher loaded: {ckpt} (merged LoRA layers={merged}, feat layers={list(head.f)}, strides={head.stride.tolist()})")
        self._store = {}
        self._register_hooks()
        with torch.no_grad():
            feats, logits = self(torch.zeros(1, 3, CFG["val_size"][1], CFG["val_size"][0], device=device))
        self.feat_channels = [f.shape[1] for f in feats]
        print("teacher feature channels:", self.feat_channels, "grids:", [tuple(f.shape[-2:]) for f in feats],
              "logit grids:", [tuple(l.shape[-2:]) for l in logits])

    def _register_hooks(self):
        for mi, model in enumerate(self.models):
            for li, layer_idx in enumerate(self.feat_layers[mi]):
                def hook(_m, _i, out, key=(mi, li)):
                    self._store[key] = out
                model.model[layer_idx].register_forward_hook(hook)

    @torch.no_grad()
    def __call__(self, images):
        self._store.clear()
        feats, logits_sum = None, None
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
            for mi, model in enumerate(self.models):
                model(images)
                neck = [self._store[(mi, li)] for li in range(len(self.feat_layers[mi]))]
                head = model.model[-1]
                logits = [head.cv3[li](neck[li]).float() for li in range(len(neck))]   # one2many class logits [B,NC,h,w]
                if feats is None:
                    feats = [n.float() for n in neck]
                logits_sum = logits if logits_sum is None else [a + b for a, b in zip(logits_sum, logits)]
        return feats, [l / len(self.models) for l in logits_sum]


class KDAdapters(nn.Module):
    """Training-only 1x1 projections student neck (80ch) -> teacher channels; not part of the deployed model."""
    def __init__(self, student_width, teacher_channels):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(student_width, c, 1) for c in teacher_channels])

    def forward(self, feats):
        return [p(f) for p, f in zip(self.proj, feats)]


def cwd_loss(student_feat, teacher_feat, tau=1.0):
    """Channel-wise distillation: KL between per-channel spatial softmax distributions."""
    b, c, h, w = teacher_feat.shape
    s = student_feat.float().reshape(b * c, h * w) / tau
    t = teacher_feat.float().reshape(b * c, h * w) / tau
    pt = t.softmax(-1)
    return (pt * (t.log_softmax(-1) - s.log_softmax(-1))).sum(-1).mean() * (tau ** 2)


def logit_kd_loss(student_outputs, teacher_logits, T=1.0):
    """Soft-label BCE per anchor, weighted by the teacher's confidence at that anchor (never zero)."""
    total, wsum = 0.0, 0.0
    for raw, t_logit in zip(student_outputs, teacher_logits):
        s = raw[:, :NC].float() / T
        t_prob = (t_logit / T).sigmoid()
        w = t_prob.max(1).values.clamp(0.02, 1.0)                       # [B,h,w]
        per_anchor = F.binary_cross_entropy_with_logits(s, t_prob, reduction="none").sum(1)
        total = total + (per_anchor * w).sum(); wsum = wsum + w.sum()
    return total / wsum.clamp_min(1e-6)


def kd_losses(student_outputs, student_feats, adapters, teacher_feats, teacher_logits):
    feat = sum(cwd_loss(sf, tf, CFG["kd_feat_tau"]) for sf, tf in zip(adapters(student_feats), teacher_feats)) / len(teacher_feats)
    logit = logit_kd_loss(student_outputs, teacher_logits, CFG["kd_logit_T"])
    return feat, logit


# ============================================================
# 7. Train the student with GT loss + online distillation
#    EMA weights are validated/saved; cosine LR with warmup; AMP; multi-scale; resumable.
# ============================================================


class ModelEMA:
    def __init__(self, model, decay):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = float(decay), 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.is_floating_point():
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


def _rel_to_root(p):
    """teacher path relative to ROOT, so the signature does not depend on where the project folder lives"""
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def student_config_signature():
    keys = ["seed", "val_fraction", "student_epochs", "student_batch", "student_lr", "student_min_lr", "student_warmup_epochs",
            "student_weight_decay", "ema_decay", "train_sizes", "val_size", "kd_feat_weight", "kd_feat_tau", "kd_logit_weight",
            "kd_logit_T", "kd_warmup_epochs", "kd_ramp_epochs"]
    return json.dumps({k: CFG[k] for k in keys}, sort_keys=True) + "|teachers=" + ",".join(_rel_to_root(p) for p in teacher_checkpoints)


def train_student_v4(teacher_ckpts):
    best_path, last_path, hist_path = STUDENT_DIR / "best.pt", STUDENT_DIR / "last.pt", STUDENT_DIR / "history.csv"
    device = DEVICE
    seed_everything(CFG["seed"])
    train_paths, val_paths = detector_train_val_paths()
    workers = 0 if SMOKE_TEST else DATA_WORKERS
    preload_images(train_paths, "train"); preload_images(val_paths, "val")
    loader_kw = dict(num_workers=workers, pin_memory=(device.type == "cuda"), collate_fn=det_collate,
                     persistent_workers=(workers > 0), **({"prefetch_factor": 4} if workers > 0 else {}))
    train_loader = DataLoader(DetDataset(train_paths, augment=True), batch_size=CFG["student_batch"], shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(DetDataset(val_paths, augment=False), batch_size=max(8, CFG["student_batch"]), shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"), collate_fn=det_collate)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    print(f"[student] train={len(train_paths)} val={len(val_paths)} batch={CFG['student_batch']} sizes={CFG['train_sizes']}")

    teacher = TeacherKD(teacher_ckpts, device)
    model = StudentV4().to(device)
    adapters = KDAdapters(model.neck_width, teacher.feat_channels).to(device)
    deploy_params = count_deploy_params(model)
    assert deploy_params <= PARAM_LIMIT
    ema = ModelEMA(model, CFG["ema_decay"])

    params_decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    params_no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([
        {"params": params_decay, "weight_decay": CFG["student_weight_decay"]},
        {"params": params_no_decay, "weight_decay": 0.0},
        {"params": list(adapters.parameters()), "weight_decay": 0.0},
    ], lr=CFG["student_lr"], betas=(0.9, 0.999))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = CFG["student_epochs"] * steps_per_epoch
    warmup_steps = CFG["student_warmup_epochs"] * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return CFG["student_lr"] * (0.05 + 0.95 * step / max(1, warmup_steps))
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return CFG["student_min_lr"] + 0.5 * (CFG["student_lr"] - CFG["student_min_lr"]) * (1 + math.cos(math.pi * min(1.0, prog)))

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    signature = student_config_signature()
    start_epoch, best_map, best_epoch, patience, history, step = 1, -1.0, 0, 0, [], 0
    if last_path.exists() and not SMOKE_TEST:
        state = load_ckpt(last_path)
        if state.get("signature") == signature:
            model.load_state_dict(state["model"]); ema.ema.load_state_dict(state["ema_model"]); ema.updates = state["ema_updates"]
            adapters.load_state_dict(state["adapters"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
            start_epoch, best_map, best_epoch = state["epoch"] + 1, state["best_map"], state["best_epoch"]
            patience, history, step = state["patience"], state["history"], state["step"]
            print(f"[student] resumed from epoch {state['epoch']} (best {best_map:.5f} @ {best_epoch})")
        else:
            print("[student] last.pt has a different configuration; starting fresh")
            for p in (best_path, last_path):
                p.unlink(missing_ok=True)
    print(f"[student] deployable params={deploy_params:,} train-time params={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(start_epoch, CFG["student_epochs"] + 1):
        if patience >= CFG["student_patience"]:
            print("[student] early stop (patience)"); break
        model.train(); adapters.train()
        kd_ramp = 0.0 if epoch <= CFG["kd_warmup_epochs"] else min(1.0, (epoch - CFG["kd_warmup_epochs"]) / max(1, CFG["kd_ramp_epochs"]))
        sums = defaultdict(float); n_seen = 0; t0 = time.time()
        for images, labels in train_loader:
            size = random.choice(CFG["train_sizes"])
            images = upsample_batch(images.to(device, non_blocking=True), size)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs, feats = model(images, return_feats=True)
            targets = build_targets(labels, [o.shape[-2:] for o in outputs], size[0], size[1], device)
            gt_loss, parts = detection_loss(outputs, targets, return_parts=True)
            if kd_ramp > 0:
                t_feats, t_logits = teacher(images)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    feat_kd, logit_kd = kd_losses(outputs, feats, adapters, t_feats, t_logits)
                loss = gt_loss + kd_ramp * (CFG["kd_feat_weight"] * feat_kd + CFG["kd_logit_weight"] * logit_kd)
            else:
                feat_kd = logit_kd = torch.zeros((), device=device)
                loss = gt_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}: {loss.item()}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapters.parameters()), 10.0)
            scaler.step(optimizer); scaler.update()
            ema.update(model); step += 1
            n = images.shape[0]; n_seen += n
            sums["loss"] += float(loss) * n; sums["gt"] += float(gt_loss) * n
            sums["feat_kd"] += float(feat_kd) * n; sums["logit_kd"] += float(logit_kd) * n
            for k, v in parts.items():
                sums[k] += float(v) * n
        val_map, per_class = evaluate_map50(ema.ema, val_loader, device, sizes=[CFG["val_size"]], hflip=False)
        row = {"epoch": epoch, "lr": lr_at(step), "kd_ramp": kd_ramp, "val_map50": val_map, "sec": time.time() - t0}
        row.update({k: v / max(n_seen, 1) for k, v in sums.items()})
        row.update({f"ap50_{k}": v for k, v in per_class.items()})
        history.append(row)
        improved = val_map > best_map + 1e-6
        if improved:
            best_map, best_epoch, patience = val_map, epoch, 0
        else:
            patience += 1
        ckpt = {"model": ema.ema.state_dict(), "architecture": "StudentV4", "epoch": epoch, "val_map50": val_map,
                "per_class_ap50": per_class, "best_map": best_map, "best_epoch": best_epoch, "deploy_params": deploy_params,
                "class_names": CLASS_NAMES, "input_size": CFG["val_size"], "strides": list(STRIDES), "reg_max": REG_MAX,
                "level_ranges": [list(r) for r in LEVEL_RANGES], "signature": signature}
        if improved or not best_path.exists():
            torch.save(ckpt, best_path)
        torch.save({**ckpt, "model": model.state_dict(), "ema_model": ema.ema.state_dict(), "ema_updates": ema.updates,
                    "adapters": adapters.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "patience": patience, "history": history, "step": step}, last_path)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        ap_text = " ".join(f"{k[:4]}={v:.3f}" for k, v in per_class.items())
        print(f"{epoch:03d}: loss={row['loss']:.4f} gt={row['gt']:.4f} (vfl={row['vfl']:.3f} giou={row['giou']:.3f} dfl={row['dfl']:.3f}) "
              f"featKD={row['feat_kd']:.4f} logitKD={row['logit_kd']:.4f} ramp={kd_ramp:.2f} lr={row['lr']:.2e} "
              f"val_mAP50={val_map:.5f} [{ap_text}] best={best_map:.5f}@{best_epoch} pat={patience}/{CFG['student_patience']} {row['sec']:.0f}s")
    print(f"[student] best EMA mAP@0.5={best_map:.5f} @ epoch {best_epoch} -> {best_path}")
    # Deployable (fused) copy of the best EMA model for submission / verification
    best_state = load_ckpt(best_path)
    model.load_state_dict(best_state["model"]); model.eval()
    deploy = to_deploy(model)
    torch.save({"model": deploy.state_dict(), "deploy_params": sum(p.numel() for p in deploy.parameters()),
                "architecture": "StudentV4-deploy", "epoch": best_state["epoch"], "val_map50": best_state["val_map50"]}, STUDENT_DIR / "deploy.pt")
    print(f"[student] deploy.pt saved: {sum(p.numel() for p in deploy.parameters()):,} params")
    return best_path


student_checkpoint = train_student_v4(teacher_checkpoints)


# ============================================================
# 8. Validation report (with/without TTA) and submission export
#    Only the two <=200k student models run at inference; the teacher is never used here.
# ============================================================
intent_state = load_ckpt(INTENT_DIR / "best.pt")
intent_model = IntentTextCNN().to(DEVICE); intent_model.load_state_dict(intent_state["model"]); intent_model.eval()

# Detector: load the already-fused deployable checkpoint directly (no best.pt, no fusing at export time)
student_ckpt = load_ckpt(STUDENT_DIR / "deploy.pt")
student_deploy = build_deploy_model().to(DEVICE)
student_deploy.load_state_dict(student_ckpt["model"], strict=True); student_deploy.eval()
_intent_params = count_params(intent_model)
_deploy_params = sum(p.numel() for p in student_deploy.parameters()) + sum(b.numel() for b in student_deploy.buffers())
print(f"intent params={_intent_params:,}  student params (deploy.pt)={_deploy_params:,}  limit < 200,000")
print(f"student deploy.pt: epoch={student_ckpt['epoch']} val mAP50={student_ckpt['val_map50']:.5f}")
assert _intent_params < 200_000 and _deploy_params < 200_000 and _deploy_params == student_ckpt["deploy_params"]

_, _val_paths = detector_train_val_paths()
_val_loader = DataLoader(DetDataset(_val_paths, augment=False), batch_size=8, shuffle=False, num_workers=0, collate_fn=det_collate)
for name, sizes, flip in [("single 640x480", [(640, 480)], False), ("single 704x528 (export)", CFG["tta_sizes"], False)]:   # no TTA
    m, per = evaluate_map50(student_deploy, _val_loader, DEVICE, sizes=sizes, hflip=flip)
    print(f"val mAP@0.5 {name:24s}: {m:.5f}  " + " ".join(f"{k}={v:.3f}" for k, v in per.items()))


def predict_intent(text):
    words = torch.tensor([encode_words(text, intent_state["word_vocab"])], dtype=torch.long, device=DEVICE)
    chars = torch.tensor([encode_chars(text)], dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        s, o = intent_model(words, chars)
        return "OUT_OF_SCOPE" if int(s.argmax(1)) == 0 else INTENT_NAMES[int(o.argmax(1))]


def predict_detection_json(paths, sizes, hflip, batch=16):
    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        tensors = torch.stack([torch.from_numpy(np.ascontiguousarray(load_native_image(p).transpose(2, 0, 1))).float().div_(255.0) for p in chunk])
        dets = predict_native(student_deploy, tensors, DEVICE, sizes=sizes, hflip=hflip, conf=0.001, max_det=CFG["max_det"])
        for p, (boxes, scores, classes) in zip(chunk, dets):
            assert len(scores) <= CFG["max_det"], f"{p.stem}: {len(scores)} boxes > {CFG['max_det']}"
            out[p.stem] = json.dumps([{"class_id": int(c), "score": round(float(s), 6), "bbox": [round(float(v), 2) for v in b.tolist()]}
                                      for b, s, c in zip(boxes, scores, classes)], separators=(",", ":"))
    return out


test_paths = sorted((DATA / "test" / "images").glob("*.png"))
assert not CFG["tta_hflip"] and len(CFG["tta_sizes"]) == 1, "TTA must be disabled"
if SMOKE_TEST:
    test_paths = test_paths[:8]
detections = predict_detection_json(test_paths, CFG["tta_sizes"], CFG["tta_hflip"])
intent_test = pd.read_csv(DATA / "intent_test.csv")
intent_by_id = {qid: predict_intent(q) for qid, q in zip(intent_test.question_id, intent_test.question)}
submission = pd.read_csv(DATA / "sample_submission.csv")
submission["prediction"] = [detections.get(sid, "[]") if t == "DETECTION" else intent_by_id.get(sid, "OUT_OF_SCOPE")
                            for t, sid in zip(submission.task_type, submission.sample_id)]
submission_path = OUTPUT_DIR / ("submission_smoke.csv" if SMOKE_TEST else "submission_v5.csv")
submission.to_csv(submission_path, index=False)
print("saved", submission_path, submission.shape, "| intent distribution:", Counter(intent_by_id.values()))

