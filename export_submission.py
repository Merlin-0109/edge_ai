import os, sys, json, math, random, time, copy, gc, csv, re, string
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("EDGE_AI_ROOT", Path(__file__).resolve().parent)).resolve()
os.environ["EDGE_AI_ROOT"] = str(ROOT)   # notebook cell 1 resolves the same folder
DATA = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
CHECKPOINTS_DIR = ROOT / "checkpoints"
INTENT_DIR = OUTPUT_DIR / "intent_v4"
STUDENT_DIR = CHECKPOINTS_DIR / "student_v4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

with open(ROOT / "train_v4_full_pipeline.ipynb") as f:
    nb = json.load(f)

# Combine cells 1, 3, 7, 9
g = {"ROOT": ROOT, "DATA": DATA, "OUTPUT_DIR": OUTPUT_DIR, "CHECKPOINTS_DIR": CHECKPOINTS_DIR,
     "INTENT_DIR": INTENT_DIR, "STUDENT_DIR": STUDENT_DIR, "DEVICE": DEVICE}

print("Loading notebook definitions...")
exec(''.join(nb['cells'][1]['source']), g)
exec(''.join(nb['cells'][3]['source']), g)
exec(''.join(nb['cells'][7]['source']), g)
exec(''.join(nb['cells'][9]['source']), g)

load_ckpt = g["load_ckpt"]
count_params = g["count_params"]
IntentTextCNN = g["IntentTextCNN"]
build_deploy_model = g["build_deploy_model"]
detector_train_val_paths = g["detector_train_val_paths"]
DataLoader = g["DataLoader"]
DetDataset = g["DetDataset"]
det_collate = g["det_collate"]
evaluate_map50 = g["evaluate_map50"]
encode_words = g["encode_words"]
encode_chars = g["encode_chars"]
INTENT_NAMES = g["INTENT_NAMES"]
load_native_image = g["load_native_image"]
predict_native = g["predict_native"]
CFG = g["CFG"]

print("Loading models...")
intent_state = load_ckpt(INTENT_DIR / "best.pt")
intent_model = IntentTextCNN().to(DEVICE)
intent_model.load_state_dict(intent_state["model"])
intent_model.eval()

# Detector: load the already-fused deployable checkpoint directly (no best.pt, no fusing at export time)
student_ckpt = load_ckpt(STUDENT_DIR / "deploy.pt")
student_deploy = build_deploy_model().to(DEVICE)
student_deploy.load_state_dict(student_ckpt["model"], strict=True)
student_deploy.eval()

intent_params = count_params(intent_model)
deploy_params = sum(p.numel() for p in student_deploy.parameters()) + sum(b.numel() for b in student_deploy.buffers())
print(f"Intent params: {intent_params:,} (limit < 200,000)")
print(f"Student params (deploy.pt): {deploy_params:,} (limit < 200,000)")
print(f"Student deploy.pt: epoch {student_ckpt['epoch']}, val mAP50: {student_ckpt['val_map50']:.5f}")
assert intent_params < 200_000 and deploy_params < 200_000 and deploy_params == student_ckpt["deploy_params"]

print("Running validation evaluation...")
_, _val_paths = detector_train_val_paths()
_val_loader = DataLoader(DetDataset(_val_paths, augment=False), batch_size=8, shuffle=False, num_workers=0, collate_fn=det_collate)

for name, sizes, flip in [("single 640x480", [(640, 480)], False), ("single 704x528 (export)", CFG["tta_sizes"], False)]:   # no TTA / no ensemble
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
    total = len(paths)
    for i in range(0, total, batch):
        if i % 400 == 0 or i + batch >= total:
            print(f"Predicting test images: {min(i + batch, total)}/{total}...")
        chunk = paths[i:i + batch]
        tensors = torch.stack([torch.from_numpy(np.ascontiguousarray(load_native_image(p).transpose(2, 0, 1))).float().div_(255.0) for p in chunk])
        dets = predict_native(student_deploy, tensors, DEVICE, sizes=sizes, hflip=hflip, conf=0.001, max_det=CFG["max_det"])
        for p, (boxes, scores, classes) in zip(chunk, dets):
            assert len(scores) <= CFG["max_det"], f"{p.stem}: {len(scores)} boxes > {CFG['max_det']}"
            out[p.stem] = json.dumps([{"class_id": int(c), "score": round(float(s), 6), "bbox": [round(float(v), 2) for v in b.tolist()]}
                                      for b, s, c in zip(boxes, scores, classes)], separators=(",", ":"))
    return out


print("Predicting test detections...")
test_paths = sorted((DATA / "test" / "images").glob("*.png"))
assert not CFG["tta_hflip"] and len(CFG["tta_sizes"]) == 1, "TTA must be disabled"
detections = predict_detection_json(test_paths, CFG["tta_sizes"], CFG["tta_hflip"])

print("Predicting test intents...")
intent_test = pd.read_csv(DATA / "intent_test.csv")
intent_by_id = {qid: predict_intent(q) for qid, q in zip(intent_test.question_id, intent_test.question)}

print("Creating submission dataframe...")
submission = pd.read_csv(DATA / "sample_submission.csv")
submission["prediction"] = [
    detections.get(sid, "[]") if t == "DETECTION" else intent_by_id.get(sid, "OUT_OF_SCOPE")
    for t, sid in zip(submission.task_type, submission.sample_id)
]

submission_path = OUTPUT_DIR / "submission_v5.csv"
submission.to_csv(submission_path, index=False)
print("Saved final submission to:", submission_path)
print("Submission shape:", submission.shape)
print("Intent distribution in test:", Counter(intent_by_id.values()))

# Quick sanity check
print("\nSubmission sanity check:")
_det_counts = [len(json.loads(x)) for t, x in zip(submission.task_type, submission.prediction) if t == "DETECTION"]
print(f"Boxes per image: max={max(_det_counts)} mean={np.mean(_det_counts):.1f} (limit {CFG['max_det']})")
assert max(_det_counts) <= CFG["max_det"]
print("Null values count:", submission["prediction"].isnull().sum())
print("Total rows:", len(submission))
print(submission.head(5))
