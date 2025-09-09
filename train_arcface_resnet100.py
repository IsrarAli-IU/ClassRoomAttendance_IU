#!/usr/bin/env python3
"""
Complete pipeline: ArcFace embeddings (InsightFace) + lightweight classifier + simple video attendance
Auto‑configured for this folder layout (no CLI args required):

project_root/
├─ train_arcface_embed_clf.py    ← this file
├─ labels.csv                    ← with columns: student_id,name,class,image_path
└─ dataset/                      ← images per student in subfolders
   ├─ 63018_Anisha_Sarhadi/
   ├─ 61412_Areka_Raza_Hashmi/
   └─ ...

What this script does when run:
1) Load `labels.csv`, resolve image paths inside ./dataset (supports paths with or without leading "dataset/").
2) Use InsightFace FaceAnalysis ("buffalo_l") to detect, align, and embed faces (512‑D ArcFace embeddings).
3) Build a training set of embeddings and train a small Logistic Regression classifier.
4) Save artifacts into ./arcface_artifacts/:
   - arcface_clf.joblib            (classifier)
   - sid_maps.json                 (student id/name maps)
   - prototypes.json               (mean embedding per student)
   - embeddings.csv                (per‑image embedding rows for audit)
5) (Optional demo) Run a very simple video attendance pass if you set VIDEO_PATH below.

Dependencies (install on your GPU machine):
    pip install "torch==2.3.*" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install insightface opencv-python pandas numpy scikit-learn tqdm joblib

Notes:
- The first run downloads model packs to ~/.insightface/models .
- For a production system, add multi‑object tracking (DeepSORT/ByteTrack) and temporal smoothing.
"""
import os
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import joblib

# InsightFace
from insightface.app import FaceAnalysis

# =====================
# CONFIG — edit as needed
# =====================
ROOT_DIR = Path(__file__).parent.resolve()
LABELS_CSV = ROOT_DIR / "labels.csv"
IMAGES_ROOT = ROOT_DIR / "dataset"
OUT_DIR = ROOT_DIR / "arcface_artifacts"

# Video demo (optional): set a valid path here to run a quick attendance pass after training
VIDEO_PATH = Path("803_30Sec.mp4")  # e.g., Path("/path/to/classroom.mp4")
VIDEO_SAMPLE_EVERY = 15            # analyze every Nth frame
SIM_THRESHOLD = 0.65               # cosine similarity vs prototype to accept identity
MIN_HITS_FOR_PRESENT = 5           # min accepted frames to mark present

# =====================
# Utilities
# =====================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resolve_image_paths(df: pd.DataFrame, images_root: Path) -> List[Tuple[Path, int, int, str]]:
    """Return list of (abs_path, cls_idx, sid, name). Supports paths with or without 'dataset/' prefix."""
    sids = sorted(df["student_id"].astype(int).unique().tolist())
    sid2idx = {int(s): i for i, s in enumerate(sids)}

    items = []
    missing = 0
    for _, row in df.iterrows():
        sid = int(row['student_id'])
        name = str(row['name'])
        cls_idx = sid2idx[sid]
        rel = str(row['image_path']).lstrip("./")
        # Style A: already starts with dataset/
        p = (images_root.parent / rel) if rel.startswith("dataset/") else (images_root / rel)
        if not p.exists():
            missing += 1
            continue
        items.append((p, cls_idx, sid, name))
    if missing:
        print(f"[WARN] {missing} image paths from CSV were not found on disk and will be skipped.")
    if not items:
        raise SystemExit("No images found. Check labels.csv and dataset/ structure.")
    return items


def init_face_app() -> FaceAnalysis:
    app = FaceAnalysis(name='buffalo_l')
    # ctx_id: 0 for GPU (if available), -1 for CPU
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    app.prepare(ctx_id=0 if has_cuda else -1, det_size=(640, 640))
    return app


def extract_embedding(app: FaceAnalysis, bgr_img: np.ndarray) -> Optional[np.ndarray]:
    faces = app.get(bgr_img)
    if not faces:
        return None
    # choose largest face
    faces.sort(key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]), reverse=True)
    f = faces[0]
    emb = f.get('normed_embedding', None)
    if emb is None:
        emb = f.get('embedding', None)
    if emb is None:
        return None
    v = np.asarray(emb, dtype=np.float32)
    # ensure normalized (some builds already return L2-normalized)
    n = np.linalg.norm(v) + 1e-9
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# =====================
# Training
# =====================

def build_embeddings(items: List[Tuple[Path,int,int,str]], app: FaceAnalysis) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    X, y, meta = [], [], []
    for p, cls_idx, sid, name in tqdm(items, desc="Embeddings"):
        img = cv2.imread(str(p))
        if img is None:
            continue
        e = extract_embedding(app, img)
        if e is None:
            continue
        X.append(e)
        y.append(cls_idx)
        meta.append({"sid": sid, "name": name, "path": str(p)})
    if not X:
        raise SystemExit("No embeddings extracted (all images failed detection).")
    return np.vstack(X).astype(np.float32), np.asarray(y, dtype=np.int64), meta


def train_classifier(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=4000, n_jobs=-1, multi_class='ovr')
    clf.fit(X, y)
    return clf


def evaluate_kfold(X: np.ndarray, y: np.ndarray, k: int = 5) -> None:
    # quick sanity check accuracy across folds
    skf = StratifiedKFold(n_splits=min(k, np.unique(y, return_counts=True)[1].min()), shuffle=True, random_state=42)
    accs = []
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=4000, n_jobs=-1, multi_class='ovr')
        clf.fit(X[tr], y[tr])
        pr = clf.predict(X[va])
        accs.append(accuracy_score(y[va], pr))
    print(f"KFold accuracy (mean±std): {np.mean(accs):.4f} ± {np.std(accs):.4f} over {len(accs)} folds")


def compute_prototypes(X: np.ndarray, y: np.ndarray, idx2sid: Dict[int,int]) -> Dict[int, List[float]]:
    protos: Dict[int, List[float]] = {}
    for cls_idx in np.unique(y):
        sid = int(idx2sid[int(cls_idx)])
        protos[sid] = X[y == cls_idx].mean(axis=0).tolist()
    return protos


def save_artifacts(clf: LogisticRegression,
                   protos: Dict[int, List[float]],
                   sid2idx: Dict[int,int],
                   idx2sid: Dict[int,int],
                   sid2name: Dict[int,str],
                   meta_rows: List[Dict]):
    ensure_dir(OUT_DIR)
    joblib.dump(clf, OUT_DIR / "arcface_clf.joblib")
    with open(OUT_DIR / "prototypes.json", 'w') as f:
        json.dump(protos, f, indent=2)
    with open(OUT_DIR / "sid_maps.json", 'w') as f:
        json.dump({"sid2idx": sid2idx, "idx2sid": idx2sid, "sid2name": sid2name}, f, indent=2)
    # embeddings CSV for audit
    import csv
    csv_path = OUT_DIR / "embeddings.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        # only save metadata (embeddings are large; omit for brevity)
        w.writerow(["sid", "name", "path"]) 
        for m in meta_rows:
            w.writerow([m['sid'], m['name'], m['path']])
    print(f"Artifacts saved in {OUT_DIR}")


# =====================
# Simple video attendance demo (frame sampling + prototype matching)
# =====================

def run_video_attendance(app: FaceAnalysis, protos: Dict[int, List[float]], sid2name: Dict[int,str], video_path: Path):
    print(f"\n[Video demo] {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Cannot open video; skipping.")
        return
    hits: Dict[int, int] = {}
    frame_idx = 0
    proto_mat = {sid: np.asarray(vec, dtype=np.float32) for sid, vec in protos.items()}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % VIDEO_SAMPLE_EVERY != 0:
            frame_idx += 1
            continue
        faces = app.get(frame)
        for f in faces:
            emb = f.get('normed_embedding', None) or f.get('embedding', None)
            if emb is None:
                continue
            e = np.asarray(emb, dtype=np.float32)
            e = e / (np.linalg.norm(e) + 1e-9)
            # nearest prototype by cosine
            best_sid, best_sim = None, -1.0
            for sid, pv in proto_mat.items():
                s = cosine(e, pv)
                if s > best_sim:
                    best_sid, best_sim = sid, s
            if best_sid is not None and best_sim >= SIM_THRESHOLD:
                hits[best_sid] = hits.get(best_sid, 0) + 1
        frame_idx += 1
    cap.release()

    # Decide attendance
    rows = []
    for sid, name in sid2name.items():
        present = 1 if hits.get(sid, 0) >= MIN_HITS_FOR_PRESENT else 0
        rows.append((sid, name, present, hits.get(sid, 0)))
    # Save CSV
    out_csv = OUT_DIR / "attendance_demo.csv"
    import csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["student_id", "name", "present", "hit_count"]) 
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv}")


# =====================
# Main
# =====================
if __name__ == "__main__":
    # 0) Load labels and resolve paths
    if not LABELS_CSV.exists():
        raise SystemExit(f"labels.csv not found at {LABELS_CSV}")
    if not IMAGES_ROOT.exists():
        raise SystemExit(f"dataset folder not found at {IMAGES_ROOT}")

    df = pd.read_csv(LABELS_CSV)
    items = resolve_image_paths(df, IMAGES_ROOT)

    # Build maps
    sids = sorted(df["student_id"].astype(int).unique().tolist())
    sid2idx = {int(s): i for i, s in enumerate(sids)}
    idx2sid = {i: s for s, i in sid2idx.items()}
    sid2name = {int(r['student_id']): str(r['name']) for _, r in df.drop_duplicates('student_id').iterrows()}

    # 1) Init face app (detector + recognition)
    app = init_face_app()

    # 2) Extract embeddings
    X, y, meta = build_embeddings(items, app)
    print(f"Embeddings built: X={X.shape}, classes={len(np.unique(y))}")

    # 3) Quick k-fold sanity check
    try:
        evaluate_kfold(X, y, k=5)
    except Exception as e:
        print(f"(KFold skip) {e}")

    # 4) Train classifier
    clf = train_classifier(X, y)
    print("Classifier trained.")

    # 5) Prototypes per student (sid)
    protos = compute_prototypes(X, y, idx2sid)

    # 6) Save artifacts
    ensure_dir(OUT_DIR)
    save_artifacts(clf, protos, sid2idx, idx2sid, sid2name, meta)

    # 7) Optional: run a simple video demo if VIDEO_PATH is set
    if VIDEO_PATH is not None and Path(VIDEO_PATH).exists():
        run_video_attendance(app, protos, sid2name, Path(VIDEO_PATH))
    else:
        print("Video demo skipped (set VIDEO_PATH in the script to run it).")
