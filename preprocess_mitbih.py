"""
=============================================================================
preprocess_mitbih.py
MIT-BIH Arrhythmia Database — Full Preprocessing Pipeline

Stages:
  1. Download raw .dat/.hea/.atr records from PhysioNet
  2. Stratified beat segmentation → 64×64 waveform PNG images
  3. Prewitt / Sobel edge detection
  4. CNN patch feature extraction (PatchCNNEncoder, 32-dim)
  5. Graph construction (8-connectivity) → TU-format .txt files

Authors : Uroosh Kamran (23i-0035), Afroz Talha (23i-2539)
Subject : ANN + PDC Joint Project — FAST-NUCES Islamabad
=============================================================================

Usage:
  python preprocess_mitbih.py --base-dir /path/to/ecg_project
  python preprocess_mitbih.py --base-dir /path/to/ecg_project --skip-download
  python preprocess_mitbih.py --base-dir /path/to/ecg_project --max-records 10
"""

import os
import gc
import sys
import time
import random
import shutil
import argparse
import multiprocessing
import os.path as osp
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = (64, 64)
EDGE_SIZE = 64
BRIGHTNESS_THR = 128
PATCH_SIZE = 7
CNN_DIM = 32
WINDOW = 180          # samples either side of R-peak
MAX_BEATS_PER_CLASS = 500
MAX_NODES = 2000

BEAT_CLASSES = {'N', 'L', 'R', 'A', 'V'}
LABELS = {'N': '0', 'L': '1', 'R': '2', 'A': '3', 'V': '4'}
REVERT = {v: k for k, v in LABELS.items()}

# All 48 standard MIT-BIH record numbers
ALL_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119,
    121, 122, 123, 124,
    200, 201, 202, 203, 205, 207, 208, 209, 210, 212,
    213, 214, 215, 217, 219, 220, 221, 222, 223, 228,
    230, 231, 232, 233, 234
]

# 20 strategically selected records covering all 5 classes
SUBSET_RECORDS = [
    100, 101, 103, 106, 109,
    111, 115, 117, 118, 119,
    124, 200, 201, 203, 208,
    212, 214, 219, 222, 228,
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def elapsed(t0):
    s = time.time() - t0
    return f"{s/60:.2f} min" if s > 90 else f"{s:.1f}s"

def section(title):
    bar = "─" * 60
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# CNN ENCODER
# ─────────────────────────────────────────────────────────────────────────────
class PatchCNNEncoder(nn.Module):
    """
    Two-layer convolutional encoder for local ECG patch features.
    No pooling — spatial resolution preserved so position (i,j) in the
    output feature map corresponds to pixel (i,j) in the input image.

    Input : (B, 1, H, W) grayscale patch batch
    Output: (B, CNN_DIM) feature vectors — one row per graph node
    """
    def __init__(self, out_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)   # (B, out_dim, H, W)


def make_encoder(out_dim=CNN_DIM, device=None):
    if device is None:
        device = torch.device("cpu")
    enc = PatchCNNEncoder(out_dim=out_dim).to(device)
    enc.eval()
    params = sum(p.numel() for p in enc.parameters())
    log(f"PatchCNNEncoder: out_dim={out_dim}  params={params:,}  device={device}")
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_mitbih(raw_dir, records):
    section("Stage 1 — Download MIT-BIH records from PhysioNet")
    os.makedirs(raw_dir, exist_ok=True)

    to_download = [r for r in records
                   if not osp.exists(osp.join(raw_dir, f"{r}.dat"))]

    if not to_download:
        log("All records already on disk — skipping download.")
        return

    log(f"Downloading {len(to_download)} missing records to {raw_dir} ...")
    files = []
    for r in to_download:
        files += [f"mitbih/{r}.dat", f"mitbih/{r}.hea", f"mitbih/{r}.atr"]
    try:
        wfdb.dl_files("mitbih", dl_dir=raw_dir, files=files)
        log("Download complete.")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("[ERROR] Manual download: https://physionet.org/content/mitdb/1.0.0/")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — STRATIFIED BEAT SEGMENTATION → IMAGES
# ─────────────────────────────────────────────────────────────────────────────
def _beats_from_record(record_num, raw_dir, window):
    """
    Extract all beats from one MIT-BIH record.
    Returns dict: class_symbol → list of beat arrays.
    Parallel task unit — no shared state.
    """
    rec_path = osp.join(raw_dir, str(record_num))
    try:
        record = wfdb.rdrecord(rec_path)
        ann    = wfdb.rdann(rec_path, "atr")
    except Exception as e:
        print(f"  [WARNING] Skipping record {record_num}: {e}")
        return {}

    signal = record.p_signal[:, 0].astype(np.float32)
    n      = len(signal)
    beats  = {cls: [] for cls in BEAT_CLASSES}

    for sym, sample in zip(ann.symbol, ann.sample):
        if sym not in BEAT_CLASSES:
            continue
        s    = max(0, sample - window)
        e    = min(n, sample + window)
        beat = signal[s:e]
        if len(beat) >= window:
            beats[sym].append((beat, record_num))

    return beats


def _save_beat_image(beat, record_num, beat_idx, cls, out_dir, img_size):
    """Render one beat as a 64×64 grayscale PNG."""
    fig = plt.figure(frameon=False, figsize=(2, 2))
    plt.plot(beat, linewidth=0.8)
    plt.xticks([]); plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.cla(); plt.clf(); plt.close("all")

    im = cv2.cvtColor(buf, cv2.COLOR_RGBA2GRAY)
    im = cv2.resize(im, img_size, interpolation=cv2.INTER_LANCZOS4)
    im = np.invert(im)   # white waveform on black background

    path = osp.join(out_dir, cls, f"{cls}_{record_num}_{beat_idx:05d}.png")
    cv2.imwrite(path, im)
    return path


def stratified_beats_to_images(records, raw_dir, img_dir,
                                 img_size, window, max_per_class,
                                 n_jobs, seed=SEED):
    """
    PDC Stage: Parallel beat collection across all records (MAP),
    then stratified sample, then sequential image saving.

    Flynn MIMD: each worker reads a different record independently.
    """
    section("Stage 2 — Stratified Beat Segmentation → Images (PARALLEL)")
    t0 = time.time()
    random.seed(seed)

    # MAP: collect beats from all records in parallel
    log(f"Collecting beats from {len(records)} records using {n_jobs} workers ...")
    all_record_beats = Parallel(n_jobs=n_jobs, verbose=5, backend="loky")(
        delayed(_beats_from_record)(rec, raw_dir, window)
        for rec in records
    )

    # REDUCE: merge per-record dicts
    merged = {cls: [] for cls in BEAT_CLASSES}
    for rec_beats in all_record_beats:
        for cls, beat_list in rec_beats.items():
            merged[cls].extend(beat_list)

    log("Beat counts before stratified sampling:")
    for cls in sorted(BEAT_CLASSES):
        log(f"  {cls}: {len(merged[cls]):,} beats")

    # Stratified sample
    selected = {}
    for cls in BEAT_CLASSES:
        beats = merged[cls]
        random.shuffle(beats)
        selected[cls] = beats[:max_per_class]

    log(f"Beat counts after sampling (max {max_per_class}/class):")
    for cls in sorted(BEAT_CLASSES):
        log(f"  {cls}: {len(selected[cls]):,} beats selected")

    # Save images
    total = 0
    for cls, beats in selected.items():
        out_dir = osp.join(img_dir, cls)
        os.makedirs(out_dir, exist_ok=True)
        lbl = LABELS[cls]
        for idx, (beat, rec_num) in enumerate(beats):
            _save_beat_image(beat, rec_num, idx, cls, img_dir, img_size)
            total += 1
        log(f"  Saved {len(beats)} images for class {cls}")

    log(f"Total images saved: {total}   time: {elapsed(t0)}")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — PREWITT EDGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _apply_prewitt(src_path, dst_path, edge_size):
    """Prewitt edge detection on one image. Parallel task unit."""
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    img    = cv2.resize(img, (edge_size, edge_size))
    kern_h = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kern_v = kern_h.T
    h_grad = cv2.filter2D(img, -1, kern_h)
    v_grad = cv2.filter2D(img, -1, kern_v)
    mag    = np.sqrt(h_grad.astype(np.float32)**2 + v_grad.astype(np.float32)**2)
    mag    = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    os.makedirs(osp.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, mag)
    return True


def prewitt_edge_filter(img_dir, edge_dir, edge_size, n_jobs):
    """
    PDC Stage: Parallel Prewitt edge detection.
    Embarrassingly parallel — each image is fully independent.
    Flynn SIMD: identical operation on different data.
    """
    section("Stage 3 — Prewitt Edge Detection (PARALLEL)")
    t0 = time.time()

    tasks = []
    for subdir, _, files in os.walk(img_dir):
        rel = osp.relpath(subdir, img_dir)
        if rel == ".":
            continue
        for fname in files:
            if fname.lower().endswith(".png"):
                src = osp.join(subdir, fname)
                dst = osp.join(edge_dir, rel, fname)
                tasks.append((src, dst))

    log(f"Prewitt tasks: {len(tasks)} images  workers: {n_jobs}")
    results = Parallel(n_jobs=n_jobs, verbose=5, backend="loky")(
        delayed(_apply_prewitt)(src, dst, edge_size)
        for src, dst in tasks
    )
    ok  = sum(1 for r in results if r)
    err = sum(1 for r in results if not r)
    log(f"Prewitt done: {ok} OK  {err} errors  time: {elapsed(t0)}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 + 5 — GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def _extract_features_fullimage(img_gray, encoder, device):
    """
    Pass the full 64×64 edge image through PatchCNNEncoder.
    Returns (32, H, W) feature map. No pooling → spatial alignment preserved.
    """
    t = torch.from_numpy(img_gray.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    with torch.no_grad():
        feat_map = encoder(t.to(device))  # (1, 32, H, W)
    return feat_map.squeeze(0).cpu().numpy()  # (32, H, W)


def _normalize_features(arr):
    """Z-score normalise per feature dimension across nodes in one graph."""
    arr = np.array(arr, dtype=np.float64)
    m   = arr.mean(axis=0)
    s   = arr.std(axis=0)
    s[s == 0] = 1.0
    return ((arr - m) / s).tolist()


def _image_to_graph(filename, node_label, encoder, brightness_thr,
                    device, max_nodes=MAX_NODES):
    """
    Convert one edge-filtered PNG into graph data structures.
    Parallel task unit — stateless, no global mutation.

    NOTE: encoder runs on CPU inside workers (CUDA cannot be used in
    joblib loky worker processes — see BUG FIX 1 in original notebook).
    """
    worker_device = torch.device("cpu")
    encoder       = encoder.cpu()

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [WARNING] Cannot read {filename} — skipping.")
        return None

    h, w = img.shape

    # Full-image CNN features (spatial alignment preserved)
    feat_map = _extract_features_fullimage(img, encoder, worker_device)
    # feat_map: (CNN_DIM, H, W)

    # Identify bright pixels → graph nodes
    hot_pixels = [
        (i, j) for i in range(h) for j in range(w)
        if img[i, j] >= brightness_thr
    ]

    if len(hot_pixels) > max_nodes:
        random.seed(0)
        hot_pixels = random.sample(hot_pixels, max_nodes)

    n_nodes = len(hot_pixels)
    if n_nodes == 0:
        print(f"  [WARNING] No bright pixels in {filename} — skipping.")
        return None

    # Build node-to-index map
    node_map = {}
    for idx, (i, j) in enumerate(hot_pixels, start=1):
        node_map[(i, j)] = idx

    # Extract CNN feature vector for each node (from pre-computed feature map)
    node_feats = []
    for i, j in hot_pixels:
        feat_vec = feat_map[:, i, j].tolist()   # (CNN_DIM,)
        node_feats.append(feat_vec)

    norm_attrs = _normalize_features(node_feats)

    # 8-connectivity edges
    local_edges = []
    for (i, j), nid in node_map.items():
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                neighbour = node_map.get((ni, nj))
                if neighbour is not None:
                    local_edges.append([nid, neighbour])

    return {
        "node_labels": [[node_label]] * n_nodes,
        "graph_label": [node_label],
        "edges":       local_edges,
        "attrs":       norm_attrs,
        "n_nodes":     n_nodes,
        "n_edges":     len(local_edges),
    }


def _assemble_graph_data(results):
    """
    Merge per-image graph dicts into global TU-format arrays.
    Sequential (global ID assignment — serial Amdahl fraction).
    """
    edges, attrs, graph_labels = [], [], []
    node_labels_all, graph_indicator = [], []
    graph_id = 1
    node_offset = 1

    for res in results:
        if res is None:
            continue
        n = res["n_nodes"]
        for src, dst in res["edges"]:
            edges.append([src + node_offset - 1, dst + node_offset - 1])
        attrs.extend(res["attrs"])
        node_labels_all.extend(res["node_labels"])
        graph_indicator.extend([graph_id] * n)
        graph_labels.append(res["graph_label"])
        node_offset += n
        graph_id    += 1

    log(f"Assembled: {graph_id-1} graphs | "
        f"{len(node_labels_all)} nodes | {len(edges)} edges")
    return edges, attrs, graph_labels, node_labels_all, graph_indicator


def _save_df(df, path):
    df.to_csv(path, header=None, index=None, sep=",")


def _save_tu_split(graph_subset, split_name, out_root,
                   df_A, df_nl, df_na, df_gl, df_gi, feat_dim):
    """Write one train/test split as 5 TU-format .txt files."""
    out_dir = osp.join(out_root, split_name, "raw")
    os.makedirs(out_dir, exist_ok=True)
    prefix  = osp.join(out_dir, split_name)

    mask          = df_gi["graph-id"].isin(graph_subset)
    df_nl_s       = df_nl[mask].copy()
    df_na_s       = df_na[mask].copy()
    df_gi_s       = df_gi[mask].copy()

    orig_node_ids = set(df_gi_s.index + 1)
    old2new_n     = {old: new + 1 for new, old in enumerate(sorted(orig_node_ids))}

    df_A_s = df_A[
        df_A["node-1"].isin(orig_node_ids) &
        df_A["node-2"].isin(orig_node_ids)
    ].copy()
    df_A_s["node-1"] = df_A_s["node-1"].map(old2new_n)
    df_A_s["node-2"] = df_A_s["node-2"].map(old2new_n)

    sorted_g    = sorted(graph_subset)
    old2new_g   = {old: new + 1 for new, old in enumerate(sorted_g)}
    df_gi_s["graph-id"] = df_gi_s["graph-id"].map(old2new_g)
    df_gl_s     = df_gl.iloc[[gid - 1 for gid in sorted_g]].copy()

    _save_df(df_A_s,  f"{prefix}_A.txt")
    _save_df(df_gi_s, f"{prefix}_graph_indicator.txt")
    _save_df(df_gl_s, f"{prefix}_graph_labels.txt")
    _save_df(df_na_s, f"{prefix}_node_attributes.txt")
    _save_df(df_nl_s, f"{prefix}_node_labels.txt")

    log(f"  {split_name}: {len(sorted_g)} graphs | "
        f"{len(df_gi_s)} nodes | {len(df_A_s)} edges → {out_dir}")


def build_graphs(edge_dir, graph_dir, encoder, brightness_thr,
                 n_jobs, seed=SEED):
    """
    PDC Stage: Parallel graph construction (image_to_graph MAP),
    then sequential assembly (REDUCE), then TU-format save.
    """
    section("Stage 4+5 — Graph Construction + TU-Format Save (PARALLEL)")
    t0 = time.time()

    tasks = []
    for subdir, _, files in os.walk(edge_dir):
        rel = osp.relpath(subdir, edge_dir)
        if rel == "." or rel not in LABELS:
            continue
        lbl = LABELS[rel]
        for fname in sorted(files):
            if fname.lower().endswith(".png"):
                tasks.append((osp.join(subdir, fname), lbl))

    log(f"Graph construction: {len(tasks)} images  n_jobs={n_jobs}")

    results = Parallel(n_jobs=n_jobs, verbose=5, backend="loky")(
        delayed(_image_to_graph)(fname, lbl, encoder, brightness_thr,
                                  torch.device("cpu"))
        for fname, lbl in tasks
    )

    log("Assembling graph data (sequential) ...")
    edges, attrs, graph_labels, node_labels, graph_indicator = \
        _assemble_graph_data(results)

    if not graph_labels:
        print("[ERROR] No graphs built. Check edge images in:", edge_dir)
        return

    feat_cols = [f"cnn_{k}" for k in range(CNN_DIM)]
    df_A  = pd.DataFrame(np.array(edges),           columns=["node-1", "node-2"])
    df_nl = pd.DataFrame(np.array(node_labels),     columns=["label"])
    df_gl = pd.DataFrame(np.array(graph_labels),    columns=["label"])
    df_na = pd.DataFrame(np.array(attrs),           columns=feat_cols)
    df_gi = pd.DataFrame(np.array(graph_indicator), columns=["graph-id"])

    log(f"Class distribution:\n{df_gl['label'].value_counts().sort_index()}")

    graph_ids  = df_gi["graph-id"].unique()
    graph_lbls = df_gl["label"].values
    try:
        train_g, test_g = train_test_split(
            graph_ids, test_size=0.2, random_state=seed, stratify=graph_lbls
        )
    except ValueError as e:
        log(f"[WARNING] Stratified split failed ({e}). Using random split.")
        train_g, test_g = train_test_split(graph_ids, test_size=0.2, random_state=seed)

    log(f"Train: {len(train_g)} graphs  Test: {len(test_g)} graphs")

    _save_tu_split(train_g, "Trainset_MITBIH_CNN", graph_dir,
                   df_A, df_nl, df_na, df_gl, df_gi, CNN_DIM)
    _save_tu_split(test_g,  "Testset_MITBIH_CNN",  graph_dir,
                   df_A, df_nl, df_na, df_gl, df_gi, CNN_DIM)

    log(f"MIT-BIH graph construction done  time: {elapsed(t0)}")
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="MIT-BIH Arrhythmia Database preprocessing pipeline."
    )
    p.add_argument("--base-dir",       default="./ecg_project",
                   help="Root output directory (default: ./ecg_project)")
    p.add_argument("--datasets-root",  default=None,
                   help="Where raw .dat files live (default: <base-dir>/data)")
    p.add_argument("--max-records",    type=int, default=20,
                   help="Number of records to use (default: 20; use 48 for full DB)")
    p.add_argument("--max-per-class",  type=int, default=MAX_BEATS_PER_CLASS,
                   help=f"Max beats per class (default: {MAX_BEATS_PER_CLASS})")
    p.add_argument("--n-jobs",         type=int, default=2,
                   help="Parallel workers (default: 2; -1 = all cores)")
    p.add_argument("--skip-download",  action="store_true",
                   help="Skip PhysioNet download (records already on disk)")
    p.add_argument("--skip-images",    action="store_true",
                   help="Skip stage 2 (images already saved)")
    p.add_argument("--skip-edges",     action="store_true",
                   help="Skip stage 3 (edge images already saved)")
    p.add_argument("--skip-graphs",    action="store_true",
                   help="Skip stages 4+5 (TU-format files already saved)")
    p.add_argument("--use-subset",     action="store_true", default=True,
                   help="Use the pre-defined 20-record stratified subset (default: True)")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir    = args.base_dir
    data_root   = args.datasets_root or osp.join(base_dir, "data")
    raw_dir     = osp.join(data_root, "mitbih")
    img_dir     = osp.join(base_dir, "mitbih", "images")
    edge_dir    = osp.join(base_dir, "mitbih", "edge_filtered")
    graph_dir   = osp.join(base_dir, "mitbih", "graphs")

    for d in [raw_dir, img_dir, edge_dir, graph_dir]:
        os.makedirs(d, exist_ok=True)

    # Select records
    if args.use_subset:
        records = SUBSET_RECORDS
        log(f"Using pre-defined 20-record stratified subset.")
    else:
        records = ALL_RECORDS[:args.max_records]
        log(f"Using first {len(records)} records from full list.")
    log(f"Records: {records}")

    n_jobs = args.n_jobs
    log(f"Parallel workers: {n_jobs}  "
        f"(CPU cores available: {multiprocessing.cpu_count()})")

    t_total = time.time()

    # Stage 1: Download
    if not args.skip_download:
        download_mitbih(raw_dir, records)

    # Stage 2: Beats → Images
    if not args.skip_images:
        stratified_beats_to_images(
            records      = records,
            raw_dir      = raw_dir,
            img_dir      = img_dir,
            img_size     = IMG_SIZE,
            window       = WINDOW,
            max_per_class= args.max_per_class,
            n_jobs       = n_jobs,
        )

    # Stage 3: Prewitt edge detection
    if not args.skip_edges:
        prewitt_edge_filter(img_dir, edge_dir, EDGE_SIZE, n_jobs)

    # Stage 4+5: Graph construction
    if not args.skip_graphs:
        encoder = make_encoder(out_dim=CNN_DIM, device=torch.device("cpu"))
        build_graphs(edge_dir, graph_dir, encoder,
                     brightness_thr=BRIGHTNESS_THR, n_jobs=n_jobs)

    log(f"\nTotal preprocessing time: {elapsed(t_total)}")
    log(f"TU-format files saved to: {graph_dir}")
    log("Ready for training — run train.py --dataset mitbih")


if __name__ == "__main__":
    main()
