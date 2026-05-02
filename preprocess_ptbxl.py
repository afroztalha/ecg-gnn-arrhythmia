"""
=============================================================================
preprocess_ptbxl.py
PTB-XL ECG Dataset — Full Preprocessing Pipeline

Stages:
  1. Download raw records + metadata CSVs from PhysioNet
  2. Stratified record selection (100 per superclass)
  3. Lead II waveform → 64×64 grayscale PNG images (parallel)
  4. Prewitt edge detection (parallel)
  5. CNN patch feature extraction (PatchCNNEncoder, 32-dim)
  6. Graph construction (8-connectivity, node cap 1,500) → TU-format

Authors : Uroosh Kamran (23i-0035), Afroz Talha (23i-2539)
Subject : ANN + PDC Joint Project — FAST-NUCES Islamabad
=============================================================================

Usage:
  python preprocess_ptbxl.py --base-dir /path/to/ecg_project
  python preprocess_ptbxl.py --base-dir /path/to/ecg_project --n-total 500
  python preprocess_ptbxl.py --base-dir /path/to/ecg_project --skip-download
"""

import os
import gc
import ast
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
SEED            = 42
IMG_SIZE        = (224, 224)   # initial render size
EDGE_SIZE       = 64           # resize to before graph construction
BRIGHTNESS_THR  = 80           # lower threshold for PTB-XL (full 10s clips)
PATCH_SIZE      = 7
CNN_DIM         = 32
MAX_NODES       = 1500         # per-graph node cap — prevents GPU OOM
LEAD_INDEX      = 1            # Lead II (0-indexed)

SUPERCLASSES    = {"NORM", "MI", "STTC", "CD", "HYP"}
LABELS          = {"NORM": "0", "MI": "1", "STTC": "2", "CD": "3", "HYP": "4"}
REVERT          = {v: k for k, v in LABELS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def elapsed(t0):
    s = time.time() - t0
    return f"{s/60:.2f} min" if s > 90 else f"{s:.1f}s"

def section(title):
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
    Two-layer convolutional encoder. No pooling — spatial resolution preserved
    so output position (i,j) maps exactly to input pixel (i,j).
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
        return self.encoder(x)


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
def download_ptbxl(raw_dir, df_meta=None):
    """
    Download PTB-XL metadata CSVs and record .dat/.hea files.
    If df_meta is provided, only download those records.
    """
    section("Stage 1 — Download PTB-XL from PhysioNet")
    os.makedirs(raw_dir, exist_ok=True)

    csv_path = osp.join(raw_dir, "ptbxl_database.csv")
    scp_path = osp.join(raw_dir, "scp_statements.csv")

    if not osp.exists(csv_path) or not osp.exists(scp_path):
        log("Downloading metadata CSVs ...")
        try:
            wfdb.dl_files("ptb-xl", dl_dir=raw_dir,
                          files=["ptbxl_database.csv", "scp_statements.csv"])
            log("Metadata CSVs downloaded.")
        except Exception as e:
            print(f"[ERROR] CSV download failed: {e}")
            raise
    else:
        log("Metadata CSVs already on disk.")

    if df_meta is None:
        df_meta = pd.read_csv(csv_path, index_col="ecg_id")

    # Download missing .dat/.hea pairs
    files_to_dl = []
    for rel_path in df_meta["filename_lr"]:
        dat = osp.join(raw_dir, rel_path + ".dat")
        hea = osp.join(raw_dir, rel_path + ".hea")
        if not osp.exists(dat):
            files_to_dl.append(rel_path + ".dat")
        if not osp.exists(hea):
            files_to_dl.append(rel_path + ".hea")

    if not files_to_dl:
        log("All records already on disk.")
        return df_meta

    log(f"Downloading {len(files_to_dl)//2} missing records ...")
    try:
        wfdb.dl_files("ptb-xl", dl_dir=raw_dir, files=files_to_dl)
        log("Download complete.")
    except Exception as e:
        print(f"[ERROR] Record download failed: {e}")
        print("[ERROR] Manual download: https://physionet.org/content/ptb-xl/1.0.3/")
        raise
    return df_meta


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — STRATIFIED RECORD SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def _build_scp_map(raw_dir):
    """Parse scp_statements.csv → fine-grained code → superclass mapping."""
    scp_path = osp.join(raw_dir, "scp_statements.csv")
    if not osp.exists(scp_path):
        raise FileNotFoundError(f"scp_statements.csv not found at {scp_path}")
    scp_df = pd.read_csv(scp_path, index_col=0)
    scp_df = scp_df[scp_df["diagnostic"] == 1]
    mapping = {}
    for code, row in scp_df.iterrows():
        sc = row.get("diagnostic_class", None)
        if pd.notna(sc) and sc in SUPERCLASSES:
            mapping[code] = sc
    log(f"SCP mappings loaded: {len(mapping)} codes")
    return mapping


def _dominant_superclass(scp_codes_str, scp_map):
    """Return the highest-likelihood superclass for one record."""
    try:
        codes = ast.literal_eval(scp_codes_str)
    except Exception:
        return None
    found = {}
    for code, likelihood in codes.items():
        sc = scp_map.get(code)
        if sc is not None:
            if sc not in found or likelihood > found[sc]:
                found[sc] = likelihood
    return max(found, key=found.get) if found else None


def stratified_subset(csv_path, scp_map, n_per_class=100, seed=SEED):
    """
    Sample n_per_class records from each of the 5 PTB-XL superclasses.
    Returns a DataFrame of the selected records.
    """
    section("Stage 2 — Stratified Record Selection")
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df["superclass"] = df["scp_codes"].apply(
        lambda x: _dominant_superclass(x, scp_map)
    )
    df = df[df["superclass"].isin(SUPERCLASSES)].copy()

    max_available = df.groupby("superclass").size().min()
    n_per_class   = min(n_per_class, max_available)
    log(f"Sampling {n_per_class} records per class ...")

    groups = []
    for cls, g in df.groupby("superclass"):
        groups.append(g.sample(n_per_class, random_state=seed))
    sampled = pd.concat(groups)

    log(f"Selected {len(sampled)} records total:")
    for cls in sorted(SUPERCLASSES):
        n = (sampled["superclass"] == cls).sum()
        log(f"  {cls}: {n}")

    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — SIGNAL → IMAGES
# ─────────────────────────────────────────────────────────────────────────────
def _ptbxl_record_to_image(ecg_id, row, raw_dir, img_dir, img_size, scp_map):
    """
    Read one PTB-XL record, render Lead II as PNG.
    Parallel task unit — stateless.
    """
    label = _dominant_superclass(row["scp_codes"], scp_map)
    if label is None:
        return False

    full_path = osp.join(raw_dir, row["filename_lr"])
    try:
        sig, _ = wfdb.rdsamp(full_path)
    except Exception as e:
        print(f"  [WARNING] Cannot read record {ecg_id}: {e}")
        return False

    matplotlib.use("Agg")
    lead = sig[:, LEAD_INDEX].astype(np.float32)
    lead_min, lead_max = lead.min(), lead.max()
    if lead_max > lead_min:
        lead = (lead - lead_min) / (lead_max - lead_min)

    fig = plt.figure(frameon=False, figsize=(4, 2))
    plt.plot(np.arange(len(lead)), lead, linewidth=1.0, color="black")
    plt.xticks([]); plt.yticks([])
    plt.ylim(-0.1, 1.1)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.cla(); plt.clf(); plt.close("all")

    im = cv2.cvtColor(buf, cv2.COLOR_RGBA2GRAY)
    im = cv2.resize(im, img_size, interpolation=cv2.INTER_LANCZOS4)
    im = np.invert(im)

    out_dir  = osp.join(img_dir, label)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"{label}_{ecg_id:05d}.png")
    cv2.imwrite(out_path, im)
    return True


def signals_to_images(df_meta, raw_dir, img_dir, img_size, scp_map, n_jobs):
    """
    PDC Stage: Parallel Lead-II waveform image generation.
    Flynn MIMD: each worker handles a different record.
    """
    section("Stage 3 — Signal → Images (PARALLEL)")
    t0 = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=5, backend="loky")(
        delayed(_ptbxl_record_to_image)(
            ecg_id, row, raw_dir, img_dir, img_size, scp_map
        )
        for ecg_id, row in df_meta.iterrows()
    )

    ok  = sum(1 for r in results if r)
    err = sum(1 for r in results if not r)
    log(f"Images saved: {ok}  skipped: {err}  time: {elapsed(t0)}")

    for cls in sorted(SUPERCLASSES):
        path = osp.join(img_dir, cls)
        n    = len(os.listdir(path)) if osp.exists(path) else 0
        log(f"  {cls}: {n} images")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — PREWITT EDGE DETECTION
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
    Embarrassingly parallel — each image is independent.
    """
    section("Stage 4 — Prewitt Edge Detection (PARALLEL)")
    t0 = time.time()

    tasks = []
    for subdir, _, files in os.walk(img_dir):
        rel = osp.relpath(subdir, img_dir)
        if rel == ".":
            continue
        for fname in files:
            if fname.lower().endswith(".png"):
                tasks.append((
                    osp.join(subdir, fname),
                    osp.join(edge_dir, rel, fname)
                ))

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
# STAGE 5+6 — GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def _extract_features_fullimage(img_gray, encoder, device):
    """Full-image CNN forward pass. Returns (CNN_DIM, H, W) feature map."""
    t = torch.from_numpy(img_gray.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        feat_map = encoder(t.to(device))
    return feat_map.squeeze(0).cpu().numpy()


def _normalize_features(arr):
    arr = np.array(arr, dtype=np.float64)
    m   = arr.mean(axis=0)
    s   = arr.std(axis=0)
    s[s == 0] = 1.0
    return ((arr - m) / s).tolist()


def _image_to_graph_ptbxl(filename, node_label, encoder,
                            brightness_thr, max_nodes=MAX_NODES):
    """
    Convert one PTB-XL edge-filtered PNG into graph data.
    Sequential (not parallelised for PTB-XL — large images use too much
    memory when results are pickled back to the main process via loky).
    """
    worker_device = torch.device("cpu")
    encoder       = encoder.cpu()

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w     = img.shape
    feat_map = _extract_features_fullimage(img, encoder, worker_device)

    hot_pixels = [
        (i, j) for i in range(h) for j in range(w)
        if img[i, j] >= brightness_thr
    ]

    if len(hot_pixels) > max_nodes:
        random.seed(0)
        hot_pixels = random.sample(hot_pixels, max_nodes)

    n_nodes = len(hot_pixels)
    if n_nodes == 0:
        return None

    node_map = {}
    for idx, (i, j) in enumerate(hot_pixels, start=1):
        node_map[(i, j)] = idx

    node_feats = [feat_map[:, i, j].tolist() for i, j in hot_pixels]
    norm_attrs = _normalize_features(node_feats)

    local_edges = []
    for (i, j), nid in node_map.items():
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = node_map.get((i + di, j + dj))
                if nb is not None:
                    local_edges.append([nid, nb])

    return {
        "node_labels": [[node_label]] * n_nodes,
        "graph_label": [node_label],
        "edges":       local_edges,
        "attrs":       norm_attrs,
        "n_nodes":     n_nodes,
        "n_edges":     len(local_edges),
    }


def _assemble_graph_data(results):
    edges, attrs, graph_labels = [], [], []
    node_labels_all, graph_indicator = [], []
    graph_id, node_offset = 1, 1
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
    out_dir = osp.join(out_root, split_name, "raw")
    os.makedirs(out_dir, exist_ok=True)
    prefix  = osp.join(out_dir, split_name)

    mask        = df_gi["graph-id"].isin(graph_subset)
    df_nl_s     = df_nl[mask].copy()
    df_na_s     = df_na[mask].copy()
    df_gi_s     = df_gi[mask].copy()

    orig_node_ids = set(df_gi_s.index + 1)
    old2new_n     = {old: new + 1 for new, old in enumerate(sorted(orig_node_ids))}

    df_A_s = df_A[
        df_A["node-1"].isin(orig_node_ids) &
        df_A["node-2"].isin(orig_node_ids)
    ].copy()
    df_A_s["node-1"] = df_A_s["node-1"].map(old2new_n)
    df_A_s["node-2"] = df_A_s["node-2"].map(old2new_n)

    sorted_g  = sorted(graph_subset)
    old2new_g = {old: new + 1 for new, old in enumerate(sorted_g)}
    df_gi_s["graph-id"] = df_gi_s["graph-id"].map(old2new_g)
    df_gl_s   = df_gl.iloc[[gid - 1 for gid in sorted_g]].copy()

    _save_df(df_A_s,  f"{prefix}_A.txt")
    _save_df(df_gi_s, f"{prefix}_graph_indicator.txt")
    _save_df(df_gl_s, f"{prefix}_graph_labels.txt")
    _save_df(df_na_s, f"{prefix}_node_attributes.txt")
    _save_df(df_nl_s, f"{prefix}_node_labels.txt")

    log(f"  {split_name}: {len(sorted_g)} graphs | "
        f"{len(df_gi_s)} nodes | {len(df_A_s)} edges → {out_dir}")


def build_graphs(edge_dir, graph_dir, encoder,
                 brightness_thr=BRIGHTNESS_THR, seed=SEED):
    """
    Graph construction for PTB-XL — sequential (memory safety with large graphs).
    node cap: MAX_NODES per graph to prevent GPU OOM.
    """
    section("Stage 5+6 — Graph Construction + TU-Format Save (SEQUENTIAL)")
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

    log(f"Graph tasks: {len(tasks)} images  (sequential — memory safety)")
    log(f"Max nodes per graph: {MAX_NODES}")

    results = []
    for k, (fname, lbl) in enumerate(tasks):
        res = _image_to_graph_ptbxl(
            fname, lbl, encoder, brightness_thr, max_nodes=MAX_NODES
        )
        results.append(res)
        if (k + 1) % 50 == 0:
            log(f"  Processed {k+1}/{len(tasks)} graphs ...")

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
        train_g, test_g = train_test_split(
            graph_ids, test_size=0.2, random_state=seed
        )

    log(f"Train: {len(train_g)} graphs  Test: {len(test_g)} graphs")

    _save_tu_split(train_g, "Trainset_PTBXL_CNN", graph_dir,
                   df_A, df_nl, df_na, df_gl, df_gi, CNN_DIM)
    _save_tu_split(test_g,  "Testset_PTBXL_CNN",  graph_dir,
                   df_A, df_nl, df_na, df_gl, df_gi, CNN_DIM)

    log(f"PTB-XL graph construction done  time: {elapsed(t0)}")
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="PTB-XL ECG Dataset preprocessing pipeline."
    )
    p.add_argument("--base-dir",       default="./ecg_project",
                   help="Root output directory (default: ./ecg_project)")
    p.add_argument("--datasets-root",  default=None,
                   help="Where raw .dat files live (default: <base-dir>/data)")
    p.add_argument("--n-total",        type=int, default=500,
                   help="Total records to use across 5 classes (default: 500)")
    p.add_argument("--n-jobs",         type=int, default=2,
                   help="Parallel workers for image stages (default: 2)")
    p.add_argument("--skip-download",  action="store_true",
                   help="Skip PhysioNet download")
    p.add_argument("--skip-images",    action="store_true",
                   help="Skip stage 3 (images already saved)")
    p.add_argument("--skip-edges",     action="store_true",
                   help="Skip stage 4 (edge images already saved)")
    p.add_argument("--skip-graphs",    action="store_true",
                   help="Skip stages 5+6 (TU-format files already saved)")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir   = args.base_dir
    data_root  = args.datasets_root or osp.join(base_dir, "data")
    raw_dir    = osp.join(data_root, "ptb-xl")
    img_dir    = osp.join(base_dir, "ptbxl", "images")
    edge_dir   = osp.join(base_dir, "ptbxl", "edge_filtered")
    graph_dir  = osp.join(base_dir, "ptbxl", "graphs")

    for d in [raw_dir, img_dir, edge_dir, graph_dir]:
        os.makedirs(d, exist_ok=True)

    n_jobs     = args.n_jobs
    n_per_cls  = args.n_total // 5
    t_total    = time.time()

    log(f"Base dir     : {base_dir}")
    log(f"Records/class: {n_per_cls}  (total ~{n_per_cls*5})")
    log(f"Workers      : {n_jobs}  (available: {multiprocessing.cpu_count()})")

    # Stage 1: Download metadata CSVs
    csv_path = osp.join(raw_dir, "ptbxl_database.csv")
    scp_map  = None
    if not args.skip_download:
        # Download CSVs first, then select, then download records
        os.makedirs(raw_dir, exist_ok=True)
        if not osp.exists(csv_path):
            log("Downloading metadata CSVs ...")
            wfdb.dl_files("ptb-xl", dl_dir=raw_dir,
                          files=["ptbxl_database.csv", "scp_statements.csv"])

    # Build SCP map (needed for stratified selection)
    scp_map = _build_scp_map(raw_dir)

    # Stage 2: Stratified record selection
    df_meta = stratified_subset(csv_path, scp_map,
                                 n_per_class=n_per_cls, seed=SEED)

    # Stage 1 continued: download selected records
    if not args.skip_download:
        download_ptbxl(raw_dir, df_meta=df_meta)

    # Stage 3: Signal → Images
    if not args.skip_images:
        signals_to_images(df_meta, raw_dir, img_dir, IMG_SIZE, scp_map, n_jobs)

    # Stage 4: Prewitt edge detection
    if not args.skip_edges:
        prewitt_edge_filter(img_dir, edge_dir, EDGE_SIZE, n_jobs)

    # Stage 5+6: Graph construction
    if not args.skip_graphs:
        encoder = make_encoder(out_dim=CNN_DIM, device=torch.device("cpu"))
        build_graphs(edge_dir, graph_dir, encoder,
                     brightness_thr=BRIGHTNESS_THR)

    log(f"\nTotal preprocessing time: {elapsed(t_total)}")
    log(f"TU-format files saved to: {graph_dir}")
    log("Ready for training — run train.py --dataset ptbxl")


if __name__ == "__main__":
    main()
