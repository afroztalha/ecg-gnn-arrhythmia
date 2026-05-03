

# =============================================================================
# SECTION 0 — IMPORTS  (identical to parallel version)
# =============================================================================
import os, sys, gc, ast, glob, json, time, shutil, random, argparse
import traceback, warnings, multiprocessing
from collections import Counter
from datetime import datetime
from pathlib import Path

from scipy import signal as scipy_signal
import numpy as np
import pandas as pd
import cv2
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GraphConv, GCNConv, GATConv, GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data as PyGData
from torch_geometric.io import read_tu_data

import os.path as osp
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL OVERRIDE: Force CPU — no GPU acceleration in sequential baseline
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cpu')   # <-- WAS: torch.device('cuda' if ... else 'cpu')
print(f"[SEQ] Device forced to CPU for sequential baseline: {DEVICE}")

print("[SEQ] All imports successful.")
print(f"[SEQ] Python  : {sys.version}")
print(f"[SEQ] PyTorch : {torch.__version__}")
print(f"[SEQ] CUDA    : {torch.cuda.is_available()} (NOT used in sequential mode)")
print(f"[SEQ] CPU cores available: {multiprocessing.cpu_count()} (using 1)")

# =============================================================================
# SECTION 1 — PATH CONFIGURATION  (same as parallel — fill in your paths)
# =============================================================================
BASE_DIR      = "E:/ecg_projectt"
DATASETS_ROOT = "E:/ecg_projectt/data"

MITBIH_RAW_DIR   = osp.join(DATASETS_ROOT, "mitbih")
PTBXL_RAW_DIR    = osp.join(DATASETS_ROOT, "ptb-xl")

MITBIH_IMG_DIR   = osp.join(BASE_DIR, "mitbih", "images_seq")    # separate output dirs
MITBIH_EDGE_DIR  = osp.join(BASE_DIR, "mitbih", "edge_filtered_seq")
MITBIH_GRAPH_DIR = osp.join(BASE_DIR, "mitbih", "graphs_seq")

PTBXL_IMG_DIR    = osp.join(BASE_DIR, "ptbxl", "images_seq")
PTBXL_EDGE_DIR   = osp.join(BASE_DIR, "ptbxl", "edge_filtered_seq")
PTBXL_GRAPH_DIR  = osp.join(BASE_DIR, "ptbxl", "graphs_seq")

CHECKPOINT_DIR   = osp.join(BASE_DIR, "checkpoints_seq")
RESULTS_DIR      = osp.join(BASE_DIR, "results_seq")

# =============================================================================
# SECTION 2 — GLOBAL CONFIG  (sequential overrides noted)
# =============================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── SEQUENTIAL OVERRIDE: N_JOBS and NUM_WORKERS both set to 1/0 ─────────────
N_JOBS      = 1    # <-- WAS: 2 (or -1). Sequential = 1 worker only
NUM_WORKERS = 0    # <-- WAS: 0 already; confirmed no parallel data loading

MITBIH_IMG_SIZE    = (64, 64)
MITBIH_BRIGHTNESS  = 128
MITBIH_PATCH_SIZE  = 7
MITBIH_CNN_DIM     = 32
MITBIH_LABELS      = {'N':'0','L':'1','R':'2','A':'3','V':'4'}
MITBIH_REVERT      = {v: k for k, v in MITBIH_LABELS.items()}
MITBIH_MAX_RECORDS = 20
MITBIH_MAX_BEATS_PER_CLASS = 500
MITBIH_BEAT_CLASSES = {'N','L','R','A','V'}
MITBIH_WINDOW       = 180

PTBXL_IMG_SIZE   = (224, 224)
PTBXL_EDGE_SIZE  = 112
PTBXL_BRIGHTNESS = 80
PTBXL_PATCH_SIZE = 7
PTBXL_CNN_DIM    = 32
PTBXL_LEAD_INDEX = 1
PTBXL_LABELS     = {'NORM':'0','MI':'1','STTC':'2','CD':'3','HYP':'4'}
PTBXL_REVERT     = {v: k for k, v in PTBXL_LABELS.items()}
PTBXL_MAX_RECORDS = 500

EPOCHS       = 150
BATCH_SIZE   = 32
LR           = 0.005
WEIGHT_DECAY = 3e-4
STEP_SIZE    = 40
LAYER_NAME   = 'GraphConv'
C_HIDDEN     = 128
NUM_LAYERS   = 3
DP_RATE      = 0.5
DP_LINEAR    = 0.5
PATIENCE     = 35
CKPT_EVERY_N = 5

# =============================================================================
# AMDAHL TIMING ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────
# Stores per-stage wall-clock seconds so you can compare directly against
# the parallel version's per-stage times.
# =============================================================================
AMDAHL_TIMES = {}

def _record_time(stage, seconds):
    AMDAHL_TIMES[stage] = seconds
    print(f"  [AMDAHL-TIMING] {stage}: {seconds:.3f}s")

def print_amdahl_summary():
    print("\n" + "="*60)
    print("  AMDAHL'S LAW — SEQUENTIAL TIMING SUMMARY")
    print("="*60)
    total = sum(AMDAHL_TIMES.values())
    for stage, t in AMDAHL_TIMES.items():
        frac = t / total * 100 if total > 0 else 0
        print(f"  {stage:<40s}: {t:8.3f}s  ({frac:5.1f}%)")
    print(f"  {'TOTAL':<40s}: {total:8.3f}s")
    print("="*60)
    print("  Compare each stage time against your parallel run to get:")
    print("    Speedup S(p) = T_sequential / T_parallel(p)")
    print("    Serial fraction f = (1/S - 1/p) / (1 - 1/p)")
    print("    Theoretical max speedup = 1/f")
    print("="*60 + "\n")

# =============================================================================
# SECTION 3 — DIRECTORY SETUP  (unchanged)
# =============================================================================
def make_all_dirs():
    dirs = [MITBIH_IMG_DIR, MITBIH_EDGE_DIR, MITBIH_GRAPH_DIR,
            PTBXL_IMG_DIR,  PTBXL_EDGE_DIR,  PTBXL_GRAPH_DIR,
            CHECKPOINT_DIR, RESULTS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def _validate_paths():
    errors = []
    if "<<<REPLACE_ME>>>" in BASE_DIR:
        errors.append("BASE_DIR is still a placeholder.")
    if "<<<REPLACE_ME>>>" in DATASETS_ROOT:
        errors.append("DATASETS_ROOT is still a placeholder.")
    if errors:
        for e in errors: print(f"[ERROR] {e}")
        sys.exit(1)

# =============================================================================
# SECTION 5 — SHARED UTILITIES  (unchanged)
# =============================================================================
def debug_section(title):
    bar = "─" * 60
    print(f"\n[SEQ] {bar}\n[SEQ] {title}\n[SEQ] {bar}")

def elapsed(t0):
    s = time.time() - t0
    return f"{s/60:.2f} min" if s > 90 else f"{s:.1f}s"

def extract_patch(img_gray, i, j, patch_size):
    h, w  = img_gray.shape
    half  = patch_size // 2
    patch = np.zeros((patch_size, patch_size), dtype=np.float32)
    r0 = max(0, i - half);  r1 = min(h, i + half + 1)
    c0 = max(0, j - half);  c1 = min(w, j + half + 1)
    pr0 = half-(i-r0);  pr1 = pr0+(r1-r0)
    pc0 = half-(j-c0);  pc1 = pc0+(c1-c0)
    patch[pr0:pr1, pc0:pc1] = img_gray[r0:r1, c0:c1]
    return patch / 255.0

def normalize_features(arr):
    arr = np.array(arr, dtype=np.float64)
    m = arr.mean(axis=0); s = arr.std(axis=0)
    s[s == 0] = 1.0
    return ((arr - m) / s).tolist()

def save_df(df, path):
    df.to_csv(path, header=None, index=None, sep=',')

# =============================================================================
# SECTION 6 — CNN NODE FEATURE ENCODER  (unchanged architecture)
# =============================================================================
class PatchCNNEncoder(nn.Module):
    def __init__(self, patch_size=7, out_dim=32):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, out_dim),
        )
    def forward(self, x):
        return self.encoder(x)

def make_cnn_encoder(patch_size, out_dim, device):
    enc = PatchCNNEncoder(patch_size=patch_size, out_dim=out_dim).to(device)
    enc.eval()
    total = sum(p.numel() for p in enc.parameters())
    print(f"[SEQ] PatchCNNEncoder: patch={patch_size} out_dim={out_dim} params={total:,}  device={device}")
    return enc

# =============================================================================
# SECTION 7 — GNN MODEL  (unchanged)
# =============================================================================
GNN_LAYER_MAP = {'GraphConv': GraphConv, 'GCN': GCNConv,
                 'GAT': GATConv, 'GATv2': GATv2Conv}

class GNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=3,
                 layer_name='GraphConv', dp_rate=0.5, **kwargs):
        super().__init__()
        gnn_layer = GNN_LAYER_MAP[layer_name]
        layers, in_ch = [], c_in
        for _ in range(num_layers - 1):
            layers += [gnn_layer(in_ch, c_hidden, **kwargs),
                       nn.ReLU(inplace=True), nn.Dropout(dp_rate)]
            in_ch = c_hidden
        layers += [gnn_layer(in_ch, c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            if isinstance(layer, pyg_nn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x

class CNNGraphGNN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_linear=0.5, **kwargs):
        super().__init__()
        self.GNN  = GNNModel(c_in, c_hidden, c_hidden, **kwargs)
        self.head = nn.Sequential(nn.Dropout(dp_linear), nn.Linear(c_hidden, c_out))

    def forward(self, x, edge_index, batch_idx):
        x = self.GNN(x, edge_index)
        x = global_mean_pool(x, batch_idx)
        return self.head(x)

# =============================================================================
# SECTION 8 — GRAPH DATASET LOADER  (unchanged)
# =============================================================================
class GraphDataset(InMemoryDataset):
    def __init__(self, root, name, use_node_attr=False, use_edge_attr=False,
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        out = torch.load(self.processed_paths[0], weights_only=False)
        self.data, self.slices = out[0], out[1]
        if self.data.x is not None and not use_node_attr:
            self.data.x = self.data.x[:, self.num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            self.data.edge_attr = self.data.edge_attr[:, self.num_edge_attributes:]

    @property
    def raw_dir(self):       return osp.join(self.root, self.name, 'raw')
    @property
    def processed_dir(self): return osp.join(self.root, self.name, 'processed')

    @property
    def num_node_labels(self):
        if self.data.x is None: return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None: return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None: return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None: return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        return [f'{self.name}_{s}.txt' for s in
                ['A','graph_indicator','graph_labels','node_attributes','node_labels']]

    @property
    def processed_file_names(self): return 'data.pt'

    def process(self):
        self.data, self.slices, _ = read_tu_data(self.raw_dir, self.name)
        if self.pre_filter is not None:
            dl = [self.get(i) for i in range(len(self))]
            dl = [d for d in dl if self.pre_filter(d)]
            self.data, self.slices = self.collate(dl)
        if self.pre_transform is not None:
            dl = [self.get(i) for i in range(len(self))]
            dl = [self.pre_transform(d) for d in dl]
            self.data, self.slices = self.collate(dl)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self): return f'{self.name}({len(self)})'

# =============================================================================
# SECTION 9 — CHECKPOINTING  (unchanged)
# =============================================================================
def ckpt_rolling_path(dataset_tag, layer_name, epoch):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return osp.join(CHECKPOINT_DIR, f"{dataset_tag}_{layer_name}_epoch{epoch:03d}_{ts}.pth")

def ckpt_best_path(dataset_tag, layer_name):
    return osp.join(CHECKPOINT_DIR, f"{dataset_tag}_{layer_name}_BEST.pth")

def save_checkpoint(epoch, model, optimizer, scheduler, train_acc, val_acc,
                    best_val_acc, train_losses, val_losses, train_accs, val_accs,
                    dataset_tag, layer_name, no_improve_count=0, is_best=False):
    state = {
        'epoch': epoch, 'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'train_acc': train_acc, 'val_acc': val_acc,
        'best_val_acc': best_val_acc, 'no_improve_count': no_improve_count,
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs': train_accs, 'val_accs': val_accs,
        'dataset_tag': dataset_tag, 'layer_name': layer_name,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(state, ckpt_rolling_path(dataset_tag, layer_name, epoch))
    if is_best:
        torch.save(state, ckpt_best_path(dataset_tag, layer_name))

def load_checkpoint(model, optimizer, scheduler, dataset_tag, layer_name):
    pattern = osp.join(CHECKPOINT_DIR, f"{dataset_tag}_{layer_name}_epoch*_*.pth")
    checkpoints = sorted(glob.glob(pattern))
    if not checkpoints:
        return 0, 0.0, 0, [], [], [], []
    state = torch.load(checkpoints[-1], map_location=DEVICE, weights_only=False)
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    scheduler.load_state_dict(state['scheduler_state'])
    return (state['epoch']+1, state.get('best_val_acc',0.0),
            state.get('no_improve_count',0),
            state.get('train_losses',[]), state.get('val_losses',[]),
            state.get('train_accs',[]), state.get('val_accs',[]))

def load_best_model(model, dataset_tag, layer_name):
    best_path = ckpt_best_path(dataset_tag, layer_name)
    if not osp.exists(best_path): return
    state = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state['model_state'])

# =============================================================================
# SECTION 10 — GRAPH CONSTRUCTION: SEQUENTIAL VERSION
# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WAS:
#   results = Parallel(n_jobs=N_JOBS, backend='loky')(
#       delayed(image_to_graph)(fname, lbl, ...) for fname, lbl in tasks
#   )
#
# SEQUENTIAL IS:
#   results = [image_to_graph(fname, lbl, ...) for fname, lbl in tasks]
#
# The inner image_to_graph function is IDENTICAL. Only the outer loop changes.
# CNN encoding is forced to CPU (same as worker processes in the parallel ver).
# =============================================================================

def image_to_graph(filename, node_label, cnn_encoder, patch_size,
                   brightness_thr, device, max_nodes=2000):
    """
    SEQUENTIAL: Identical logic to the parallel version. No joblib wrapper.
    CPU is used for CNN encoding (device is already set to cpu globally).
    """
    # [AMDAHL-TIMING] This function body is the parallel fraction per image.
    img = cv2.imread(filename)
    if img is None: return None

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w     = gray.shape
    node_map = np.full((h, w), -1, dtype=np.int64)
    hot_pixels = []

    for i in range(h):
        for j in range(w):
            if gray[i, j] >= brightness_thr:
                node_map[i, j] = len(hot_pixels) + 1
                hot_pixels.append((i, j))

    if len(hot_pixels) > max_nodes:
        random.seed(0)
        hot_pixels = random.sample(hot_pixels, max_nodes)
        node_map = np.full((h, w), -1, dtype=np.int64)
        for local_id, (pi, pj) in enumerate(hot_pixels, start=1):
            node_map[pi, pj] = local_id

    n_nodes = len(hot_pixels)
    if n_nodes == 0: return None

    patches_np = np.stack(
        [extract_patch(gray, i, j, patch_size)[np.newaxis]
         for i, j in hot_pixels]
    )

    with torch.no_grad():
        t_in  = torch.tensor(patches_np, dtype=torch.float32).to(device)
        feats = cnn_encoder(t_in).numpy()

    norm_attrs = normalize_features(feats)

    local_edges = []
    for i in range(h):
        for j in range(w):
            if node_map[i, j] == -1: continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0: continue
                    ni, nj = i+di, j+dj
                    if (0 <= ni < h and 0 <= nj < w
                            and node_map[ni, nj] != -1):
                        local_edges.append([node_map[i,j], node_map[ni,nj]])

    return {
        'node_labels': [[node_label]] * n_nodes,
        'graph_label': [node_label],
        'edges':       local_edges,
        'attrs':       norm_attrs,
        'n_nodes':     n_nodes,
        'n_edges':     len(local_edges),
    }


def assemble_graph_data(results):
    """Sequential assembly — identical to parallel version (was already serial)."""
    edges, attrs, graph_labels, node_labels_all, graph_indicator = [], [], [], [], []
    graph_id = 1; node_id_offset = 1

    for res in results:
        if res is None: continue
        n = res['n_nodes']
        for local_src, local_dst in res['edges']:
            edges.append([local_src+node_id_offset-1, local_dst+node_id_offset-1])
        attrs.extend(res['attrs'])
        node_labels_all.extend(res['node_labels'])
        graph_indicator.extend([graph_id] * n)
        graph_labels.append(res['graph_label'])
        node_id_offset += n; graph_id += 1

    print(f"  [SEQ] Assembled: {graph_id-1} graphs | "
          f"{len(node_labels_all)} nodes | {len(edges)} edges")
    return edges, attrs, graph_labels, node_labels_all, graph_indicator


def save_tu_split(graph_subset, split_name, out_root,
                  df_A, df_node_label, df_node_attr,
                  df_graph_label, df_graph_indicator, cnn_feat_dim):
    out_dir = osp.join(out_root, split_name, 'raw')
    os.makedirs(out_dir, exist_ok=True)
    prefix  = osp.join(out_dir, split_name)

    mask_nodes = df_graph_indicator['graph-id'].isin(graph_subset)
    df_nl_s = df_node_label[mask_nodes].copy()
    df_na_s = df_node_attr[mask_nodes].copy()
    df_gi_s = df_graph_indicator[mask_nodes].copy()

    orig_node_ids = set(df_gi_s.index + 1)
    old_to_new_n  = {old: new+1 for new, old in enumerate(sorted(orig_node_ids))}

    df_A_s = df_A[
        df_A['node-1'].isin(orig_node_ids) &
        df_A['node-2'].isin(orig_node_ids)
    ].copy()
    df_A_s['node-1'] = df_A_s['node-1'].map(old_to_new_n)
    df_A_s['node-2'] = df_A_s['node-2'].map(old_to_new_n)

    sorted_graphs = sorted(graph_subset)
    old_to_new_g  = {old: new+1 for new, old in enumerate(sorted_graphs)}
    df_gi_s['graph-id'] = df_gi_s['graph-id'].map(old_to_new_g)
    df_gl_s = df_graph_label.iloc[[gid-1 for gid in sorted_graphs]].copy()

    save_df(df_A_s,  f'{prefix}_A.txt')
    save_df(df_gi_s, f'{prefix}_graph_indicator.txt')
    save_df(df_gl_s, f'{prefix}_graph_labels.txt')
    save_df(df_na_s, f'{prefix}_node_attributes.txt')
    save_df(df_nl_s, f'{prefix}_node_labels.txt')
    print(f"  [SEQ] {split_name}: {len(sorted_graphs)} graphs → {out_dir}")

# =============================================================================
# SECTION 11 — MIT-BIH PREPROCESSING: SEQUENTIAL VERSION
# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WAS: joblib.Parallel across records / images
# SEQUENTIAL IS: plain Python for-loop — one record/image at a time
# =============================================================================

def _mitbih_record_to_images_seq(record_num, raw_dir, img_dir,
                                   img_size, window, beat_classes):
    """
    SEQUENTIAL: Process one MIT-BIH record — identical logic, no joblib.
    [AMDAHL-TIMING] This is the bottleneck: was parallelised across records.
    """
    try:
        rec_path = osp.join(raw_dir, str(record_num))
        record   = wfdb.rdrecord(rec_path)
        ann      = wfdb.rdann(rec_path, 'atr')
    except Exception as e:
        print(f"  [SEQ-WARNING] Record {record_num}: {e}")
        return 0

    signal = record.p_signal[:, 0].astype(np.float32)
    n = len(signal); saved = 0

    for idx, (sym, sample) in enumerate(zip(ann.symbol, ann.sample)):
        if sym not in beat_classes: continue
        label = MITBIH_LABELS.get(sym)
        if label is None: continue
        s = max(0, sample-window); e = min(n, sample+window)
        beat = signal[s:e]
        if len(beat) < window: continue

        fig = plt.figure(frameon=False, figsize=(2, 2))
        plt.plot(beat, linewidth=0.8)
        plt.xticks([]); plt.yticks([])
        for spine in plt.gca().spines.values(): spine.set_visible(False)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.cla(); plt.clf(); plt.close('all')

        im = cv2.cvtColor(buf, cv2.COLOR_RGBA2GRAY)
        im = cv2.resize(im, img_size, interpolation=cv2.INTER_LANCZOS4)
        im = np.invert(im)

        beat_class_name = MITBIH_REVERT[label]
        out_dir_cls = osp.join(img_dir, beat_class_name)
        os.makedirs(out_dir_cls, exist_ok=True)
        cv2.imwrite(osp.join(out_dir_cls,
                             f"{beat_class_name}_{record_num}_{idx:05d}.png"), im)
        saved += 1
    return saved


def mitbih_signal_to_images_seq(record_list, raw_dir, img_dir,
                                  img_size, window):
    """
    SEQUENTIAL replacement for:
        Parallel(n_jobs=N_JOBS)(_mitbih_record_to_images(...) for rec in records)

    [AMDAHL-TIMING] Stage: MIT-BIH Signal→Images
    Parallel fraction: all record processing (loop body).
    Serial fraction  : list aggregation (sum).
    """
    debug_section("MIT-BIH — Stage 1 [SEQ]: Signal → Images")
    t0 = time.time()

    # ── SEQUENTIAL LOOP (was: joblib.Parallel) ────────────────────────────────
    total = 0
    for rec in tqdm(record_list, desc="MIT-BIH records"):
        total += _mitbih_record_to_images_seq(
            rec, raw_dir, img_dir, img_size, window, MITBIH_BEAT_CLASSES
        )
    # ─────────────────────────────────────────────────────────────────────────

    dt = time.time() - t0
    _record_time("MITBIH_signal_to_images", dt)
    print(f"[SEQ] MIT-BIH images saved: {total}  time: {elapsed(t0)}")
    return total


def _apply_sobel_seq(src_path, dst_path, edge_size):
    """Sobel on one image — identical to parallel version, no wrapper."""
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    img = cv2.resize(img, (edge_size, edge_size))
    sx  = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    os.makedirs(osp.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, mag)
    return True


def sobel_edge_filter_seq(img_dir, edge_dir, edge_size, dataset_tag):
    """
    SEQUENTIAL replacement for:
        Parallel(n_jobs=N_JOBS)(_apply_sobel(src, dst, ...) for src, dst in tasks)

    [AMDAHL-TIMING] Stage: Sobel Edge Filter
    Parallel fraction: each _apply_sobel call.
    Serial fraction  : task list construction, counting.
    """
    debug_section(f"{dataset_tag} — Stage 2 [SEQ]: Sobel Edge Filter")
    t0 = time.time()
    tasks = []
    for subdir, _, files in os.walk(img_dir):
        rel = os.path.relpath(subdir, img_dir)
        if rel == '.': continue
        for fname in files:
            if fname.lower().endswith('.png'):
                tasks.append((osp.join(subdir, fname),
                               osp.join(edge_dir, rel, fname)))

    # ── SEQUENTIAL LOOP (was: joblib.Parallel) ────────────────────────────────
    ok = 0
    for src, dst in tqdm(tasks, desc=f"{dataset_tag} Sobel"):
        if _apply_sobel_seq(src, dst, edge_size): ok += 1
    # ─────────────────────────────────────────────────────────────────────────

    dt = time.time() - t0
    _record_time(f"{dataset_tag}_sobel_filter", dt)
    print(f"[SEQ] Sobel done: {ok}/{len(tasks)}  time: {elapsed(t0)}")
    return ok


def mitbih_build_graphs_seq(edge_dir, graph_dir, cnn_encoder,
                              patch_size, brightness_thr, device,
                              labels, feat_dim, seed):
    """
    SEQUENTIAL replacement for:
        Parallel(n_jobs=N_JOBS)(delayed(image_to_graph)(...) for ...)

    [AMDAHL-TIMING] Stage: MIT-BIH Graph Construction
    Parallel fraction: image_to_graph per image.
    Serial fraction  : assemble_graph_data (was always serial in original).
    """
    debug_section("MIT-BIH — Stage 3 [SEQ]: Graph Construction")
    t0 = time.time()

    tasks = []
    for subdir, _, files in os.walk(edge_dir):
        rel = os.path.relpath(subdir, edge_dir)
        if rel == '.' or rel not in labels: continue
        lbl = labels[rel]
        for fname in sorted(files):
            if fname.lower().endswith('.png'):
                tasks.append((osp.join(subdir, fname), lbl))

    print(f"[SEQ] Graph tasks: {len(tasks)} images (sequential, 1 worker)")

    # ── SEQUENTIAL LOOP (was: joblib.Parallel) ────────────────────────────────
    t_graph_start = time.time()
    results = []
    for fname, lbl in tqdm(tasks, desc="MITBIH graphs"):
        results.append(image_to_graph(
            fname, lbl, cnn_encoder, patch_size, brightness_thr, device
        ))
    dt_graph = time.time() - t_graph_start
    _record_time("MITBIH_image_to_graph_loop", dt_graph)
    # ─────────────────────────────────────────────────────────────────────────

    t_assemble = time.time()
    edges, attrs, graph_labels, node_labels, graph_indicator = \
        assemble_graph_data(results)
    _record_time("MITBIH_assemble_graph_data", time.time() - t_assemble)

    if not graph_labels:
        print("[SEQ-ERROR] No MIT-BIH graphs built."); return

    feat_cols = [f'cnn_{k}' for k in range(feat_dim)]
    df_A  = pd.DataFrame(np.array(edges),           columns=['node-1','node-2'])
    df_nl = pd.DataFrame(np.array(node_labels),     columns=['label'])
    df_gl = pd.DataFrame(np.array(graph_labels),    columns=['label'])
    df_na = pd.DataFrame(np.array(attrs),           columns=feat_cols)
    df_gi = pd.DataFrame(np.array(graph_indicator), columns=['graph-id'])

    graph_ids = df_gi['graph-id'].unique(); graph_lbls = df_gl['label'].values
    try:
        train_g, test_g = train_test_split(
            graph_ids, test_size=0.2, random_state=seed, stratify=graph_lbls)
    except ValueError:
        train_g, test_g = train_test_split(graph_ids, test_size=0.2, random_state=seed)

    save_tu_split(train_g, 'Trainset_MITBIH_CNN', graph_dir,
                  df_A, df_nl, df_na, df_gl, df_gi, feat_dim)
    save_tu_split(test_g,  'Testset_MITBIH_CNN',  graph_dir,
                  df_A, df_nl, df_na, df_gl, df_gi, feat_dim)

    _record_time("MITBIH_graph_construction_total", time.time() - t0)
    gc.collect()


def mitbih_stratified_beats_seq(record_list, raw_dir, img_dir,
                                  img_size, window, max_per_class, seed=42):
    """
    SEQUENTIAL version of stratified beat sampling.
    No joblib — single-threaded.
    """
    random.seed(seed)
    debug_section("MIT-BIH — Stage 1 [SEQ]: Stratified Beat Sampling → Images")
    t0 = time.time()

    all_beats = {cls: [] for cls in MITBIH_BEAT_CLASSES}
    for record_num in tqdm(record_list, desc="Collecting beats"):
        rec_path = osp.join(raw_dir, str(record_num))
        try:
            record = wfdb.rdrecord(rec_path)
            ann    = wfdb.rdann(rec_path, 'atr')
        except Exception as e:
            print(f"  [SEQ-WARNING] Skipping {record_num}: {e}"); continue

        signal = record.p_signal[:, 0].astype(np.float32)
        n      = len(signal)
        for idx, (sym, sample) in enumerate(zip(ann.symbol, ann.sample)):
            if sym not in MITBIH_BEAT_CLASSES: continue
            label = MITBIH_LABELS.get(sym)
            if label is None: continue
            s = max(0, sample-window); e = min(n, sample+window)
            beat = signal[s:e]
            if len(beat) < window: continue
            all_beats[sym].append((beat, record_num, idx))

    selected = {}
    for cls in MITBIH_BEAT_CLASSES:
        beats = all_beats[cls]; random.shuffle(beats)
        selected[cls] = beats[:max_per_class]

    total_saved = 0
    for cls, beats in selected.items():
        label     = MITBIH_LABELS[cls]
        class_dir = osp.join(img_dir, cls)
        os.makedirs(class_dir, exist_ok=True)
        for beat, record_num, idx in tqdm(beats, desc=f"Saving {cls}", leave=False):
            fig = plt.figure(frameon=False, figsize=(2, 2))
            plt.plot(beat, linewidth=0.8)
            plt.xticks([]); plt.yticks([])
            for spine in plt.gca().spines.values(): spine.set_visible(False)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.cla(); plt.clf(); plt.close('all')
            im = cv2.cvtColor(buf, cv2.COLOR_RGBA2GRAY)
            im = cv2.resize(im, img_size, interpolation=cv2.INTER_LANCZOS4)
            im = np.invert(im)
            cv2.imwrite(osp.join(class_dir, f"{cls}_{record_num}_{idx:05d}.png"), im)
            total_saved += 1

    dt = time.time() - t0
    _record_time("MITBIH_stratified_beats", dt)
    print(f"[SEQ] Total images saved: {total_saved}  time: {elapsed(t0)}")
    return total_saved

# =============================================================================
# SECTION 12 — PTB-XL PREPROCESSING: SEQUENTIAL VERSION
# =============================================================================
def _build_scp_map(ptbxl_dir):
    scp_path = osp.join(ptbxl_dir, 'scp_statements.csv')
    scp_df   = pd.read_csv(scp_path, index_col=0)
    scp_df   = scp_df[scp_df['diagnostic'] == 1]
    mapping  = {}
    valid_sc = {'NORM','MI','STTC','CD','HYP'}
    for code, row in scp_df.iterrows():
        sc = row.get('diagnostic_class', None)
        if pd.notna(sc) and sc in valid_sc:
            mapping[code] = sc
    return mapping

def get_ptbxl_superclass(scp_codes_str, scp_map):
    try: codes = ast.literal_eval(scp_codes_str)
    except: return None
    found = {}
    for code, likelihood in codes.items():
        sc = scp_map.get(code)
        if sc is not None:
            if sc not in found or likelihood > found[sc]:
                found[sc] = likelihood
    return max(found, key=found.get) if found else None

def _ptbxl_record_to_image_seq(ecg_id, row, ptbxl_dir, img_dir,
                                 img_size, lead_index, scp_map):
    """SEQUENTIAL: identical logic, no joblib wrapper."""
    label = get_ptbxl_superclass(row['scp_codes'], scp_map)
    if label is None: return False
    full_path = osp.join(ptbxl_dir, row['filename_lr'])
    try: sig, _ = wfdb.rdsamp(full_path)
    except: return False

    fig, axes = plt.subplots(3, 1, figsize=(4, 3), frameon=False)
    for ax, idx in zip(axes, [1, 6, 11]):
        lead = sig[:, idx].astype(np.float32)
        lmin, lmax = lead.min(), lead.max()
        if lmax > lmin: lead = (lead-lmin)/(lmax-lmin)
        ax.plot(np.arange(len(lead)), lead, linewidth=1.0, color='black')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-0.1, 1.1)
        for spine in ax.spines.values(): spine.set_visible(False)
    plt.subplots_adjust(hspace=0.05)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.cla(); plt.clf(); plt.close('all')
    im = cv2.cvtColor(buf, cv2.COLOR_RGBA2GRAY)
    im = cv2.resize(im, img_size, interpolation=cv2.INTER_LANCZOS4)
    im = np.invert(im)
    out_dir = osp.join(img_dir, label)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(osp.join(out_dir, f"{label}_{ecg_id:05d}.png"), im)
    return True

def ptbxl_signal_to_images_seq(df_meta, ptbxl_dir, img_dir,
                                 img_size, lead_index, scp_map,
                                 max_per_class=None):
    """
    SEQUENTIAL replacement for:
        Parallel(n_jobs=N_JOBS)(_ptbxl_record_to_image(...) for ecg_id, row in ...)

    [AMDAHL-TIMING] Stage: PTB-XL Signal→Images
    """
    debug_section("PTB-XL — Stage 1 [SEQ]: Signal → Images")
    t0 = time.time()

    # ── SEQUENTIAL LOOP (was: joblib.Parallel) ────────────────────────────────
    saved = skipped = 0
    for ecg_id, row in tqdm(df_meta.iterrows(), total=len(df_meta),
                             desc="PTB-XL records"):
        if _ptbxl_record_to_image_seq(ecg_id, row, ptbxl_dir, img_dir,
                                       img_size, lead_index, scp_map):
            saved += 1
        else:
            skipped += 1
    # ─────────────────────────────────────────────────────────────────────────

    if max_per_class is not None:
        for cls in ['NORM','MI','STTC','CD','HYP']:
            cls_dir = osp.join(img_dir, cls)
            if not osp.exists(cls_dir): continue
            imgs = sorted(os.listdir(cls_dir))
            for f in imgs[max_per_class:]:
                os.remove(osp.join(cls_dir, f))

    dt = time.time() - t0
    _record_time("PTBXL_signal_to_images", dt)
    print(f"[SEQ] PTB-XL images: {saved} saved  {skipped} skipped  time: {elapsed(t0)}")
    return saved


def _apply_prewitt_seq(src_path, dst_path, edge_size):
    """Prewitt on one image — identical to parallel version, no wrapper."""
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False
    img    = cv2.resize(img, (edge_size, edge_size))
    kern_h = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
    kern_v = kern_h.T
    h_grad = cv2.filter2D(img, -1, kern_h); v_grad = cv2.filter2D(img, -1, kern_v)
    mag    = np.sqrt(h_grad.astype(np.float32)**2 + v_grad.astype(np.float32)**2)
    mag    = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    os.makedirs(osp.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, mag)
    return True

def ptbxl_prewitt_filter_seq(img_dir, edge_dir, edge_size):
    """
    SEQUENTIAL replacement for:
        Parallel(n_jobs=N_JOBS)(_apply_prewitt(src, dst, ...) for ...)

    [AMDAHL-TIMING] Stage: PTB-XL Prewitt Filter
    """
    debug_section("PTB-XL — Stage 2 [SEQ]: Prewitt Edge Filter")
    t0 = time.time()
    tasks = []
    for subdir, _, files in os.walk(img_dir):
        rel = os.path.relpath(subdir, img_dir)
        if rel == '.': continue
        for fname in files:
            if fname.lower().endswith('.png'):
                tasks.append((osp.join(subdir, fname),
                               osp.join(edge_dir, rel, fname)))

    # ── SEQUENTIAL LOOP (was: joblib.Parallel) ────────────────────────────────
    ok = 0
    for src, dst in tqdm(tasks, desc="PTB-XL Prewitt"):
        if _apply_prewitt_seq(src, dst, edge_size): ok += 1
    # ─────────────────────────────────────────────────────────────────────────

    dt = time.time() - t0
    _record_time("PTBXL_prewitt_filter", dt)
    print(f"[SEQ] Prewitt done: {ok}/{len(tasks)}  time: {elapsed(t0)}")
    return ok


def ptbxl_build_graphs_seq(edge_dir, graph_dir, cnn_encoder,
                             patch_size, brightness_thr, device,
                             labels, revert_labels, feat_dim, seed):
    """
    SEQUENTIAL graph construction for PTB-XL.
    (Original was already sequential for memory safety — timing still captured.)

    [AMDAHL-TIMING] Stage: PTB-XL Graph Construction
    """
    debug_section("PTB-XL — Stage 3 [SEQ]: Graph Construction")
    t0 = time.time()

    tasks = []
    for subdir, _, files in os.walk(edge_dir):
        rel = os.path.relpath(subdir, edge_dir)
        if rel == '.' or rel not in labels: continue
        lbl = labels[rel]
        for fname in sorted(files):
            if fname.lower().endswith('.png'):
                tasks.append((osp.join(subdir, fname), lbl))

    results = []
    for k, (fname, lbl) in enumerate(tqdm(tasks, desc="PTB-XL graphs")):
        res = image_to_graph(fname, lbl, cnn_encoder, patch_size,
                             brightness_thr, device, max_nodes=1500)
        results.append(res)

    edges, attrs, graph_labels, node_labels, graph_indicator = \
        assemble_graph_data(results)

    if not graph_labels:
        print("[SEQ-ERROR] No PTB-XL graphs built."); return

    feat_cols = [f'cnn_{k}' for k in range(feat_dim)]
    df_A  = pd.DataFrame(np.array(edges),           columns=['node-1','node-2'])
    df_nl = pd.DataFrame(np.array(node_labels),     columns=['label'])
    df_gl = pd.DataFrame(np.array(graph_labels),    columns=['label'])
    df_na = pd.DataFrame(np.array(attrs),           columns=feat_cols)
    df_gi = pd.DataFrame(np.array(graph_indicator), columns=['graph-id'])

    graph_ids = df_gi['graph-id'].unique(); graph_lbls = df_gl['label'].values
    try:
        train_g, test_g = train_test_split(
            graph_ids, test_size=0.2, random_state=seed, stratify=graph_lbls)
    except ValueError:
        train_g, test_g = train_test_split(graph_ids, test_size=0.2, random_state=seed)

    save_tu_split(train_g, 'Trainset_PTBXL_CNN', graph_dir,
                  df_A, df_nl, df_na, df_gl, df_gi, feat_dim)
    save_tu_split(test_g,  'Testset_PTBXL_CNN',  graph_dir,
                  df_A, df_nl, df_na, df_gl, df_gi, feat_dim)

    _record_time("PTBXL_graph_construction_total", time.time() - t0)
    gc.collect()

# =============================================================================
# SECTION 12b — CLASS IMBALANCE: SEQUENTIAL VERSION
# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WAS: joblib.Parallel map-reduce for label counting + weight assignment
# SEQUENTIAL IS: single Counter loop + list comprehension
# =============================================================================

def compute_class_weights_seq(dataset, n_classes, device):
    """
    SEQUENTIAL replacement for:
        _parallel_count_labels(dataset, N_JOBS)
    which used joblib map-reduce.

    [AMDAHL-TIMING] Stage: Class Weight Computation
    Serial fraction dominates here (trivial for small n_classes).
    """
    debug_section("Class imbalance [SEQ] — weighted loss computation")
    t0 = time.time()

    # ── SEQUENTIAL COUNTER (was: joblib map-reduce across chunks) ─────────────
    counts = Counter()
    for i in range(len(dataset)):
        label = int(dataset[i].y.item())
        counts[label] += 1
    # ─────────────────────────────────────────────────────────────────────────

    total = sum(counts.values())
    weights = []
    for c in range(n_classes):
        cnt = counts.get(c, 1)
        weights.append(total / (n_classes * cnt))
        print(f"  [SEQ] Class {c}: {cnt:7,d} samples  weight={weights[-1]:.4f}")

    dt = time.time() - t0
    _record_time("class_weight_computation", dt)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def make_imbalanced_sampler_seq(dataset, n_classes):
    """
    SEQUENTIAL replacement for parallel sampler build.
    Uses PyG ImbalancedSampler directly (its internal build is sequential here).

    [AMDAHL-TIMING] Stage: Sampler Build
    Parallel fraction: label counting + weight assignment loops.
    """
    from torch_geometric.loader import ImbalancedSampler
    debug_section("Class imbalance [SEQ] — ImbalancedSampler build")
    t0 = time.time()

    # ── SEQUENTIAL weight computation (was: two parallel joblib stages) ────────
    counts = Counter(int(dataset[i].y.item()) for i in range(len(dataset)))
    # ─────────────────────────────────────────────────────────────────────────

    sampler = ImbalancedSampler(dataset)

    dt = time.time() - t0
    _record_time("imbalanced_sampler_build", dt)
    return sampler

# =============================================================================
# SECTION 13 — TRAINING LOOP: SEQUENTIAL (CPU) VERSION
# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WAS: GPU (CUDA) accelerated forward/backward pass
# SEQUENTIAL IS: identical logic but on CPU (DEVICE = 'cpu')
#
# DataLoader num_workers=0 (already set above) — no parallel data loading.
# =============================================================================

def run_one_epoch(model, loader, optimizer, criterion, device, is_train, split_label):
    """Identical to parallel version — GPU calls become CPU calls via device."""
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true_all, y_pred_all = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=split_label, leave=False):
            batch = batch.to(device)
            if batch.x is None or batch.edge_index is None: continue
            out  = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = out.argmax(dim=1)
            total_loss += loss.item() * batch.num_graphs
            correct    += int((preds == batch.y).sum())
            total      += batch.num_graphs
            y_true_all.extend(batch.y.cpu().numpy().tolist())
            y_pred_all.extend(preds.cpu().numpy().tolist())

    if total == 0: return 0.0, float("inf"), [], []
    return correct/total, total_loss/total, y_true_all, y_pred_all


def train_pipeline_seq(dataset_tag, graph_dir, train_name, test_name,
                        cnn_feat_dim, num_classes, revert_labels,
                        layer_name, epochs, batch_size, lr, weight_decay,
                        step_size, c_hidden, num_layers, dp_rate, dp_linear,
                        patience, ckpt_every_n, device, seed, resume=True):
    """
    SEQUENTIAL training pipeline.
    Key differences from parallel version:
      1. device = CPU (no CUDA acceleration)
      2. DataLoader num_workers=0 (no parallel data loading)
      3. compute_class_weights_seq — sequential Counter
      4. make_imbalanced_sampler_seq — sequential build

    [AMDAHL-TIMING] Stage: Training Loop (per epoch)
    """
    debug_section(f"TRAINING [SEQ] — {dataset_tag} — {layer_name}")

    try:
        train_ds = GraphDataset(root=graph_dir, name=train_name, use_node_attr=True)
        test_ds  = GraphDataset(root=graph_dir, name=test_name,  use_node_attr=True)
    except Exception as e:
        print(f"[SEQ-ERROR] Dataset load failed: {e}"); return

    if train_ds._data.x is not None:
        train_ds._data.x = train_ds._data.x[:, :cnn_feat_dim]
    if test_ds._data.x is not None:
        test_ds._data.x  = test_ds._data.x[:,  :cnn_feat_dim]

    feat_dim  = train_ds.num_features
    n_classes = train_ds.num_classes

    train_ds   = train_ds.shuffle()
    val_split  = int(len(train_ds) * 0.8)
    train_part = train_ds[:val_split]
    val_part   = train_ds[val_split:]

    # ── SEQUENTIAL class weight computation (was: parallel map-reduce) ─────────
    class_weights = compute_class_weights_seq(train_part, n_classes, device)
    sampler       = make_imbalanced_sampler_seq(train_part, n_classes)
    # ─────────────────────────────────────────────────────────────────────────

    # ── DataLoaders: num_workers=0 — no parallel loading ─────────────────────
    train_loader = DataLoader(train_part, batch_size=batch_size,
                              sampler=sampler, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_part,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,    batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    # ─────────────────────────────────────────────────────────────────────────

    model = CNNGraphGNN(c_in=feat_dim, c_hidden=c_hidden, c_out=n_classes,
                        dp_linear=dp_linear, num_layers=num_layers,
                        layer_name=layer_name, dp_rate=dp_rate).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    if resume:
        (start_epoch, best_val_acc, no_improve_count,
         train_losses, val_losses, train_accs, val_accs) = \
            load_checkpoint(model, optimizer, scheduler, dataset_tag, layer_name)
    else:
        start_epoch, best_val_acc, no_improve_count = 0, 0.0, 0
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

    t_train_start = time.time()
    epoch_times   = []

    for epoch in range(start_epoch, epochs):
        t_ep = time.time()

        train_acc, train_loss, _, _ = run_one_epoch(
            model, train_loader, optimizer, criterion,
            device, True, f"Train {epoch+1:03d}")
        val_acc, val_loss, y_true_val, y_pred_val = run_one_epoch(
            model, val_loader, optimizer, criterion,
            device, False, f"Val   {epoch+1:03d}")

        scheduler.step()
        train_accs.append(train_acc); train_losses.append(train_loss)
        val_accs.append(val_acc);     val_losses.append(val_loss)

        is_best = val_acc > best_val_acc
        if is_best: best_val_acc, no_improve_count = val_acc, 0
        else:        no_improve_count += 1

        ep_time = time.time() - t_ep
        epoch_times.append(ep_time)
        print(f"  Epoch {epoch+1:03d}/{epochs}"
              f"  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
              f"  loss={train_loss:.4f}  ep_time={ep_time:.1f}s"
              f"  {'★ BEST' if is_best else ''}")

        if (epoch+1) % ckpt_every_n == 0 or is_best:
            save_checkpoint(epoch, model, optimizer, scheduler,
                            train_acc, val_acc, best_val_acc,
                            train_losses, val_losses, train_accs, val_accs,
                            dataset_tag, layer_name,
                            no_improve_count=no_improve_count, is_best=is_best)

        if no_improve_count >= patience:
            print(f"  [SEQ] Early stopping at epoch {epoch+1}"); break

    total_train_time = time.time() - t_train_start
    avg_epoch_time   = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    _record_time(f"{dataset_tag}_training_total", total_train_time)
    _record_time(f"{dataset_tag}_avg_epoch_time", avg_epoch_time)

    print(f"\n[SEQ] Training complete: {elapsed(t_train_start)}"
          f"  avg epoch={avg_epoch_time:.1f}s  best_val={best_val_acc:.4f}")

    load_best_model(model, dataset_tag, layer_name)
    test_acc, test_loss, y_true, y_pred = run_one_epoch(
        model, test_loader, optimizer, criterion, device, False, "Test")
    print(f"[SEQ] Test accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

    class_names = [revert_labels[str(i)] for i in range(n_classes)]
    print(classification_report(y_true, y_pred, labels=list(range(n_classes)),
                                target_names=class_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=30, values_format="d")
    ax.set_title(f"{dataset_tag} Test [SEQ] — {layer_name}")
    plt.tight_layout()
    cm_path = osp.join(RESULTS_DIR, f"cm_{dataset_tag}_{layer_name}_seq.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()

    return {"dataset": dataset_tag, "layer": layer_name,
            "test_acc": test_acc, "best_val_acc": best_val_acc,
            "train_time": elapsed(t_train_start)}

# =============================================================================
# SECTION 14 — MAIN ORCHESTRATOR: SEQUENTIAL VERSION
# =============================================================================
def download_mitbih():
    """Same as parallel version — network I/O, unchanged."""
    print("\n[SEQ] MIT-BIH download — sequential (same as parallel, I/O bound)")
    os.makedirs(MITBIH_RAW_DIR, exist_ok=True)
    record_nums = [
        100,101,102,103,104,105,106,107,108,109,
        111,112,113,114,115,116,117,118,119,
        121,122,123,124,
        200,201,202,203,205,207,208,209,210,212,213,214,215,217,
        219,220,221,222,223,228,230,231,232,233,234
    ]
    already = [r for r in record_nums
               if osp.exists(osp.join(MITBIH_RAW_DIR, f"{r}.dat"))]
    if len(already) == len(record_nums):
        print("[SEQ] MIT-BIH already downloaded."); return
    to_dl = [r for r in record_nums if r not in already]
    files_to_dl = []
    for r in to_dl:
        for ext in ['.dat','.hea','.atr']:
            files_to_dl.append(f"mitbih/{r}{ext}")
    wfdb.dl_files('mitbih', dl_dir=MITBIH_RAW_DIR, files=files_to_dl)

def download_ptbxl(max_records=None):
    """Same as parallel version — network I/O, unchanged."""
    os.makedirs(PTBXL_RAW_DIR, exist_ok=True)
    csv_path = osp.join(PTBXL_RAW_DIR, 'ptbxl_database.csv')
    if not osp.exists(csv_path):
        wfdb.dl_files('ptb-xl', dl_dir=PTBXL_RAW_DIR,
                      files=['ptbxl_database.csv', 'scp_statements.csv'])
    df_meta = pd.read_csv(csv_path, index_col='ecg_id')
    if max_records is not None: df_meta = df_meta.head(max_records)
    files_to_dl = []
    for rel_path in df_meta['filename_lr']:
        for ext in ['.dat','.hea']:
            p = osp.join(PTBXL_RAW_DIR, rel_path+ext)
            if not osp.exists(p): files_to_dl.append(rel_path+ext)
    if files_to_dl:
        wfdb.dl_files('ptb-xl', dl_dir=PTBXL_RAW_DIR, files=files_to_dl)
    return df_meta


def main(args):
    _validate_paths()
    make_all_dirs()

    t_total = time.time()
    run_mitbih = args.dataset in ('mitbih', 'both')
    run_ptbxl  = args.dataset in ('ptbxl',  'both')
    results_summary = []

    # ── MIT-BIH ───────────────────────────────────────────────────────────────
    if run_mitbih:
        debug_section("PIPELINE START [SEQ] — MIT-BIH")
        t_mitbih = time.time()

        if not args.skip_download: download_mitbih()

        all_records = [
            100,101,102,103,104,105,106,107,108,109,
            111,112,113,114,115,116,117,118,119,
            121,122,123,124,
            200,201,202,203,205,207,208,209,210,212,213,214,215,217,
            219,220,221,222,223,228,230,231,232,233,234
        ]
        if MITBIH_MAX_RECORDS is not None:
            all_records = all_records[:MITBIH_MAX_RECORDS]

        if not args.skip_preprocess:
            mitbih_stratified_beats_seq(
                record_list=all_records, raw_dir=MITBIH_RAW_DIR,
                img_dir=MITBIH_IMG_DIR, img_size=MITBIH_IMG_SIZE,
                window=MITBIH_WINDOW, max_per_class=MITBIH_MAX_BEATS_PER_CLASS,
                seed=SEED)
            sobel_edge_filter_seq(
                MITBIH_IMG_DIR, MITBIH_EDGE_DIR,
                MITBIH_IMG_SIZE[0], 'MIT-BIH')

        if not args.skip_graphs:
            enc = make_cnn_encoder(MITBIH_PATCH_SIZE, MITBIH_CNN_DIM, DEVICE)
            mitbih_build_graphs_seq(
                MITBIH_EDGE_DIR, MITBIH_GRAPH_DIR, enc,
                MITBIH_PATCH_SIZE, MITBIH_BRIGHTNESS,
                DEVICE, MITBIH_LABELS, MITBIH_CNN_DIM, SEED)

        res = train_pipeline_seq(
            dataset_tag='MITBIH', graph_dir=MITBIH_GRAPH_DIR,
            train_name='Trainset_MITBIH_CNN', test_name='Testset_MITBIH_CNN',
            cnn_feat_dim=MITBIH_CNN_DIM, num_classes=len(MITBIH_LABELS),
            revert_labels=MITBIH_REVERT, layer_name=args.layer,
            epochs=args.epochs, batch_size=BATCH_SIZE, lr=LR,
            weight_decay=WEIGHT_DECAY, step_size=STEP_SIZE,
            c_hidden=C_HIDDEN, num_layers=NUM_LAYERS,
            dp_rate=DP_RATE, dp_linear=DP_LINEAR,
            patience=PATIENCE, ckpt_every_n=CKPT_EVERY_N,
            device=DEVICE, seed=SEED, resume=args.resume)
        if res: results_summary.append(res)
        _record_time("MITBIH_pipeline_total", time.time() - t_mitbih)

    # ── PTB-XL ────────────────────────────────────────────────────────────────
    if run_ptbxl:
        debug_section("PIPELINE START [SEQ] — PTB-XL")
        t_ptbxl = time.time()

        if not args.skip_download:
            df_meta = download_ptbxl(max_records=PTBXL_MAX_RECORDS)
        else:
            csv_path = osp.join(PTBXL_RAW_DIR, 'ptbxl_database.csv')
            df_meta  = pd.read_csv(csv_path, index_col='ecg_id')
            if PTBXL_MAX_RECORDS: df_meta = df_meta.head(PTBXL_MAX_RECORDS)

        scp_map = _build_scp_map(PTBXL_RAW_DIR)

        if not args.skip_preprocess:
            ptbxl_signal_to_images_seq(
                df_meta, PTBXL_RAW_DIR, PTBXL_IMG_DIR,
                PTBXL_IMG_SIZE, PTBXL_LEAD_INDEX, scp_map,
                max_per_class=PTBXL_MAX_RECORDS//5)
            # Direct copy (no Prewitt) — same decision as parallel version
            for subdir, _, files in os.walk(PTBXL_IMG_DIR):
                rel = os.path.relpath(subdir, PTBXL_IMG_DIR)
                for fname in files:
                    if fname.lower().endswith('.png'):
                        dst_dir = osp.join(PTBXL_EDGE_DIR, rel)
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.copy2(osp.join(subdir, fname),
                                     osp.join(dst_dir, fname))

        if not args.skip_graphs:
            enc = make_cnn_encoder(PTBXL_PATCH_SIZE, PTBXL_CNN_DIM, DEVICE)
            ptbxl_build_graphs_seq(
                PTBXL_EDGE_DIR, PTBXL_GRAPH_DIR, enc,
                PTBXL_PATCH_SIZE, PTBXL_BRIGHTNESS,
                DEVICE, PTBXL_LABELS, PTBXL_REVERT,
                PTBXL_CNN_DIM, SEED)

        res = train_pipeline_seq(
            dataset_tag='PTBXL', graph_dir=PTBXL_GRAPH_DIR,
            train_name='Trainset_PTBXL_CNN', test_name='Testset_PTBXL_CNN',
            cnn_feat_dim=PTBXL_CNN_DIM, num_classes=len(PTBXL_LABELS),
            revert_labels=PTBXL_REVERT, layer_name=args.layer,
            epochs=args.epochs, batch_size=BATCH_SIZE, lr=LR,
            weight_decay=WEIGHT_DECAY, step_size=STEP_SIZE,
            c_hidden=C_HIDDEN, num_layers=NUM_LAYERS,
            dp_rate=DP_RATE, dp_linear=DP_LINEAR,
            patience=PATIENCE, ckpt_every_n=CKPT_EVERY_N,
            device=DEVICE, seed=SEED, resume=args.resume)
        if res: results_summary.append(res)
        _record_time("PTBXL_pipeline_total", time.time() - t_ptbxl)

    _record_time("GRAND_TOTAL", time.time() - t_total)
    print_amdahl_summary()

    debug_section("FINAL RESULTS SUMMARY [SEQ]")
    for r in results_summary:
        print(f"  Dataset={r['dataset']}  Layer={r['layer']}"
              f"  TestAcc={r['test_acc']:.4f}  Time={r['train_time']}")

# =============================================================================
# SECTION 15 — ENTRY POINT
# =============================================================================
import types

RUN_DATASET     = 'both'
RUN_LAYER       = 'GraphConv'
RUN_EPOCHS      = EPOCHS
SKIP_DOWNLOAD   = True
SKIP_PREPROCESS = False
SKIP_GRAPHS     = False
RESUME          = False

args = types.SimpleNamespace(
    dataset=RUN_DATASET, layer=RUN_LAYER, epochs=RUN_EPOCHS,
    skip_download=SKIP_DOWNLOAD, skip_preprocess=SKIP_PREPROCESS,
    skip_graphs=SKIP_GRAPHS, resume=RESUME,
)

print(f"\n[SEQ] Sequential baseline configuration:")
print(f"  dataset={RUN_DATASET}  layer={RUN_LAYER}  epochs={RUN_EPOCHS}")
print(f"  skip_download={SKIP_DOWNLOAD}  skip_preprocess={SKIP_PREPROCESS}"
      f"  skip_graphs={SKIP_GRAPHS}")
print(f"  N_JOBS=1  NUM_WORKERS=0  DEVICE=cpu")
print(f"\n[SEQ] Run this script and record stage times for Amdahl's Law.\n")

try:
    main(args)
except KeyboardInterrupt:
    print("\n[SEQ] Interrupted. Partial timing:")
    print_amdahl_summary()
except Exception as e:
    print(f"\n[SEQ-ERROR] {e}")
    traceback.print_exc()
    print_amdahl_summary()
