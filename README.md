# ecg-gnn-arrhythmia

**ECG Arrhythmia Classification via CNN-Seeded Graph Neural Networks**  
Joint ANN + Parallel & Distributed Computing Project  
Authors: Uroosh Kamran (23i-0035) · Afroz Talha (23i-2539)

---

## Overview

This pipeline classifies ECG arrhythmias by converting beat waveforms into
graphs and training a GNN on them.

**ANN angle**: CNN node-feature extraction → GNN graph classification
(GraphConv / GCN / GAT / GATv2) on MIT-BIH and PTB-XL datasets.

**PDC angle**: GPU acceleration, parallel preprocessing (joblib / multiprocessing),
parallel dataset pipelines, checkpointing, Amdahl/Gustafson analysis.

---

## Repository Structure

```
ecg-gnn-arrhythmia/
├── main.py                        # Entry point — run the full pipeline
├── requirements.txt
├── configs/
│   └── config.py                  # All hyperparameters and paths (edit this first)
├── src/
│   ├── preprocessing/
│   │   ├── mitbih.py              # MIT-BIH: download, beat segmentation, graph build
│   │   ├── ptbxl.py               # PTB-XL: download, signal→image, graph build
│   │   ├── edge_filters.py        # Sobel + Prewitt parallel edge detection
│   │   └── graph_builder.py       # image_to_graph, assemble_graph_data, save_tu_split
│   ├── models/
│   │   ├── cnn_encoder.py         # PatchCNNEncoder (node feature extractor)
│   │   ├── gnn.py                 # GNNModel + CNNGraphGNN
│   │   └── dataset.py             # PyG GraphDataset (TU-format loader)
│   ├── training/
│   │   ├── trainer.py             # Training loop, evaluation, curves
│   │   ├── checkpointing.py       # Rolling + best-model checkpoints
│   │   └── imbalance.py           # WeightedLoss + ImbalancedSampler (parallel)
│   └── utils/
│       └── helpers.py             # debug_section, elapsed, extract_patch, etc.
├── experiments/
│   └── experiment_log.md          # Fill in results for each experiment run
└── notebooks/
    └── (copy original notebook here for reference)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch torchvision torch_geometric wfdb
pip install opencv-python-headless scikit-learn joblib tqdm pandas numpy matplotlib
```

### 2. Set your paths
Edit `configs/config.py` — change `BASE_DIR` and `DATASETS_ROOT` to your local paths.

### 3. Run the pipeline
```bash
# Full run — both datasets
python main.py

# MIT-BIH only, GAT layer, skip download if data is already on disk
python main.py --dataset mitbih --layer GAT --skip-download

# Resume from last checkpoint
python main.py --dataset both

# Train from scratch (ignore checkpoints)
python main.py --no-resume
```

---

## Pipeline Stages

For each dataset (MIT-BIH and PTB-XL):

| Stage | Description | PDC |
|-------|-------------|-----|
| 0 | Dataset download | Sequential (PhysioNet rate limits) |
| 1 | Signal → waveform PNG | Parallel (joblib loky, MIMD) |
| 2 | Edge detection | Parallel Sobel/Prewitt (SIMD) |
| 3 | Graph construction | Parallel `image_to_graph` + serial assembly |
| 4 | GNN training | GPU (CUDA), DataLoader workers |

---

## Datasets

| Dataset | Records | Leads | Classes |
|---------|---------|-------|---------|
| [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) | 48 | 1 (Lead I) | N, L, R, A, V |
| [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 21,837 | 12 | NORM, MI, STTC, CD, HYP |

---

## GNN Layers

Select with `--layer` or set `LAYER_NAME` in `configs/config.py`:

| Name | Class |
|------|-------|
| `GraphConv` | `torch_geometric.nn.GraphConv` (default) |
| `GCN` | `GCNConv` |
| `GAT` | `GATConv` |
| `GATv2` | `GATv2Conv` |

---

## Known Fixes

| Bug | Fix |
|-----|-----|
| CUDA in joblib workers | `image_to_graph` forces `torch.device('cpu')` in workers |
| Early stopping resets on resume | `no_improve_count` saved in checkpoint state |
