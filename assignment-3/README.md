# AI651 Assignment 3 — Spatio-Temporal Traffic Forecasting on LargeST-SD

Graph-based traffic-flow forecasting on the San Diego subset of LargeST (716 sensors, 2020–2021).
Input: 8 × 15-min steps (2 h); output: next 4 × 15-min steps (1 h).
Compares a temporal-only model against an STGNN with fixed (road-distance), adaptive (learned) and hybrid adjacency.

## Repository structure

| Path | Purpose |
|---|---|
| `requirements.txt` | Python dependencies |
| `utils.py` | `set_seed()` — fixed seed (42) for `random`, NumPy and PyTorch |
| `inspect_data.py` | Inspects the raw LargeST files without loading them into memory |
| `data_prep.py` | Q3(a): extracts SD sensors, resamples to 15 min, handles missing values, builds `A_SD` |
| `q3_graph.py` | Q3(b–c): graph statistics, adjacency/degree plots, 1-/2-hop neighbourhood analysis |
| `models.py` | Temporal encoder, adjacency module (none / fixed / adaptive / hybrid), STGNN |
| `train.py` | Q4–Q5: windows, splits, normalisation, masked training, evaluation |
| `analysis.py` | Q6: comparison tables, predicted-vs-actual plots, learned-graph analysis |
| `q1_verify.py` | Q1: numerical checks for (a), (b) and the (c) table |
| `q2_ring.py` | Q2: ring-road experiments and heatmaps |
| `kaggle_kernel/` | Kaggle script (`run_all.py`) + metadata used to train all four models on GPU |
| `results/` | Metrics (`*.json`), comparison tables (`*.csv`), training logs, learned adjacency (`*.npy`) |
| `figures/` | All figures in the report |

Not committed: `data/` (raw and processed data), `*.pt` checkpoints, `kaggle_upload/`.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12 (local) and Python 3.13 (Kaggle).

## 2. Data

The raw data come from the official LargeST release on Kaggle (`liuxu77/largest`).
Create a Kaggle API token (kaggle.com → Settings → API) and save it to `~/.kaggle/access_token`
(or export `KAGGLE_API_TOKEN`), then download only the four files needed:

```bash
for f in ca_his_raw_2020.h5 ca_his_raw_2021.h5 ca_meta.csv ca_rn_adj.npy; do
  kaggle datasets download liuxu77/largest -f $f -p data/raw --unzip
done
```

## 3. Reproducing the experiments

Run from the repository root, in this order.

```bash
# Q3(a): SD extraction, 15-min resampling, missing values -> data/processed/
python3 data_prep.py

# Q3(b), Q3(c): graph statistics and neighbourhood analysis
python3 q3_graph.py

# Q4, Q5: train the four models (see "Training on GPU" below for the Kaggle route)
for a in none fixed adaptive hybrid; do
  python3 train.py --adj $a | tee results/log_$a.txt
done

# Optional: hybrid with fixed alpha = 0.5
python3 train.py --adj hybrid --alpha 0.5 | tee results/log_hybrid_a0.5.txt

# Q6: tables, prediction plots, learned-graph analysis (CPU is fine)
python3 analysis.py

# Q1, Q2
python3 q1_verify.py
python3 q2_ring.py
```

`data_prep.py` reads the raw `.h5` files in row chunks and keeps only the 716 SD columns, so peak
memory stays below about 1 GB even though each raw file holds 8,600 sensors.

### Training on GPU (Kaggle) — how the reported results were produced

1. Copy the training inputs and code into an upload folder and publish them as a private Kaggle dataset:
   ```bash
   mkdir -p kaggle_upload
   cp data/processed/{X.npy,mask.npy,timestamps.npy,A_sd.npy} models.py utils.py train.py kaggle_upload/
   kaggle datasets init -p kaggle_upload      # set "title" and "id": "<username>/largest-sd-processed"
   kaggle datasets create -p kaggle_upload
   ```
2. In `kaggle_kernel/kernel-metadata.json`, replace the username in `id` and `dataset_sources`, then:
   ```bash
   kaggle kernels push -p kaggle_kernel
   kaggle kernels status <username>/ai651-a3-stgnn
   kaggle kernels output <username>/ai651-a3-stgnn -p results/
   ```
   `run_all.py` trains `none`, `fixed`, `adaptive` and `hybrid` sequentially on one GPU and writes everything to `results/`.

## 4. Configuration

### Data
| Setting | Value |
|---|---|
| Region / sensors | San Diego (Caltrans District 11), 716 sensors |
| Period | 2020-01-01 to 2021-12-31 |
| Resampling | 5 min → 15 min. Flow is a vehicle count per interval, so 15-min flow = sum of three 5-min counts, computed as 3 × mean of available values (equals the sum when complete; NaN only if all three are missing). Units: vehicles / 15 min |
| Missing values | Linear interpolation for gaps ≤ 1 h; all other gaps set to 0 and excluded through an observation mask from loss and metrics |
| Splits | Train 2020-01-01 – 2020-12-31; validation 2021-01-01 – 2021-03-31; test 2021-04-01 – 2021-12-31. Windows are built within each split and never cross a boundary |
| Normalisation | Global z-score; mean and std from observed training values only. Missing inputs set to 0 after normalisation |
| Windows | 8 input steps → 4 output steps; 35,125 / 8,629 / 26,389 windows (train / val / test) |

### Graph
| Setting | Value |
|---|---|
| Fixed adjacency | LargeST road-network adjacency restricted to SD (716 × 716, 17,319 directed edges, self-loops present). Directed, so row normalisation `Â = D⁻¹A`; no extra `+I` because the diagonal is already 1 |
| Adaptive adjacency | `softmax(ReLU(E1 E2ᵀ))`, row-wise, `E1, E2 ∈ R^{716×10}`, random-normal initialisation, dense (no top-k). Optional `--topk k` masks entries outside each row's top k to −∞ *before* the softmax |
| Hybrid adjacency | `α Â_fixed + (1 − α) A_adaptive`; both terms row-stochastic. `α = sigmoid(θ)` learned (initial 0.5), or fixed with `--alpha` |

### Model and training
| Setting | Value |
|---|---|
| Temporal encoder | 2 × Conv1d (kernel 3, d = 32, ReLU) over each sensor's 8-step history → Linear → `Z ∈ R^{B×N×32}` |
| Graph layer | `H = ReLU(Â Z W_g) + Z` (residual). Temporal-only: `H = ReLU(Z W_g) + Z` |
| Head | Linear(32, 32) → ReLU → Linear(32, 4) |
| Loss | Masked MAE on normalised values |
| Optimiser | Adam, lr 1e-3, batch 64, gradient clipping 5.0 |
| Epochs | 30, early stopping on validation MAE with patience 5; best checkpoint restored |
| Seed | 42 (`utils.set_seed`, cuDNN deterministic) |
| Metrics | MAE and RMSE on observed targets only, in vehicles / 15 min, overall and per horizon |

Without the residual connection, aggregating over the road graph made the model underfit
(fixed graph: test MAE 139.1 vs 64.9 for temporal-only). These ablation logs are kept in `results/`.

## 5. Results (test set, Apr–Dec 2021)

| Model | Adjacency | MAE | RMSE | Params | Training time |
|---|---|---|---|---|---|
| Temporal-only | None | 64.66 | 101.44 | 13,668 | 10.0 min |
| STGNN | Fixed | 59.67 | 93.93 | 13,668 | 10.4 min |
| STGNN | Adaptive | **51.77** | **82.41** | 27,988 | 10.9 min |
| STGNN | Hybrid (learned α = 0.015) | 52.40 | 83.47 | 27,989 | 10.9 min |

Test MAE per horizon (15 / 30 / 45 / 60 min):
None 44.90 / 58.34 / 71.44 / 83.98 · Fixed 42.54 / 54.42 / 65.47 / 76.25 ·
Adaptive 39.66 / 48.56 / 55.91 / 62.95 · Hybrid 39.81 / 49.17 / 56.72 / 63.88.

Full metrics, per-epoch logs, hyperparameters and hardware for each run are in `results/<model>.json`.

## 6. Hardware

- **Training (all four models):** Kaggle kernel, 1 × NVIDIA Tesla T4 (16 GB), CUDA 13.0. All models were trained on the same machine, so training times are comparable.
- **Preprocessing and analysis:** local CPU.

GPU training uses cuDNN deterministic mode with a fixed seed. Some GPU indexing and scatter operations
are not bit-for-bit deterministic, so reruns may differ slightly in the last decimal places.

## 7. Figures

| File | Content |
|---|---|
| `figures/q3a_week.png` | Flow for three sensors over one week |
| `figures/q3b_adjacency.png` | Adjacency heatmap and zoom |
| `figures/q3b_degree.png` | In-/out-degree distributions |
| `figures/q3c_neighbourhood.png` | 1- and 2-hop neighbourhood of sensor 1126411 on a map |
| `figures/q5b_adjacency_fixed_vs_learned.png` | Fixed vs learned adjacency |
| `figures/q6a_pred_vs_actual_h15.png`, `..._h60.png` | Predicted vs actual flow, 15 and 60 min ahead |
| `figures/q6b_edge_distances.png` | Distance distributions of road, learned and random edges |
| `figures/q6b_learned_nonphysical_edges.png` | Strongest learned edges absent from the road graph |
| `figures/q2a_heatmaps.png` | Q2(a)(ii) ring-road rollout vs true motion |
