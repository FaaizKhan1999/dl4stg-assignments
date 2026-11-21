# %% setup
import json, os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from models import Adjacency, STGNN
from train import load_data, batch, T_IN, T_OUT, DATA, SPLITS

RES, FIG = "results", "figures"
os.makedirs(FIG, exist_ok=True)
MODELS = [("none", "Temporal-only", "None"), ("fixed", "STGNN", "Fixed"),
          ("adaptive", "STGNN", "Adaptive"), ("hybrid", "STGNN", "Hybrid (learned α)")]
if os.path.exists(f"{RES}/hybrid_a0.5.json"):
    MODELS.append(("hybrid_a0.5", "STGNN", "Hybrid (α=0.5)"))

meta = pd.read_csv(f"{DATA}/sd_meta.csv")
A = np.load(f"{DATA}/A_sd.npy")
A_row = A / A.sum(1, keepdims=True)
N = A.shape[0]

# %% Q6(a) table
rows, horizon_rows = [], []
for tag, model, adj in MODELS:
    r = json.load(open(f"{RES}/{tag}.json"))
    rows.append({"Model": model, "Adjacency": adj,
                 "Test MAE": r["test"]["MAE"], "Test RMSE": r["test"]["RMSE"],
                 "Val MAE": r["val"]["MAE"], "Val RMSE": r["val"]["RMSE"],
                 "Params": r["n_params"], "Best epoch": r["best_epoch"],
                 "Train time (min)": r["train_time_s"] / 60,
                 "alpha": r.get("alpha", np.nan)})
    for h in range(T_OUT):
        horizon_rows.append({"Adjacency": adj, "Horizon (min)": 15 * (h + 1),
                             "MAE": r["test"]["MAE_per_horizon"][h],
                             "RMSE": r["test"]["RMSE_per_horizon"][h]})
table = pd.DataFrame(rows)
print(table.to_string(index=False, float_format="%.2f"))
table.to_csv(f"{RES}/comparison_table.csv", index=False)
hz = pd.DataFrame(horizon_rows).pivot(index="Adjacency", columns="Horizon (min)", values="MAE")
print("\nTest MAE per horizon:\n", hz.round(2).to_string())
hz.to_csv(f"{RES}/mae_per_horizon.csv")

# %% test-set predictions for 3 sensors
data, mu, sd = load_data("cpu")
test = data["test"]
ts = pd.DatetimeIndex(np.load(f"{DATA}/timestamps.npy"))
ts_test = ts[(ts >= SPLITS["test"][0]) & (ts < SPLITS["test"][1])]

obs_frac = test["m"].float().mean(0).numpy()
mean_flow = test["y"].mean(0).numpy()
clean = np.where(obs_frac > 0.99)[0]
sel = [33,                                                   # Q3(c) sensor
       clean[np.argmax(mean_flow[clean])],                   # busiest
       clean[np.argmin(np.abs(mean_flow[clean] - np.median(mean_flow[clean])))]]  # typical
sel = list(dict.fromkeys(sel))

def build(tag):
    a = json.load(open(f"{RES}/{tag}.json"))["args"]
    adj = Adjacency(a["adj"], torch.tensor(A_row, dtype=torch.float32), N,
                    a["emb_dim"], a["topk"], a["alpha"])
    m = STGNN(adj, a["d"], T_IN, T_OUT)
    m.load_state_dict(torch.load(f"{RES}/{tag}.pt", map_location="cpu"))
    return m.eval()

preds = {}
with torch.no_grad():
    for tag, _, adj in MODELS:
        m, out = build(tag), []
        for s in range(0, test["n"], 512):
            x, _, _ = batch(test, torch.arange(s, min(s + 512, test["n"])))
            out.append((m(x)[..., 0] * sd + mu)[:, :, sel].numpy())   # (b, 4, |sel|)
        preds[adj] = np.concatenate(out)
        print("predicted:", adj)

start, days = pd.Timestamp("2021-06-07"), 3                  # Mon–Wed in the test period
for h in (0, T_OUT - 1):
    t_pred = ts_test[T_IN + h: T_IN + h + test["n"]]         # time each window's horizon-h forecast targets
    win = (t_pred >= start) & (t_pred < start + pd.Timedelta(days=days))
    fig, axes = plt.subplots(len(sel), 1, figsize=(13, 3.2 * len(sel)), sharex=True)
    for ax, (j, sid) in zip(axes, enumerate(sel)):
        k = T_IN + h + np.arange(test["n"])[win]
        actual = np.where(test["m"][k, sid].numpy(), test["y"][k, sid].numpy(), np.nan)
        ax.plot(t_pred[win], actual, "k", lw=2, label="actual")
        for adj, P in preds.items():
            ax.plot(t_pred[win], P[win, h, j], lw=1, label=adj)
        r = meta.iloc[sid]
        ax.set_title(f"sensor {r.ID} ({r.Fwy}), {15*(h+1)}-min ahead")
        ax.set_ylabel("veh / 15 min")
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/q6a_pred_vs_actual_h{15*(h+1)}.png", dpi=150)
    plt.close(fig)

# %% Q6(b) learned graph analysis
A_ad = np.load(f"{RES}/adaptive_A_adaptive.npy")
B = A > 0
np.fill_diagonal(B, False)
density = B.sum() / (N * (N - 1))

lat, lng = np.radians(meta.Lat.values), np.radians(meta.Lng.values)
hav = (np.sin((lat[:, None] - lat[None]) / 2) ** 2
       + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lng[:, None] - lng[None]) / 2) ** 2)
D = 2 * 6371 * np.arcsin(np.sqrt(hav))                         # km

def topk_mask(W, k):
    W = W.copy()
    np.fill_diagonal(W, -np.inf)
    idx = np.argsort(-W, axis=1)[:, :k]
    M = np.zeros_like(B)
    M[np.arange(N)[:, None], idx] = True
    return M

def eff_neighbours(P):                                         # exp(row entropy)
    P = np.clip(P, 1e-12, 1)
    return np.exp(-(P * np.log(P)).sum(1))

print("\n--- Q6(b) ---")
print(f"self-weight: fixed (row-norm) median {np.median(np.diag(A_row)):.3f} | "
      f"learned median {np.median(np.diag(A_ad)):.4f}")
print(f"effective neighbours per row: fixed {np.median(eff_neighbours(A_row)):.1f} | "
      f"learned {np.median(eff_neighbours(A_ad)):.1f}")
for k in (1, 5, 10):
    Mk = topk_mask(A_ad, k)
    print(f"top-{k:2d} learned neighbours that are road-graph edges: "
          f"{(Mk & B).sum() / Mk.sum():.3f} (random baseline {density:.3f})")

off = ~np.eye(N, dtype=bool)
strongest = np.argsort(-A_ad[off])[:100]
print("100 strongest learned edges present in road graph:", f"{B[off][strongest].mean():.2f}")

T10 = topk_mask(A_ad, 10)
new_edges = T10 & ~B
rng = np.random.default_rng(42)
rand = np.zeros_like(B)
rand[rng.integers(0, N, 7000), rng.integers(0, N, 7000)] = True
np.fill_diagonal(rand, False)
print(f"median distance (km): road edges {np.median(D[B]):.1f} | learned top-10 {np.median(D[T10]):.1f}"
      f" | learned-not-in-road {np.median(D[new_edges]):.1f} | random pairs {np.median(D[rand]):.1f}")

# residual (de-seasonalised) correlation on 2020 — do learned non-physical edges link similar sensors?
X = np.load(f"{DATA}/X.npy")[..., 0]
Mk_obs = np.load(f"{DATA}/mask.npy")
yr = ts.year == 2020
df = pd.DataFrame(np.where(Mk_obs[yr], X[yr], np.nan), index=ts[yr])
R = (df - df.groupby([df.index.day_of_week, df.index.time]).transform("mean")).fillna(0).values
s = R.std(0)
valid = s > 0
Rz = (R - R.mean(0)) / np.where(valid, s, 1)
C = Rz.T @ Rz / len(Rz)
C[~valid] = np.nan
C[:, ~valid] = np.nan
print(f"median residual corr: road edges {np.nanmedian(C[B]):.3f} | learned top-10 {np.nanmedian(C[T10]):.3f}"
      f" | learned-not-in-road {np.nanmedian(C[new_edges]):.3f} | random {np.nanmedian(C[rand]):.3f}")

# figures: adjacency heatmaps
fig, ax = plt.subplots(1, 3, figsize=(19, 6))
for a_, M_, t_ in ((ax[0], A_row, "fixed (row-normalised)"), (ax[1], A_ad, "learned (adaptive)")):
    im = a_.imshow(np.where(M_ > 1e-4, M_, np.nan), norm=LogNorm(1e-4, 1), cmap="viridis",
                   interpolation="nearest")
    a_.set_title(t_)
ax[2].imshow(A_ad[:80, :80], cmap="viridis", interpolation="nearest")
ax[2].set_title("learned, zoom nodes 0–79")
fig.colorbar(im, ax=ax[:2], shrink=0.8, label="weight (log)")
fig.savefig(f"{FIG}/q5b_adjacency_fixed_vs_learned.png", dpi=150)
plt.close(fig)

# distance distributions
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 80, 60)
for M_, lab in ((B, "road-graph edges"), (T10, "learned top-10"),
                (new_edges, "learned, not in road graph"), (rand, "random pairs")):
    ax.hist(D[M_], bins=bins, density=True, histtype="step", lw=1.5, label=lab)
ax.set_xlabel("great-circle distance (km)")
ax.set_ylabel("density")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG}/q6b_edge_distances.png", dpi=150)
plt.close(fig)

# map of the strongest learned edges absent from the road graph
cand = np.argwhere(new_edges)
top_new = cand[np.argsort(-A_ad[new_edges])[:80]]
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(meta.Lng, meta.Lat, s=3, c="lightgrey")
for i, j in top_new:
    ax.plot(meta.Lng[[i, j]], meta.Lat[[i, j]], "r-", lw=0.6, alpha=0.6)
ax.set_title("80 strongest learned edges absent from the road graph")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
fig.tight_layout()
fig.savefig(f"{FIG}/q6b_learned_nonphysical_edges.png", dpi=150)
plt.close(fig)
print("figures saved")

col = A_ad.sum(0)                                  # total attention each sensor receives (sums to N)
order = np.argsort(-col)
dead = (1 - np.load(f"{DATA}/mask.npy").mean(0)) > 0.99
print(f"attention share to top-10 hubs {col[order[:10]].sum()/N:.3f} | top-50 {col[order[:50]].sum()/N:.3f}"
      f" | uniform would be {10/N:.3f} / {50/N:.3f}")
print("dead sensors among top-20 hubs:", int(dead[order[:20]].sum()),
      f"| attention share on dead sensors {col[dead].sum()/N:.3f} (they are {dead.mean():.3f} of nodes)")
fwy = meta["Fwy"].to_numpy(dtype=str)
same = fwy[:, None] == fwy[None]
print(f"top-10 learned on same freeway+direction {same[T10].mean():.3f} | road edges {same[B].mean():.3f}"
      f" | random {same[rand].mean():.3f}")
