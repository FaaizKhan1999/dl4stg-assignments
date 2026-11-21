import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, FIG = "data/processed", "figures"
os.makedirs(FIG, exist_ok=True)

A = np.load(f"{OUT}/A_sd.npy")
meta = pd.read_csv(f"{OUT}/sd_meta.csv")
X = np.load(f"{OUT}/X.npy")[..., 0]
mask = np.load(f"{OUT}/mask.npy")
ts = pd.DatetimeIndex(np.load(f"{OUT}/timestamps.npy"))
N = A.shape[0]

B = A > 0                      # binary edges
np.fill_diagonal(B, False)     # exclude self-loops for edge/degree counts


def q3b():
    print("--- Q3(b) ---")
    n_edges = int(B.sum())
    recip = int((B & B.T).sum())
    print("nodes:", N)
    print("nonzero entries incl. self-loops:", int((A > 0).sum()))
    print("directed edges excl. self-loops:", n_edges)
    print("symmetric:", np.allclose(A, A.T))
    print("edges whose reverse also exists:", recip, "| one-way edges:", n_edges - recip)
    print("mean |A_ij - A_ji| over reciprocal pairs:",
          float(np.abs(A - A.T)[B & B.T].mean()))
    print(f"density (excl. diag): {n_edges / (N * (N - 1)):.4f}")

    # heatmap: full matrix + zoom on first 80 nodes
    cmap = plt.cm.viridis.copy(); cmap.set_bad("white")
    Ash = np.where(A > 0, A, np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(Ash, cmap=cmap, interpolation="nearest")
    ax[0].set_title("A_SD (716×716), white = no edge")
    ax[1].imshow(Ash[:80, :80], cmap=cmap, interpolation="nearest")
    ax[1].set_title("zoom: nodes 0–79 (note asymmetry)")
    for a in ax: a.set_xlabel("j"); a.set_ylabel("i")
    fig.colorbar(im, ax=ax, shrink=0.8, label="A_ij")
    fig.savefig(f"{FIG}/q3b_adjacency.png", dpi=150); plt.close(fig)

    # degree distributions
    out_deg, in_deg = B.sum(1), B.sum(0)
    for name, d in (("out", out_deg), ("in", in_deg)):
        print(f"{name}-degree: min {d.min()}, median {np.median(d):.0f}, "
              f"mean {d.mean():.1f}, max {d.max()}, zero-degree nodes {(d == 0).sum()}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax[0].hist(out_deg, bins=40); ax[0].set_title("out-degree (row nonzeros, excl. self)")
    ax[1].hist(in_deg, bins=40); ax[1].set_title("in-degree (column nonzeros, excl. self)")
    for a in ax: a.set_xlabel("number of neighbours")
    ax[0].set_ylabel("number of sensors")
    fig.tight_layout(); fig.savefig(f"{FIG}/q3b_degree.png", dpi=150); plt.close(fig)


def q3c():
    print("\n--- Q3(c) ---")
    nan_frac = 1 - mask.mean(0)
    out_deg = B.sum(1)
    cands = np.where((nan_frac < 0.01) & (out_deg > 0))[0]
    i = cands[np.argmin(np.abs(out_deg[cands] - np.median(out_deg[cands])))]

    one = np.where(B[i])[0]
    reach2 = B[i] | ((B[i].astype(int) @ B.astype(int)) > 0)
    reach2[i] = False
    two_only = np.where(reach2 & ~B[i])[0]

    r = meta.iloc[i]
    print(f"selected sensor: idx {i}, ID {r.ID}, {r.Fwy} {r.Direction}, lanes {r.Lanes}")
    print("1-hop neighbours:", len(one), "| extra nodes at 2 hops:", len(two_only),
          "| total within 2 hops:", len(one) + len(two_only))
    print("1-hop freeways:\n", meta.iloc[one]["Fwy"].value_counts().to_string())

    # correlation on 2020 (training year), observed values only.
    # Raw correlation is inflated by the shared daily cycle, so we also use
    # residuals after removing each sensor's day-of-week x time-of-day profile.
    yr = ts.year == 2020
    df = pd.DataFrame(np.where(mask[yr], X[yr], np.nan), index=ts[yr])
    resid = df - df.groupby([df.index.day_of_week, df.index.time]).transform("mean")
    diff = df.diff()
    rest = np.setdiff1d(np.arange(N), np.r_[i, one, two_only])
    rand = np.random.default_rng(42).choice(rest, size=min(100, len(rest)), replace=False)
    for label, js in (("1-hop", one), ("2-hop only", two_only), ("random non-neighbours", rand)):
        raw = df[js].corrwith(df[i]).median()
        res = resid[js].corrwith(resid[i]).median()
        dif = diff[js].corrwith(diff[i]).median()
        print(f"median corr with {label:22s}: raw {raw:.3f} | residual {res:.3f} | Δ15min {dif:.3f}")

    # map
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(meta.Lng, meta.Lat, s=4, c="lightgrey", label="other sensors")
    ax.scatter(meta.Lng[two_only], meta.Lat[two_only], s=12, c="tab:orange", label="2-hop")
    ax.scatter(meta.Lng[one], meta.Lat[one], s=18, c="tab:blue", label="1-hop")
    ax.scatter(r.Lng, r.Lat, s=150, c="red", marker="*", label=f"sensor {r.ID}")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
    ax.set_title(f"1- and 2-hop neighbourhood of sensor {r.ID}")
    fig.tight_layout(); fig.savefig(f"{FIG}/q3c_neighbourhood.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    q3b()
    q3c()
