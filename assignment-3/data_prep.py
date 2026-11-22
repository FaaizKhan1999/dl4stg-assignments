import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_GAP = 4  

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)


def decode(arr):
    return np.char.strip(arr.astype(str))


def load_sd_year(year, sd_ids, chunk=4096):
    """Read one year's h5 in row chunks, keeping only the SD sensor columns."""
    with h5py.File(f"{RAW}/ca_his_raw_{year}.h5", "r") as f:
        g = f["t"]
        cols = decode(g["axis0"][:])
        times = pd.to_datetime(g["axis1"][:])          # int64 ns -> timestamps
        col_pos = pd.Index(cols).get_indexer(sd_ids)
        assert (col_pos >= 0).all(), "some SD IDs missing from h5 columns"

        ds = g["block0_values"]
        out = np.empty((ds.shape[0], len(col_pos)), dtype=np.float32)
        for s in range(0, ds.shape[0], chunk):
            out[s:s + chunk] = ds[s:s + chunk][:, col_pos]
            print(f"  {year}: rows {s}/{ds.shape[0]}", end="\r")
        print()
    return pd.DataFrame(out, index=times, columns=sd_ids), cols


def extract():
    meta = pd.read_csv(f"{RAW}/ca_meta.csv")
    sd_meta = meta[meta["District"] == 11].reset_index(drop=True)
    assert len(sd_meta) == 716
    sd_ids = sd_meta["ID"].astype(str).values          # the canonical node order

    frames = []
    for year in (2020, 2021):
        df, cols = load_sd_year(year, sd_ids)
        frames.append(df)
    flow = pd.concat(frames)

    # --- alignment checks ---
    same_order = (cols == meta["ID"].astype(str).values).all()
    print("h5 column order == meta row order:", same_order)
    print("ID2 == row position:", (meta["ID2"].values == np.arange(len(meta))).all())

    # --- adjacency submatrix, same node order ---
    A_full = np.load(f"{RAW}/ca_rn_adj.npy", mmap_mode="r")
    idx = sd_meta["ID2"].values
    A_sd = np.asarray(A_full[idx][:, idx], dtype=np.float32)

    # --- quick report ---
    print("flow shape:", flow.shape, "| range:", flow.index.min(), "->", flow.index.max())
    steps = flow.index.to_series().diff().value_counts()
    print("time step counts:\n", steps.head())
    print("NaN fraction overall:", flow.isna().mean().mean())
    print("worst sensors by NaN fraction:\n", flow.isna().mean().sort_values(ascending=False).head())
    print("exact zeros fraction:", (flow == 0).mean().mean())
    print("A_sd:", A_sd.shape, "| nonzeros:", np.count_nonzero(A_sd),
          "| diag nonzeros:", np.count_nonzero(np.diag(A_sd)),
          "| symmetric:", np.allclose(A_sd, A_sd.T))

    flow.to_pickle(f"{OUT}/sd_flow_5min.pkl")
    sd_meta.to_csv(f"{OUT}/sd_meta.csv", index=False)
    np.save(f"{OUT}/A_sd.npy", A_sd)

def missing_report(flow5):
    frac = flow5.isna().mean()
    print("\n--- missingness (5-min) ---")
    print("sensors fully missing (both years):", int((frac == 1).sum()))
    for y in ("2020", "2021"):
        print(f"  fully missing in {y}:", int(flow5.loc[y].isna().all().sum()))
    print("sensors with 0% NaN:", int((frac == 0).sum()),
          "| <5%:", int(((frac > 0) & (frac < .05)).sum()),
          "| 5-50%:", int(((frac >= .05) & (frac < .5)).sum()),
          "| 50-<100%:", int(((frac >= .5) & (frac < 1)).sum()))
    q = np.nanpercentile(flow5.values, [1, 25, 50, 75, 99])
    print("flow percentiles [1,25,50,75,99]:", np.round(q, 1))


def resample_15(flow5):
    # PeMS flow = vehicle count per 5 min. 15-min count = sum of three 5-min counts.
    # 3 * mean(available) equals the sum when all three are present and stays
    # unbiased when one or two are missing; the result is NaN only if all three are.
    return 3 * flow5.resample("15min").mean()


def fill_short_gaps(s, max_gap=MAX_GAP):
    isn = s.isna()
    run_len = isn.groupby((~isn).cumsum()).transform("sum")   # length of each NaN run
    interp = s.interpolate(method="linear", limit_area="inside")
    return s.where(~(isn & (run_len <= max_gap)), interp)


def clean(flow15):
    observed = flow15.notna()
    filled = flow15.apply(fill_short_gaps)
    n_obs = observed.values.sum()
    n_interp = int((filled.notna() & ~observed).values.sum())
    n_zero = int(filled.isna().values.sum())
    total = flow15.size
    print("\n--- 15-min cleaning ---")
    print(f"observed: {n_obs/total:.2%} | interpolated (<=1h gaps): {n_interp/total:.2%}"
          f" | zero-filled + masked: {n_zero/total:.2%}")
    return filled.fillna(0.0), observed


def plot_week(flow15, observed, start="2020-02-03", days=7):
    end = pd.Timestamp(start) + pd.Timedelta(days=days) - pd.Timedelta("15min")
    nan_frac = (~observed).mean()
    clean_ids = nan_frac[nan_frac < 0.01].index
    means = flow15[clean_ids].mean()
    busy = means.idxmax()
    typical = (means - means.median()).abs().idxmin()
    wk = observed.loc[start:end]
    wk_frac = wk.mean()                                   # observed share within the plotted week
    gappy_pool = nan_frac[(nan_frac > 0.05) & (nan_frac < 0.5)
                          & (wk_frac.reindex(nan_frac.index) > 0.3)
                          & (wk_frac.reindex(nan_frac.index) < 0.95)].index
    if len(gappy_pool) == 0:                              # fall back: any sensor with some gaps that week
        gappy_pool = wk_frac[(wk_frac > 0.3) & (wk_frac < 0.99)].index
    picks = [(busy, "busiest"), (typical, "median-volume")]
    if len(gappy_pool):
        picks.append((gappy_pool[0], "with gaps"))

    fig, ax = plt.subplots(figsize=(12, 4))
    for sid, label in picks:
        ax.plot(flow15.loc[start:end, sid], label=f"{sid} ({label})", lw=1)
    ax.set_ylabel("flow (vehicles / 15 min)")
    ax.set_title(f"SD traffic flow, week of {start}")
    ax.legend()
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/q3a_week.png", dpi=150)
    print("saved figures/q3a_week.png")


def adjacency_check():
    A = np.load(f"{OUT}/A_sd.npy")
    nz = A[A > 0]
    print("\n--- adjacency ---")
    print("diag values (unique, first 5):", np.unique(np.diag(A))[:5])
    print("weight min/max:", nz.min(), nz.max(), "| distinct values:", len(np.unique(nz)))
    print("entries with an edge one way only:", int(((A > 0) != (A.T > 0)).sum()))


if __name__ == "__main__":
    if not os.path.exists(f"{OUT}/sd_flow_5min.pkl"):
        extract()
    flow5 = pd.read_pickle(f"{OUT}/sd_flow_5min.pkl")
    missing_report(flow5)

    flow15 = resample_15(flow5)
    print("15-min shape:", flow15.shape)          # expect (70176, 716)
    X, observed = clean(flow15)

    np.save(f"{OUT}/X.npy", X.values[..., None].astype(np.float32))   # (T, 716, 1)
    np.save(f"{OUT}/mask.npy", observed.values)                          # True = really observed
    np.save(f"{OUT}/timestamps.npy", X.index.values)

    plot_week(flow15, observed)
    adjacency_check()
