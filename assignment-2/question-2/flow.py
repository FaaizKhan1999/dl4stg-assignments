import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Exactly the handout's settings: calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
FARNEBACK_ARGS = (None, 0.5, 3, 15, 3, 5, 1.2, 0)
THRESH_U8 = 51        # intensity > 0.2 on [0, 1]  <=>  > 51 on the uint8 scale
MIN_SPEED = 0.5       # keep only flow magnitude > 0.5 px
N_VIDEOS = 50         # first 50 training videos
DIR_EDGES = np.arange(0, 361, 20)   # 18 bins of 20 degrees on [0, 360)


# ---------- per-pair flow ----------
def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Frames are converted to uint8 by multiplying [0,1] values by 255 (rounded,
    so 254.9999 does not truncate to 254). Already-uint8 frames pass through."""
    if frame.dtype == np.uint8:
        return frame
    return np.rint(frame * 255).astype(np.uint8)

def flow_pair(prev: np.ndarray, nxt: np.ndarray):
    """Directions (deg, [0,360)) and speeds (px/frame) of the kept pixels of one pair."""
    prev, nxt = to_uint8(prev), to_uint8(nxt)
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, *FARNEBACK_ARGS)
    dx, dy = flow[..., 0], flow[..., 1]          # dx along columns, dy along rows (down)
    speed = np.hypot(dx, dy)
    keep = (prev > THRESH_U8) & (nxt > THRESH_U8) & (speed > MIN_SPEED)
    theta = np.degrees(np.arctan2(-dy[keep], dx[keep])) % 360.0   # 0 = right, 90 = up
    return theta, speed[keep]

def n_blobs(frame: np.ndarray) -> int:
    """Number of bright connected components (rough proxy for 'digits are separate')."""
    n, _ = cv2.connectedComponents((to_uint8(frame) > THRESH_U8).astype(np.uint8))
    return n - 1                                  # minus background

# ---------- pooled over videos ----------
def pooled_flow(videos_u8: np.ndarray, separate_only: bool = False):
    """videos_u8: (V, 20, 1, 64, 64) uint8. Pairs (x_s, x_{s+1}), s = 1..19."""
    thetas, speeds, n_pairs = [], [], 0
    for v in videos_u8:
        for s in range(v.shape[0] - 1):           # tensor index s, s+1  <=>  frames s+1, s+2
            a, b = v[s, 0], v[s + 1, 0]
            if separate_only and (n_blobs(a) < 2 or n_blobs(b) < 2):
                continue
            th, sp = flow_pair(a, b)
            thetas.append(th); speeds.append(sp); n_pairs += 1
    return np.concatenate(thetas), np.concatenate(speeds), n_pairs

# ---------- histograms and diagnostics ----------
def dir_hist(theta: np.ndarray) -> np.ndarray:
    return np.histogram(theta % 360.0, bins=DIR_EDGES)[0] / len(theta)

def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between two histograms (0 = identical, 1 = disjoint)."""
    return 0.5 * float(np.abs(p - q).sum())

def diagnostics(theta: np.ndarray, speed: np.ndarray, n_pairs: int) -> dict:
    """Numbers that help answer Q2.1: how the direction histogram would move if every
    training video were mirrored or rotated, and the typical speed."""
    h = dir_hist(theta)
    med = float(np.median(speed))
    return {
        "pairs": n_pairs,
        "pixels_kept": int(len(theta)),
        "median_speed": med,
        "median_speed_x2": 2 * med,
        "median_speed_x3": 3 * med,
        "tv_to_uniform": tv(h, np.full(18, 1 / 18)),
        "tv_original_vs_mirrored": tv(h, dir_hist(180.0 - theta)),   # M: theta -> 180 - theta
        "tv_original_vs_rotated90": tv(h, dir_hist(theta + 90.0)),   # R: theta -> theta + 90
        "dir_hist": h.tolist(),
    }

# ---------- Figure 1 ----------
def plot_figure1(theta, speed, out_stem: Path, title_suffix: str = ""):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    w = np.full(len(theta), 1 / len(theta))      # fractions of kept pixels, not counts
    med = float(np.median(speed))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(theta, bins=DIR_EDGES, weights=w, edgecolor="white")
    ax[0].axhline(1 / 18, color="k", ls="--", lw=1, label="uniform (1/18)")
    ax[0].set_xticks(np.arange(0, 361, 60))
    ax[0].set_xlabel(r"flow direction $\theta$ (deg; 0 = right, 90 = up)")
    ax[0].set_ylabel("fraction of kept pixels")
    ax[0].set_title("(a) direction" + title_suffix)
    ax[0].legend(frameon=False)

    hi = np.ceil(np.percentile(speed, 99.5) * 4) / 4         # overflow goes in the last bin
    edges = np.arange(MIN_SPEED, hi + 0.25, 0.25)
    ax[1].hist(np.clip(speed, None, edges[-1] - 1e-9), bins=edges, weights=w, edgecolor="white")
    ax[1].axvline(med, color="r", lw=1.5, label=f"median = {med:.3g} px/frame")
    ax[1].set_xlabel("flow speed (px/frame)")
    ax[1].set_ylabel("fraction of kept pixels")
    ax[1].set_title("(b) speed" + title_suffix)
    ax[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=200)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)

def caption(d: dict) -> str:
    return (f"Figure 1: Farneback optical flow on the first {N_VIDEOS} training videos "
            f"({d['pairs']} consecutive frame pairs), pooling {d['pixels_kept']:,} pixels with "
            f"intensity > 0.2 in both frames and flow magnitude > 0.5 px. "
            f"(a) Flow direction (0° = right, 90° = up; dashed line = uniform 1/18). "
            f"(b) Flow speed; median {d['median_speed']:.3g} px/frame (red line). "
            f"Both shown as fractions of kept pixels.")

# ---------- self-test: sign conventions on synthetic blobs ----------
def selftest():
    """A Gaussian blob moved by a known (dx, dy) must give the right speed and direction.
    Catches swapped axes or a wrong sign in atan2(-dy, dx)."""
    yy, xx = np.mgrid[0:64, 0:64]
    def blob(r, c):
        return np.rint(255 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 4.0 ** 2))).astype(np.uint8)
    # (d_col, d_row) on screen -> expected theta
    cases = {"right": ((2, 0), 0.0), "up": ((0, -2), 90.0),
             "left": ((-2, 0), 180.0), "down-left": ((-2, 2), 225.0)}
    ok = True
    for name, ((dc, dr), expected) in cases.items():
        th, sp = flow_pair(blob(32, 32), blob(32 + dr, 32 + dc))
        mean_dir = np.degrees(np.arctan2(np.sin(np.radians(th)).mean(),
                                         np.cos(np.radians(th)).mean())) % 360
        err = min(abs(mean_dir - expected), 360 - abs(mean_dir - expected))
        good = err < 10 and abs(np.median(sp) - np.hypot(dc, dr)) < 0.6
        ok &= good
        print(f"{name:9s}: expected {expected:5.1f}°, got {mean_dir:5.1f}° | "
              f"speed {np.median(sp):.2f} vs {np.hypot(dc, dr):.2f} | {'OK' if good else 'FAIL'}")
    # the M and R maps used in diagnostics: mirror sends 30° to 150°, rotation sends 30° to 120°
    assert dir_hist(np.array([180.0 - 30]))[7] == 1 and dir_hist(np.array([30 + 90.0]))[6] == 1
    print("\nSELFTEST PASS" if ok else "\n*** SELFTEST FAILED ***")
    assert ok

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--separate-only", action="store_true",
                    help="robustness check: skip pairs where digits may overlap (<2 blobs)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from data import load_moving_mnist, get_split              # needs torch/torchvision
    root = os.environ.get("MMNIST_ROOT", "data")
    train = get_split(load_moving_mnist(root), "train")[:N_VIDEOS].numpy()   # uint8

    theta, speed, n_pairs = pooled_flow(train, separate_only=args.separate_only)
    if not args.separate_only:
        assert n_pairs == N_VIDEOS * 19 == 950
    d = diagnostics(theta, speed, n_pairs)

    tag = "_separate" if args.separate_only else ""
    plot_figure1(theta, speed, Path(f"figures/fig1{tag}"),
                 " (separate digits only)" if args.separate_only else "")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/fig1{tag}.json").write_text(json.dumps(d, indent=2))
    np.savez(f"results/fig1_pooled{tag}.npz", theta=theta, speed=speed)
    if not args.separate_only:
        Path("results/fig1_caption.txt").write_text(caption(d) + "\n")
        print(caption(d), "\n")
    print(json.dumps({k: v for k, v in d.items() if k != "dir_hist"}, indent=2))

if __name__ == "__main__":
    main()
