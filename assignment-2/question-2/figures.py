import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from data import load_moving_mnist, get_split, to_float, make_window, window_frames
from transforms import TRANSFORMS
from models import Symmetrised, synthetic_videos
from evaluate import load_run
from flow import n_blobs
from metrics import fmt

FIG, RES, GIFS, CAP = Path("figures"), Path("results"), Path("gifs"), Path("results/captions")
TEST_OFFSET = 9000                         # test video i is dataset index 9000 + i
NEAREST = getattr(Image, "Resampling", Image).NEAREST


# ---------- helpers ----------
def frames_np(t):
    """(5, 1, 64, 64) tensor -> list of five (64, 64) numpy arrays."""
    return [f[0].detach().float().cpu().numpy() for f in t]

def has_overlap(video_u8, frame_numbers):
    """Heuristic: digits overlap (or touch) in a frame if it shows fewer than 2 bright blobs."""
    v = video_u8.cpu().numpy()
    return any(n_blobs(v[f - 1, 0]) < 2 for f in frame_numbers)

def save(fig, stem):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def image_grid(rows, row_labels, titles, suptitle, stem):
    """rows: list of lists of (64, 64) images; titles: dict {(r, c): text}."""
    R, C = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(R, C, figsize=(1.25 * C, 1.3 * R + 0.6), squeeze=False)
    for r in range(R):
        for c in range(C):
            ax = axes[r][c]
            if c >= len(rows[r]) or rows[r][c] is None:
                ax.axis("off"); continue
            ax.imshow(rows[r][c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if (r, c) in titles:
                ax.set_title(titles[(r, c)], fontsize=7)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    save(fig, stem)

def write_gif(rows, labels, path: Path, scale=3):
    """One GIF frame per horizon h: each of the four rows shows (input frame h, future frame h)."""
    GIFS.mkdir(exist_ok=True)
    tile, lab_w, pad, head = 64 * scale, 170, 6, 22
    W, H = lab_w + 2 * tile + 3 * pad, head + len(rows) * (tile + pad) + pad
    frames = []
    for h in range(5):
        im = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(im)
        d.text((lab_w + pad, 5), f"input frame {h + 1}", fill=0)
        d.text((lab_w + 2 * pad + tile, 5), f"future, horizon h = {h + 1}", fill=0)
        for r, (inp, fut) in enumerate(rows):
            y = head + r * (tile + pad)
            d.text((5, y + tile // 2 - 6), labels[r], fill=0)
            for c, img in enumerate((inp[h], fut[h])):
                t = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((tile, tile), NEAREST)
                im.paste(t, (lab_w + pad + c * (tile + pad), y))
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=800, loop=0)

def write_caption(name, text):
    CAP.mkdir(parents=True, exist_ok=True)
    (CAP / f"{name}.txt").write_text(text + "\n")
    print(text, "\n")


# ---------- Figure 2: predictions of model A ----------
@torch.no_grad()
def figure2(model, test_u8, videos=None):
    one_x = list(range(1, 11))                                 # frames 1-10 shown at 1x, t = 1
    if videos is None:
        clean = [i for i in range(len(test_u8)) if not has_overlap(test_u8[i], one_x)][:2]
        over = [i for i in range(len(test_u8)) if has_overlap(test_u8[i], range(6, 11))][:1]
        if not over:
            print("warning: no overlapping video found; pass --fig2-videos")
        videos = clean + over
    rows, labels, titles = [], [], {}
    for b, i in enumerate(videos):
        X, Y, _ = make_window(test_u8[i:i + 1], 1, 1)
        X, Y = to_float(X), to_float(Y)
        P = model(X)
        for r, (name, imgs) in enumerate((("input", frames_np(X[0])), ("prediction", frames_np(P[0])),
                                          ("ground truth", frames_np(Y[0])))):
            rows.append(imgs); labels.append(f"test {i}\n{name}")
            for c in range(5):
                frame = c + 1 if name == "input" else c + 6
                titles[(3 * b + r, c)] = f"frame {frame}"
    image_grid(rows, labels, titles, "Figure 2: model A (seed 0), 1×, t = 1", "fig2")
    write_caption("fig2", f"Figure 2: input (frames 1–5), prediction and ground truth (frames 6–10) "
                          f"of seed-0 model A for test videos {videos} (dataset indices "
                          f"{[TEST_OFFSET + i for i in videos]}). Video {videos[-1]} has overlapping digits.")


# ---------- Figure 3 / 6(a): consistency panels + GIFs ----------
@torch.no_grad()
def figure3(model, test_u8, videos, stem, model_name):
    """videos: list of (T, test index). Four rows of ten frames per panel (handout layout)."""
    cap = []
    for n, (T, i) in enumerate(videos):
        Tf = TRANSFORMS[T]
        X, Y, _ = make_window(test_u8[i:i + 1], 1, 1)
        X, Y = to_float(X), to_float(Y)
        fX, TX = model(X), Tf(X)
        fTX, TfX, TY = model(TX), Tf(fX), Tf(Y)
        C = ((fTX - TfX) ** 2).mean().item()
        Et = ((fTX - TY) ** 2).mean().item()
        pairs = [(X, fX), (TX, TY), (TX, fTX), (TX, TfX)]
        labels = ["(1) X | f(X)", f"(2) {T}(X) | {T}(Y)", f"(3) {T}(X) | f({T}(X))", f"(4) {T}(X) | {T}(f(X))"]
        rows = [frames_np(a[0]) + frames_np(b[0]) for a, b in pairs]
        titles = {(0, c): f"input {c + 1}" for c in range(5)}
        titles.update({(0, 5 + h): f"h = {h + 1}" for h in range(5)})
        panel = chr(ord("a") + n)
        image_grid(rows, labels, titles,
                   f"{model_name}, T = {T}, test video {i} (dataset {TEST_OFFSET + i}): "
                   f"C_{T} = {fmt(C)}, E_{T} = {fmt(Et)}", f"{stem}{panel}")
        gif_path = GIFS / f"{stem}{panel}_{T}_test{i}.gif"
        write_gif([(frames_np(a[0]), frames_np(b[0])) for a, b in pairs], labels, gif_path)
        cap.append(f"({panel}) T = {T}, test video {i}, C_{T} = {fmt(C)}, E_{T} = {fmt(Et)}; GIF: {gif_path}")
    return cap

def pick_fig3_videos(npz_path):
    d = np.load(npz_path)
    return [(T, int(i)) for T in ("M", "R") for i in np.argsort(-d[f"C_{T}"])[:2]]


# ---------- Figure 4: per-video consistency histograms ----------
def figure4(npz_path, videos):
    d = np.load(npz_path)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, T in zip(ax, ("M", "R")):
        C = d[f"C_{T}"]
        pos = C[C > 0]
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 40) if len(pos) else 40
        a.hist(C, bins=bins, edgecolor="white")
        a.set_xscale("log")
        for T2, i in videos:
            if T2 == T:
                a.axvline(C[i], color="r", lw=1.2)
                a.text(C[i], a.get_ylim()[1] * 0.9, f" {i}", color="r", fontsize=8, rotation=90, va="top")
        a.set_xlabel(f"per-video C_{T} (log scale)"); a.set_ylabel("number of test videos")
        a.set_title(f"({'a' if T == 'M' else 'b'}) C_{T}, model A (seed 0)")
    fig.tight_layout()
    save(fig, "fig4")
    write_caption("fig4", "Figure 4: histograms of per-video consistency gaps C_M (a) and C_R (b) of "
                          "seed-0 model A over the 1,000 test videos. Red lines: the videos in Figure 3 "
                          f"({videos}).")


# ---------- Figure 5 / 6(b): speeds ----------
def fig5_frames(k):
    inp, tgt = window_frames(1, k)
    return inp + tgt[:2]

def pick_fig5_video(test_u8):
    """First test video with no overlap in any displayed frame (inputs + first two targets) at 1x, 2x, 3x."""
    shown = sorted(set(f for k in (1, 2, 3) for f in fig5_frames(k)))
    for i in range(len(test_u8)):
        if not has_overlap(test_u8[i], shown):
            return i
    raise RuntimeError("no video without overlap found")

@torch.no_grad()
def figure5(model, test_u8, i, stem, model_name):
    rows, labels, titles = [], [], {}
    for r, k in enumerate((1, 2, 3)):
        X, Y, _ = make_window(test_u8[i:i + 1], 1, k)
        X, Y = to_float(X), to_float(Y)
        P = model(X)
        inp, tgt = window_frames(1, k)
        rows.append(frames_np(X[0]) + frames_np(P[0])[:2] + frames_np(Y[0])[:2])
        labels.append(f"{k}×")
        for c, f in enumerate(inp):
            titles[(r, c)] = f"input f{f}"
        for h in range(2):
            titles[(r, 5 + h)] = f"pred f{tgt[h]}"
            titles[(r, 7 + h)] = f"true f{tgt[h]}"
    image_grid(rows, labels, titles,
               f"{model_name}, test video {i} (dataset {TEST_OFFSET + i}), t = 1", stem)
    return (f"{model_name} on test video {i} (dataset index {TEST_OFFSET + i}) at 1×, 2× and 3× (t = 1): "
            f"five input frames, then the first two predicted and true future frames. "
            f"No digit overlap in any displayed frame (blob-count heuristic).")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--run", default="A", help="model for Figures 2-5 (seed 0)")
    ap.add_argument("--final-run", default="D", help="run wrapped as the final symmetrised model (Fig 6)")
    ap.add_argument("--fig2-videos", nargs="*", type=int, default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    runs_dir = Path(args.runs_dir)
    RES.mkdir(exist_ok=True)
    if args.synthetic:
        test_u8 = (synthetic_videos(n=64, seed=7) * 255).round().to(torch.uint8)
    else:
        test_u8 = get_split(load_moving_mnist(os.environ.get("MMNIST_ROOT", "data")), "test")
    test_u8 = test_u8.to(device)
    npz_A = RES / "eval" / f"{args.run}_seed0.npz"

    if any(f in args.figs for f in (2, 3, 5)):
        loaded = load_run(runs_dir, args.run, 0, device)
        assert loaded, f"no checkpoint {runs_dir}/{args.run}_seed0/final.pt"
        model_A = loaded[0].eval()

    if 2 in args.figs:
        figure2(model_A, test_u8, args.fig2_videos)

    if 3 in args.figs or 4 in args.figs:
        assert npz_A.exists(), f"run evaluate.py first ({npz_A} missing)"
        videos = pick_fig3_videos(npz_A)
        (RES / "fig3_videos.json").write_text(json.dumps(videos))
        if 3 in args.figs:
            cap = figure3(model_A, test_u8, videos, "fig3", "model A (seed 0)")
            write_caption("fig3", "Figure 3: the two test videos with the largest per-video C_M and the two "
                                  "with the largest C_R for seed-0 model A. Rows: (1) X | f(X); "
                                  "(2) T(X) | T(Y); (3) T(X) | f(T(X)); (4) T(X) | T(f(X)).\n" + "\n".join(cap))
        if 4 in args.figs:
            figure4(npz_A, videos)

    if 5 in args.figs:
        i = pick_fig5_video(test_u8)
        (RES / "fig5_video.json").write_text(json.dumps(i))
        write_caption("fig5", "Figure 5: " + figure5(model_A, test_u8, i, "fig5", "model A (seed 0)"))

    if 6 in args.figs:
        loaded = load_run(runs_dir, args.final_run, 0, device)
        assert loaded, f"no checkpoint {runs_dir}/{args.final_run}_seed0/final.pt"
        final = Symmetrised(loaded[0]).eval()
        name = f"final model (symmetrised {args.final_run}, seed 0)"
        videos = [tuple(v) for v in json.loads((RES / "fig3_videos.json").read_text())]
        cap = figure3(final, test_u8, videos, "fig6a_", name)
        i = json.loads((RES / "fig5_video.json").read_text())
        cap_b = figure5(final, test_u8, i, "fig6b", name)
        write_caption("fig6", "Figure 6: (a) Figure 3 repeated with the " + name + " on the same videos:\n"
                              + "\n".join(cap) + "\n(b) Figure 5 repeated on the same video: " + cap_b)

if __name__ == "__main__":
    main()
