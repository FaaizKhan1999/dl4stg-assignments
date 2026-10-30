import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import load_moving_mnist, get_split, to_float, make_window
from transforms import TRANSFORMS
from metrics import per_video_horizon_mse, consistency_and_error, predict, fmt, fmt_seeds
from models import ConvLSTMForecaster, Blank, CopyLast, Symmetrised, count_params, synthetic_videos

SEEDS = (0, 1)


# ---------- loading ----------
def load_run(runs_dir: Path, run: str, seed: int, device):
    """Rebuild a trained model from runs/<run>_seed<seed>/final.pt. Returns (model, log) or None."""
    d = runs_dir / f"{run}_seed{seed}"
    if not (d / "final.pt").exists():
        return None
    ck = torch.load(d / "final.pt", map_location=device)
    model = ConvLSTMForecaster(**ck["arch"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    log = json.loads((d / "log.json").read_text()) if (d / "log.json").exists() else {}
    return model, log

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------- measurements ----------
@torch.no_grad()
def horizons(f, test_u8, k, bs):
    """Per-video MSE_h, shape (N, 5), at speed k with t = 1 (NaN where no target frame)."""
    X, Y, m = make_window(test_u8, 1, k)
    return per_video_horizon_mse(predict(f, to_float(X), bs), to_float(Y), m).cpu().numpy()

@torch.no_grad()
def inference_time(f, test_u8, device, bs=32):
    """Wall-clock time to predict all original (1x, t = 1) test videos at batch size 32,
    under eval() and no_grad(), timed from when each batch is already on the device.
    One untimed warm-up batch first (CUDA kernel selection)."""
    X = to_float(make_window(test_u8, 1, 1)[0])              # already on the device
    if isinstance(f, nn.Module):
        f.eval()
    f(X[:bs]); sync(device)
    total = 0.0
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        sync(device); t0 = time.perf_counter()
        f(xb)
        sync(device); total += time.perf_counter() - t0
    return total

def evaluate(f, test_u8, device, bs=32):
    """All metrics needed by Tables 2-6 for one forecaster. Returns (summary dict, per-video arrays)."""
    per = {}
    for k in (1, 2, 3):
        per[f"h_{k}x"] = horizons(f, test_u8, k, bs)           # (N, 5)
    X, Y, m = make_window(test_u8, 1, 1)
    X, Y = to_float(X), to_float(Y)
    for name, T in TRANSFORMS.items():                          # "M", "R"
        C, Et = consistency_and_error(f, X, Y, m, T, bs)
        per[f"C_{name}"], per[f"E_{name}"] = C.cpu().numpy(), Et.cpu().numpy()

    s = {}
    h1 = per["h_1x"]
    for h in range(5):
        s[f"MSE{h + 1}"] = float(h1[:, h].mean())
    s["E5_1x"] = float(h1.mean(1).mean())
    s["E5_2x"] = float(per["h_2x"].mean(1).mean())             # all 5 targets exist at 2x, t = 1
    for k in (1, 2, 3):
        s[f"E2_{k}x"] = float(per[f"h_{k}x"][:, :2].mean(1).mean())
    for name in TRANSFORMS:
        s[f"C_{name}"] = float(per[f"C_{name}"].mean())
        s[f"E_{name}"] = float(per[f"E_{name}"].mean())
    s["infer_time"] = inference_time(f, test_u8, device, bs)
    s["params"] = count_params(f)
    return s, per

def run_and_save(label, seed, f, log, test_u8, device, out_dir: Path):
    s, per = evaluate(f, test_u8, device)
    s["train_time"] = log.get("mean_epoch_time")
    s["hardware"] = log.get("hardware")
    tag = f"{label}_seed{seed}" if seed is not None else label
    (out_dir / f"{tag}.json").write_text(json.dumps(s, indent=2))
    np.savez(out_dir / f"{tag}.npz", **per)
    print(f"  {tag:14s} E5_1x {fmt(s['E5_1x'])}  C_M {fmt(s['C_M'])}  C_R {fmt(s['C_R'])}  "
          f"E2_3x {fmt(s['E2_3x'])}  infer {s['infer_time']:.2f}s")
    return s


# ---------- tables ----------
def cell(results, label, key):
    """Learned models: 'mean (s0, s1)'. References: single value. Missing: '—'."""
    vals = [r[key] for r in results.get(label, []) if r.get(key) is not None]
    if not vals:
        return "—"
    if key == "params":
        return f"{int(vals[0]):,}"
    if len(vals) == 2:
        return fmt_seeds(*vals)
    return fmt(vals[0]) + (" (seed 0 only)" if label not in ("Blank", "Copy-last") else "")

def table(results, title, rows, cols):
    present = [r for r in rows if r in results]
    if not present:
        return f"### {title}\n(skipped: no results yet)\n"
    head = "| Model | " + " | ".join(h for h, _ in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    body = [f"| {r} | " + " | ".join(cell(results, r, k) for _, k in cols) + " |" for r in present]
    missing = [r for r in rows if r not in results]
    note = f"\n(missing: {', '.join(missing)})" if missing else ""
    return "\n".join([f"### {title}", head, sep, *body]) + note + "\n"

def write_tables(results, path: Path, hardware: str):
    refs = ["Blank", "Copy-last"]
    t5_cols = [("E(5) 1×", "E5_1x"), ("C_M", "C_M"), ("E_M", "E_M"), ("C_R", "C_R"), ("E_R", "E_R"),
               ("E(2) 1×", "E2_1x"), ("E(2) 2×", "E2_2x"), ("E(2) 3×", "E2_3x"),
               ("Params", "params"), ("Train s/epoch", "train_time"), ("Inference s", "infer_time")]
    parts = [f"Hardware: {hardware}\n",
             table(results, "Table 2", ["A"] + refs,
                   [(f"MSE{h}", f"MSE{h}") for h in range(1, 6)] + [("E(5)", "E5_1x"), ("Params", "params")]),
             table(results, "Table 3", ["A"] + refs,
                   [("E(5)", "E5_1x"), ("C_M", "C_M"), ("E_M", "E_M"), ("C_R", "C_R"), ("E_R", "E_R")]),
             table(results, "Table 4", ["A", "Copy-last"],
                   [("E(2) 1×", "E2_1x"), ("E(2) 2×", "E2_2x"), ("E(2) 3×", "E2_3x"),
                    ("E(5) 1×", "E5_1x"), ("E(5) 2×", "E5_2x")]),
             table(results, "Table 5", ["A", "B", "C", "D"], t5_cols),
             table(results, "Table 6", ["sym B", "sym D", "sym D+B"], t5_cols)]
    text = "\n".join(parts)
    path.write_text(text)
    print("\n" + text)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--d-run", default="D", help="run name of model D, e.g. D_dil2")
    ap.add_argument("--db-run", default="DB", help="run name of optional D+B")
    ap.add_argument("--only", nargs="*", default=None,
                    help="labels to (re)evaluate, e.g. --only A Blank Copy-last; default: all found")
    ap.add_argument("--synthetic", action="store_true", help="synthetic test videos (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    runs_dir, out_dir = Path(args.runs_dir), Path("results/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        test_u8 = (synthetic_videos(n=64, seed=7) * 255).round().to(torch.uint8)
    else:
        test_u8 = get_split(load_moving_mnist(os.environ.get("MMNIST_ROOT", "data")), "test")
    test_u8 = test_u8.to(device)
    print(f"evaluating on {len(test_u8)} {'synthetic' if args.synthetic else 'test'} videos, {device}")

    want = lambda lab: args.only is None or lab in args.only
    results = {}

    # references (deterministic -> one value)
    for lab, f in (("Blank", Blank()), ("Copy-last", CopyLast())):
        if want(lab):
            results[lab] = [run_and_save(lab, None, f.to(device), {}, test_u8, device, out_dir)]

    # learned models and their symmetrised versions
    learned = {"A": "A", "B": "B", "C": "C", "D": args.d_run, "D+B": args.db_run}
    sym_of = {"B": "sym B", "D": "sym D", "D+B": "sym D+B"}
    for lab, run in learned.items():
        for seed in SEEDS:
            loaded = load_run(runs_dir, run, seed, device)
            if loaded is None:
                continue
            model, log = loaded
            if want(lab):
                results.setdefault(lab, []).append(
                    run_and_save(lab.replace("+", ""), seed, model, log, test_u8, device, out_dir))
            if lab in sym_of and want(sym_of[lab]):
                results.setdefault(sym_of[lab], []).append(
                    run_and_save(sym_of[lab].replace(" ", "").replace("+", ""), seed,
                                 Symmetrised(model).eval(), log, test_u8, device, out_dir))

        # merge with previously saved results for labels NOT evaluated in this run
    fresh = set(results)
    for p in sorted(out_dir.glob("*.json")):
        name = p.stem.rsplit("_seed", 1)[0]
        label = {"Blank": "Blank", "Copy-last": "Copy-last", "A": "A", "B": "B", "C": "C",
                 "D": "D", "DB": "D+B", "symB": "sym B", "symD": "sym D", "symDB": "sym D+B"}.get(name)
        if label and label not in fresh:
            results.setdefault(label, []).append(json.loads(p.read_text()))

    hw = next((r["hardware"] for rs in results.values() for r in rs if r.get("hardware")), "see logs")
    hw_eval = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    write_tables(results, Path("results/tables.md"), f"training: {hw} | inference: {hw_eval}")

if __name__ == "__main__":
    main()
