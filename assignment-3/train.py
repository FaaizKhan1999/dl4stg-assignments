import argparse, copy, json, os, platform, time
import numpy as np
import pandas as pd
import torch
from utils import set_seed, SEED
from models import Adjacency, STGNN

DATA, RES = "data/processed", "results"
T_IN, T_OUT = 8, 4
SPLITS = {"train": ("2020-01-01", "2021-01-01"),      # [start, end)
          "val":   ("2021-01-01", "2021-04-01"),
          "test":  ("2021-04-01", "2022-01-01")}


def hardware():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":")[1].strip() for l in f if l.startswith("model name"))
    except Exception:
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"cpu": cpu, "torch_threads": torch.get_num_threads(), "gpu": gpu,
            "torch": torch.__version__}


def load_data(device):
    X = np.load(f"{DATA}/X.npy")[..., 0]
    M = np.load(f"{DATA}/mask.npy")
    ts = pd.DatetimeIndex(np.load(f"{DATA}/timestamps.npy"))
    tr = (ts >= SPLITS["train"][0]) & (ts < SPLITS["train"][1])
    mu, sd = X[tr][M[tr]].mean(), X[tr][M[tr]].std()          # training, observed only
    data = {}
    for k, (a, b) in SPLITS.items():
        sel = (ts >= a) & (ts < b)
        x, m = X[sel], M[sel]
        z = np.where(m, (x - mu) / sd, 0.0)                    # missing inputs -> 0 (= train mean)
        data[k] = {"z": torch.tensor(z, dtype=torch.float32, device=device),
                   "y": torch.tensor(x, dtype=torch.float32, device=device),
                   "m": torch.tensor(m, dtype=torch.bool, device=device),
                   "n": len(x) - T_IN - T_OUT + 1}             # windows stay inside the split
        print(f"{k}: {sel.sum()} steps, {data[k]['n']} windows")
    return data, float(mu), float(sd)


def batch(d, starts):
    idx = starts[:, None] + torch.arange(T_IN + T_OUT, device=starts.device)
    x = d["z"][idx[:, :T_IN]].unsqueeze(-1)                   # (B, 8, N, 1) normalised
    y = d["y"][idx[:, T_IN:]].unsqueeze(-1)                   # (B, 4, N, 1) raw units
    m = d["m"][idx[:, T_IN:]].unsqueeze(-1).float()
    return x, y, m


@torch.no_grad()
def evaluate(model, d, mu, sd, bs=256):
    model.eval()
    dev = d["z"].device
    abs_s = torch.zeros(T_OUT, device=dev); sq_s = torch.zeros_like(abs_s); cnt = torch.zeros_like(abs_s)
    for s in range(0, d["n"], bs):
        x, y, m = batch(d, torch.arange(s, min(s + bs, d["n"]), device=dev))
        err = (model(x) * sd + mu - y) * m                     # back to vehicles/15 min
        abs_s += err.abs().sum((0, 2, 3)); sq_s += (err ** 2).sum((0, 2, 3)); cnt += m.sum((0, 2, 3))
    model.train()
    return {"MAE": (abs_s.sum() / cnt.sum()).item(),
            "RMSE": (sq_s.sum() / cnt.sum()).sqrt().item(),
            "MAE_per_horizon": (abs_s / cnt).tolist(),
            "RMSE_per_horizon": (sq_s / cnt).sqrt().tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", choices=["none", "fixed", "adaptive", "hybrid"], required=True)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--emb_dim", type=int, default=10)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None, help="hybrid: fix alpha; omit to learn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_batches", type=int, default=None, help="debug: cap batches/epoch")
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, mu, sd = load_data(device)

    A = np.load(f"{DATA}/A_sd.npy")                            # diag already 1 -> no +I
    A_hat = torch.tensor(A / A.sum(1, keepdims=True), dtype=torch.float32)   # D^-1 A
    adj = Adjacency(args.adj, A_hat, A.shape[0], args.emb_dim, args.topk, args.alpha)
    model = STGNN(adj, args.d, T_IN, T_OUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.adj}: {n_params} params on {device}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr = data["train"]
    best, best_val, best_ep, bad, log = None, float("inf"), 0, 0, []
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        perm = torch.randperm(tr["n"], device=device)
        for b, s in enumerate(range(0, tr["n"], args.bs)):
            if args.max_batches and b >= args.max_batches:
                break
            x, y, m = batch(tr, perm[s:s + args.bs])
            loss = ((model(x) - (y - mu) / sd).abs() * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        val = evaluate(model, data["val"], mu, sd)
        ep_time = time.time() - t0
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_MAE": val["MAE"],
                    "val_RMSE": val["RMSE"], "epoch_time_s": ep_time})
        print(f"ep {ep:02d} | train MAE(z) {np.mean(losses):.4f} | "
              f"val MAE {val['MAE']:.2f} RMSE {val['RMSE']:.2f} | {ep_time:.1f}s")
        if val["MAE"] < best_val:
            best_val, best, best_ep, bad = val["MAE"], copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
    train_time = time.time() - t_start

    model.load_state_dict(best)
    res = {"adj": args.adj, "args": vars(args), "seed": SEED, "n_params": n_params,
           "epochs_run": ep, "best_epoch": best_ep, "train_time_s": train_time,
           "mean_epoch_time_s": float(np.mean([l["epoch_time_s"] for l in log])),
           "val": evaluate(model, data["val"], mu, sd),
           "test": evaluate(model, data["test"], mu, sd),
           "norm": {"mean": mu, "std": sd}, "hardware": hardware(), "log": log}
    if args.adj == "hybrid":
        res["alpha"] = float(model.adj.alpha().detach())

    tag = args.adj + (f"_top{args.topk}" if args.topk else "") + (f"_a{args.alpha}" if args.alpha is not None else "")
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/{tag}.json", "w") as f:
        json.dump(res, f, indent=2)
    torch.save(best, f"{RES}/{tag}.pt")
    if args.adj in ("adaptive", "hybrid"):
        with torch.no_grad():
            np.save(f"{RES}/{tag}_A_adaptive.npy", model.adj.adaptive().cpu().numpy())
            np.save(f"{RES}/{tag}_A_used.npy", model.adj().cpu().numpy())
    print(f"TEST  MAE {res['test']['MAE']:.2f}  RMSE {res['test']['RMSE']:.2f}  "
          f"| train time {train_time/60:.1f} min" + (f" | alpha {res['alpha']:.3f}" if "alpha" in res else ""))


if __name__ == "__main__":
    main()
