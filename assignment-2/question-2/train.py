import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch

from data import load_moving_mnist, get_split, to_float, make_window, sample_windows
from transforms import apply_g_per_sample
from metrics import masked_loss, per_video_horizon_mse, E, predict
from models import ConvLSTMForecaster, model_A, count_params, synthetic_videos

PARAMS_A = 480_784
PARAMS_D_MAX = 721_176                      # 1.5 x model A

# Per-model training rules (handout table in 'What Can Data Fix?')
MODELS = {
    "A":  dict(mode="normal", sym=False, arch="A"),   # 1x only, no symmetry augmentation
    "B":  dict(mode="normal", sym=True,  arch="A"),   # 1x only, g uniform from G
    "C":  dict(mode="speed",  sym=False, arch="A"),   # k uniform from {1,2,3}
    "D":  dict(mode="speed",  sym=False, arch="D"),   # larger-reach design, k uniform
    "DB": dict(mode="speed",  sym=True,  arch="D"),   # optional D+B
}


# ---------- helpers ----------
def build_model(arch_kind: str, arch_args: dict):
    if arch_kind == "A":
        model, arch = model_A(), dict(patch=4, hidden=64, n_layers=2, kernel=3, dilation=1)
        assert count_params(model) == PARAMS_A
    else:
        arch = dict(arch_args)
        model = ConvLSTMForecaster(**arch)
        n = count_params(model)
        assert n <= PARAMS_D_MAX, f"model D has {n:,} params > {PARAMS_D_MAX:,}"
    return model, arch

def hardware() -> str:
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}"
    return f"CPU: {platform.processor() or platform.machine()} | torch {torch.__version__}"

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def validate(model, videos_val, batch_size=32):
    """Monitoring only (allowed on the validation split): E(5) at 1x and E(2) at 1x/2x/3x, t = 1."""
    out = {}
    for k in (1, 2, 3):
        X, Y, m = make_window(videos_val, 1, k)
        ph = per_video_horizon_mse(predict(model, to_float(X), batch_size), to_float(Y), m)
        out[f"E2_{k}x"] = E(ph, 2)
        if k == 1:
            out["E5_1x"] = E(ph, 5)
    model.train()
    return out


# ---------- main training loop ----------
def train(args):
    device = torch.device(args.device)
    rules = MODELS[args.model]
    name = args.model + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out) / f"{name}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # data: uint8 on the device, converted to float per batch
    if args.synthetic:
        vids = (synthetic_videos(n=96, seed=123) * 255).round().to(torch.uint8)
        train_v, val_v = vids[:64], vids[64:]
    else:
        allv = load_moving_mnist(os.environ.get("MMNIST_ROOT", "data"))   # download not timed
        train_v, val_v = get_split(allv, "train"), get_split(allv, "val")
        if args.quick:
            train_v, val_v = train_v[:64], val_v[:32]
    train_v, val_v = train_v.to(device), val_v.to(device)
    N = len(train_v)

    # seed: initialisation (global RNG) and all sampling (dedicated generator)
    torch.manual_seed(args.seed)
    model, arch = build_model(rules["arch"], dict(patch=args.patch, hidden=args.hidden,
                              n_layers=args.layers, kernel=args.kernel,
                              dilation=args.dilation[0] if len(args.dilation) == 1 else args.dilation))
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)

    log = dict(model=args.model, tag=args.tag, seed=args.seed, arch=arch, rules=rules,
               params=count_params(model), hardware=hardware(), batch_size=args.batch,
               lr=args.lr, epochs=args.epochs, n_train=N, epoch_times=[], train_loss=[], val={})
    start_epoch = 1

    # resume (useful if a Kaggle session dies)
    ckpt_path = out_dir / "last.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"]); torch.set_rng_state(ck["rng"]); log = ck["log"]
        start_epoch = ck["epoch"] + 1
        print(f"resumed from epoch {ck['epoch']}")

    print(f"{name} seed {args.seed} | {log['params']:,} params | {log['hardware']} | N={N}")
    for epoch in range(start_epoch, args.epochs + 1):
        sync(device); t0 = time.perf_counter()
        perm = torch.randperm(N, generator=gen)               # shuffled order (seeded)
        total, nb = 0.0, 0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch].to(device)
            X, Y, mask, _, _ = sample_windows(train_v[idx], rules["mode"], gen)
            if rules["sym"]:                                  # same g for input and target
                g_idx = torch.randint(0, 8, (len(idx),), generator=gen)
                X, Y = apply_g_per_sample(X, g_idx), apply_g_per_sample(Y, g_idx)
            loss = masked_loss(model(to_float(X)), to_float(Y), mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
        sync(device); dt = time.perf_counter() - t0           # training pass only

        log["epoch_times"].append(dt); log["train_loss"].append(total / nb)
        msg = f"epoch {epoch:2d}/{args.epochs}  loss {total / nb:.4g}  {dt:.1f}s"
        if epoch % args.val_every == 0 or epoch == args.epochs:
            v = validate(model, val_v); log["val"][epoch] = v      # not included in epoch time
            msg += "  | val " + "  ".join(f"{k} {x:.4g}" for k, x in v.items())
        print(msg, flush=True)

        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), gen=gen.get_state(),
                        rng=torch.get_rng_state(), epoch=epoch, log=log), ckpt_path)

    # final-epoch weights (no checkpoint selection) + log
    log["mean_epoch_time"] = sum(log["epoch_times"]) / len(log["epoch_times"])
    torch.save(dict(state_dict=model.state_dict(), arch=arch, model=args.model,
                    tag=args.tag, seed=args.seed), out_dir / "final.pt")
    (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    print(f"done: mean epoch time {log['mean_epoch_time']:.1f}s -> {out_dir}/final.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="", help="name for a model-D design, e.g. k5")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # model-D architecture (ignored for A, B, C)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dilation", type=int, nargs="+", default=[1])
    # smoke tests
    ap.add_argument("--quick", action="store_true", help="64 train videos, 2 epochs, batch 8")
    ap.add_argument("--synthetic", action="store_true", help="synthetic squares, no dataset")
    args = ap.parse_args()
    if args.quick or args.synthetic:
        args.epochs, args.batch, args.val_every = 2, 8, 1
        args.out = "runs_smoke"
    train(args)
