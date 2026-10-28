import math
import torch

# ---------- core error ----------
def per_video_horizon_mse(pred, target, mask):
    """MSE_h for each video: mean over the 64x64 pixels of horizon h.
    pred, target: (N, 5, C, H, W); mask: (N, 5) bool. Returns (N, 5), NaN where masked."""
    se = ((pred.float() - target.float()) ** 2).flatten(2).mean(-1)
    return se.masked_fill(~mask.to(se.device), float("nan"))

def E(per_h: torch.Tensor, H: int) -> float:
    """E(H): mean over horizons 1..H for each video, then mean over videos."""
    sub = per_h[:, :H]
    assert not torch.isnan(sub).any(), f"horizons 1..{H} must all be scored"
    return sub.mean(1).mean(0).item()

def masked_loss(pred, target, mask):
    """Training loss: per video, squared error averaged over pixels and the
    AVAILABLE target frames; then averaged over videos in the batch."""
    se = ((pred - target) ** 2).flatten(2).mean(-1)            # (B, 5)
    m = mask.float()
    per_video = (se * m).sum(1) / m.sum(1)
    return per_video.mean()

# ---------- prediction helper ----------
@torch.no_grad()
def predict(f, X, batch_size: int = 32):
    """Run forecaster f on X (N, 5, 1, 64, 64) float in batches. f may be an
    nn.Module or any callable. Returns (N, 5, 1, 64, 64)."""
    if isinstance(f, torch.nn.Module):
        f.eval()
    return torch.cat([f(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])

# ---------- consistency gap and transformed error (Eq. 6) ----------
@torch.no_grad()
def consistency_and_error(f, X, Y, mask, T, batch_size: int = 32):
    """Per-video C_T = MSE(f(T X), T f(X)) and E_T = MSE(f(T X), T Y),
    averaged over the five horizons. Returns two (N,) tensors."""
    p = predict(f, X, batch_size)
    pT = predict(f, T(X), batch_size)
    C = per_video_horizon_mse(pT, T(p), mask).mean(1)
    Et = per_video_horizon_mse(pT, T(Y), mask).mean(1)
    return C, Et

# ---------- reporting format ----------
def fmt(x: float) -> str:
    """3 significant figures; scientific notation below 1e-3."""
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:#.3g}".rstrip(".")

def fmt_seeds(v0: float, v1: float) -> str:
    """'mean (seed 0, seed 1)' as required for every learned model."""
    return f"{fmt((v0 + v1) / 2)} ({fmt(v0)}, {fmt(v1)})"

if __name__ == "__main__":
    # tiny self-test with synthetic tensors
    p = torch.zeros(2, 5, 1, 64, 64); y = torch.ones(2, 5, 1, 64, 64)
    m = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    ph = per_video_horizon_mse(p, y, m)
    print("per-horizon:", ph.tolist())
    print("E(2) =", E(ph, 2), "| loss =", masked_loss(p, y, m).item())
    print([fmt(v) for v in (0.0123456, 0.1, 12.0, 4.2e-5, 3e-15)])
