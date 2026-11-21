"""Q2: ring-road checks. Coefficients are FITTED by least squares, to verify the hand derivations."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, STEPS = 40, 20
os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(42)

back = lambda x: np.roll(x, -1)           # true rule: x_i(t+1) = x_{i+1}(t)
fwd = lambda x: np.roll(x, 1)             # forward rule: x_i(t+1) = x_{i-1}(t)
left = lambda x: np.roll(x, 1)            # x_{i-1}
right = lambda x: np.roll(x, -1)          # x_{i+1}

def fit(features, target):
    coef, res, *_ = np.linalg.lstsq(np.column_stack(features), target, rcond=None)
    sse = float(((np.column_stack(features) @ coef - target) ** 2).sum())
    return coef, sse

# ---------- (a)(i) single isolated slowdown ----------
x0 = np.zeros(N); x0[20] = 1.0
(a, b), sse = fit([x0, left(x0) + right(x0)], back(x0))
print(f"(a)(i)  a={a:.4f}, b={b:.4f}, SSE={sse:.4f}")
x_hat = a * x0 + b * (left(x0) + right(x0))
print("        predictions at 19, 20, 21:", np.round(x_hat[19:22], 4), "| truth:", back(x0)[19:22])

# ---------- (a)(ii) roll out the fitted layer on a smooth bump ----------
i = np.arange(N)
bump = np.exp(-(i - 20) ** 2 / 8)
pred, true = [bump], [bump]
for _ in range(STEPS):
    p = pred[-1]
    pred.append(a * p + b * (left(p) + right(p)))
    true.append(back(true[-1]))
pred, true = np.array(pred), np.array(true)

vmin, vmax = min(pred.min(), true.min()), max(pred.max(), true.max())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for a_, M, title in ((ax[0], pred, "fitted symmetric layer (a)"), (ax[1], true, "true motion")):
    im = a_.imshow(M, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis",
                   extent=[-0.5, N - 0.5, -0.5, STEPS + 0.5])
    a_.set_title(title); a_.set_xlabel("sensor index")
ax[0].set_ylabel("time step t")
fig.colorbar(im, ax=ax, label="slowdown")
fig.savefig("figures/q2a_heatmaps.png", dpi=150)
for t in (0, 10, 20):
    print(f"        t={t:2d}: layer peak {pred[t].max():.3f} at {pred[t].argmax()} | "
          f"true peak {true[t].max():.3f} at {true[t].argmax()}")

# ---------- (b) direction-aware weights, fitted on random snapshots ----------
X = rng.normal(size=(200, N))
feats = [X.ravel(), np.apply_along_axis(left, 1, X).ravel(), np.apply_along_axis(right, 1, X).ravel()]
(a_b, wm, wp), sse = fit(feats, np.apply_along_axis(back, 1, X).ravel())
print(f"\n(b)     a={a_b:.4f}, w-={wm:.4f}, w+={wp:.4f}, SSE={sse:.2e}")

conv = torch.nn.Conv1d(1, 1, kernel_size=3, padding=1, padding_mode="circular", bias=False)
with torch.no_grad():
    conv.weight[:] = torch.tensor([[[wm, a_b, wp]]], dtype=torch.float32)  # order [x_{i-1}, x_i, x_{i+1}]
    out = conv(torch.tensor(X[:5], dtype=torch.float32).unsqueeze(1)).squeeze(1).numpy()
print("        Conv1d (circular padding) max error vs true step:",
      float(np.abs(out - np.apply_along_axis(back, 1, X[:5])).max()))

# ---------- (c) equal neighbour weights + history ----------
Xprev = rng.normal(size=(200, N))
Xcur = np.apply_along_axis(back, 1, Xprev)              # x(t) follows the backward rule
feats = [Xcur.ravel(),
         (np.apply_along_axis(left, 1, Xcur) + np.apply_along_axis(right, 1, Xcur)).ravel(),
         Xprev.ravel()]
(a_c, b_c, c_c), sse = fit(feats, np.apply_along_axis(back, 1, Xcur).ravel())
print(f"\n(c)(i)  a={a_c:.4f}, b={b_c:.4f}, c={c_c:.4f}, SSE={sse:.2e}")

Xcur_f = np.apply_along_axis(fwd, 1, Xprev)             # forward motion instead
pred_f = (a_c * Xcur_f + b_c * (np.apply_along_axis(left, 1, Xcur_f) + np.apply_along_axis(right, 1, Xcur_f))
          + c_c * Xprev)
print("(c)(ii) same coefficients on forward motion, max error:",
      float(np.abs(pred_f - np.apply_along_axis(fwd, 1, Xcur_f)).max()))
