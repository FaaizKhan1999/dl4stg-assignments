import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transforms import G, apply_g, apply_g_inv


class Symmetrised(nn.Module):
    """f_bar(X) = 1/8 * sum_g g^{-1}( f( g(X) ) )  (handout Eq. 7). No retraining needed."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, X):
        out = torch.zeros_like(X)
        for g in G:
            out = out + apply_g_inv(self.base(apply_g(X, g)), g)
        return out / len(G)


# ---------- ConvLSTM cell: Eqs. (3)-(4), no peepholes ----------
class ConvLSTMCell(nn.Module):
    """[i, f, o, g] = Conv([u, h_prev]);  c = s(f)*c_prev + s(i)*tanh(g);  h = s(o)*tanh(c).
    One convolution produces all four gates. Zero padding keeps the spatial size."""

    def __init__(self, in_ch: int, hid_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        assert kernel % 2 == 1, "odd kernels keep the grid size with symmetric padding"
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size=kernel, stride=1,
                              padding=dilation * (kernel - 1) // 2, dilation=dilation,
                              bias=True)                       # default PyTorch init

    def forward(self, u, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([u, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


# ---------- forecaster with the 9-step rollout ----------
class ConvLSTMForecaster(nn.Module):
    """PixelUnshuffle(patch) -> stacked ConvLSTM layers -> 1x1 conv -> PixelShuffle(patch) -> sigmoid.

    Rollout (handout, Section 2): 9 steps. Step tau takes u_tau and outputs x_hat_{tau+1}.
    tau = 1..5: u_tau is the tau-th input frame; tau = 6..9: u_tau is the network's own
    previous prediction (no teacher forcing, no detach: gradients flow through it).
    Outputs of steps 5..9 form Y_hat. States start at zero for every video.
    Defaults give model A exactly.
    """

    def __init__(self, patch: int = 4, hidden=64, n_layers: int = 2, kernel: int = 3,
                 dilation: int = 1, in_frames: int = 5, out_frames: int = 5):
        super().__init__()
        hidden = [hidden] * n_layers if isinstance(hidden, int) else list(hidden)
        assert len(hidden) == n_layers
        self.patch, self.hidden = patch, hidden
        self.in_frames, self.out_frames = in_frames, out_frames
        fold = patch * patch                                    # 1 x 64 x 64 -> fold x 16 x 16
        ins = [fold] + hidden[:-1]                              # layer l>1 takes h of layer l-1
        dil = [dilation] * n_layers if isinstance(dilation, int) else list(dilation)
        assert len(dil) == n_layers
        self.cells = nn.ModuleList(ConvLSTMCell(i, h, kernel, d) for i, h, d in zip(ins, hidden, dil))
        self.head = nn.Conv2d(hidden[-1], fold, kernel_size=1, bias=True)

    def step(self, frame, states):
        """One time step: frame (B, 1, 64, 64) -> next-frame prediction (B, 1, 64, 64)."""
        u = F.pixel_unshuffle(frame, self.patch)
        new_states = []
        for cell, st in zip(self.cells, states):
            h, c = cell(u, st)
            new_states.append((h, c))
            u = h
        out = torch.sigmoid(F.pixel_shuffle(self.head(u), self.patch))
        return out, new_states

    def forward(self, X):
        """X: (B, 5, 1, 64, 64) in [0, 1]. Returns Y_hat: (B, 5, 1, 64, 64)."""
        B, T, C, H, W = X.shape
        assert T == self.in_frames and C == 1, X.shape
        gh, gw = H // self.patch, W // self.patch
        states = [(X.new_zeros(B, h, gh, gw), X.new_zeros(B, h, gh, gw)) for h in self.hidden]

        preds, pred = [], None
        n_steps = self.in_frames + self.out_frames - 1          # 9
        for tau in range(1, n_steps + 1):
            u = X[:, tau - 1] if tau <= self.in_frames else pred   # own prediction fed back
            pred, states = self.step(u, states)
            if tau >= self.in_frames:                           # keep outputs of steps 5..9
                preds.append(pred)
        return torch.stack(preds, dim=1)


def model_A() -> ConvLSTMForecaster:
    return ConvLSTMForecaster(patch=4, hidden=64, n_layers=2, kernel=3, dilation=1)


# ---------- reference predictors (no training) ----------
class Blank(nn.Module):
    """All-zero frames."""
    def forward(self, X):
        return torch.zeros_like(X)                              # (B, 5, 1, 64, 64)

class CopyLast(nn.Module):
    """Repeat the last input frame five times."""
    def forward(self, X):
        return X[:, -1:].expand(-1, 5, -1, -1, -1).clone()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------- self-tests ----------
def selftest():
    torch.manual_seed(0)
    A = model_A()
    n = count_params(A)
    print(f"model A trainable parameters: {n:,}")
    assert n == 480_784, "architecture does not match the handout"

    X = torch.rand(2, 5, 1, 64, 64)
    Yh = A(X)
    assert Yh.shape == (2, 5, 1, 64, 64)
    assert (Yh > 0).all() and (Yh < 1).all(), "sigmoid output must lie in (0, 1)"

    # the rollout must feed back its own predictions (same tensor, still in the graph)
    seen_in, seen_out = [], []
    orig_step = A.step
    def logged_step(frame, states):
        out, st = orig_step(frame, states)
        seen_in.append(frame); seen_out.append(out)
        return out, st
    A.step = logged_step
    Yh = A(X)
    A.step = orig_step
    assert len(seen_in) == 9, "rollout must run exactly 9 steps"
    for tau in range(1, 5):                                     # steps 1..5 take real frames
        assert torch.equal(seen_in[tau - 1], X[:, tau - 1])
    for tau in range(6, 10):                                    # steps 6..9 take own prediction
        assert seen_in[tau - 1] is seen_out[tau - 2], f"step {tau} is not fed its own prediction"
        assert seen_in[tau - 1].requires_grad, "fed-back prediction was detached"
    for h in range(5):                                          # Y_hat = outputs of steps 5..9
        assert torch.equal(Yh[:, h], seen_out[4 + h])

    # gradient reaches the weights through the fed-back path
    A.zero_grad()
    A(X)[:, 4].mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in A.parameters())

    # zero initial states: two different videos in one batch must not interact
    X2 = X.clone(); X2[1] = torch.rand(5, 1, 64, 64)
    assert torch.allclose(A(X)[0], A(X2)[0], atol=1e-6), "batch elements leak into each other"

    # reference predictors
    assert torch.equal(Blank()(X), torch.zeros_like(X))
    cl = CopyLast()(X)
    assert all(torch.equal(cl[:, h], X[:, -1]) for h in range(5))
    assert count_params(Blank()) == count_params(CopyLast()) == 0
    print("SELFTEST PASS")


# ---------- optional: overfit a few videos to catch training bugs ----------
def synthetic_videos(n=16, seed=0):
    """Two 8x8 bright squares moving at constant velocity with elastic bounces."""
    g = torch.Generator().manual_seed(seed)
    V = torch.zeros(n, 20, 1, 64, 64)
    for v in range(n):
        for _ in range(2):
            pos = torch.randint(0, 56, (2,), generator=g).float()
            vel = (torch.randint(1, 4, (2,), generator=g) * (torch.randint(0, 2, (2,), generator=g) * 2 - 1)).float()
            for f in range(20):
                r, c = pos.long().tolist()
                V[v, f, 0, r:r + 8, c:c + 8] = 1.0
                pos += vel
                for d in range(2):
                    if pos[d] < 0 or pos[d] > 56:
                        vel[d] = -vel[d]; pos[d] = pos[d].clamp(0, 56)
    return V

def overfit(data: str, steps: int, device: str, n_videos: int, loss_name: str):
    from data import make_window, to_float
    from metrics import masked_loss
    if data == "real":
        from data import load_moving_mnist, get_split
        videos = to_float(get_split(load_moving_mnist(os.environ.get("MMNIST_ROOT", "data")), "train")[:n_videos])
    else:
        videos = synthetic_videos(n=n_videos)
    X, Y, mask = make_window(videos.to(device), 1, 1)

    def loss_fn(pred):
        if loss_name == "mse":
            return masked_loss(pred, Y, mask)
        # diagnostic only: BCE has no vanishing gradient through the sigmoid
        return F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), Y)

    torch.manual_seed(0)
    A = model_A().to(device)
    opt = torch.optim.Adam(A.parameters(), lr=1e-3)
    blank = masked_loss(Blank()(X), Y, mask).item()
    copy = masked_loss(CopyLast()(X), Y, mask).item()
    print(f"{n_videos} {data} videos | training loss: {loss_name} | "
          f"Blank MSE {blank:.4g} | Copy-last MSE {copy:.4g} | target mean {Y.mean().item():.4g}")
    t0 = time.time()
    for s in range(1, steps + 1):
        pred = A(X)
        loss = loss_fn(pred)
        opt.zero_grad(); loss.backward()
        if s % 100 == 0 or s == 1:
            gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in A.parameters())).item()
            mse = masked_loss(pred.detach(), Y, mask).item()
            on = pred.detach()[Y > 0.5]                       # predictions where the target is bright
            print(f"step {s:5d}  MSE {mse:.4g} ({mse / blank:.2f} x Blank)  "
                  f"grad-norm {gnorm:.2e}  pred max {pred.max().item():.3f}  "
                  f"pred on targets {on.mean().item():.3f}  ({time.time() - t0:.0f}s)")
        opt.step()
    mse = masked_loss(A(X).detach(), Y, mask).item()
    print(f"\nfinal MSE = {mse / blank:.2f} x Blank")
    print("PASS: learned more than 'predict nothing'." if mse / blank < 0.5 else "STILL NEAR BLANK.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--loss", choices=["mse", "bce"], default="mse")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--videos", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    selftest()
    if args.overfit:
        overfit(args.data, args.steps, args.device, args.videos, args.loss)
