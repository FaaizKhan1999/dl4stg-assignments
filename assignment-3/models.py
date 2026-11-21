import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Conv1d over each sensor's 8-step history -> d-dim feature per sensor."""
    def __init__(self, d=32, t_in=8, in_dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.proj = nn.Linear(d * t_in, d)

    def forward(self, x):                          # x: (B, T_in, N, F)
        B, T, N, Fd = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N, Fd, T)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = self.proj(h.flatten(1))                # (B*N, d)
        return h.view(B, N, -1)                    # Z: (B, N, d)


class Adjacency(nn.Module):
    """Returns the normalised adjacency used by the graph conv (None for temporal-only)."""
    def __init__(self, mode, A_fixed=None, n_nodes=716, emb_dim=10, topk=None, alpha=None):
        super().__init__()
        assert mode in ("none", "fixed", "adaptive", "hybrid")
        self.mode, self.topk = mode, topk
        if mode in ("fixed", "hybrid"):
            self.register_buffer("A_fixed", A_fixed)            # row-normalised, never trained
        if mode in ("adaptive", "hybrid"):
            self.E1 = nn.Parameter(torch.randn(n_nodes, emb_dim))
            self.E2 = nn.Parameter(torch.randn(n_nodes, emb_dim))
        if mode == "hybrid":
            if alpha is None:
                self.alpha_logit = nn.Parameter(torch.zeros(()))  # sigmoid(0) = 0.5
            else:
                self.register_buffer("alpha_fixed", torch.tensor(float(alpha)))

    def alpha(self):
        return torch.sigmoid(self.alpha_logit) if hasattr(self, "alpha_logit") else self.alpha_fixed

    def adaptive(self):
        S = F.relu(self.E1 @ self.E2.T)
        if self.topk:
            keep = torch.zeros_like(S, dtype=torch.bool)
            keep.scatter_(1, S.topk(self.topk, dim=1).indices, True)
            S = S.masked_fill(~keep, float("-inf"))              # sparsify BEFORE softmax
        return torch.softmax(S, dim=1)                           # row-wise

    def forward(self):
        if self.mode == "none":
            return None
        if self.mode == "fixed":
            return self.A_fixed
        if self.mode == "adaptive":
            return self.adaptive()
        a = self.alpha()
        return a * self.A_fixed + (1 - a) * self.adaptive()


class STGNN(nn.Module):
    def __init__(self, adjacency, d=32, t_in=8, t_out=4):
        super().__init__()
        self.encoder = TemporalEncoder(d, t_in)
        self.adj = adjacency
        self.Wg = nn.Linear(d, d, bias=False)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, t_out))

    def forward(self, x):                          # (B, T_in, N, 1)
        Z = self.encoder(x)
        A_hat = self.adj()
        agg = Z if A_hat is None else torch.einsum("nm,bmd->bnd", A_hat, Z)
        H = F.relu(self.Wg(agg)) + Z               # residual keeps each sensor's own features
        return self.head(H).permute(0, 2, 1).unsqueeze(-1)   # (B, T_out, N, 1)
