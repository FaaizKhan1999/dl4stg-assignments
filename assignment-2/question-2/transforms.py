import torch

# ---------- primitives (used directly by Table 1) ----------
def mirror(x: torch.Tensor) -> torch.Tensor:
    """(Mx)[i, j] = x[i, 63 - j]. Acts on the last axis (columns) only."""
    return x.flip(-1)

def rot(x: torch.Tensor, a: int = 1) -> torch.Tensor:
    """R^a: a quarter-turns anticlockwise on the last two axes (rows, columns)."""
    return torch.rot90(x, k=a, dims=(-2, -1))

# ---------- the group G = {R^a, M R^a : a = 0..3} ----------
# Each element is (m, a): rotate by a first, then mirror if m == 1.
G = [(m, a) for m in (0, 1) for a in range(4)]   # index 0 is the identity (0, 0)

def apply_g(x: torch.Tensor, g) -> torch.Tensor:
    m, a = g
    y = rot(x, a)
    return mirror(y) if m else y

def apply_g_inv(x: torch.Tensor, g) -> torch.Tensor:
    """Inverse of R^a is R^{-a}; each M R^a is its own inverse."""
    m, a = g
    return apply_g(x, g) if m else rot(x, -a)

def apply_g_per_sample(x: torch.Tensor, g_idx: torch.Tensor) -> torch.Tensor:
    """Apply a (possibly different) group element to each sample in a batch.
    x: (B, ...), g_idx: (B,) integers in 0..7 indexing G."""
    out = torch.empty_like(x)
    g_idx = g_idx.to(x.device)
    for i, g in enumerate(G):
        sel = g_idx == i
        if sel.any():
            out[sel] = apply_g(x[sel], g)
    return out

# Transformations used in the consistency tests (Section 3)
TRANSFORMS = {"M": mirror, "R": lambda x: rot(x, 1)}
