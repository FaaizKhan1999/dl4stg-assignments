import sys
import torch

N_FRAMES = 20
SPLITS = {"train": (0, 8000), "val": (8000, 9000), "test": (9000, 10000)}

# ---------- loading ----------
def load_moving_mnist(root: str = "data") -> torch.Tensor:
    """All 10,000 complete videos as uint8, shape (10000, 20, 1, 64, 64).
    split=None is required: torchvision's 'train'/'test' split frames, not videos."""
    from torchvision.datasets import MovingMNIST
    ds = MovingMNIST(root=root, split=None, download=True)
    videos = getattr(ds, "data", None)
    if videos is None:                       # fallback if the attribute ever changes
        videos = torch.stack([ds[i] for i in range(len(ds))])
    assert videos.shape == (10000, N_FRAMES, 1, 64, 64), videos.shape
    assert videos.dtype == torch.uint8, videos.dtype
    return videos

def get_split(videos: torch.Tensor, name: str) -> torch.Tensor:
    a, b = SPLITS[name]
    return videos[a:b]

def to_float(x: torch.Tensor) -> torch.Tensor:
    """float32 in [0, 1], exactly as specified: convert, then divide by 255."""
    return x.float() / 255

# ---------- windows (frame numbers are 1..20 throughout) ----------
def window_frames(t: int, k: int):
    """Frame numbers (1-based) of a window, before dropping missing targets."""
    inp = [t + i * k for i in range(5)]
    tgt = [t + (5 + i) * k for i in range(5)]
    return inp, tgt

def make_window(videos: torch.Tensor, t, k):
    """X = (x_t, ..., x_{t+4k}), Y = (x_{t+5k}, ..., x_{t+9k}).

    videos: (20, C, H, W) for one video or (B, 20, C, H, W) for a batch.
    t, k:   ints, or (B,) integer tensors (one start/speed per video).
    Returns X (.., 5, C, H, W), Y (.., 5, C, H, W) with missing targets zero-padded,
    and mask (.., 5) bool, True where the target frame exists (frame number <= 20).
    """
    single = videos.dim() == 4
    v = videos.unsqueeze(0) if single else videos
    B, dev = v.shape[0], v.device

    t = torch.as_tensor(t, dtype=torch.long, device=dev)
    k = torch.as_tensor(k, dtype=torch.long, device=dev)
    if t.dim() == 0: t = t.expand(B)
    if k.dim() == 0: k = k.expand(B)

    offsets = torch.arange(10, device=dev)
    frames = t[:, None] + offsets[None, :] * k[:, None]        # (B, 10), 1-based
    exists = frames <= N_FRAMES
    assert exists[:, :5].all(), "all five input frames must exist"

    idx = (frames - 1).clamp(max=N_FRAMES - 1)                 # frame f -> tensor index f-1
    out = v[torch.arange(B, device=dev)[:, None], idx]         # (B, 10, C, H, W)
    X, Y = out[:, :5], out[:, 5:]
    mask = exists[:, 5:]
    Y = torch.where(mask[:, :, None, None, None], Y, torch.zeros_like(Y))

    if single:
        return X[0], Y[0], mask[0]
    return X, Y, mask

# ---------- allowed start frames ----------
def max_start(k: int, mode: str) -> int:
    """'normal' (models A, B): 1x only, all five targets exist -> t + 9k <= 20.
    'speed' (models C, D): all inputs and >= 2 targets exist -> t + 6k <= 20."""
    if mode == "normal":
        assert k == 1, "normal-speed training uses 1x only"
        return N_FRAMES - 9 * k
    if mode == "speed":
        return N_FRAMES - 6 * k
    raise ValueError(mode)

def allowed_starts(k: int, mode: str):
    return list(range(1, max_start(k, mode) + 1))

def sample_windows(videos: torch.Tensor, mode: str, generator: torch.Generator):
    """One random window per video.
    'normal': k = 1, t uniform on {1..11}.
    'speed' : k uniform on {1,2,3}, THEN t uniform on that k's allowed starts.
    Sampling uses `generator` (CPU) so the seed controls it.
    Returns X, Y, mask, t, k."""
    B = videos.shape[0]
    if mode == "normal":
        k = torch.ones(B, dtype=torch.long)
        t = torch.randint(1, max_start(1, "normal") + 1, (B,), generator=generator)
    elif mode == "speed":
        k = torch.randint(1, 4, (B,), generator=generator)
        maxt = torch.tensor([0] + [max_start(kk, "speed") for kk in (1, 2, 3)])
        u = torch.rand(B, generator=generator)
        t = 1 + (u * maxt[k]).long()                           # uniform on 1..maxt[k]
    else:
        raise ValueError(mode)
    X, Y, mask = make_window(videos, t, k)
    return X, Y, mask, t, k

# ---------- quick sanity run: python data.py [root] ----------
if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    videos = load_moving_mnist(root)
    print("all videos:", tuple(videos.shape), videos.dtype)
    for name in SPLITS:
        print(f"{name:5s}:", tuple(get_split(videos, name).shape))
    x = to_float(videos[:2])
    print("float range:", float(x.min()), float(x.max()), x.dtype)
    X, Y, m = make_window(get_split(videos, "test")[:4], 1, 3)
    print("3x window on 4 test videos:", tuple(X.shape), tuple(Y.shape), m[0].tolist())
    g = torch.Generator().manual_seed(0)
    _, _, m, t, k = sample_windows(get_split(videos, "train")[:8], "speed", g)
    print("sampled k:", k.tolist(), "t:", t.tolist(), "n targets:", m.sum(1).tolist())
