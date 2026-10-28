from pathlib import Path
import torch
from transforms import mirror, rot, G, apply_g, apply_g_inv
from data import make_window, allowed_starts

def bright_pixel(v):
    """(frame, row, column) of the single nonzero pixel, indices from 0."""
    nz = torch.nonzero(v)
    assert len(nz) == 1, f"expected one bright pixel, found {len(nz)}"
    f, _, r, c = nz[0].tolist()
    return (f, r, c)

def frames_of(video_value_is_frame, t, k):
    """Read frame numbers back from a video whose frame f has value f."""
    X, Y, mask = make_window(video_value_is_frame, t, k)
    inp = [int(X[i, 0, 0, 0]) for i in range(5)]
    tgt = [int(Y[i, 0, 0, 0]) for i in range(5) if mask[i]]   # existing targets only
    return inp, tgt

def show(frames):
    """[1,2,3,4,5] -> '1-5'; otherwise comma-separated."""
    if frames == list(range(frames[0], frames[-1] + 1)) and len(frames) > 2:
        return f"{frames[0]}–{frames[-1]}"
    return ", ".join(map(str, frames))

def main():
    # synthetic videos, exactly as in the handout
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64); p[7, 0, 10, 20] = 1
    fv = torch.arange(1, 21, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 64, 64).contiguous()

    rows = []
    def exact(name, a, b, expected="0"):
        d = (a - b).abs().max().item()
        rows.append((name, expected, f"{d:g} (torch.equal: {torch.equal(a, b)})"))
        return d == 0 and torch.equal(a, b)

    ok = True
    ok &= exact("(a) max|M(M(x)) − x|", mirror(mirror(x)), x)
    ok &= exact("(b) max|R⁴(x) − x|", rot(x, 4), x)
    ok &= exact("(c) max|MRM(x) − R⁻¹(x)|", mirror(rot(mirror(x), 1)), rot(x, -1))

    bm, br = bright_pixel(mirror(p)), bright_pixel(rot(p, 1))
    rows.append(("(d) bright pixel of M(p) (frame, row, column)", "(7, 10, 43)", str(bm)))
    rows.append(("(e) bright pixel of R(p) (frame, row, column)", "(7, 43, 10)", str(br)))
    ok &= bm == (7, 10, 43) and br == (7, 43, 10)

    expected = {1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
                2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
                3: ([1, 4, 7, 10, 13], [16, 19])}
    for label, k in (("(f)", 1), ("(g)", 2), ("(h)", 3)):
        inp, tgt = frames_of(fv, 1, k)
        e_inp, e_tgt = expected[k]
        rows.append((f"{label} {k}× input / target frames",
                     f"{show(e_inp)} / {show(e_tgt)}", f"{show(inp)} / {show(tgt)}"))
        ok &= inp == e_inp and tgt == e_tgt

    # extra (not reported): start-frame rules and group inverses
    assert len(allowed_starts(1, "normal")) == 11
    assert [len(allowed_starts(k, "speed")) for k in (1, 2, 3)] == [14, 8, 2]
    for g in G:
        assert torch.equal(apply_g_inv(apply_g(x, g), g), x), f"inverse wrong for {g}"
    assert len({apply_g(p, g).nonzero()[0].tolist().__str__() for g in G}) == 8, \
        "the 8 group elements must send the marker to 8 different places"

    # print and save Table 1 (markdown)
    lines = ["| Check | Expected | Yours |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    table = "\n".join(lines)
    print(table)
    Path("results").mkdir(exist_ok=True)
    Path("results/table1.md").write_text(table + "\n")
    print("\nALL CHECKS PASS" if ok else "\n*** SOME CHECKS FAILED — fix before continuing ***")
    assert ok

if __name__ == "__main__":
    main()
