"""Q1: numerical checks for parts (a), (b) and (c)."""
from fractions import Fraction
import numpy as np

K = 8
TRIPLES = [(s, z, p) for s in range(K + 1) for z in range(K + 1) for p in range(K + 1)
           if 1 <= s + z + p <= K]                      # (slowing, steady, speeding-up) counts
print(f"{len(TRIPLES)} valid neighbourhoods (count triples) for K={K}\n")


def trends(t):
    s, z, p = t
    return [-1] * s + [0] * z + [1] * p


# ---------- (a) affine message: sum = w*S + c*n, so equal (n, S) => equal input to rho ----------
print("--- (a) ---")
N1, N2 = [-1, 1], [0, 0]                                # n=2, S=0 for both; steady counts 0 vs 2
rng = np.random.default_rng(42)
for w, c in rng.normal(size=(3, 2)):
    agg = lambda N: sum(w * x + c for x in N)
    print(f"w={w:+.3f}, c={c:+.3f}: agg(N1)={agg(N1):+.4f}  agg(N2)={agg(N2):+.4f}")
print("identical for every (w, c) -> rho cannot tell them apart\n")


# ---------- (b) base-(K+1) encoding ----------
print("--- (b) ---")
B = K + 1
phi = {-1: 1, 0: B, 1: B ** 2}                          # place values: slowing | steady | speeding

def decode(code):
    return code % B, (code // B) % B, code // B ** 2

codes = [sum(phi[x] for x in trends(t)) for t in TRIPLES]   # sum over actual neighbour lists
assert len(set(codes)) == len(TRIPLES), "collision!"
assert all(decode(c) == t for c, t in zip(codes, TRIPLES)), "decode failed!"
print(f"phi(-1)={phi[-1]}, phi(0)={phi[0]}, phi(+1)={phi[1]}: "
      f"all {len(TRIPLES)} sums distinct and decode correctly (max sum {max(codes)})")

phi_ns = {-1: 1, 0: 0, 1: 1}                           # counts non-steady neighbours
for n in range(1, 30):                                  # any neighbourhood size
    for s in range(n + 1):
        for z in range(n - s + 1):
            t = (s, z, n - s - z)
            assert sum(phi_ns[x] for x in trends(t)) == s + t[2]
print("non-steady counter phi(-1)=1, phi(0)=0, phi(+1)=1 verified for sizes 1..29\n")


# ---------- (c) brute-force the table ----------
print("--- (c) ---")
def aggregate(t, method):
    s, z, p = t
    n = s + z + p
    if method == "Sum":
        return (s, z, p)
    if method == "Mean":
        return (Fraction(s, n), Fraction(z, n), Fraction(p, n))
    return (int(s > 0), int(z > 0), int(p > 0))         # Max of one-hot vectors

TASKS = {"(i) count steady": lambda t: t[1],
         "(ii) more than half slowing": lambda t: 2 * t[0] > sum(t),
         "(iii) any slowing": lambda t: t[0] > 0}

for task, answer in TASKS.items():
    for method in ("Sum", "Mean", "Max"):
        seen, clash = {}, None
        for t in TRIPLES:
            v = aggregate(t, method)
            if v in seen and answer(seen[v]) != answer(t):
                clash = (seen[v], t, v)
                break
            seen.setdefault(v, t)
        if clash is None:
            print(f"{task:30s} {method:4s}: Yes")
        else:
            t1, t2, v = clash
            vs = tuple(str(x) for x in v)
            print(f"{task:30s} {method:4s}: No  e.g. counts {t1} vs {t2} -> same vector {vs}, "
                  f"answers {answer(t1)} vs {answer(t2)}")
