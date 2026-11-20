import h5py, numpy as np, pandas as pd

# 1. Look at the h5 layout without loading the data
with h5py.File("data/raw/ca_his_raw_2020.h5", "r") as f:
    print("attrs:", dict(f["t"].attrs))
    f["t"].visititems(
        lambda name, obj: print(name, getattr(obj, "shape", ""), getattr(obj, "dtype", ""))
    )

# 2. Metadata (small)
meta = pd.read_csv("data/raw/ca_meta.csv")
print(meta.head())
print(meta.columns.tolist())
print(meta["District"].value_counts())

# 3. Adjacency, memory-mapped so it isn't loaded into RAM
A = np.load("data/raw/ca_rn_adj.npy", mmap_mode="r")
print(A.shape, A.dtype)
