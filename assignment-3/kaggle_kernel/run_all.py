import glob, os, shutil, subprocess, sys

src = os.path.dirname(glob.glob("/kaggle/input/**/X.npy", recursive=True)[0])
work = "/kaggle/working"
os.chdir(work)
for f in ("models.py", "utils.py", "train.py"):
    shutil.copy(os.path.join(src, f), work)
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/processed"):
    os.symlink(src, "data/processed")          # train.py expects data/processed/
os.makedirs("results", exist_ok=True)

subprocess.run(["nvidia-smi"])                  # hardware record for the report
for a in ["none", "fixed", "adaptive", "hybrid"]:
    with open(f"results/log_{a}.txt", "w") as log:
        p = subprocess.Popen([sys.executable, "-u", "train.py", "--adj", "hybrid", "--alpha", "0.5"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="")
            log.write(line)
        p.wait()

os.remove("data/processed")                     # keep the output folder clean
