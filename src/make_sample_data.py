"""Create a SMALL SYNTHETIC dataset in CIC-IDS2017 style so you can test the
pipeline without downloading 1GB+. It is NOT real traffic - results on it
mean nothing scientifically. Use the real dataset for your final results."""
import numpy as np
import pandas as pd
from config import DATA_DIR

rng = np.random.default_rng(0)


def gen(n, attack):
    if attack:  # e.g. DoS/PortScan-like: short, many packets, high rate
        dur = rng.exponential(2e4, n)
        fwd = rng.poisson(12, n) + 1
        bwd = rng.poisson(1, n)
        length = rng.normal(60, 30, n).clip(0)
        rate = rng.exponential(5e4, n)
        syn = rng.binomial(1, 0.7, n)
    else:
        dur = rng.exponential(5e5, n)
        fwd = rng.poisson(8, n) + 1
        bwd = rng.poisson(8, n) + 1
        length = rng.normal(400, 200, n).clip(0)
        rate = rng.exponential(2e3, n)
        syn = rng.binomial(1, 0.1, n)
    return pd.DataFrame({
        " Flow Duration": dur, " Total Fwd Packets": fwd,
        " Total Backward Packets": bwd, " Packet Length Mean": length,
        " Flow Packets/s": rate, " SYN Flag Count": syn,
        " Destination Port": rng.choice([80, 443, 22, 53, 8080], n),
    })


benign = gen(8000, False); benign[" Label"] = "BENIGN"
dos = gen(1500, True); dos[" Label"] = "DoS"
scan = gen(1000, True); scan[" Label"] = "PortScan"
df = pd.concat([benign, dos, scan]).sample(frac=1, random_state=0)
DATA_DIR.mkdir(exist_ok=True)
df.to_csv(DATA_DIR / "sample_synthetic.csv", index=False)
print("Wrote", DATA_DIR / "sample_synthetic.csv", df.shape)
