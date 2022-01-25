import torch
import numpy as np

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

s = t.add(1)
print(f"t: {t}")
print(f"n: {n}")
print(f"s: {s}")