import cupy as cp
import numpy as np

A1 = 1280
B1 = 2816
CHUNK = 16

# Load existing candidates
cx = np.load("candidates_x.npy")
cz = np.load("candidates_z.npy")
N = len(cx)
print(f"Loaded {N} candidates")

cx_gpu = cp.array(cx.astype(np.int64))
cz_gpu = cp.array(cz.astype(np.int64))
cand_keys = cx_gpu * 100000 + cz_gpu
sort_idx = cp.argsort(cand_keys)
sorted_keys = cand_keys[sort_idx]

def run_simulation_gpu(n_sims, seed=42):
    hits = cp.zeros(N, dtype=cp.int64)
    rng = cp.random.default_rng(seed)
    BATCH = min(n_sims, 1_000_000)
    done = 0
    while done < n_sims:
        n = min(BATCH, n_sims - done)
        phi = rng.uniform(0, 2 * cp.pi, n)
        r   = rng.uniform(A1, B1, n)
        x = (cp.floor(r * cp.cos(phi) / CHUNK) * CHUNK + 8).astype(cp.int64)
        z = (cp.floor(r * cp.sin(phi) / CHUNK) * CHUNK + 8).astype(cp.int64)
        x = x + rng.integers(-7, 8, n) * CHUNK
        z = z + rng.integers(-7, 8, n) * CHUNK
        sim_keys = x * 100000 + z
        pos = cp.searchsorted(sorted_keys, sim_keys)
        pos = cp.clip(pos, 0, N - 1)
        found = sorted_keys[pos] == sim_keys
        hit_indices = sort_idx[pos[found]]
        hits += cp.bincount(hit_indices, minlength=N)
        done += n
    probs = hits / hits.sum()
    return cp.asnumpy(probs)

scales = [10_000, 100_000, 1_000_000]
prev_probs = None
lines = [f"Valid ring-1 candidates: {N}\n"]

for n in scales:
    print(f"Running {n:,} simulations...", flush=True)
    probs = run_simulation_gpu(n)
    line = (f"n = {n:>12,} | sum = {probs.sum():.8f} | "
            f"non-zero = {(probs > 0).sum():>6} / {N} | "
            f"max_prob = {probs.max():.8f}")
    if prev_probs is not None:
        diff = np.max(np.abs(prev_probs - probs))
        line += f" | max_diff_vs_prev = {diff:.8f}"
    print(line)
    lines.append(line)
    prev_probs = probs

with open("convergence_results.txt", "w") as f:
    f.write("\n".join(lines))
print("\nSaved: convergence_results.txt")
