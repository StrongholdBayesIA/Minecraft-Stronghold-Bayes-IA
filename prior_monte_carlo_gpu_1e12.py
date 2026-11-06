import cupy as cp
import numpy as np

# Ring 1 parameters
A1 = 1280
B1 = 2816
CHUNK = 16

# Step 1: enumerate all valid chunk centres in ring 1
xs = np.arange(-B1 - CHUNK, B1 + 2 * CHUNK, CHUNK) + 8
zs = np.arange(-B1 - CHUNK, B1 + 2 * CHUNK, CHUNK) + 8
xx, zz = np.meshgrid(xs, zs)
dist = np.sqrt(xx**2 + zz**2)
mask = (dist >= A1) & (dist <= B1)
cx = xx[mask].astype(np.int32)
cz = zz[mask].astype(np.int32)
N = len(cx)
print(f"Valid ring-1 candidate chunks: {N}")

# Move candidates to GPU
cx_gpu = cp.array(cx.astype(np.int64))
cz_gpu = cp.array(cz.astype(np.int64))
cand_keys = cx_gpu * 100000 + cz_gpu
sort_idx = cp.argsort(cand_keys)
sorted_keys = cand_keys[sort_idx]

def run_simulation_gpu(n_sims):
    BATCH = 50_000_000  # 50M per batch
    hits = cp.zeros(N, dtype=cp.int64)
    done = 0
    rng = cp.random.default_rng(42)

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
        if n_sims >= 10_000_000_000:
            print(f"  {done:,} / {n_sims:,} done", flush=True)

    probs = hits / hits.sum()
    return cp.asnumpy(probs)

scales = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
]

prev_probs = None
all_results = {}

for n in scales:
    print(f"\nRunning {n:,} simulations on GPU...", flush=True)
    probs = run_simulation_gpu(n)
    all_results[n] = probs
    print(f"  Sum of probs    : {probs.sum():.8f}")
    print(f"  Non-zero chunks : {(probs > 0).sum()} / {N}")
    print(f"  Max prob        : {probs.max():.8f}")
    if prev_probs is not None:
        diff = np.max(np.abs(prev_probs - probs))
        print(f"  Max diff vs prev: {diff:.8f}")
    prev_probs = probs

# Save best prior
np.save("candidates_x.npy", cx)
np.save("candidates_z.npy", cz)
np.save("prior_probs.npy", all_results[scales[-1]])
print(f"\nSaved prior from {scales[-1]:,} simulations.")
