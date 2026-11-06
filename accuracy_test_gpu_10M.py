import cupy as cp
import numpy as np
import json
import time

# Load prior
cx = np.load("candidates_x.npy").astype(np.float32)
cz = np.load("candidates_z.npy").astype(np.float32)
prior = np.load("prior_probs.npy").astype(np.float32)
N = len(cx)
print(f"Loaded {N} candidates")
print(f"Prior sums to: {prior.sum():.8f}")

cx_gpu = cp.array(cx)
cz_gpu = cp.array(cz)
prior_gpu = cp.array(prior)

# Candidate lookup
cx_key = cx.astype(np.int64) * 100000 + cz.astype(np.int64)
sort_order = np.argsort(cx_key)
sorted_cx_key = cx_key[sort_order]
sorted_cx_key_gpu = cp.array(sorted_cx_key)
sort_order_gpu = cp.array(sort_order)

A1 = 1280
B1 = 2816
CHUNK = 16

# sigma values — special strings "perfect" and "f3_only" handled separately
SIGMAS = ["perfect", "f3_only", 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
N_TRIALS = 1_000_000
PLAYER_RADIUS = 1280
BATCH = 2_000

# F3 rounding std dev: U(-0.005, 0.005) has std = 0.005 / sqrt(3)
F3_SIGMA = np.float32(0.005 / np.sqrt(3))

np.random.seed(123)

def run_accuracy_test_gpu(sigma):
    t0 = time.time()

    if sigma == "perfect":
        label = "0 (perfect)"
        sigma_eff = np.float32(0.0001)
        use_H = False
        use_F = False
    elif sigma == "f3_only":
        label = "F3 only"
        sigma_eff = F3_SIGMA
        use_H = False
        use_F = True
    else:
        label = str(sigma)
        sigma_eff = np.float32(sigma)
        use_H = True
        use_F = True

    # Generate all trials on CPU
    phi = np.random.uniform(0, 2 * np.pi, N_TRIALS).astype(np.float32)
    r_s = np.random.uniform(A1, B1, N_TRIALS).astype(np.float32)
    tx = (np.floor(r_s * np.cos(phi) / CHUNK) * CHUNK + 8).astype(np.int32)
    tz = (np.floor(r_s * np.sin(phi) / CHUNK) * CHUNK + 8).astype(np.int32)
    tx = tx + np.random.randint(-7, 8, N_TRIALS).astype(np.int32) * CHUNK
    tz = tz + np.random.randint(-7, 8, N_TRIALS).astype(np.int32) * CHUNK

    pa = np.random.uniform(0, 2 * np.pi, N_TRIALS).astype(np.float32)
    pr = np.random.uniform(0, PLAYER_RADIUS, N_TRIALS).astype(np.float32)
    px = pr * np.cos(pa)
    pz = pr * np.sin(pa)

    true_angle = np.degrees(np.arctan2(-(px - tx), tz - pz)).astype(np.float32)

    theta = true_angle.copy()
    if use_H:
        theta += np.random.normal(0, float(sigma), N_TRIALS).astype(np.float32)
    if use_F:
        theta += np.random.uniform(-0.005, 0.005, N_TRIALS).astype(np.float32)

    # Find valid trials
    tx_key = tx.astype(np.int64) * 100000 + tz.astype(np.int64)
    pos = np.searchsorted(sorted_cx_key, tx_key)
    pos = np.clip(pos, 0, N - 1)
    valid = sorted_cx_key[pos] == tx_key
    true_idx = sort_order[pos]

    n_valid = int(valid.sum())
    theta_v    = cp.array(theta[valid])
    px_v       = cp.array(px[valid])
    pz_v       = cp.array(pz[valid])
    true_idx_v = cp.array(true_idx[valid])

    top1 = 0
    top3 = 0
    post_true = []
    rank_true = []

    for i in range(0, n_valid, BATCH):
        sl   = slice(i, min(i + BATCH, n_valid))
        t_v  = theta_v[sl]
        p_xv = px_v[sl]
        p_zv = pz_v[sl]
        ti_v = true_idx_v[sl]
        b    = len(t_v)

        gamma = cp.degrees(cp.arctan2(
            -(p_xv[:, None] - cx_gpu[None, :]),
             cz_gpu[None, :] - p_zv[:, None]
        ))

        diff      = t_v[:, None] - gamma
        weights   = cp.exp(-diff**2 / (2 * sigma_eff**2)) * prior_gpu[None, :]
        row_sum   = weights.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        posterior = weights / row_sum

        tp    = posterior[cp.arange(b), ti_v]
        ranks = (posterior > tp[:, None]).sum(axis=1) + 1

        top1 += int((ranks == 1).sum())
        top3 += int((ranks <= 3).sum())
        post_true.extend(cp.asnumpy(tp).tolist())
        rank_true.extend(cp.asnumpy(ranks).tolist())

        if (i // BATCH) % 100 == 0:
            print(f"  {i+b:,} / {n_valid:,}", flush=True)

    n = len(rank_true)
    elapsed = time.time() - t0
    return {
        "sigma": sigma if isinstance(sigma, (int, float)) else label,
        "label": label,
        "top1_accuracy": top1 / n * 100,
        "top3_accuracy": top3 / n * 100,
        "mean_posterior": float(np.mean(post_true)) * 100,
        "mean_rank": float(np.mean(rank_true)),
        "n_valid": n,
        "n_trials": N_TRIALS,
        "time_s": elapsed
    }

print(f"Baseline (random): {1/N*100:.6f}%")
print(f"Running {N_TRIALS:,} trials per sigma on GPU...\n")

results = []
for sigma in SIGMAS:
    label = sigma if isinstance(sigma, str) else str(sigma)
    print(f"sigma = {label}...", flush=True)
    r = run_accuracy_test_gpu(sigma)
    results.append(r)
    print(f"  Top-1: {r['top1_accuracy']:.2f}%  Top-3: {r['top3_accuracy']:.2f}%  "
          f"Mean post: {r['mean_posterior']:.4f}%  Mean rank: {r['mean_rank']:.1f}  "
          f"Valid: {r['n_valid']:,}  Time: {r['time_s']:.1f}s\n")

print("\n" + "="*85)
print(f"{'sigma':>12} {'Top-1%':>10} {'Top-3%':>10} {'Mean post%':>12} {'Mean rank':>10}")
print("="*85)
for r in results:
    print(f"{r['label']:>12} {r['top1_accuracy']:>10.2f} {r['top3_accuracy']:>10.2f} "
          f"{r['mean_posterior']:>12.4f} {r['mean_rank']:>10.1f}")
print(f"\nBaseline (random guess): {1/N*100:.6f}%")

with open("accuracy_results_10M.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: accuracy_results_10M.json")
