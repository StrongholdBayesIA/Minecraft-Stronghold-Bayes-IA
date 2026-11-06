import numpy as np

# Check of prior_probs.npy: are the probabilities whole counts out of one total?
p = np.load("prior_probs.npy")
H = 962308854054            # total simulated strongholds that landed on a candidate
counts = p * H

print("Candidates            :", len(p))
print("Sum of probs          :", float(p.sum()))
print("Non-zero chunks       :", int((p > 0).sum()), "/", len(p))
print("Max prob              :", p.max())
print("Min prob              :", p.min())
print("Total hits H          :", f"{H:,}")
print("Whole-count check     : max |p*H - round(p*H)| =", np.abs(counts - np.round(counts)).max())
print("Sum of counts         :", f"{int(np.round(counts).sum()):,}")
print("Min / max count       :", f"{int(round(counts.min())):,}", f"{int(round(counts.max())):,}")
