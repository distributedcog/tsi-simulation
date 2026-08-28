"""Topological Sheaves of Irreducibility (TSI) degree-1 benchmark.

Exhaustive evaluation of deterministic one-round protocols on C_m.

Registered instance: m=6, B=2 -> h_B=3, q_{m,B}=5, congestion beta=5 with
q <= beta < (m-1)h_B = 15. All 5^6 = 15,625 instances enumerated (uniform prior).

COMPLETE 1-SKELETON CONTROL (pairwise, coordinator agent 0). Two lossless reductions:
 (R1) For any fixed messages, the optimal deterministic coordinator rule picks,
      per view, the majority label (T1) / modal correction (T3). Exact.
 (R2) A b-bit message from agent i is a partition of {-2..2} into <= 2^b blocks;
      accuracy depends only on the induced partition, and refining a partition
      never decreases the optimum. Hence it suffices to search maximal
      partitions: 2 blocks (15) for 1 bit, 4 blocks (10) for 2 bits, discrete (1)
      for >=3 bits; 0 bits = constant. Unused budget never helps, and agents
      1..5 are exchangeable, so bit allocations reduce to multisets summing to 5:
      (1,1,1,1,1), (2,1,1,1), (2,2,1), (3,1,1), (3,2)  [(4,1),(5) dominated].
So the search below covers ALL deterministic one-round protocols on the complete
pairwise 1-skeleton with receive budget beta=5 at agent 0, exactly. Extra
pairwise edges among non-coordinators cannot affect agent 0 in the same round,
so the complete-pairwise optimum equals the star-centred optimum.

HIGHER-ORDER ABLATION: the clean ablation removes only the all-agent joint
primitive from the filled simplex, leaving the same complete pairwise
1-skeleton. Its one-round performance is therefore the complete-pairwise row.

BASE-RING SANITY: if one instead restores only the original ring C_6, agent 0
hears only ring neighbours 1 and 5 in one round. This is evaluated separately,
but it is no longer the main ablation because its 1-skeleton differs.

CONTROL TASK (trivial relevant class): decide c_0 + c_1 = 0 (needs h_B=3 bits
over one edge, feasible for complete pairwise and ring systems alike).
"""
import itertools, math
import numpy as np

m, B = 6, 2
V = np.arange(-B, B + 1)          # 5 values, index = value + B
nV = len(V)
hB = math.ceil(math.log2(2 * B + 1))
q_mB = math.ceil(math.log2(2 * m * B + 1))
beta = 5
assert q_mB <= beta < (m - 1) * hB

grid = np.array(list(itertools.product(range(nV), repeat=m)), dtype=np.int64)  # value indices
c = grid - B
l = c.sum(axis=1)                                  # -12..12
t1_label = (l == 0).astype(np.int64)
n_inst = len(grid)
n_real = int(t1_label.sum())
unreal = l != 0
l_idx = (l + m * B)                                # 0..24 index of l (mode of l == mode of -l)
n_lvals = 2 * m * B + 1

# ---- maximal partitions of the 5 value-indices ----
P1 = []   # 2 blocks (1 bit): nonempty proper subsets up to complement
for mask in range(1, 1 << nV):
    comp = ((1 << nV) - 1) ^ mask
    if mask < comp:
        P1.append(np.array([(mask >> i) & 1 for i in range(nV)], dtype=np.int64))
P2 = []   # 4 blocks (2 bits): one merged pair
for i, j in itertools.combinations(range(nV), 2):
    lab, nxt = np.zeros(nV, dtype=np.int64), 0
    for k in range(nV):
        if k == j: lab[k] = lab[i] if i < k else 0
    nxt = 0
    lab = np.full(nV, -1, dtype=np.int64)
    for k in range(nV):
        if k == j: lab[k] = lab[i]
        else: lab[k] = nxt; nxt += 1
    P2.append(lab)
P3 = [np.arange(nV, dtype=np.int64)]  # discrete (>=3 bits)
P0 = [np.zeros(nV, dtype=np.int64)]   # constant (0 bits)
assert len(P1) == 15 and len(P2) == 10

def blocks(lab): return int(lab.max()) + 1

def evaluate(parts):
    """parts: list of 5 label arrays for agents 1..5. Returns (t1_correct, t3_exact, n_unreal)."""
    key = grid[:, 0].copy()
    for j, lab in enumerate(parts, start=1):
        key = key * blocks(lab) + lab[grid[:, j]]
    nkeys = int(key.max()) + 1
    pos = np.bincount(key, weights=t1_label, minlength=nkeys)
    tot = np.bincount(key, minlength=nkeys)
    t1_correct = int(np.maximum(pos, tot - pos).sum())
    ku = key[unreal]
    comb = ku * n_lvals + l_idx[unreal]
    cnt = np.bincount(comb, minlength=nkeys * n_lvals).reshape(nkeys, n_lvals)
    t3_exact = int(cnt.max(axis=1).sum())
    return t1_correct, t3_exact, int(unreal.sum())

def search(alloc_pools):
    """alloc_pools: list of (pool, multiplicity). Exhaust multisets; return best."""
    best = (-1, -1, None)
    iters = []
    for pool, mult in alloc_pools:
        iters.append(list(itertools.combinations_with_replacement(range(len(pool)), mult)))
    bt1, bt3, nu_ = -1, -1, None
    for choice in itertools.product(*iters):
        parts = []
        for (pool, mult), idxs in zip(alloc_pools, choice):
            parts += [pool[i] for i in idxs]
        parts += [P0[0]] * (5 - len(parts))
        t1c, t3e, nu = evaluate(parts)
        bt1, bt3, nu_ = max(bt1, t1c), max(bt3, t3e), nu
    return (bt1, bt3, nu_)

allocations = {
    "(1,1,1,1,1)": [(P1, 5)],
    "(2,1,1,1)":   [(P2, 1), (P1, 3)],
    "(2,2,1)":     [(P2, 2), (P1, 1)],
    "(3,1,1)":     [(P3, 1), (P1, 2)],
    "(3,2)":       [(P3, 1), (P2, 1)],
}
print(f"m={m} B={B} h_B={hB} q_mB={q_mB} beta={beta} (m-1)h_B={(m-1)*hB}")
print(f"instances={n_inst} realizable={n_real} ({n_real/n_inst:.6f}) majority correct={n_inst-n_real}")
complete_pairwise_best = (-1, -1, None)
for name, pools in allocations.items():
    t1c, t3e, nu = search(pools)
    print(f"complete-pairwise alloc {name:12s}: T1 correct={t1c} ({t1c/n_inst:.6f})  "
          f"T3 exact={t3e}/{nu} ({t3e/nu:.4f})")
    complete_pairwise_best = (max(complete_pairwise_best[0], t1c), max(complete_pairwise_best[1], t3e), nu)

# ---- base-ring sanity: C_6, agent 0 hears only agents 1 and 5 ----
def evaluate_ring(lab1, lab5):
    key = grid[:, 0] * blocks(lab1) + lab1[grid[:, 1]]
    key = key * blocks(lab5) + lab5[grid[:, 5]]
    nkeys = int(key.max()) + 1
    pos = np.bincount(key, weights=t1_label, minlength=nkeys)
    tot = np.bincount(key, minlength=nkeys)
    t1c = int(np.maximum(pos, tot - pos).sum())
    ku = key[unreal]
    cnt = np.bincount(ku * n_lvals + l_idx[unreal], minlength=nkeys * n_lvals).reshape(nkeys, n_lvals)
    return t1c, int(cnt.max(axis=1).sum())

ring_best = (-1, -1)
for lab1, lab5 in itertools.chain(
        ((P3[0], p) for p in P2), ((p, P3[0]) for p in P2)):   # (3,2) and (2,3); coarser dominated
    r = evaluate_ring(lab1, lab5)
    ring_best = (max(ring_best[0], r[0]), max(ring_best[1], r[1]))
n_unreal = int(unreal.sum())
print(f"S_fill (atomic g_A): T1 correct={n_inst} (1.000000)  "
      f"T3 exact={n_unreal}/{n_unreal} (1.0000)")
print(f"S_pair (same 1-skeleton): "
      f"T1 correct={complete_pairwise_best[0]} ({complete_pairwise_best[0]/n_inst:.6f})  "
      f"T3 exact={complete_pairwise_best[1]}/{complete_pairwise_best[2]} "
      f"({complete_pairwise_best[1]/complete_pairwise_best[2]:.4f})")
print(f"S_ring (sanity check): T1 correct={ring_best[0]} ({ring_best[0]/n_inst:.6f})  "
      f"T3 exact={ring_best[1]}/{n_unreal} ({ring_best[1]/n_unreal:.4f})")
print(f"complete-pairwise T1 gain over majority baseline: "
      f"{complete_pairwise_best[0]-(n_inst-n_real)} instances")
print(f"base-ring T1 gain over majority baseline: {ring_best[0]-(n_inst-n_real)} instances")
print(f"filled gain over same 1-skeleton ablation: "
      f"T1 +{n_inst-complete_pairwise_best[0]} instances, "
      f"T3 +{n_unreal-complete_pairwise_best[1]} exact repairs")

# control task: decide c_0 + c_1 == 0, agent 1 sends full value (3 bits <= beta)
key = grid[:, 0] * nV + grid[:, 1]
lab = ((c[:, 0] + c[:, 1]) == 0).astype(np.int64)
pos = np.bincount(key, weights=lab); tot = np.bincount(key)
print(f"control task, one full-value message: correct={int(np.maximum(pos, tot-pos).sum())}/{n_inst}")
