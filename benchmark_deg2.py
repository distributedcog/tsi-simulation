"""
benchmark_deg2.py -- Topological Sheaves of Irreducibility (TSI) degree-2 benchmark
(Section 9.2).

Task model  (K_T, F_T) = (dDelta^3, R), boundary of the tetrahedron, constant sheaf.
            H^0 = R, H^1 = 0, H^2 = R;  beta^2(K_T; F_T) = 1 and the critical
            support is all four 2-simplices.

Datum       c in C^2(K_T; R), one integer per triangle,
            c_sigma = sum_{v in sigma} s_{v,sigma},  shares s_{v,sigma} in {-B..B}
            (12 shares in total).
Signature   ell(c) = c_123 - c_023 + c_013 - c_012  generates (im delta^1)^perp.
            The realization task R^2(c) is solvable iff ell(c) = 0.

Tasks       T1  agent 0 outputs whether [c] = 0.
            T3  when [c] != 0, agent 0 outputs the minimum-support correction on a
                designated 2-simplex, i.e. it must identify ell(c) exactly.
Control     single 2-simplex {0,1,2}, decide c_012 = 0.  H^2 = 0, no class;
            two 2-bit messages suffice, so every system is exact.

Systems (four agents, agent 0 = coordinator, one synchronous round; every message
and every primitive output is a simultaneous function of the initial shares, so a
primitive output cannot be relayed):
  S_fill  complete pairwise + four registered 3-agent primitives + registered
          all-agent primitive g_A = ell.
          induced complex Delta^3, H^2 = 0        ->  class-filling.
  S_face  the same system with the all-agent primitive removed; the four
          registered 3-agent primitives g_sigma = c_sigma remain.
          induced complex dDelta^3 = K_T           ->  class extends.
  S_pair  complete pairwise only, induced complex skel_1(Delta^3).

Run:  python3 benchmark_deg2.py
"""
import itertools, math
import numpy as np

B = 1                                   # share bound
TRIS = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
EPS = {(1, 2, 3): +1, (0, 2, 3): -1, (0, 1, 3): +1, (0, 1, 2): -1}
ST0 = [t for t in TRIS if 0 in t]


def bits(n):
    return 0 if n <= 1 else math.ceil(math.log2(n))


def signed_sum(k):
    """Exact counts of a signed sum of k i.i.d. uniform shares on {-B..B}.
    Signs are irrelevant: the distribution is symmetric."""
    cur = np.array([1], dtype=np.int64)
    unit = np.ones(2 * B + 1, dtype=np.int64)
    for _ in range(k):
        cur = np.convolve(cur, unit)
    return -k * B, cur


def exact_partitions(n, blocks):
    res = []

    def rec(i, assign, used):
        if n - i < blocks - used:
            return
        if i == n:
            if used == blocks:
                res.append(list(assign))
            return
        for b in range(used):
            assign.append(b); rec(i + 1, assign, used); assign.pop()
        if used < blocks:
            assign.append(used); rec(i + 1, assign, used + 1); assign.pop()

    rec(0, [], 0)
    return res


def block_vectors(counts, k):
    """Maximal partitions only: refining a partition never lowers the
    coordinator's optimum, so it suffices to search partitions with exactly
    min(k, |support|) blocks (same reduction as Section 9.1)."""
    n = len(counts)
    k = min(k, n)
    P = exact_partitions(n, k)
    out = np.zeros((len(P), k, n), dtype=np.int64)
    for p, a in enumerate(P):
        for v, b in enumerate(a):
            out[p, b, v] = counts[v]
    return out


def optimum(off_lo, off, senders, budget):
    """Exact optimum over every deterministic one-round coordinator-output
    protocol within `budget` received bits.

    off_*   : distribution of the statistic the coordinator knows exactly
    senders : list of (lo, counts), the residual statistic of each non-coordinator

    Sufficient-statistic reduction: ell = offset + sum_v b_v, with the b_v
    supported on pairwise disjoint share sets, hence independent; so w.l.o.g.
    each message is a function of b_v alone and the search over maximal
    partitions of each support is exhaustive.

    T1 and T3 are optimised independently, as in Section 9.1.
    """
    caps = [bits(len(c)) for _, c in senders]
    allocs = [a for a in itertools.product(*[range(c + 1) for c in caps])
              if sum(a) <= budget]
    maximal = [a for a in allocs
               if not any(b != a and all(y >= x for y, x in zip(b, a))
                          for b in allocs)]
    bT1 = bT3 = 0
    total = nonzero = 0
    for al in maximal:
        BVs = [(lo, block_vectors(c, 2 ** b)) for (lo, c), b in zip(senders, al)]
        for idx in itertools.product(*[range(bv.shape[0]) for _, bv in BVs]):
            chosen = [(lo, bv[i]) for (lo, bv), i in zip(BVs, idx)]
            a1 = a3 = 0
            T = N = 0
            for combo in itertools.product(*[range(v.shape[0]) for _, v in chosen]):
                w = np.array([1], dtype=np.int64)
                lo = 0
                for (l, v), j in zip(chosen, combo):
                    w = np.convolve(w, v[j]); lo += l
                W = int(w.sum())
                if W == 0:
                    continue
                srt = np.sort(w)
                mx = int(srt[-1]); m2 = int(srt[-2]) if len(w) > 1 else 0
                am = int(w.argmax())
                for iu, cu in enumerate(off):
                    if cu == 0:
                        continue
                    u = off_lo + iu
                    it = -u - lo
                    z = int(w[it]) if 0 <= it < len(w) else 0
                    a1 += cu * max(z, W - z)
                    a3 += cu * (m2 if am == it else mx)
                    T += cu * W
                    N += cu * (W - z)
            total, nonzero = T, N
            bT1 = max(bT1, a1)
            bT3 = max(bT3, a3)
    return bT1, bT3, total, nonzero


def cohomology_check():
    verts, edges = [0, 1, 2, 3], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    d0 = np.zeros((len(edges), len(verts)))
    for i, (a, b) in enumerate(edges):
        d0[i, b] += 1; d0[i, a] -= 1
    d1 = np.zeros((len(TRIS), len(edges)))
    for i, (a, b, c) in enumerate(TRIS):
        for (x, y), s in [((b, c), +1), ((a, c), -1), ((a, b), +1)]:
            d1[i, edges.index((x, y))] += s
    r0 = np.linalg.matrix_rank(d0); r1 = np.linalg.matrix_rank(d1)
    assert np.allclose(d1 @ d0, 0)
    assert np.allclose(np.array([EPS[t] for t in TRIS]) @ d1, 0)
    return (len(verts) - r0, len(edges) - r1 - r0, len(TRIS) - r1)


def run(beta, verbose=True):
    q_sigma = bits(6 * B + 1)
    q_A = bits(24 * B + 1)
    w_a = bits(6 * B + 1)
    lo_ell, ell = signed_sum(12)
    total = int(ell.sum())
    realizable = int(ell[-lo_ell])
    assert beta >= q_A, "budget below the all-agent primitive width"

    if verbose:
        print(f"--- beta = {beta}    registered window: {q_A} <= beta < {3*w_a}")
        print(f"    inputs {total}    realizable {realizable} "
              f"({realizable/total:.4%})    unrealizable {total-realizable}")
        print(f"    majority-class baseline T1 = {(total-realizable)/total:.6f}")
        print(f"    S_fill                     T1 = 1.000000   T3 = 1.000000 "
              f"   [{q_A} bits received]")

    rows = []
    for r in range(len(ST0) + 1):
        if q_sigma * r > beta:
            continue
        T = ST0[:r]
        R = [t for t in TRIS if t not in T]
        nU = 3 * r + (len(ST0) - r)
        ks = [sum(1 for t in R if v in t) for v in (1, 2, 3)]
        assert nU + sum(ks) == 12
        lo_u, U = signed_sum(nU)
        t1, t3, tot, nz = optimum(lo_u, U, [signed_sum(k) for k in ks],
                                  beta - q_sigma * r)
        rows.append((r, t1, t3, tot, nz))
        if verbose:
            tag = "S_pair == S_face(|T|=0)" if r == 0 else f"S_face, |T| = {r}   "
            print(f"    {tag}    T1 = {t1/tot:.6f}   T3 = {t3/nz:.6f}"
                  f"   [{t1}/{tot}, {t3}/{nz}]")

    best = max(rows, key=lambda x: (x[1], x[2]))
    if verbose:
        msg = ("S_face optimum attained at |T| = 0: the four registered 3-agent "
               "primitives are never used." if best[0] == 0 else
               "S_face strictly beats S_pair.")
        print("    " + msg)
    return rows


if __name__ == "__main__":
    print("H^0, H^1, H^2 of dDelta^3 with constant coefficients:",
          cohomology_check(), "\n")
    for beta in (7, 8):
        run(beta)
        print()
