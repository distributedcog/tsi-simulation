"""fastopt3.py -- exact one-round optimum with up to THREE simultaneously
compressing senders (degree-2 benchmark, Section 9.2).

`fastopt.py` folds every sender whose budget already exceeds its support width
into the coordinator's known offset, which leaves at most two genuine
compressors.  That is enough for beta in {7,8}; for beta in {5,6} some
undominated allocations -- (2,2,1) at beta = 5 and (2,2,2) at beta = 6 -- leave
three compressing senders, and the partition triple count (350^3 = 4.3e7 for
(2,2,2)) makes the naive triple loop of `benchmark_deg2.optimum` hopeless.

Reduction used here.  The objective is additive over block triples:

    F(P1,P2,P3) = sum_{B1 in P1, B2 in P2, B3 in P3} g(B1,B2,B3),

where g depends only on the three blocks as *sets*, not on the partitions that
contain them.  A partition into m blocks only ever uses blocks of size at most
n-m+1, so the number of distinct blocks is small (98 for n = 7, m = 4).  We
therefore tabulate g once on the block triples (98^3 = 9.4e5 entries) and then
maximise the sum by gathering, which costs O(|P1| * |P2| * |P3| * m) integer
adds with no convolution inside the loop.  Exactly the same optima as
`benchmark_deg2.optimum`, just reachable.

T1 and T3 are maximised independently, as in the text.
"""
import itertools, math
import numpy as np


def bits(n):
    return 0 if n <= 1 else math.ceil(math.log2(n))


def exact_partitions(n, blocks):
    """All set partitions of range(n) into exactly `blocks` blocks."""
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


def _blocks_and_partitions(counts, m):
    """Distinct blocks used by the partitions of the support into exactly m
    blocks, as count vectors, plus the (partition -> block id) matrix."""
    n = len(counts)
    m = min(m, n)
    P = exact_partitions(n, m)
    index, rows = {}, []
    M = np.zeros((len(P), m), dtype=np.intp)
    for p, a in enumerate(P):
        cells = [[] for _ in range(m)]
        for v, b in enumerate(a):
            cells[b].append(v)
        for b, elems in enumerate(cells):
            key = tuple(elems)
            if key not in index:
                index[key] = len(rows)
                vec = np.zeros(n, dtype=np.int64)
                for v in elems:
                    vec[v] = counts[v]
                rows.append(vec)
            M[p, b] = index[key]
    return np.array(rows, dtype=np.int64), M


def _pair_table(b1, b2):
    """(N1, N2, n1+n2-1) convolution of every block of sender 1 with every
    block of sender 2."""
    n1, n2 = b1.shape[1], b2.shape[1]
    C = np.zeros((b1.shape[0], b2.shape[0], n1 + n2 - 1), dtype=np.int64)
    for x in range(n1):
        col = b1[:, x]
        if not col.any():
            continue
        for y in range(n2):
            row = b2[:, y]
            if not row.any():
                continue
            C[:, :, x + y] += np.outer(col, row)
    return C


def _g_tables(off_lo, off, comps):
    """g_T1, g_T3 indexed by (block of 1, block of 2, block of 3)."""
    (lo1, b1), (lo2, b2), (lo3, b3) = comps
    lo_tot = lo1 + lo2 + lo3
    C12 = _pair_table(b1, b2)
    N1, N2, L12 = C12.shape
    C12 = C12.reshape(N1 * N2, L12)
    N3, n3 = b3.shape
    L = L12 + n3 - 1

    g1 = np.zeros((N1 * N2, N3), dtype=np.int64)
    g3 = np.zeros((N1 * N2, N3), dtype=np.int64)
    us = [(off_lo + i, int(c)) for i, c in enumerate(off) if c]

    full = np.empty((N1 * N2, L), dtype=np.int64)
    for s3 in range(N3):
        full[:] = 0
        for y in range(n3):
            w = int(b3[s3, y])
            if w:
                full[:, y:y + L12] += w * C12
        W = full.sum(axis=1)
        srt = np.sort(full, axis=1)
        mx = srt[:, -1]
        m2 = srt[:, -2] if L > 1 else np.zeros_like(mx)
        am = full.argmax(axis=1)
        a1 = np.zeros(N1 * N2, dtype=np.int64)
        a3 = np.zeros(N1 * N2, dtype=np.int64)
        for u, cu in us:
            it = -u - lo_tot
            if 0 <= it < L:
                z = full[:, it]
                a1 += cu * np.maximum(z, W - z)
                a3 += cu * np.where(am == it, m2, mx)
            else:
                a1 += cu * W
                a3 += cu * mx
        g1[:, s3] = a1
        g3[:, s3] = a3
    return g1.reshape(N1, N2, N3), g3.reshape(N1, N2, N3)


def _best_over_partitions(g, M1, M2, M3):
    """max over partition triples of the sum of g over their block triples."""
    best = None
    for p1 in range(M1.shape[0]):
        G1 = g[M1[p1]].sum(axis=0)                 # (N2, N3)
        H = G1[M2].sum(axis=1)                     # (|P2|, N3)
        S = H[:, M3[:, 0]]
        if M3.shape[1] > 1:
            S = S.copy()
            for j in range(1, M3.shape[1]):
                S += H[:, M3[:, j]]
        v = int(S.max())
        best = v if best is None or v > best else best
    return best


def _alloc_optimum(off_lo, off, senders, alloc):
    """Exact optimum for one receive-bit allocation."""
    off_lo, off = int(off_lo), np.asarray(off, dtype=np.int64)
    comps = []
    for (lo, counts), b in zip(senders, alloc):
        counts = np.asarray(counts, dtype=np.int64)
        n = len(counts)
        if 2 ** b >= n:                     # exact: fold into the known offset
            off = np.convolve(off, counts)
            off_lo += lo
        else:
            comps.append((lo, counts, 2 ** b))
    while len(comps) < 3:                   # pad with a constant sender
        comps.append((0, np.array([1], dtype=np.int64), 1))
    if len(comps) > 3:
        raise ValueError("more than three compressing senders")

    packed, Ms = [], []
    for lo, counts, m in comps:
        blocks, M = _blocks_and_partitions(counts, m)
        packed.append((lo, blocks))
        Ms.append(M)
    g1, g3 = _g_tables(off_lo, off, packed)
    t1 = _best_over_partitions(g1, *Ms)
    t3 = _best_over_partitions(g3, *Ms)
    return t1, t3


def optimum(off_lo, off, senders, budget):
    """Drop-in replacement for benchmark_deg2.optimum (same return tuple)."""
    off = np.asarray(off, dtype=np.int64)
    caps = [bits(len(c)) for _, c in senders]
    allocs = [a for a in itertools.product(*[range(c + 1) for c in caps])
              if sum(a) <= budget]
    maximal = [a for a in allocs
               if not any(b != a and all(y >= x for y, x in zip(b, a))
                          for b in allocs)]

    # totals are partition-independent: convolve everything
    dist, lo = off, int(off_lo)
    for l, c in senders:
        dist = np.convolve(dist, np.asarray(c, dtype=np.int64))
        lo += l
    total = int(dist.sum())
    z = int(dist[-lo]) if 0 <= -lo < len(dist) else 0
    nonzero = total - z

    bT1 = bT3 = 0
    for al in maximal:
        t1, t3 = _alloc_optimum(off_lo, off, senders, al)
        bT1 = max(bT1, t1)
        bT3 = max(bT3, t3)
    return bT1, bT3, total, nonzero
