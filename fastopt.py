"""Vectorized exhaustive search for the degree-2 one-round benchmark.

Key reduction (proved in the text): a sender whose budget b satisfies
2**b >= |support| transmits its statistic exactly, so its partition is the
discrete one and its value can be folded into the coordinator's known offset.
This leaves at most two genuinely compressing senders, which is what makes the
exhaustive search tractable.
"""
import numpy as np, itertools, json, math


def bits(n):
    return 0 if n <= 1 else math.ceil(math.log2(n))


def signed_sum_counts(k, B):
    cur = np.array([1], dtype=np.int64)
    unit = np.ones(2 * B + 1, dtype=np.int64)
    for _ in range(k):
        cur = np.convolve(cur, unit)
    lo = -k * B
    return lo, cur


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


def block_vectors(counts, assigns, blocks):
    """(len(assigns), blocks, len(counts)) block-restricted count vectors."""
    n = len(counts)
    out = np.zeros((len(assigns), blocks, n), dtype=np.int64)
    for p, a in enumerate(assigns):
        for v, b in enumerate(a):
            out[p, b, v] = counts[v]
    return out


def two_compressor_optimum(off_lo, off_counts, s_lo, s_counts, k1, k2):
    """Exhaustive optimum with exactly two compressing senders.

    off_* : coordinator's exactly known offset distribution
    s_*   : support/counts shared by both compressing senders
    k1,k2 : number of blocks allowed to each (2**b, capped at support size)
    Returns (T1_correct, T3_correct, total, nonzero).
    """
    n = len(s_counts)
    P1 = exact_partitions(n, min(k1, n)); P2 = exact_partitions(n, min(k2, n))
    BV1 = block_vectors(s_counts, P1, min(k1, n))
    BV2 = block_vectors(s_counts, P2, min(k2, n))
    n1, b1, _ = BV1.shape; n2, b2, _ = BV2.shape
    F1 = BV1.reshape(n1 * b1, n); F2 = BV2.reshape(n2 * b2, n)

    L = 2 * n - 1
    C = np.zeros((F1.shape[0], F2.shape[0], L), dtype=np.int32)
    for x in range(n):
        col = F1[:, x]
        if not col.any():
            continue
        for y in range(n):
            row = F2[:, y]
            if not row.any():
                continue
            C[:, :, x + y] += np.outer(col, row)

    Wtot = C.sum(axis=2)
    order = np.sort(C, axis=2)
    m1 = order[:, :, -1]; m2 = order[:, :, -2]
    am1 = C.argmax(axis=2)

    conv_lo = 2 * s_lo
    total = 0; nonzero = 0; T1 = 0; T3 = 0
    best1 = None; best3 = None
    acc1 = np.zeros((n1, n2), dtype=np.int64)
    acc3 = np.zeros((n1, n2), dtype=np.int64)
    acc_tot = 0; acc_nz = np.zeros((n1, n2), dtype=np.int64)

    for iu, cu in enumerate(off_counts):
        if cu == 0:
            continue
        u = off_lo + iu
        it = -u - conv_lo              # index of the value that makes ell = 0
        if 0 <= it < L:
            z = C[:, :, it]
        else:
            z = np.zeros_like(Wtot)
        f1 = np.maximum(z, Wtot - z)
        f3 = np.where(am1 == it, m2, m1) if 0 <= it < L else m1
        acc1 += cu * f1.reshape(n1, b1, n2, b2).sum(axis=(1, 3))
        acc3 += cu * f3.reshape(n1, b1, n2, b2).sum(axis=(1, 3))
        acc_nz += cu * (Wtot - z).reshape(n1, b1, n2, b2).sum(axis=(1, 3))
        acc_tot += cu * int(Wtot.reshape(n1, b1, n2, b2).sum(axis=(1, 3))[0, 0])

    i1 = np.unravel_index(acc1.argmax(), acc1.shape)
    i3 = np.unravel_index(acc3.argmax(), acc3.shape)
    return int(acc1.max()), int(acc3.max()), acc_tot, int(acc_nz[i1])


def one_compressor_optimum(off_lo, off_counts, s_lo, s_counts, k):
    n = len(s_counts)
    P = exact_partitions(n, min(k, n))
    BV = block_vectors(s_counts, P, min(k, n))
    nP, nb, _ = BV.shape
    Wtot = BV.sum(axis=2)
    order = np.sort(BV, axis=2)
    m1 = order[:, :, -1]; m2 = order[:, :, -2] if n > 1 else np.zeros_like(m1)
    am1 = BV.argmax(axis=2)
    acc1 = np.zeros(nP, dtype=np.int64); acc3 = np.zeros(nP, dtype=np.int64)
    acc_nz = np.zeros(nP, dtype=np.int64); tot = 0
    for iu, cu in enumerate(off_counts):
        if cu == 0:
            continue
        u = off_lo + iu
        it = -u - s_lo
        z = BV[:, :, it] if 0 <= it < n else np.zeros_like(Wtot)
        acc1 += cu * np.maximum(z, Wtot - z).sum(axis=1)
        f3 = np.where(am1 == it, m2, m1) if 0 <= it < n else m1
        acc3 += cu * f3.sum(axis=1)
        acc_nz += cu * (Wtot - z).sum(axis=1)
        tot += cu * int(Wtot[0].sum())
    return int(acc1.max()), int(acc3.max()), tot, int(acc_nz[acc1.argmax()])
