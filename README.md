# TSI Simulation

Reproducible Python simulations for the finite benchmarks in "Topological Signatures of Irreducibility".

Two independent benchmarks are included, one per degree:

| File | Benchmark | Complex |
| --- | --- | --- |
| `benchmark_deg1.py` | Degree-1, `C_m` ring (Section 9.1) | `C_6` and its filled simplex |
| `benchmark_deg2.py` | Degree-2, exhaustive (Section 9.2) | `dDelta^3`, boundary of the tetrahedron |
| `fastopt.py` | Vectorized search helpers for the degree-2 case (up to two compressing senders) | — |
| `fastopt3.py` | Same, extended to three simultaneously compressing senders (`beta` in `{5,6}`) | — |

## Setup

Both scripts need only NumPy:

```bash
python3 -m pip install numpy
```

## Degree-1 benchmark (`benchmark_deg1.py`)

The script exhaustively enumerates the registered instance `m=6`, `B=2`, `beta=5`, covering all `5^6 = 15,625` inputs. It compares:

- a filled simplex with the registered all-agent joint primitive,
- the same complete 1-skeleton with only pairwise communication,
- the base ring `C_6` as a sanity check,
- a trivial-cohomology control task.

```bash
python3 benchmark_deg1.py
```

### Expected headline result

```text
filled simplex + joint primitive: T1 correct=15625 (1.000000)  T3 exact=13874/13874 (1.0000)
same 1-skeleton ablation / complete pairwise only: T1 correct=13875 (0.888000)  T3 exact=3694/13874 (0.2663)
base-ring sanity best: T1 correct=13874 (0.887936)  T3 exact=2340/13874 (0.1687)
control task, one full-value message: correct=15625/15625
```

## Degree-2 benchmark (`benchmark_deg2.py`)

The task model is `(K_T, F_T) = (dDelta^3, R)`: the boundary of the tetrahedron with constant
coefficients, so `H^0 = R`, `H^1 = 0`, `H^2 = R`. The datum is a cochain `c` in `C^2(K_T; R)`
built from 12 shares in `{-1,0,1}`, giving `3^12 = 531,441` inputs; the signature
`ell(c) = c_123 - c_023 + c_013 - c_012` generates `(im delta^1)^perp`, and the realization
task is solvable exactly when `ell(c) = 0`.

Four agents, agent 0 the coordinator, one synchronous round. Three systems are compared:

- `S_fill` — complete pairwise plus the registered all-agent primitive `g_A = ell`; the induced complex is `Delta^3`, so `H^2 = 0` and the class is filled,
- `S_face` — complete pairwise plus the four registered 3-agent primitives `g_sigma = c_sigma`; the induced complex is `dDelta^3 = K_T`, so the class extends,
- `S_pair` — complete pairwise only, induced complex `skel_1(Delta^3)`.

The search over deterministic one-round protocols is exhaustive, via the sufficient-statistic
and maximal-partition reductions described in the module docstring.

```bash
python3 benchmark_deg2.py
```

### Expected headline result

Run at two congestion levels inside the registered window `5 <= beta < 9`:

```text
H^0, H^1, H^2 of dDelta^3 with constant coefficients: (1, 0, 1)

--- beta = 7    registered window: 5 <= beta < 9
    inputs 531441    realizable 73789 (13.8847%)    unrealizable 457652
    majority-class baseline T1 = 0.861153
    S_fill                     T1 = 1.000000   T3 = 1.000000    [5 bits received]
    S_pair == S_face(|T|=0)    T1 = 0.945281   T3 = 0.719348   [502361/531441, 329211/457652]
    S_face, |T| = 1       T1 = 0.876344   T3 = 0.458130   [465725/531441, 209664/457652]
    S_face, |T| = 2       T1 = 0.861153   T3 = 0.274827   [457652/531441, 125775/457652]

--- beta = 8    registered window: 5 <= beta < 9
    inputs 531441    realizable 73789 (13.8847%)    unrealizable 457652
    majority-class baseline T1 = 0.861153
    S_fill                     T1 = 1.000000   T3 = 1.000000    [5 bits received]
    S_pair == S_face(|T|=0)    T1 = 0.968139   T3 = 0.840132   [514509/531441, 384488/457652]
    S_face, |T| = 1       T1 = 0.894430   T3 = 0.536226   [475337/531441, 245405/457652]
    S_face, |T| = 2       T1 = 0.861153   T3 = 0.338257   [457652/531441, 154804/457652]
```

Only `S_fill` is exact on both tasks. At both budgets the `S_face` optimum is attained at
`|T| = 0`, i.e. the four registered 3-agent primitives are never used: spending budget on them
costs more than the class-extending structure returns.

Note: on NumPy 2.x the cohomology line prints as `(np.int64(1), np.int64(0), np.int64(1))`.

### Lower budgets, `beta` in `{5,6}`

`benchmark_deg2.optimum` folds every sender whose budget already covers its support into the
coordinator's known offset, which leaves at most two genuinely compressing senders — enough for
`beta` in `{7,8}`. At `beta = 5` and `beta = 6` some undominated allocations, `(2,2,1)` and
`(2,2,2)`, leave three compressing senders, and the resulting triple loop over partitions
(`350^3` for `(2,2,2)`) is out of reach. `fastopt3.optimum` is a drop-in replacement that
handles that case (see below), so the remaining two budgets in the registered window run in
seconds:

```bash
python3 -c "
import benchmark_deg2 as bd, fastopt3 as f3
bd.optimum = f3.optimum
for beta in (5, 6):
    bd.run(beta)
"
```

```text
--- beta = 5    registered window: 5 <= beta < 9
    inputs 531441    realizable 73789 (13.8847%)    unrealizable 457652
    majority-class baseline T1 = 0.861153
    S_fill                     T1 = 1.000000   T3 = 1.000000    [5 bits received]
    S_pair == S_face(|T|=0)    T1 = 0.888432   T3 = 0.466649   [472149/531441, 213563/457652]
    S_face, |T| = 1       T1 = 0.861153   T3 = 0.278956   [457652/531441, 127665/457652]

--- beta = 6    registered window: 5 <= beta < 9
    inputs 531441    realizable 73789 (13.8847%)    unrealizable 457652
    majority-class baseline T1 = 0.861153
    S_fill                     T1 = 1.000000   T3 = 1.000000    [5 bits received]
    S_pair == S_face(|T|=0)    T1 = 0.929482   T3 = 0.629861   [493965/531441, 288257/457652]
    S_face, |T| = 1       T1 = 0.867199   T3 = 0.384076   [460865/531441, 175773/457652]
    S_face, |T| = 2       T1 = 0.861153   T3 = 0.238563   [457652/531441, 109179/457652]
```

The picture is the same as at `beta` in `{7,8}`: only `S_fill` is exact, and the `S_face`
optimum is again attained at `|T| = 0`.

## Notes

`fastopt.py` is a standalone module of vectorized search routines for the degree-2 case
(`one_compressor_optimum`, `two_compressor_optimum`, and the partition helpers). It relies on
the reduction that a sender whose budget `b` satisfies `2**b >= |support|` transmits its
statistic exactly, leaving at most two genuinely compressing senders. It is kept as a
cross-check on the search in `benchmark_deg2.py`, which is self-contained and does not import it.

`fastopt3.py` lifts that reduction to three simultaneously compressing senders, which is what the
`beta` in `{5,6}` allocations need. The objective is additive over block triples, and `g` depends
only on the three blocks as sets, so the module tabulates `g` once on the distinct blocks a
partition into `m` parts can use (98 blocks for `n = 7`, `m = 4`, hence `98^3` entries) and then
maximises by gathering, at `O(|P1| * |P2| * |P3| * m)` integer adds with no convolution in the
loop. Its `optimum(off_lo, off, senders, budget)` has the same signature and return tuple as
`benchmark_deg2.optimum` and agrees with it on every case small enough for the naive search to
finish, including the three-compressor `(2,2,2)` shape.
