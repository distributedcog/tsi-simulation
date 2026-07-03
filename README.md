# TSI Simulation

Reproducible Python simulation for the finite benchmark in "Topological Signatures of Irreducibility".

The script exhaustively enumerates the registered instance `m=6`, `B=2`, `beta=5`, covering all `5^6 = 15,625` inputs. It compares:

- a filled simplex with the registered all-agent joint primitive,
- the same complete 1-skeleton with only pairwise communication,
- the base ring `C_6` as a sanity check,
- a trivial-cohomology control task.

## Run

```bash
python3 -m pip install numpy
python3 benchmark_sim.py
```

## Expected headline result

```text
filled simplex + joint primitive: T1 correct=15625 (1.000000)  T3 exact=13874/13874 (1.0000)
same 1-skeleton ablation / complete pairwise only: T1 correct=13875 (0.888000)  T3 exact=3694/13874 (0.2663)
base-ring sanity best: T1 correct=13874 (0.887936)  T3 exact=2340/13874 (0.1687)
control task, one full-value message: correct=15625/15625
```

