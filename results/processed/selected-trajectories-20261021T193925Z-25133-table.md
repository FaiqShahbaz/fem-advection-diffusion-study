# Selected full-trajectory solver work and reuse

CG P2, beta=(1,1), dt=0.1, T=1, ten steps per completed trajectory; one process/thread. Eight candidate plus three physical-LU reference trajectories passed all 110 steps. No elapsed-cost measurement.

| Point | n | kappa | Solver | Total outer iterations | Physical PCApply | Physical MatSolve | Rana factor entries |
|---|---:|---:|---|---:|---:|---:|---:|
| baseline | 16 | 1 | Jacobi | 7258 | 7408 | 0 | — |
| baseline | 16 | 1 | ASM | 600 | 620 | 2480 | — |
| baseline | 16 | 1 | RanaLD | 70 | 80 | 80 | 167432 |
| mesh_refined | 32 | 1 | Jacobi | not attempted: pilot cap | — | — | — |
| mesh_refined | 32 | 1 | ASM | 1500 | 1530 | 6120 | — |
| mesh_refined | 32 | 1 | RanaLD | 70 | 80 | 80 | 1194936 |
| lower_diffusion | 16 | 0.1 | Jacobi | 2280 | 2330 | 0 | — |
| lower_diffusion | 16 | 0.1 | ASM | 350 | 360 | 1440 | — |
| lower_diffusion | 16 | 0.1 | RanaLD | 80 | 90 | 90 | 167432 |

No new cap occurred. Finer-mesh Jacobi remains explicitly excluded because its first step failed at the unchanged 1000-iteration limit; no full trajectory is claimed.
Physical and factor object IDs/states stayed unchanged; no PC/factor rebuild occurred after step one. Each Rana trajectory used one auxiliary LU factorization; each ASM trajectory used four local ILU numeric setups. KSPSetUp counts do not count factorizations.
ASM MatSolve counts are actual local ILU actions; fixed composition corroborates four per outer PCApply. The final additive probe adds eight separate diagnostic MatSolve actions. Each final Rana probe adds one separate auxiliary action.
Rana factor entries grow 7.136844 times on refinement. They are a storage proxy, not measured bytes or elapsed time. No scalable-subsolve or elapsed-cost ranking follows.
Per-step actual update/slopes agree with independent numerical references. Final MMS errors combine spatial, temporal and algebraic effects; no new convergence order is inferred.

[Raw record](../raw/selected-trajectories-20261021T193925Z-25133-serial.json); [accepted review](selected_trajectories_review.json).
