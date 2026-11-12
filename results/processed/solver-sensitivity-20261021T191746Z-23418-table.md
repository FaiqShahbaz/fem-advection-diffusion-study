# Selected first-step solver sensitivity

CG P2, beta=(1,1), dt=0.1; only [0,0.1], one process/thread, no elapsed-cost measurement. Full research T=1 is not completed by this pilot.

| Point | n | kappa | Coupled size | Jacobi | ASM / ILU(0) | RanaLD / auxiliary LU | Rana factor nonzeros |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16 | 1 | 2178 | 734 | 60 | 7 | 167432 |
| mesh_refined | 32 | 1 | 8450 | 1000 (cap; not converged) | 150 | 7 | 1194936 |
| lower_diffusion | 16 | 0.1 | 2178 | 228 | 35 | 8 | 167432 |

Eight candidate solutions and all three independent physical-LU references passed. Finer-mesh Jacobi reached 1000 iterations with reason -3; no update or physical slope field was accepted. Its recomputed KSP correction relative residual was 4.107492169e−6; that is not the independent physical residual of a completed step.
Rana auxiliary factor nonzeros grew by 7.136844 from n16 to n32, while its first-step count stayed seven. Factor entries are a storage proxy, not measured bytes or elapsed time.
The LU reference uses PREONLY: displayed default KSP tolerances are not its stopping rule. Its physical residual is checked separately against max(1e−14,1e−12 norm(g)) with the fixed allowance.
ASM physical MatSolve counts corroborate four local actions per outer PCApply. Its additive probe includes one global action and four independent direct local actions; those eight diagnostic MatSolve operations are separate from physical work.
A PETSc MatLUFactorNum event can include the local ILU numeric factor work here; actual recorded PC types identify ILU(0) versus full auxiliary LU.
No mesh-independent bound, full-time accuracy, elapsed-cost ranking or scalable-subsolve result is established.

[Raw record](../raw/solver-sensitivity-20261021T191746Z-23418-serial.json); [accepted review](solver_sensitivity_review.json).
