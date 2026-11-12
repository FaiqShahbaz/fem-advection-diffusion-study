# Study scope

The completed study concerns one smooth scalar advection-diffusion equation on the unit square, with velocity `(1,1)`, positive constant diffusion, zero boundary values and a manufactured exact solution. It uses continuous triangular finite elements and runs to `T=1`.

## Completed comparisons

1. Initial interpolation and transient P1/P2 spatial convergence on `n=8,16,32`.
2. Backward Euler versus two-stage Radau IIA temporal convergence on P2/`n=16`.
3. Four selected time-step choices at two temporal-error targets.
4. Jacobi, overlapping ASM/ILU(0), and existing RanaLD with full auxiliary sparse LU.
5. Three selected solver points: `(n,kappa)=(16,1),(32,1),(16,0.1)`, with full ten-step trajectories and repeated matched timing for converged candidates.

The finer-mesh Jacobi first step reached its fixed 1000-iteration limit. Its failed outcome is retained; no successful trajectory or runtime is assigned to it.

## Limits

Performance measurements use one MPI process and one thread on one laptop. Early multi-process checks establish correctness for their small cases, not parallel scaling. Rana's full auxiliary LU is not a scalable multigrid stage solver. Three timing repetitions show observed ranges; they do not provide confidence intervals.

The accuracy-cost targets concern time error on a fixed mesh. They are not total PDE error targets, and the four candidate steps do not establish an optimal step. Sharp fronts, stabilization, adaptive steps, new preconditioners, three-dimensional problems and cluster scaling are outside this study.

The result is a reproducible technical report about existing methods. A journal extension needs a distinct, literature-supported question. No additional numerical study is needed to substantiate the present bounded conclusions.
