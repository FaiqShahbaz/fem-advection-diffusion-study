# Numerical implementation

Firedrake constructs the spatial finite-element problem, Irksome generates implicit Runge-Kutta stages, and PETSc solves the linear systems. All experiments use the same manufactured advection-diffusion problem.

The files keep their original phase names because run manifests and source hashes refer to them. Earlier modules provide shared helpers used by later experiments; they are not redundant copies to delete.

| Files | Purpose |
|---|---|
| `phase0_be_smoke.py`, `phase0_irksome_be.py`, `phase0_radau_smoke.py` | Manual BE, independent BE comparison, and Radau construction |
| `phase1_interpolation.py`, `phase1_spatial.py` | Initial representation and transient spatial convergence |
| `phase2_temporal.py` | Same-mesh BE/Radau temporal convergence and reference controls |
| `phase3_cost_pilot.py`, `phase3_accuracy_cost.py` | Timing instrumentation and the four accuracy-cost candidates |
| `phase3_asm_construction.py`, `phase3_asm_trajectory.py`, `phase3_asm_cost.py` | ASM construction, full-time agreement/reuse and matched manual timings |
| `phase3_rana_construction.py`, `phase3_rana_trajectory.py`, `phase3_rana_cost.py` | Rana auxiliary operator, full trajectories/reuse and matched residual-form timings |
| `phase3_solver_sensitivity.py` | Three first-step parameter points, including the preserved Jacobi cap |
| `phase3_selected_trajectories.py` | Eight converged candidate trajectories and three physical-LU references |
| `phase3_selected_cost.py` | Common-path repeated preconditioner timing at the same points |

Use the wrappers in `environment/` for numerical reruns: they select the recorded installation, check prerequisites and create timestamped manifests. Follow [the reproduction guide](../docs/reproducibility.md). The source files alone do not set all native library/thread settings.

The final source, configurations and original measurements are unchanged by the November documentation revision. New parameter choices or solver changes require separate verification and new records.
