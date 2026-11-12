# Experiment configurations

Each JSON file records the PDE, mesh, finite element degree, time method, solver options and numerical tolerances for one experiment. The numerical scripts validate these inputs. The [mathematical notes](../docs/mathematics.md) explain the model.

| Configuration | Purpose |
|---|---|
| `phase0_mms.json` | Mathematical benchmark specification; not a runner input |
| `phase0_be_smoke.json`, `phase0_irksome_be_crosscheck.json`, `phase0_radau_smoke.json` | One-step construction checks |
| `phase1_interpolation.json` | Initial P1/P2 interpolation on three meshes |
| `phase1_spatial.json` | Transient spatial convergence and step/tolerance controls |
| `archive/phase1_spatial-max200.json` | Exact input of the earlier unsuccessful spatial run |
| `phase2_temporal.json` | Time refinement, fine-step references and solver controls |
| `phase3_cost_pilot.json` | Preliminary timing/equivalence pilot |
| `phase3_accuracy_cost.json` | Four time-step candidates and two temporal-error targets |
| `phase3_asm_construction.json`, `phase3_asm_trajectory.json`, `phase3_asm_cost.json` | ASM verification and matched timing |
| `phase3_rana_construction.json`, `phase3_rana_trajectory.json`, `phase3_rana_cost.json` | Rana verification and matched timing |
| `phase3_solver_sensitivity.json` | Three first-step mesh/diffusion points |
| `phase3_selected_trajectories.json` | Ten-step trajectories with an evidence-linked finer-Jacobi exclusion |
| `phase3_selected_cost.json` | Final matched timing: one warmup and three repetitions per converged case |

P2 is the main space for time and cost comparisons. The final selected-cost study uses `dt=0.1`, `T=1` and `(n,kappa)=(16,1),(32,1),(16,0.1)`. Its finer-Jacobi exclusion follows a failed first-step pilot; it does not provide a successful timing sample.

Do not edit configurations to reinterpret an existing measurement. A changed experiment needs a new identified input and fresh numerical records.
