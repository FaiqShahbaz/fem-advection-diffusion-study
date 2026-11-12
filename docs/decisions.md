# Main design choices

| Choice | Reason |
|---|---|
| Use a manufactured exact solution | Numerical error can be measured directly and the forcing can be checked by substitution. |
| Verify P1 and P2; use P2 for later comparisons | Both spatial implementations are checked; P2 lowers spatial error enough to study time error separately. |
| Compare BE and two-stage Radau IIA | They provide a clear first-order/third-order comparison with different per-step costs. |
| Check finer time references and tighter solves | A time-error curve should not be determined by reference, quadrature or algebraic error. |
| Compare existing Jacobi, ASM/ILU(0) and RanaLD | These expose diagonal, spatial-block and stage-based approaches without claiming a new preconditioner. |
| Use one common residual-form path for final PC timings | Construction and solve work have the same measurement boundary within each parameter point. |
| Preserve failures and every timing repeat | Exclusions and variability are part of the evidence. |
| Keep plotting separate from the numerical environment | Figure generation does not modify the installed numerical packages. |
| Keep serial and parallel claims separate | The current solver/factor configuration has only been benchmarked serially. |
