# Research log

| Date | Work and outcome |
|---|---|
| 7 October 2026 | Defined the smooth manufactured benchmark and narrowed the study to existing implicit methods and preconditioners. Built the native Python/MPI/PETSc stack. Corrected a Python build-dependency conflict. |
| 8 October 2026 | Verified runtime bindings, small serial/MPI cases, manual/Irksome backward Euler and Radau stage construction. Verified initial interpolation and transient spatial convergence after preserving an earlier iteration-limit attempt. |
| 18 October 2026 | Verified same-mesh BE/Radau temporal orders with reference and tolerance controls. Checked timing instrumentation; battery-power pilot timings remain preliminary. |
| 19 October 2026 | Completed the four-candidate AC accuracy-cost study. Verified standard overlapping ASM/local ILU(0), solution agreement and factor reuse over full trajectories. |
| 21 October 2026 | Measured matched ASM costs. Corrected a Rana form-interface mismatch while retaining its failed first attempt. Verified Rana construction, full trajectories and repeated costs. Tested the three selected mesh/diffusion points; retained finer-mesh Jacobi's iteration cap. |
| 22 October 2026 | Completed the common-path selected-cost study: 43 trajectories and 430 steps, with all timing samples retained. Prepared the original technical report. |
| 12 November 2026 | Following the author's report review, rewrote the report in simpler language, consolidated documentation, and added a portable saved-data check. Numerical implementations, configurations and measurements were preserved. Added the MIT license for original project code/documentation and retained upstream terms for the vendored helper. Prepared a clean source archive; public visibility remains a separate action. |
