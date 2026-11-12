# References and software

The report cites the methods it uses. [The BibTeX file](../references.bib) includes additional background sources.

- [Firedrake](https://doi.org/10.1145/2998441): finite-element abstractions and assembly.
- [Irksome](https://arxiv.org/abs/2006.16282): Runge-Kutta forms generated from a semi-discrete finite-element problem.
- [Extending Irksome](https://arxiv.org/abs/2403.08084v2): stage formulations, boundary treatment and preconditioning support.
- [Rana et al.](https://arxiv.org/abs/2010.11377v2): existing block LDU-based preconditioning for implicit Runge-Kutta systems, including advection-diffusion tests.
- [Official Rana demonstration](https://www.firedrakeproject.org/Irksome/demos/demo_heat_rana.py.html): configuring the stage-aware auxiliary operator. The present study instead uses full auxiliary sparse LU.
- [Hairer and Wanner](https://doi.org/10.1007/978-3-642-05221-7): stiff ordinary differential equations and implicit Runge-Kutta theory.
- PETSc [GMRES](https://petsc.org/release/manualpages/KSP/KSPGMRES/), [ASM](https://petsc.org/release/manualpages/PC/PCASM/) and [ILU](https://petsc.org/release/manualpages/PC/PCILU/): algebraic solvers and options.
- [Manufactured solutions](https://www.osti.gov/biblio/759450): the verification approach used to prescribe an exact test solution and derive its source.

Online documentation can change. Installed implementation versions and source hashes used for the measurements are recorded under `environment/` and in the run manifests.
