# Mathematical notes

The report gives a slower introduction. These notes collect the equations needed to read the code.

## Problem

On the unit square, solve

\[
u_t + \boldsymbol\beta\cdot\nabla u - \kappa\Delta u=f,
\qquad \boldsymbol\beta=(1,1),\quad \kappa>0,
\]

with zero boundary values and final time `T=1`. Choose the exact test solution

\[
u_\star=e^{-t}\sin(\pi x)\sin(\pi y).
\]

The manufactured source follows by substitution:

\[
f=e^{-t}\big[(2\kappa\pi^2-1)\sin(\pi x)\sin(\pi y)
+\pi\cos(\pi x)\sin(\pi y)+\pi\sin(\pi x)\cos(\pi y)\big].
\]

Changing diffusion requires changing the source. The initial value is the exact function at `t=0`.

## Weak form and finite elements

Multiply by a zero-boundary test function `v`, integrate, and integrate diffusion by parts:

\[
(u_t,v)+(\boldsymbol\beta\cdot\nabla u,v)
+\kappa(\nabla u,\nabla v)=(f,v).
\]

Here `(a,b)` is the integral of `a*b` over the square. P1/P2 continuous finite elements use linear/quadratic polynomials on each triangle. Writing `u_h = sum_j U_j phi_j` gives

\[
M\dot U+LU=F(t),\qquad
M_{ij}=(\phi_j,\phi_i),\quad
L_{ij}=(\boldsymbol\beta\cdot\nabla\phi_j,\phi_i)
+\kappa(\nabla\phi_j,\nabla\phi_i),\quad F_i=(f,\phi_i).
\]

The derivation uses free coefficients; assembled code retains constrained boundary slots. A mesh with `n` subdivisions on each edge has `2*n*n` triangles and maximum diameter `h=sqrt(2)/n`.

## Time methods

With time level indexed by `k`, backward Euler solves

\[
(M+\Delta tL)U^{k+1}=MU^k+\Delta tF(t_{k+1}).
\]

Two-stage Radau IIA uses

\[
A=\begin{pmatrix}5/12&-1/12\\3/4&1/4\end{pmatrix},\quad
b=(3/4,1/4)^T,\quad c=(1/3,1)^T.
\]

Its derivative stages solve

\[
MZ_i+L\left(U^k+\Delta t\sum_j a_{ij}Z_j\right)=F(t_k+c_i\Delta t),
\]

or `BZ=g` with `B = I tensor M + dt*A tensor L`. The update is

\[
U^{k+1}=U^k+\Delta t(3Z_1/4+Z_2/4).
\]

The last tableau row equals the update weights, so the last stage state equals the final solution (stiff accuracy). BE is first order and Radau is third order. Observed orders need spatial, solver and reference errors to be small enough.

## Linear solution

GMRES solves the nonsymmetric stage equations. Right preconditioning means solving `B P^-1 y=g` and then applying `Z=P^-1 y`; the inverse is an action, not an explicitly stored dense matrix.

Jacobi scales by the diagonal. ASM adds corrections from four overlapping blocks, each solved with ILU(0). RanaLD replaces the small tableau by

\[
\widetilde A=L_A D_A=\begin{pmatrix}5/12&0\\3/4&2/5\end{pmatrix},
\qquad P=I\otimes M+\Delta t\widetilde A\otimes L.
\]

The present Rana configuration uses full sparse LU on `P`. The physical equation still uses the original `A`. Each preconditioner action is fixed and linear, so ordinary GMRES is suitable.

## Errors and convergence

The `L2` norm measures solution-value error; the `H1` seminorm measures gradient error. Spatial errors compare the computed solution with `u_star`. Time errors compare solutions on the same mesh with a fine-step numerical reference. For two successive time steps,

\[
p=\log(E(\Delta t)/E(\Delta t/2))/\log2.
\]

Expected smooth-solution spatial orders are P1: `L2/H1 = 2/1`, and P2: `3/2`. The analytic exact-function norms are `exp(-t)/2` and `pi*exp(-t)/sqrt(2)` respectively. Numerical integration and solver tolerances are checked separately. A small linear residual is not an error bound for the PDE solution.
