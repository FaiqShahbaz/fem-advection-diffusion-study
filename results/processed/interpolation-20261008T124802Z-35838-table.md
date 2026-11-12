# Measured initial interpolation errors

Researcher data at t=0; primary quadrature degree 10. h is the maximum triangle diameter.

| Space | n | Total DOFs | h | L2 error | L2 rate | H1 seminorm error | H1 rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 8 | 81 | 0.176776695 | 1.555346831e-02 | — | 4.328319497e-01 | — |
| P1 | 16 | 289 | 0.088388348 | 3.923151886e-03 | 1.987151 | 2.176696101e-01 | 0.991667 |
| P1 | 32 | 1089 | 0.044194174 | 9.829737581e-04 | 1.996788 | 1.089922187e-01 | 0.997915 |
| P2 | 8 | 289 | 0.176776695 | 5.469140992e-04 | — | 3.356942168e-02 | — |
| P2 | 16 | 1089 | 0.088388348 | 6.871217524e-05 | 2.992677 | 8.431481022e-03 | 1.993290 |
| P2 | 32 | 4225 | 0.044194174 | 8.599931904e-06 | 2.998169 | 2.110323498e-03 | 1.998322 |

Rates use adjacent meshes and log(E_coarse/E_fine)/log(h_coarse/h_fine).

[Raw norms and rates](../raw/interpolation-20261008T124802Z-35838-serial.json); [accepted review](interpolation_review.json).

Expected interpolation orders: P1 L2/H1 seminorm = 2/1; P2 = 3/2.
This dataset does not establish transient Galerkin convergence or temporal order.
