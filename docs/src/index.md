# ReducedShiftedKrylov.jl

Reduced-Shifted Conjugate Gradient (RSCG) method for solving multiply shifted linear systems.

## Overview

This package solves the family of shifted linear systems:

```math
(A + \sigma_j I) \mathbf{x}(\sigma_j) = \mathbf{b}, \quad j = 1, 2, \ldots, M
```

where ``A`` is a Hermitian matrix and ``\sigma_j`` are complex shifts.

**Key features:**
- Solves all systems simultaneously with cost of a single system
- Reduced mode: computes ``\Xi = V \mathbf{x}`` directly (memory efficient)
- Complex shifts supported
- Krylov.jl-compatible API

## Mathematical Background

### Shift-Invariance of Krylov Subspace

The Krylov subspace is shift-invariant:

```math
\mathcal{K}_k(\sigma_j I + A, \mathbf{b}) = \mathcal{K}_k(A, \mathbf{b})
```

This property allows solving multiple shifted systems using a single Krylov subspace construction.

### RSCG Algorithm

The RSCG method extends shifted CG with:

1. **Seed system**: Standard CG iteration on reference system
2. **Shift systems**: Update using collinear residual property
   ```math
   \mathbf{r}_k(\sigma_j) = \rho_k(\sigma_j) \mathbf{r}_k
   ```
3. **Reduced mode**: Compute ``\Xi(\sigma_j) = V \mathbf{x}(\sigma_j)`` directly

### Collinear Residual Update

The shift coefficient ``\rho_k(\sigma_j)`` follows the recurrence:

```math
\rho_{k+1}(\sigma_j) = \frac{\rho_k \rho_{k-1} \alpha_{k-1}}{\rho_{k-1} \alpha_{k-1} (1 + \alpha_k \sigma_j) + \alpha_k \beta_{k-1} (\rho_{k-1} - \rho_k)}
```

## Applications

### Green's Function

```math
G_{\alpha\beta}(z) = \langle \alpha | (zI - H)^{-1} | \beta \rangle
```

Set ``A = -H``, ``\sigma = z``, ``\mathbf{b} = |\beta\rangle``, ``V = \langle\alpha|``.

### Local Density of States (LDOS)

```math
\rho_\alpha(\omega) = -\frac{1}{\pi} \mathrm{Im}\, G_{\alpha\alpha}(\omega + i\eta)
```

### Sakurai-Sugiura Method

Contour integral for eigenvalue problems uses shifted systems at quadrature points.

## References

- Y. Nagai, Y. Shinohara, Y. Futamura, T. Sakurai, "Reduced-Shifted Conjugate-Gradient Method for a Green's Function", J. Phys. Soc. Jpn. 86, 014708 (2017). [DOI:10.7566/JPSJ.86.014708](https://doi.org/10.7566/JPSJ.86.014708), [arXiv:1607.03992](https://arxiv.org/abs/1607.03992)
- R. Takayama et al., Phys. Rev. B 73, 165108 (2006). [DOI:10.1103/PhysRevB.73.165108](https://doi.org/10.1103/PhysRevB.73.165108) - Shifted COCG method
- S. Hidaka et al., Comput. Phys. Commun. 314, 109679 (2025). [DOI:10.1016/j.cpc.2025.109679](https://doi.org/10.1016/j.cpc.2025.109679) - Shifted MINRES comparison
