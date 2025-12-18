# ReducedShiftedKrylov.jl

[![CI](https://github.com/hsugawa8651/ReducedShiftedKrylov.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/hsugawa8651/ReducedShiftedKrylov.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/hsugawa8651/ReducedShiftedKrylov.jl/actions/workflows/Documentation.yml/badge.svg)](https://hsugawa8651.github.io/ReducedShiftedKrylov.jl/dev/)

Reduced-Shifted Conjugate Gradient (RSCG) method for solving multiply shifted linear systems with Hermitian matrices.

## Overview

Solves `(σⱼI + A) x(σⱼ) = b` for multiple shifts σ₁, σ₂, ..., σₘ simultaneously, with the same computational cost as a single system.

**Key feature**: Reduced mode computes `Ξ = V * x` directly without storing full solution vectors, enabling memory-efficient Green's function calculations.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/hsugawa8651/ReducedShiftedKrylov.jl")
```

## Quick Start

```julia
using LinearAlgebra
using ReducedShiftedKrylov

# Hermitian matrix and right-hand side
n = 100
H = rand(ComplexF64, n, n)
A = Hermitian(H + H' + 2n * I)
b = rand(ComplexF64, n)

# Multiple shifts (can be complex)
shifts = [0.1im, 0.2im, 0.3im, 0.4im, 0.5im]

# Solve all systems simultaneously
x, stats = rscg(A, b, shifts)

# x[j] is the solution for (A + shifts[j]*I) x = b
```

### Reduced Mode (for Green's function)

```julia
# Reduction matrix: extract specific components
m = 5  # reduced dimension
V = zeros(ComplexF64, m, n)
for i in 1:m
    V[i, i] = 1.0
end

# Compute Ξ = V * x directly (memory efficient)
Ξ, stats = rscg(A, b, shifts, V)

# Ξ[j] ∈ C^m instead of x[j] ∈ C^n
```

## References

- Y. Nagai et al., "Reduced-Shifted Conjugate-Gradient Method for a Green's Function", J. Phys. Soc. Jpn. 86, 014708 (2017). [DOI:10.7566/JPSJ.86.014708](https://doi.org/10.7566/JPSJ.86.014708), [arXiv:1607.03992](https://arxiv.org/abs/1607.03992)

## License

MIT License
