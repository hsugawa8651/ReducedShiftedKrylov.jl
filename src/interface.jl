# Last-Modified: 2025-12-15T09:45:00+09:00

# High-level interface for ReducedShiftedKrylov
# Following Krylov.jl conventions

export krylov_solve, krylov_workspace

"""
    rscg(A, b, shifts; kwargs...)

Out-of-place RSCG solver (non-reduced mode).

Solves the family of shifted linear systems:
    (σⱼ I + A) x(σⱼ) = b,  j = 1, 2, ..., nshifts

Returns `(x, stats)` where `x` is `Vector{Vector{FC}}` of solutions.
"""
function rscg(A, b::AbstractVector, shifts::AbstractVector;
              atol::Real = √eps(real_type(eltype(b))),
              rtol::Real = √eps(real_type(eltype(b))),
              itmax::Int = 0,
              timemax::Float64 = Inf,
              verbose::Int = 0,
              history::Bool = false,
              callback = workspace -> false)

    T = real_type(eltype(b))
    nshifts = length(shifts)
    workspace = RscgWorkspace(A, b, nshifts)

    rscg!(workspace, A, b, shifts;
          atol = T(atol),
          rtol = T(rtol),
          itmax = itmax,
          timemax = timemax,
          verbose = verbose,
          history = history,
          callback = callback)

    return results(workspace)
end

"""
    rscg(A, b, shifts, V; kwargs...)

Out-of-place RSCG solver (reduced mode).

Solves the family of shifted linear systems and returns reduced solutions:
    Ξ(σⱼ) = V * x(σⱼ),  j = 1, 2, ..., nshifts

Returns `(Ξ, stats)` where `Ξ` is `Vector{Vector{FC}}` of reduced solutions.
"""
function rscg(A, b::AbstractVector, shifts::AbstractVector, V::AbstractMatrix;
              atol::Real = √eps(real_type(eltype(b))),
              rtol::Real = √eps(real_type(eltype(b))),
              itmax::Int = 0,
              timemax::Float64 = Inf,
              verbose::Int = 0,
              history::Bool = false,
              callback = workspace -> false)

    T = real_type(eltype(b))
    nshifts = length(shifts)
    workspace = RscgWorkspace(A, b, nshifts, V)

    rscg!(workspace, A, b, shifts, V;
          atol = T(atol),
          rtol = T(rtol),
          itmax = itmax,
          timemax = timemax,
          verbose = verbose,
          history = history,
          callback = callback)

    return results(workspace)
end

"""
    krylov_solve(Val(:rscg), A, b, shifts; kwargs...) -> (x, stats)
    krylov_solve(Val(:rscg), A, b, shifts, V; kwargs...) -> (Ξ, stats)

Generic interface for reduced-shifted Krylov solvers.

This provides a unified API for selecting different Krylov methods at runtime
using a `Val` type selector. Currently supports:
- `Val(:rscg)`: Reduced-Shifted Conjugate Gradient

# Arguments
- `Val(:rscg)`: Solver selector (use `Val(:rscg)` for RSCG)
- `A`: Hermitian positive semi-definite linear operator
- `b::AbstractVector`: Right-hand side vector
- `shifts::AbstractVector`: Shift values ``σⱼ``
- `V::AbstractMatrix`: (optional) Reduction matrix for reduced mode

# Keyword Arguments
See [`rscg!`](@ref) for the full list of keyword arguments.

# Returns
Same as [`rscg`](@ref):
- `x::Vector{Vector{FC}}`: Solution vectors (non-reduced mode)
- `Ξ::Vector{Vector{FC}}`: Reduced solutions (reduced mode)
- `stats::ReducedShiftStats`: Solver statistics

# Example
```julia
using ReducedShiftedKrylov

# Select solver at runtime
solver = :rscg
x, stats = krylov_solve(Val(solver), A, b, shifts)
```

# Extensibility
To add a new solver (e.g., `:rsbicg`), define:
```julia
krylov_solve(::Val{:rsbicg}, A, b, shifts; kwargs...) = rsbicg(A, b, shifts; kwargs...)
```

See also: [`rscg`](@ref), [`krylov_workspace`](@ref)
"""
krylov_solve(::Val{:rscg}, A, b, shifts; kwargs...) = rscg(A, b, shifts; kwargs...)
krylov_solve(::Val{:rscg}, A, b, shifts, V; kwargs...) = rscg(A, b, shifts, V; kwargs...)

# Error fallback for unsupported solver types
krylov_solve(::Val{S}, A, b, shifts; kwargs...) where S =
    throw(ArgumentError("Unknown solver: :$S. Supported solvers: :rscg"))
krylov_solve(::Val{S}, A, b, shifts, V; kwargs...) where S =
    throw(ArgumentError("Unknown solver: :$S. Supported solvers: :rscg"))

"""
    krylov_workspace(Val(:rscg), A, b, nshifts) -> RscgWorkspace
    krylov_workspace(Val(:rscg), A, b, nshifts, V) -> RscgWorkspace

Create a pre-allocated workspace for reduced-shifted Krylov solvers.

This provides a unified API for creating workspaces, enabling efficient
memory reuse when solving multiple systems with the same dimensions.

# Arguments
- `Val(:rscg)`: Solver selector
- `A`: Linear operator (used to infer dimensions)
- `b::AbstractVector`: Right-hand side vector (used to infer element type)
- `nshifts::Integer`: Number of shift points
- `V::AbstractMatrix`: (optional) Reduction matrix for reduced mode

# Returns
- `workspace::RscgWorkspace`: Pre-allocated workspace

# Example
```julia
using ReducedShiftedKrylov
using LinearAlgebra

n = 100
A = Hermitian(rand(ComplexF64, n, n) + rand(ComplexF64, n, n)' + 2n * I)
b = rand(ComplexF64, n)
shifts = [0.1im, 0.2im, 0.3im]

# Create workspace once
workspace = krylov_workspace(Val(:rscg), A, b, length(shifts))

# Reuse for multiple solves
for i in 1:10
    b_i = rand(ComplexF64, n)
    rscg!(workspace, A, b_i, shifts)
    x_i, stats_i = results(workspace)
    # Process results...
end
```

# Reduced Mode Example
```julia
# Create workspace for reduced mode
m = 5
V = zeros(ComplexF64, m, n)
for i in 1:m
    V[i, i] = 1.0
end

workspace = krylov_workspace(Val(:rscg), A, b, length(shifts), V)
rscg!(workspace, A, b, shifts, V)
Ξ, stats = results(workspace)
```

See also: [`RscgWorkspace`](@ref), [`rscg!`](@ref), [`krylov_solve`](@ref)
"""
krylov_workspace(::Val{:rscg}, A, b, nshifts::Integer) = RscgWorkspace(A, b, nshifts)
krylov_workspace(::Val{:rscg}, A, b, nshifts::Integer, V) = RscgWorkspace(A, b, nshifts, V)

# Error fallback for unsupported solver types
krylov_workspace(::Val{S}, A, b, nshifts::Integer) where S =
    throw(ArgumentError("Unknown solver: :$S. Supported solvers: :rscg"))
krylov_workspace(::Val{S}, A, b, nshifts::Integer, V) where S =
    throw(ArgumentError("Unknown solver: :$S. Supported solvers: :rscg"))
