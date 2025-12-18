# Last-Modified: 2025-12-14T23:00:00+09:00

# Workspace structures for Reduced-Shifted Krylov solvers
# Following Krylov.jl conventions

export RSKrylovWorkspace, RscgWorkspace, results

"""
    RSKrylovWorkspace{T,FC,S}

Abstract type for workspaces used by reduced-shifted Krylov solvers.

# Type Parameters
- `T <: AbstractFloat`: Real floating-point type (e.g., `Float64`)
- `FC`: Float or Complex type (e.g., `ComplexF64`)
- `S <: AbstractVector{FC}`: Vector storage type

Concrete subtypes include [`RscgWorkspace`](@ref).
"""
abstract type RSKrylovWorkspace{T,FC,S} end

"""
    RscgWorkspace{T,FC,S} <: RSKrylovWorkspace{T,FC,S}

Pre-allocated workspace for the Reduced-Shifted Conjugate Gradient (RSCG) method.

Using a workspace allows efficient memory reuse when solving multiple systems
with the same dimensions but different right-hand sides or shifts.

# Type Parameters
- `T <: AbstractFloat`: Real floating-point type (e.g., `Float64`)
- `FC`: Float or Complex type (e.g., `ComplexF64`)
- `S <: AbstractVector{FC}`: Vector storage type (e.g., `Vector{ComplexF64}`)

# Constructors
```julia
RscgWorkspace(m, n, nshifts, S::Type; reduced=false, mreduced=0)
RscgWorkspace(A, b, nshifts)           # Non-reduced mode
RscgWorkspace(A, b, nshifts, V)        # Reduced mode
```

# Fields

## Problem Dimensions
- `m::Int`, `n::Int`: Matrix dimensions (must be square, m == n)
- `nshifts::Int`: Number of shift points
- `reduced::Bool`: Whether workspace is configured for reduced mode
- `mreduced::Int`: Dimension of reduced vectors (number of rows in V)

## Seed System Vectors (dimension n)
These are used for the "seed" linear system (σ=0):
- `x::S`: Solution vector
- `r::S`: Residual vector ``r_k = b - A x_k``
- `p::S`: Search direction
- `Ap::S`: Matrix-vector product ``A p_k``

## Non-Reduced Mode Arrays (dimension n × nshifts)
Used when `reduced=false`:
- `x_shifts::Vector{S}`: Solution vectors ``x(σⱼ)`` for each shift
- `p_shifts::Vector{S}`: Search directions ``p_k(σⱼ)`` for each shift

## Reduced Mode Arrays (dimension mreduced × nshifts)
Used when `reduced=true`:
- `Σ::S`: Current reduced residual ``Σ_k = V r_k``
- `Ξ::Vector{S}`: Reduced solutions ``Ξ(σⱼ) = V x(σⱼ)`` for each shift
- `Π::Vector{S}`: Reduced search directions ``Π_k(σⱼ) = V p_k(σⱼ)``

## Scalar Coefficients
- `ρ::Vector{FC}`, `ρ_prev::Vector{FC}`: Shift coefficients ``ρ_k(σⱼ)``, ``ρ_{k-1}(σⱼ)``
- `α_shifts::Vector{FC}`, `β_shifts::Vector{FC}`: Per-shift CG coefficients
- `α::Ref{FC}`, `α_prev::Ref{FC}`: Seed system ``α_k``, ``α_{k-1}``
- `β::Ref{FC}`, `β_prev::Ref{FC}`: Seed system ``β_k``, ``β_{k-1}``

## Convergence Tracking
- `rNorms::Vector{T}`: Current residual norms ``\\|r_k(σⱼ)\\|`` for each shift
- `converged::BitVector`: Convergence flags for each shift
- `not_cv::BitVector`: Not-converged flags (inverse of `converged`)
- `stagnated::BitVector`: Flags indicating ρ coefficient instability (loss of significance)

## Statistics
- `stats::ReducedShiftStats{T}`: Solver statistics (iterations, timing, status)

# Example: Workspace Reuse
```julia
using LinearAlgebra
using ReducedShiftedKrylov

n = 100
A = Hermitian(rand(ComplexF64, n, n) + rand(ComplexF64, n, n)' + 2n * I)
shifts = [0.1im, 0.2im, 0.3im]

# Create workspace once
b = rand(ComplexF64, n)
workspace = RscgWorkspace(A, b, length(shifts))

# Solve multiple systems with different right-hand sides
for i in 1:10
    b_i = rand(ComplexF64, n)
    rscg!(workspace, A, b_i, shifts)
    x_i, stats_i = results(workspace)
    # Process results...
end
```

# Memory Comparison

| Mode | Memory per shift | Total for N shifts |
|------|------------------|-------------------|
| Non-reduced | O(n) | O(N × n) |
| Reduced | O(m) | O(N × m) |

For Green's function calculations where m << n (e.g., m=10, n=10000),
reduced mode provides significant memory savings.

See also: [`rscg!`](@ref), [`rscg`](@ref), [`results`](@ref), [`ReducedShiftStats`](@ref)
"""
mutable struct RscgWorkspace{T,FC,S} <: RSKrylovWorkspace{T,FC,S}
    # Problem dimensions
    m          :: Int
    n          :: Int
    nshifts    :: Int
    reduced    :: Bool
    mreduced   :: Int

    # Seed system vectors (n-dimensional)
    x          :: S
    r          :: S
    p          :: S
    Ap         :: S

    # Non-reduced mode vectors (n-dimensional × nshifts)
    x_shifts   :: Vector{S}
    p_shifts   :: Vector{S}

    # Reduced mode vectors (mreduced-dimensional × nshifts)
    Σ          :: S
    Ξ          :: Vector{S}
    Π          :: Vector{S}

    # Scalar coefficients for shifts
    ρ          :: Vector{FC}
    ρ_prev     :: Vector{FC}
    α_shifts   :: Vector{FC}
    β_shifts   :: Vector{FC}

    # Seed scalars
    α          :: Base.RefValue{FC}
    α_prev     :: Base.RefValue{FC}
    β          :: Base.RefValue{FC}
    β_prev     :: Base.RefValue{FC}

    # Convergence tracking
    rNorms     :: Vector{T}
    converged  :: BitVector
    not_cv     :: BitVector
    stagnated  :: BitVector   # ρ coefficient instability detection

    # Statistics
    stats      :: ReducedShiftStats{T}
end

"""
    RscgWorkspace(m, n, nshifts, ::Type{S}; reduced=false, mreduced=0)

Create workspace for RSCG solver.

# Arguments
- `m`, `n`: Matrix dimensions
- `nshifts`: Number of shift values
- `S`: Vector type

# Keyword Arguments
- `reduced`: Enable reduced mode
- `mreduced`: Dimension of reduced space (required if reduced=true)
"""
function RscgWorkspace(m::Integer, n::Integer, nshifts::Integer, ::Type{S};
                       reduced::Bool=false, mreduced::Integer=0) where {S <: AbstractVector}
    FC = eltype(S)
    T = real_type(FC)

    # Seed vectors
    x  = S(undef, n)
    r  = S(undef, n)
    p  = S(undef, n)
    Ap = S(undef, n)

    if reduced
        # Reduced mode: allocate mreduced-dimensional vectors
        x_shifts = S[]
        p_shifts = S[]
        Σ = S(undef, mreduced)
        Ξ = [S(undef, mreduced) for _ in 1:nshifts]
        Π = [S(undef, mreduced) for _ in 1:nshifts]
    else
        # Non-reduced mode: allocate n-dimensional vectors for each shift
        x_shifts = [S(undef, n) for _ in 1:nshifts]
        p_shifts = [S(undef, n) for _ in 1:nshifts]
        Σ = S(undef, 0)
        Ξ = S[]
        Π = S[]
    end

    # Scalar coefficients
    ρ        = Vector{FC}(undef, nshifts)
    ρ_prev   = Vector{FC}(undef, nshifts)
    α_shifts = Vector{FC}(undef, nshifts)
    β_shifts = Vector{FC}(undef, nshifts)

    α      = Ref(one(FC))
    α_prev = Ref(one(FC))
    β      = Ref(zero(FC))
    β_prev = Ref(zero(FC))

    # Convergence
    rNorms    = Vector{T}(undef, nshifts)
    converged = BitVector(undef, nshifts)
    not_cv    = BitVector(undef, nshifts)
    stagnated = BitVector(undef, nshifts)
    fill!(stagnated, false)

    stats = ReducedShiftStats(nshifts, T)

    return RscgWorkspace{T,FC,S}(
        m, n, nshifts, reduced, mreduced,
        x, r, p, Ap,
        x_shifts, p_shifts,
        Σ, Ξ, Π,
        ρ, ρ_prev, α_shifts, β_shifts,
        α, α_prev, β, β_prev,
        rNorms, converged, not_cv, stagnated,
        stats
    )
end

"""
    RscgWorkspace(A, b, nshifts)

Create workspace for non-reduced RSCG, inferring types from A and b.
"""
function RscgWorkspace(A, b::AbstractVector, nshifts::Integer)
    m, n = size(A)
    S = typeof(b)
    return RscgWorkspace(m, n, nshifts, S; reduced=false)
end

"""
    RscgWorkspace(A, b, nshifts, V)

Create workspace for reduced RSCG with reduction matrix V.
"""
function RscgWorkspace(A, b::AbstractVector, nshifts::Integer, V::AbstractMatrix)
    m, n = size(A)
    mreduced = size(V, 1)
    S = typeof(b)
    return RscgWorkspace(m, n, nshifts, S; reduced=true, mreduced=mreduced)
end

"""
    results(workspace::RscgWorkspace) -> (solutions, stats)

Extract solutions and statistics from a solved workspace.

# Arguments
- `workspace::RscgWorkspace`: Workspace after calling `rscg!`

# Returns
For non-reduced mode (`workspace.reduced == false`):
- `x::Vector{Vector{FC}}`: Solution vectors, where `x[j]` solves ``(σⱼ I + A) x = b``
- `stats::ReducedShiftStats{T}`: Solver statistics

For reduced mode (`workspace.reduced == true`):
- `Ξ::Vector{Vector{FC}}`: Reduced solutions, where ``Ξ[j] = V x(σⱼ)``
- `stats::ReducedShiftStats{T}`: Solver statistics

# Example
```julia
workspace = RscgWorkspace(A, b, length(shifts))
rscg!(workspace, A, b, shifts)
x, stats = results(workspace)

println("Solved: \$(stats.solved)")
println("Iterations: \$(stats.niter)")
println("Time: \$(stats.timer) seconds")
```

# Note
The returned vectors are references to the workspace's internal storage.
If you need to keep the results while reusing the workspace, make a copy:
```julia
x_copy = deepcopy(x)
```

See also: [`rscg!`](@ref), [`RscgWorkspace`](@ref), [`ReducedShiftStats`](@ref)
"""
function results(workspace::RscgWorkspace)
    if workspace.reduced
        return (workspace.Ξ, workspace.stats)
    else
        return (workspace.x_shifts, workspace.stats)
    end
end
