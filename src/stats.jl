# Last-Modified: 2025-12-14T19:40:00+09:00

# Statistics structures for Reduced-Shifted Krylov solvers
# Following Krylov.jl conventions

export RSKrylovStats, ReducedShiftStats, reset!

import Base: copyto!, show

"""
    RSKrylovStats{T}

Abstract type for statistics returned by reduced-shifted Krylov solvers.

# Type Parameter
- `T <: AbstractFloat`: Real floating-point type for residual norms and timing

Concrete subtypes include [`ReducedShiftStats`](@ref).
"""
abstract type RSKrylovStats{T} end

"""
    ReducedShiftStats{T} <: RSKrylovStats{T}

Statistics returned by RSCG and related reduced-shifted Krylov methods.

This structure captures convergence information, timing, and optionally
the residual history for each shift point.

# Type Parameter
- `T <: AbstractFloat`: Real floating-point type (e.g., `Float64`)

# Fields
- `niter::Int`: Total number of iterations performed
- `solved::Bool`: `true` if all shifts converged within tolerance
- `residuals::Vector{Vector{T}}`: Residual norm history for each shift
  (only populated if `history=true` was passed to the solver)
- `indefinite::BitVector`: Flags indicating indefinite matrix detection per shift
  (reserved for future use)
- `timer::Float64`: Elapsed wall-clock time in seconds
- `status::String`: Human-readable description of the termination reason

# Status Messages
Possible values for `status`:
- `"solution good enough given atol and rtol"`: All shifts converged
- `"maximum number of iterations exceeded"`: Reached `itmax` without convergence
- `"user-requested exit"`: Callback returned `true`
- `"time limit exceeded"`: Exceeded `timemax`
- `"x = 0 is exact solution"`: Zero right-hand side vector
- `"unknown"`: Initial state before solving

# Example
```julia
x, stats = rscg(A, b, shifts; history=true)

println("Converged: \$(stats.solved)")
println("Iterations: \$(stats.niter)")
println("Time: \$(stats.timer) s")
println("Status: \$(stats.status)")

# Access residual history (if history=true)
if !isempty(stats.residuals[1])
    using Plots
    for j in eachindex(shifts)
        plot!(stats.residuals[j], label="Shift \$j", yscale=:log10)
    end
end
```

See also: [`rscg`](@ref), [`rscg!`](@ref), [`RscgWorkspace`](@ref), [`reset!`](@ref)
"""
mutable struct ReducedShiftStats{T} <: RSKrylovStats{T}
    niter      :: Int
    solved     :: Bool
    residuals  :: Vector{Vector{T}}
    indefinite :: BitVector
    timer      :: Float64
    status     :: String
end

"""
    ReducedShiftStats(nshifts::Integer, ::Type{T}) -> ReducedShiftStats{T}

Create a new `ReducedShiftStats` for `nshifts` shift points.

# Arguments
- `nshifts::Integer`: Number of shift points
- `T::Type{<:AbstractFloat}`: Real floating-point type for residual norms

# Returns
- `stats::ReducedShiftStats{T}`: Initialized statistics structure

# Example
```julia
stats = ReducedShiftStats(5, Float64)
# stats.niter == 0
# stats.solved == false
# stats.status == "unknown"
```
"""
function ReducedShiftStats(nshifts::Integer, ::Type{T}) where T <: AbstractFloat
    residuals = [T[] for _ in 1:nshifts]
    indefinite = BitVector(undef, nshifts)
    fill!(indefinite, false)
    return ReducedShiftStats{T}(0, false, residuals, indefinite, 0.0, "unknown")
end

"""
    reset!(stats::ReducedShiftStats) -> ReducedShiftStats

Reset statistics to initial state for reuse.

Clears all fields:
- `niter` → 0
- `solved` → false
- `residuals` → empty vectors
- `indefinite` → all false
- `timer` → 0.0
- `status` → "unknown"

# Example
```julia
# After solving
x, stats = rscg(A, b, shifts)
println(stats.niter)  # e.g., 42

# Reset for reuse
reset!(stats)
println(stats.niter)  # 0
```

This is called automatically at the start of `rscg!`.
"""
function reset!(stats::ReducedShiftStats)
    for vec in stats.residuals
        empty!(vec)
    end
    fill!(stats.indefinite, false)
    stats.niter = 0
    stats.solved = false
    stats.timer = 0.0
    stats.status = "unknown"
    return stats
end

function copyto!(dest::ReducedShiftStats, src::ReducedShiftStats)
    dest.niter = src.niter
    dest.solved = src.solved
    dest.residuals = deepcopy(src.residuals)
    dest.indefinite = copy(src.indefinite)
    dest.timer = src.timer
    dest.status = src.status
    return dest
end

function show(io::IO, stats::ReducedShiftStats{T}) where T
    print(io, "ReducedShiftStats{$T}")
    print(io, "(niter=$(stats.niter), solved=$(stats.solved), ")
    print(io, "nshifts=$(length(stats.residuals)), status=\"$(stats.status)\")")
end
