# Last-Modified: 2025-12-14T19:30:00+09:00

# Reduced-Shifted Conjugate Gradient (RSCG) algorithm
#
# Reference:
# Y. Nagai, Y. Shinohara, Y. Futamura, T. Sakurai,
# "Reduced-Shifted Conjugate-Gradient Method for a Green's Function",
# J. Phys. Soc. Jpn. 86, 014708 (2017)

export rscg, rscg!, compute_true_residuals!

"""
    rscg!(workspace, A, b, shifts; kwargs...)
    rscg!(workspace, A, b, shifts, V; kwargs...)

In-place Reduced-Shifted Conjugate Gradient (RSCG) method.

Solves the family of shifted linear systems simultaneously:

```math
(σⱼ I + A) x(σⱼ) = b,  \\quad j = 1, 2, \\ldots, N_{\\text{shifts}}
```

The algorithm exploits the shift-invariance of Krylov subspaces:
``K_k(σI + A, b) = K_k(A, b)``, requiring only one matrix-vector product
per iteration regardless of the number of shifts.

# Arguments
- `workspace::RscgWorkspace{T,FC,S}`: Pre-allocated workspace (created by `RscgWorkspace`)
- `A`: Hermitian positive semi-definite linear operator (matrix or LinearMap)
- `b::AbstractVector{FC}`: Right-hand side vector
- `shifts::AbstractVector`: Shift values ``σⱼ`` (can be complex)
- `V::AbstractMatrix`: (optional) Reduction matrix ``V ∈ \\mathbb{C}^{m × n}`` for reduced mode.
   When provided, computes ``Ξ(σⱼ) = V x(σⱼ)`` instead of full solutions.

# Keyword Arguments
- `atol::Real=√eps(T)`: Absolute tolerance for convergence ``\\|r_k(σⱼ)\\| ≤ \\text{atol}``
- `rtol::Real=√eps(T)`: Relative tolerance ``\\|r_k(σⱼ)\\| ≤ \\text{rtol} \\cdot \\|b\\|``
- `itmax::Int=0`: Maximum iterations (0 means 2n)
- `timemax::Float64=Inf`: Time limit in seconds
- `verbose::Int=0`: Verbosity level (0=silent, k>0 prints every k iterations)
- `history::Bool=false`: If true, store residual norm history in `stats.residuals`
- `callback=workspace->false`: Early termination callback. Return `true` to stop.

# Returns
- `workspace::RscgWorkspace`: Updated workspace. Use `results(workspace)` to extract `(x, stats)`.

# Convergence
The residual for each shift is estimated as:
```math
\\|r_k(σⱼ)\\| = |ρ_k(σⱼ)| \\cdot \\|r_k\\|
```
where ``r_k`` is the seed system residual and ``ρ_k(σⱼ)`` is the shift coefficient.

# Example
```julia
using LinearAlgebra
using ReducedShiftedKrylov

# Create Hermitian positive definite matrix
n = 100
H = rand(ComplexF64, n, n)
A = Hermitian(H + H' + 2n * I)
b = rand(ComplexF64, n)
shifts = [0.1im, 0.2im, 0.3im]

# Create workspace and solve
workspace = RscgWorkspace(A, b, length(shifts))
rscg!(workspace, A, b, shifts; verbose=1)
x, stats = results(workspace)

# Verify: x[j] solves (A + shifts[j]*I) x = b
for j in eachindex(shifts)
    r = (A + shifts[j] * I) * x[j] - b
    println("Shift \$j: residual = \$(norm(r))")
end
```

# Reduced Mode Example
```julia
# Compute only V * x(σⱼ) for memory efficiency
m = 5
V = zeros(ComplexF64, m, n)
for i in 1:m
    V[i, i] = 1.0
end

workspace = RscgWorkspace(A, b, length(shifts), V)
rscg!(workspace, A, b, shifts, V)
Ξ, stats = results(workspace)
# Ξ[j] ∈ ℂᵐ instead of x[j] ∈ ℂⁿ
```

# References
- Y. Nagai et al., "Reduced-Shifted Conjugate-Gradient Method for a Green's Function",
  J. Phys. Soc. Jpn. 86, 014708 (2017). [DOI:10.7566/JPSJ.86.014708](https://doi.org/10.7566/JPSJ.86.014708)

See also: [`rscg`](@ref), [`RscgWorkspace`](@ref), [`results`](@ref)
"""
function rscg! end

"""
    rscg(A, b, shifts; kwargs...) -> (x, stats)
    rscg(A, b, shifts, V; kwargs...) -> (Ξ, stats)

Out-of-place Reduced-Shifted Conjugate Gradient (RSCG) method.

Solves the family of shifted linear systems:
```math
(σⱼ I + A) x(σⱼ) = b,  \\quad j = 1, 2, \\ldots, N_{\\text{shifts}}
```

This is a convenience wrapper that allocates a workspace internally.
For repeated solves with the same dimensions, use `rscg!` with a pre-allocated workspace.

# Arguments
- `A`: Hermitian positive semi-definite linear operator
- `b::AbstractVector`: Right-hand side vector
- `shifts::AbstractVector`: Shift values ``σⱼ``
- `V::AbstractMatrix`: (optional) Reduction matrix for reduced mode

# Keyword Arguments
See [`rscg!`](@ref) for the full list.

# Returns
- `x::Vector{Vector{FC}}`: Solution vectors, `x[j]` solves ``(σⱼ I + A) x = b``
- `stats::ReducedShiftStats`: Solver statistics including iteration count and convergence status

For reduced mode (with `V`):
- `Ξ::Vector{Vector{FC}}`: Reduced solutions, ``Ξ[j] = V x(σⱼ) ∈ \\mathbb{C}^m``
- `stats::ReducedShiftStats`: Solver statistics

# Example
```julia
using LinearAlgebra
using ReducedShiftedKrylov

n = 100
A = Hermitian(rand(n, n) + rand(n, n)' + 2n * I)
b = rand(ComplexF64, n)
shifts = [0.1 + 0.01im, 0.2 + 0.01im, 0.3 + 0.01im]

# Solve all shifted systems
x, stats = rscg(A, b, shifts)

println("Converged: \$(stats.solved)")
println("Iterations: \$(stats.niter)")
```

# Green's Function Application
For computing Green's function ``G(z) = (zI - H)^{-1}``:
```julia
# Set A = -H, shifts = z_values
# Then (σI + A)x = b becomes (zI - H)x = b
H = Hermitian(...)  # Hamiltonian
z_values = [ω + im*η for ω in ω_range]  # complex frequencies

x, stats = rscg(-H, b, z_values)
# x[j] = G(z_values[j]) * b
```

See also: [`rscg!`](@ref), [`RscgWorkspace`](@ref), [`ReducedShiftStats`](@ref)
"""
function rscg end

# Argument definitions for metaprogramming (kept for reference)
# const def_args_rscg = (:(A),
#                        :(b::AbstractVector{FC}),
#                        :(shifts::AbstractVector))

#=============================================================================
  Internal helper functions for RSCG algorithm
==============================================================================#

"""
Update shift coefficients and solutions for a single shift point.

This implements lines 12-16 of Algorithm III from Nagai et al. (2017):
- Compute ρ_{k+1}(σ_j), α_k(σ_j), β_k(σ_j)
- Update Ξ or x depending on mode
- Update Π or p depending on mode
- Check convergence

Returns the new residual norm for this shift.

# Stagnation Detection
When `|ρ_{k-1} - ρ_k|` becomes small relative to the ρ values (loss of significance),
the denominator computation loses precision. This is detected and the shift is marked
as stagnated, preserving the last reliable solution.
"""
@inline function _update_shift!(
    j::Int, σ_j::FC, α_k::FC, β_k::FC, α_prev_val::FC, β_prev_val::FC,
    rNorm_seed::T, ε::T, reduced::Bool,
    ρ::Vector{FC}, ρ_prev::Vector{FC},
    α_shifts::Vector{FC}, β_shifts::Vector{FC},
    Σ::S, Ξ::Vector{S}, Π::Vector{S},
    r::S, x_shifts::Vector{S}, p_shifts::Vector{S},
    rNorms::Vector{T}, converged::BitVector, not_cv::BitVector,
    stagnated::BitVector
) where {T <: AbstractFloat, FC, S <: AbstractVector{FC}}

    # ρ_{k+1}(σ_j) calculation (line 12)
    # ρ_{k+1} = ρ_k * ρ_{k-1} * α_{k-1} /
    #           [ρ_{k-1} * α_{k-1} * (1 + α_k * σ_j) + α_k * β_{k-1} * (ρ_{k-1} - ρ_k)]
    ρ_k = ρ[j]
    ρ_km1 = ρ_prev[j]

    # ===== Stagnation detection (loss of significance in ρ_{k-1} - ρ_k) =====
    ρ_diff = ρ_km1 - ρ_k
    ρ_scale = max(abs(ρ_km1), abs(ρ_k))

    # Skip stagnation check in initial state (both ρ values are 1.0)
    # This happens when the iteration hasn't progressed enough for ρ to evolve
    is_initial_state = (ρ_k == one(FC)) && (ρ_km1 == one(FC))

    # Check for loss of significance: when |ρ_diff| is near machine epsilon relative to |ρ|
    # Use a conservative threshold (100 * eps) to detect before severe precision loss
    stag_threshold = T(100) * eps(T) * ρ_scale

    if !is_initial_state && abs(ρ_diff) < stag_threshold && ρ_scale > eps(T)
        # ρ coefficients are stagnating - denominator computation loses precision
        stagnated[j] = true
        # Keep current solution, mark as not converged but not updating
        not_cv[j] = false  # Stop updating this shift
        return rNorms[j]
    end

    denom = ρ_km1 * α_prev_val * (one(FC) + α_k * σ_j) +
            α_k * β_prev_val * ρ_diff
    ρ_kp1 = (ρ_k * ρ_km1 * α_prev_val) / denom

    # α_k(σ_j) = (ρ_{k+1} / ρ_k) * α_k  (line 13)
    ratio = ρ_kp1 / ρ_k
    α_j = ratio * α_k

    # β_k(σ_j) = (ρ_{k+1} / ρ_k)² * β_k  (line 15)
    β_j = ratio^2 * β_k

    if reduced
        # Ξ_{k+1}(σ_j) = Ξ_k(σ_j) + α_k(σ_j) * Π_k(σ_j)  (line 14)
        axpy!(α_j, Π[j], Ξ[j])

        # Π_{k+1}(σ_j) = ρ_{k+1}(σ_j) * Σ_{k+1} + β_k(σ_j) * Π_k(σ_j)  (line 16)
        axpby!(ρ_kp1, Σ, β_j, Π[j])
    else
        # x_{k+1}(σ_j) = x_k(σ_j) + α_k(σ_j) * p_k(σ_j)
        axpy!(α_j, p_shifts[j], x_shifts[j])

        # p_{k+1}(σ_j) = ρ_{k+1}(σ_j) * r_{k+1} + β_k(σ_j) * p_k(σ_j)
        axpby!(ρ_kp1, r, β_j, p_shifts[j])
    end

    # Update ρ values
    ρ_prev[j] = ρ_k
    ρ[j] = ρ_kp1

    # Store α, β for shifts (for debugging/analysis)
    α_shifts[j] = α_j
    β_shifts[j] = β_j

    # Residual norm: ‖r_k(σ_j)‖ = |ρ_k(σ_j)| * ‖r_k‖
    rNorm_j = abs(ρ_kp1) * rNorm_seed
    rNorms[j] = rNorm_j

    # Check convergence
    converged[j] = rNorm_j ≤ ε
    not_cv[j] = !converged[j]

    return rNorm_j
end

"""
Initialize RSCG workspace for a new solve.

Sets up:
- Seed system: x_0 = 0, r_0 = p_0 = b
- Scalar coefficients: α_{-1} = 1, β_{-1} = 0, ρ_{-1} = ρ_0 = 1
- Shift solutions/directions: Ξ_0 = 0 or x_0(σ_j) = 0
- Convergence tracking

Returns (rNorm_seed, ε, rTr) where:
- rNorm_seed: initial residual norm ||b||
- ε: convergence threshold
- rTr: initial inner product (r, r)
"""
function _rscg_init!(
    workspace::RscgWorkspace{T,FC,S}, b::AbstractVector{FC}, V,
    nshifts::Int, atol::T, rtol::T
) where {T <: AbstractFloat, FC, S <: AbstractVector{FC}}

    reduced = V !== nothing

    # Extract workspace arrays
    x, r, p = workspace.x, workspace.r, workspace.p
    x_shifts, p_shifts = workspace.x_shifts, workspace.p_shifts
    Σ, Ξ, Π = workspace.Σ, workspace.Ξ, workspace.Π
    ρ, ρ_prev = workspace.ρ, workspace.ρ_prev
    rNorms, converged, not_cv, stagnated = workspace.rNorms, workspace.converged, workspace.not_cv, workspace.stagnated

    # Reset statistics
    reset!(workspace.stats)

    # ===== Initialization (Algorithm lines 1-3) =====
    # x_0 = 0, r_0 = p_0 = b
    fill!(x, zero(FC))
    copyto!(r, b)
    copyto!(p, b)

    # α_{-1} = 1, β_{-1} = 0
    workspace.α_prev[] = one(FC)
    workspace.β_prev[] = zero(FC)
    workspace.α[] = one(FC)
    workspace.β[] = zero(FC)

    # ρ_{-1}(σ_j) = ρ_0(σ_j) = 1
    fill!(ρ, one(FC))
    fill!(ρ_prev, one(FC))

    # Initialize shift solutions/directions
    if reduced
        # Σ_0 = V * b
        mul!(Σ, V, b)
        # Ξ_0(σ_j) = 0, Π_0(σ_j) = Σ_0
        for j in 1:nshifts
            fill!(Ξ[j], zero(FC))
            copyto!(Π[j], Σ)
        end
    else
        # x_0(σ_j) = 0, p_0(σ_j) = b
        for j in 1:nshifts
            fill!(x_shifts[j], zero(FC))
            copyto!(p_shifts[j], b)
        end
    end

    # Initial residual norm
    rNorm_seed = norm(r)
    fill!(rNorms, rNorm_seed)

    # Convergence threshold
    ε = atol + rtol * rNorm_seed

    # Initialize convergence flags
    for j in 1:nshifts
        converged[j] = rNorms[j] ≤ ε
        not_cv[j] = !converged[j]
        stagnated[j] = false
    end

    # Inner product for CG
    rTr = dot(r, r)

    return rNorm_seed, ε, rTr
end

# Main implementation (non-reduced mode)
function rscg!(workspace::RscgWorkspace{T,FC,S}, A, b::AbstractVector{FC},
               shifts::AbstractVector;
               atol::Real = √eps(T),
               rtol::Real = √eps(T),
               itmax::Int = 0,
               timemax::Float64 = Inf,
               verbose::Int = 0,
               history::Bool = false,
               callback = workspace -> false) where {T <: AbstractFloat, FC, S <: AbstractVector{FC}}
    _rscg_impl!(workspace, A, b, shifts, nothing;
                atol=T(atol), rtol=T(rtol), itmax=itmax, timemax=timemax,
                verbose=verbose, history=history, callback=callback)
end

# Main implementation (reduced mode with V)
function rscg!(workspace::RscgWorkspace{T,FC,S}, A, b::AbstractVector{FC},
               shifts::AbstractVector, V::AbstractMatrix;
               atol::Real = √eps(T),
               rtol::Real = √eps(T),
               itmax::Int = 0,
               timemax::Float64 = Inf,
               verbose::Int = 0,
               history::Bool = false,
               callback = workspace -> false) where {T <: AbstractFloat, FC, S <: AbstractVector{FC}}
    _rscg_impl!(workspace, A, b, shifts, V;
                atol=T(atol), rtol=T(rtol), itmax=itmax, timemax=timemax,
                verbose=verbose, history=history, callback=callback)
end

# Internal implementation
function _rscg_impl!(workspace::RscgWorkspace{T,FC,S}, A, b::AbstractVector{FC},
                     shifts::AbstractVector, V;
                     atol::T = √eps(T),
                     rtol::T = √eps(T),
                     itmax::Int = 0,
                     timemax::Float64 = Inf,
                     verbose::Int = 0,
                     history::Bool = false,
                     callback = workspace -> false) where {T <: AbstractFloat, FC, S <: AbstractVector{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    # Dimensions and validation
    m, n = size(A)
    nshifts = length(shifts)

    (m == workspace.m && n == workspace.n) || error("Workspace size mismatch: workspace is ($(workspace.m), $(workspace.n)), A is ($m, $n)")
    m == n || error("Matrix A must be square")
    length(b) == n || error("Dimension mismatch: length(b) = $(length(b)), n = $n")
    nshifts == workspace.nshifts || error("Shift count mismatch: workspace has $(workspace.nshifts), got $nshifts")

    # Determine mode and validate
    reduced = V !== nothing
    if reduced
        mreduced = size(V, 1)
        size(V, 2) == n || error("V must have $n columns, got $(size(V, 2))")
        workspace.reduced || error("Workspace not configured for reduced mode")
        workspace.mreduced == mreduced || error("Reduced dimension mismatch")
    end

    (verbose > 0) && @printf("RSCG: %d × %d system with %d shifts%s\n",
                             n, n, nshifts, reduced ? " (reduced to $(mreduced))" : "")

    # Extract workspace arrays (for main loop)
    x, r, p, Ap = workspace.x, workspace.r, workspace.p, workspace.Ap
    x_shifts, p_shifts = workspace.x_shifts, workspace.p_shifts
    Σ, Ξ, Π = workspace.Σ, workspace.Ξ, workspace.Π
    ρ, ρ_prev = workspace.ρ, workspace.ρ_prev
    α_shifts, β_shifts = workspace.α_shifts, workspace.β_shifts
    rNorms, converged, not_cv, stagnated = workspace.rNorms, workspace.converged, workspace.not_cv, workspace.stagnated
    stats = workspace.stats

    # ===== Initialization =====
    rNorm_seed, ε, rTr = _rscg_init!(workspace, b, V, nshifts, atol, rtol)

    # Record initial history
    if history
        for j in 1:nshifts
            push!(stats.residuals[j], rNorms[j])
        end
    end

    # Check for zero residual (exact solution)
    if rNorm_seed == 0
        stats.niter = 0
        stats.solved = true
        stats.timer = ktimer(start_time)
        stats.status = "x = 0 is exact solution"
        return workspace
    end

    # Iteration setup
    iter = 0
    itmax == 0 && (itmax = 2n)

    # Status flags
    solved = !any(not_cv)
    tired = iter ≥ itmax
    user_requested_exit = false
    overtimed = false

    # Verbose header
    if verbose > 0
        @printf("%5s  %12s  %12s  %8s\n", "iter", "‖r‖ (seed)", "‖r‖ (max)", "time")
        @printf("%5d  %12.5e  %12.5e  %8.2f\n", iter, rNorm_seed, maximum(rNorms), ktimer(start_time))
    end

    # ===== Main loop (Algorithm lines 4-18) =====
    while !(solved || tired || user_requested_exit || overtimed)
        # ----- Seed system update (lines 5-9) -----
        mul!(Ap, A, p)

        pAp = dot(p, Ap)
        α_k = rTr / pAp
        workspace.α_prev[] = workspace.α[]
        workspace.α[] = α_k

        axpy!(α_k, p, x)           # x_{k+1} = x_k + α_k * p_k
        axpy!(-α_k, Ap, r)         # r_{k+1} = r_k - α_k * A*p_k

        rTr_new = dot(r, r)
        β_k = rTr_new / rTr
        workspace.β_prev[] = workspace.β[]
        workspace.β[] = β_k

        axpby!(one(FC), r, β_k, p)  # p_{k+1} = r_{k+1} + β_k * p_k

        rTr = rTr_new
        rNorm_seed = sqrt(real(rTr))

        # ----- Reduced mode: compute Σ_{k+1} = V * r_{k+1} (line 10) -----
        reduced && mul!(Σ, V, r)

        # ----- Shift system updates (lines 11-17) -----
        α_prev_val = workspace.α_prev[]
        β_prev_val = workspace.β_prev[]

        for j in 1:nshifts
            if not_cv[j]
                _update_shift!(
                    j, FC(shifts[j]), α_k, β_k, α_prev_val, β_prev_val,
                    rNorm_seed, ε, reduced,
                    ρ, ρ_prev, α_shifts, β_shifts,
                    Σ, Ξ, Π, r, x_shifts, p_shifts,
                    rNorms, converged, not_cv, stagnated
                )
            end
        end

        # Store history
        if history
            for j in 1:nshifts
                push!(stats.residuals[j], rNorms[j])
            end
        end

        iter += 1

        # Verbose output
        kdisplay(iter, verbose) && @printf("%5d  %12.5e  %12.5e  %8.2f\n",
                    iter, rNorm_seed, maximum(rNorms), ktimer(start_time))

        # Check termination conditions
        user_requested_exit = callback(workspace)::Bool
        solved = !any(not_cv)
        tired = iter ≥ itmax
        overtimed = (time_ns() - start_time) > timemax_ns
    end

    # ===== Finalization =====
    (verbose > 0) && @printf("\n")

    # Count stagnated shifts
    nstagnated = count(stagnated)
    nconverged = count(converged)

    stats.status = if solved && nstagnated == 0
        "solution good enough given atol and rtol"
    elseif solved && nstagnated > 0
        "converged ($nconverged/$nshifts), stagnated ($nstagnated/$nshifts)"
    elseif tired
        if nstagnated > 0
            "maximum iterations exceeded, stagnated ($nstagnated/$nshifts)"
        else
            "maximum number of iterations exceeded"
        end
    elseif user_requested_exit
        "user-requested exit"
    elseif overtimed
        "time limit exceeded"
    else
        "unknown"
    end

    stats.niter = iter
    stats.solved = solved
    stats.timer = ktimer(start_time)

    return workspace
end

#=============================================================================
  True residual computation (for verification and stagnation fallback)
==============================================================================#

"""
    compute_true_residuals!(workspace, A, b, shifts) -> Vector{T}
    compute_true_residuals!(workspace, A, b, shifts, V) -> Vector{T}

Compute the true residual norms for all shift solutions.

This function computes the exact residual ``\\|b - (σⱼ I + A) x(σⱼ)\\|`` for each shift,
which can be used to:
1. Verify the estimated residuals from the RSCG algorithm
2. Check solution quality when stagnation is detected
3. Post-iteration quality assessment

# Arguments
- `workspace::RscgWorkspace`: Workspace after calling `rscg!`
- `A`: The linear operator (same as used in `rscg!`)
- `b::AbstractVector`: Right-hand side vector (same as used in `rscg!`)
- `shifts::AbstractVector`: Shift values (same as used in `rscg!`)
- `V::AbstractMatrix`: (optional) Reduction matrix for reduced mode

# Returns
- `true_residuals::Vector{T}`: True residual norms for each shift

# Note
This function performs O(nshifts) matrix-vector products, so it should be used
sparingly (e.g., only for stagnated shifts or final verification).

# Example
```julia
workspace = RscgWorkspace(A, b, length(shifts))
rscg!(workspace, A, b, shifts)

# Check if any shifts stagnated
if any(workspace.stagnated)
    true_res = compute_true_residuals!(workspace, A, b, shifts)
    for j in eachindex(shifts)
        if workspace.stagnated[j]
            println("Shift \$j stagnated, true residual = \$(true_res[j])")
        end
    end
end
```

See also: [`rscg!`](@ref), [`RscgWorkspace`](@ref)
"""
function compute_true_residuals!(
    workspace::RscgWorkspace{T,FC,S}, A, b::AbstractVector{FC},
    shifts::AbstractVector
) where {T <: AbstractFloat, FC, S <: AbstractVector{FC}}

    nshifts = workspace.nshifts
    true_residuals = Vector{T}(undef, nshifts)

    # Use workspace.Ap as temporary storage for (σI + A)x
    tmp = workspace.Ap

    for j in 1:nshifts
        x_j = workspace.x_shifts[j]
        σ_j = FC(shifts[j])

        # Compute (σI + A)x_j
        mul!(tmp, A, x_j)
        axpy!(σ_j, x_j, tmp)  # tmp = A*x + σ*x = (σI + A)x

        # Compute residual: r = b - (σI + A)x
        # We reuse tmp: tmp = b - tmp
        axpby!(one(FC), b, -one(FC), tmp)

        true_residuals[j] = norm(tmp)
    end

    return true_residuals
end

# Reduced mode version
function compute_true_residuals!(
    workspace::RscgWorkspace{T,FC,S}, A, b::AbstractVector{FC},
    shifts::AbstractVector, V::AbstractMatrix
) where {T <: AbstractFloat, FC, S <: AbstractVector{FC}}

    nshifts = workspace.nshifts
    n = workspace.n
    mreduced = workspace.mreduced
    true_residuals = Vector{T}(undef, nshifts)

    # For reduced mode, we don't have the full solution x(σ_j)
    # We can only compute the reduced residual norm ||V*(b - (σI + A)x)||
    # which requires reconstructing x from Ξ, which is not stored

    # Alternative: compute ||Σ - σ*Ξ - V*A*x|| but we don't have x

    # For reduced mode, return the estimated residuals (ρ-based)
    # True residual computation would require storing the full x vectors
    @warn "compute_true_residuals! in reduced mode returns estimated residuals (true residuals require full solution vectors)"

    for j in 1:nshifts
        # Return the estimated residual norm
        true_residuals[j] = workspace.rNorms[j]
    end

    return true_residuals
end
