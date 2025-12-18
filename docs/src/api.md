# API Reference

## Main Functions

```@docs
rscg
rscg!
```

## Generic Interface

```@docs
krylov_solve
krylov_workspace
```

## Types

```@docs
RscgWorkspace
ReducedShiftStats
results
```

## Utility Functions

```@docs
reset!
```

---

## Function Signatures

### `rscg`

```julia
rscg(A, b, shifts; kwargs...) -> (x, stats)
rscg(A, b, shifts, V; kwargs...) -> (Ξ, stats)
```

Out-of-place RSCG solver.

**Arguments:**
- `A`: Hermitian matrix or linear operator
- `b::AbstractVector`: Right-hand side vector
- `shifts::AbstractVector`: Shift values σⱼ
- `V::AbstractMatrix`: (optional) Reduction matrix, enables reduced mode

**Keyword Arguments:**
- `atol::Real=√eps(T)`: Absolute tolerance
- `rtol::Real=√eps(T)`: Relative tolerance
- `itmax::Int=0`: Maximum iterations (0 = 2n)
- `timemax::Float64=Inf`: Time limit in seconds
- `verbose::Int=0`: Verbosity level (0=silent, 1=summary, 2=per-iteration)
- `history::Bool=false`: Store residual history in `stats.residuals`
- `callback=workspace->false`: Early termination callback

**Returns:**
- `x::Vector{Vector}`: Solution vectors (non-reduced mode)
- `Ξ::Vector{Vector}`: Reduced solution vectors (reduced mode with V)
- `stats::ReducedShiftStats`: Solver statistics

### `rscg!`

```julia
rscg!(workspace, A, b, shifts; kwargs...) -> workspace
rscg!(workspace, A, b, shifts, V; kwargs...) -> workspace
```

In-place RSCG solver. Use `results(workspace)` to extract solution.

### `krylov_workspace`

```julia
krylov_workspace(Val(:rscg), A, b, nshifts) -> RscgWorkspace
krylov_workspace(Val(:rscg), A, b, nshifts, V) -> RscgWorkspace
```

Create pre-allocated workspace for RSCG solver.

### `results`

```julia
results(workspace::RscgWorkspace) -> (x, stats)
```

Extract solution and statistics from workspace.

---

## Type Details

### `RscgWorkspace{T,FC,S}`

Workspace for RSCG method.

**Type Parameters:**
- `T`: Real float type (e.g., `Float64`)
- `FC`: Float or Complex type (e.g., `ComplexF64`)
- `S`: Vector type (e.g., `Vector{ComplexF64}`)

**Key Fields:**
- `m, n`: Matrix dimensions
- `nshifts`: Number of shifts
- `reduced`: Whether in reduced mode
- `stats`: Solver statistics

### `ReducedShiftStats{T}`

Statistics returned by RSCG solver.

**Fields:**
- `niter::Int`: Total iterations
- `solved::Bool`: Convergence flag
- `residuals::Vector{Vector{T}}`: Residual history (if `history=true`)
- `timer::Float64`: Elapsed time (seconds)
- `status::String`: Outcome description

---

See [Workflow and Applications](workflow.md) for detailed usage examples.
