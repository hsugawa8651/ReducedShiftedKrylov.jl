# Workflow and Applications

## Basic Usage

### Solving Shifted Linear Systems

```julia
using LinearAlgebra
using ReducedShiftedKrylov

# Create a Hermitian matrix
n = 100
H = rand(ComplexF64, n, n)
A = Hermitian(H + H' + 2n * I)
b = rand(ComplexF64, n)

# Define shifts
shifts = [0.1im, 0.2im, 0.3im]

# Solve (A + σⱼI) x = b for all shifts
x, stats = rscg(A, b, shifts)

# Check convergence
println("Converged: ", stats.solved)
println("Iterations: ", stats.niter)

# Verify solution
for (j, σ) in enumerate(shifts)
    residual = norm((A + σ * I) * x[j] - b)
    println("Shift $j residual: $residual")
end
```

### Reduced Mode

When you only need specific components of the solution (e.g., Green's function matrix elements), use reduced mode to save memory:

```julia
# Reduction matrix V: extracts components 1:m
m = 10
V = zeros(ComplexF64, m, n)
for i in 1:m
    V[i, i] = 1.0
end

# Compute Ξ = V * x directly
Ξ, stats = rscg(A, b, shifts, V)

# Ξ[j] has dimension m instead of n
println("Reduced solution size: ", length(Ξ[1]))
```

### Workspace Reuse

For solving multiple systems with different right-hand sides:

```julia
# Create workspace once
workspace = krylov_workspace(Val(:rscg), A, b, length(shifts))

# Solve multiple systems
for i in 1:10
    b_i = rand(ComplexF64, n)
    rscg!(workspace, A, b_i, shifts)
    x_i, stats_i = results(workspace)
    # Process results...
end
```

## Green's Function Calculation

### Single Matrix Element

Compute ``G_{\alpha\beta}(z) = \langle \alpha | (zI - H)^{-1} | \beta \rangle``:

```julia
using LinearAlgebra
using ReducedShiftedKrylov

# Hamiltonian
n = 50
H_raw = rand(ComplexF64, n, n)
H = Hermitian(H_raw + H_raw' + 2n * I)

# Frequency points
η = 0.1  # broadening
ω_values = range(-5, 5, length=100)
z_values = [ω + im * η for ω in ω_values]

# Indices for G_αβ
α, β = 1, 1

# Set up RSCG: (zI - H)⁻¹ → A = -H, σ = z
A = Hermitian(-Matrix(H))
b = zeros(ComplexF64, n)
b[β] = 1.0

# Reduction matrix for α-th component
V = zeros(ComplexF64, 1, n)
V[1, α] = 1.0

# Solve
Ξ, stats = rscg(A, b, z_values, V)

# Extract G_αβ(z)
G_αβ = [Ξ[j][1] for j in eachindex(z_values)]
```

### Local Density of States (LDOS)

```julia
# LDOS at site α
LDOS = [-imag(G_αβ[j]) / π for j in eachindex(ω_values)]

# Plot
using Plots
plot(ω_values, LDOS, xlabel="ω", ylabel="LDOS", title="Local Density of States")
```

### Multiple Matrix Elements

For a block of Green's function elements:

```julia
# Compute G_αβ for α, β ∈ {1, 2, ..., m}
m = 5

# Reduction matrix: extract first m components
V = zeros(ComplexF64, m, n)
for i in 1:m
    V[i, i] = 1.0
end

# Solve for each β
G = zeros(ComplexF64, m, m, length(z_values))
for β in 1:m
    b = zeros(ComplexF64, n)
    b[β] = 1.0
    Ξ, _ = rscg(A, b, z_values, V)
    for (j, z) in enumerate(z_values)
        G[:, β, j] = Ξ[j]
    end
end
```

## Comparison with Direct Diagonalization

For small systems, you can verify RSCG results against diagonalization:

```julia
# Exact Green's function via diagonalization
λ, U = eigen(H)
function G_exact(z)
    return U * Diagonal(1 ./ (z .- λ)) * U'
end

# Compare
for (j, z) in enumerate(z_values)
    G_rscg = Ξ[j][1]
    G_diag = G_exact(z)[α, β]
    error = abs(G_rscg - G_diag) / abs(G_diag)
    println("z = $z: relative error = $error")
end
```

## Integration Pattern for Application Packages

ReducedShiftedKrylov.jl provides a **generic RSCG solver**. Application packages should create their own domain-specific APIs that wrap the generic solver.

### Recommended Pattern

```julia
# MyPhysicsPackage.jl

using ReducedShiftedKrylov
using LinearAlgebra

# 1. Define domain-specific operators
struct EffectiveHamiltonian{TL, TR, TF}
    LHS::TL
    RHS::TR
    RHS_fact::TF  # LU factorization of RHS
end

function EffectiveHamiltonian(LHS, RHS)
    EffectiveHamiltonian(LHS, RHS, lu(RHS))
end

# H * x = RHS⁻¹ * (LHS * x)
function LinearAlgebra.mul!(y, H::EffectiveHamiltonian, x)
    tmp = H.LHS * x
    ldiv!(y, H.RHS_fact, tmp)
    return y
end

Base.:*(H::EffectiveHamiltonian, x) = (y = similar(x); mul!(y, H, x))
Base.size(H::EffectiveHamiltonian) = size(H.LHS)

# 2. Define negated operator for RSCG interface
struct NegatedOperator{T}
    op::T
end

function LinearAlgebra.mul!(y, A::NegatedOperator, x)
    mul!(y, A.op, x)
    y .*= -1
    return y
end

Base.:*(A::NegatedOperator, x) = -1 .* (A.op * x)
Base.size(A::NegatedOperator) = size(A.op)

# 3. Domain-specific convenience API
"""
    compute_greens_function(solver, k, ω_values, source; η=1e-3)

Compute G(ω) = (ω² + iη - H)⁻¹ * source for multiple frequencies.
"""
function compute_greens_function(solver, k, ω_values, source; η=1e-3)
    # Build matrices from your physics solver
    LHS, RHS = build_matrices(solver, k)

    # Effective Hamiltonian: H = RHS⁻¹ * LHS
    H = EffectiveHamiltonian(LHS, RHS)

    # For RSCG: solve (σI + A)x = b where A = -H, σ = z
    # This gives (zI - H)x = b, i.e., x = G(z) * b
    A = NegatedOperator(H)

    # Shifts: z = ω² + iη
    shifts = [ω^2 + im*η for ω in ω_values]

    # Solve using ReducedShiftedKrylov.jl
    x_solutions, stats = rscg(A, ComplexF64.(source), shifts)

    return x_solutions, stats
end

# 4. With reduction matrix for specific components
function compute_greens_element(solver, k, ω_values, α, β; η=1e-3)
    LHS, RHS = build_matrices(solver, k)
    n = size(LHS, 1)

    H = EffectiveHamiltonian(LHS, RHS)
    A = NegatedOperator(H)

    # Source: e_β
    b = zeros(ComplexF64, n)
    b[β] = 1.0

    # Reduction: extract α-th component
    V = zeros(ComplexF64, 1, n)
    V[1, α] = 1.0

    shifts = [ω^2 + im*η for ω in ω_values]

    # Reduced solve: returns Ξ = V * x = G_αβ directly
    Ξ, stats = rscg(A, b, shifts, V)

    return [Ξ[j][1] for j in eachindex(ω_values)], stats
end
```

### Key Points

1. **ReducedShiftedKrylov.jl solves**: ``(σI + A) x = b``

2. **For Green's function** ``G(z) = (zI - H)^{-1}``:
   - Set ``A = -H``
   - Set ``σ = z``
   - Then ``(σI + A)x = b`` becomes ``(zI - H)x = b``

3. **Domain-specific conversions** belong in the application package:
   - `EffectiveHamiltonian`: ``H = RHS^{-1} \cdot LHS``
   - `NegatedOperator`: ``A = -H``
   - Shift construction: ``z = ω^2 + iη``

4. **Reduced mode** for memory efficiency:
   - Use `rscg(A, b, shifts, V)` to compute ``Ξ = V x`` directly
   - Avoids storing full solution vectors

### Example: PhoXonic.jl

PhoXonic.jl uses this pattern for photonic/phononic crystal calculations:

```julia
using PhoXonic
using ReducedShiftedKrylov

solver = Solver(...)

# PhoXonic provides domain-specific API
G_values = compute_greens_function(solver, k, ω_values, source, RSKGF(); η=0.01)
ldos = compute_ldos(solver, position, ω_values, k_points, MatrixFreeGF())
```

Internally, PhoXonic wraps ReducedShiftedKrylov.jl's generic `rscg` solver.

## Tips and Best Practices

### Convergence

- Shifts close to eigenvalues converge slowly
- Add small imaginary part `iη` to avoid singularities
- Monitor `stats.niter` and `stats.residuals`

### Memory Efficiency

- Use reduced mode when possible
- Reuse workspace for multiple solves
- For very large `nshifts`, consider batching

### Performance

- Ensure `A` is truly Hermitian (use `Hermitian()` wrapper)
- For sparse matrices, RSCG only needs matrix-vector products
- Complex Hermitian: 1 matvec per iteration (same as real)
