# Last-Modified: 2025-12-14T19:00:00+09:00

# Green's function test: Compare RSCG with exact diagonalization
#
# G(z) = (zI - H)^{-1}
# G_αβ(z) = <α| (zI - H)^{-1} |β>
#
# RSCG formulation:
#   (σI + A) x = b  where A = -H, σ = z
#   G_αβ(z) = e_α' * x = V * x  (with V = e_α')

using Test
using LinearAlgebra
using Random
using ReducedShiftedKrylov

@testset "Green's function via RSCG" begin

    @testset "Single element G_αβ(z)" begin
        Random.seed!(42)  # Fix seed for reproducibility
        # Test matrix: random Hermitian
        n = 20
        H_raw = rand(ComplexF64, n, n)
        H = Hermitian(H_raw + H_raw' + 2n * I)  # Ensure positive definite

        # Diagonalize for exact reference
        λ, U = eigen(H)

        # Exact Green's function
        function G_exact(z)
            return U * Diagonal(1 ./ (z .- λ)) * U'
        end

        # Test indices
        α, β = 3, 5

        # Test frequencies (complex, away from real axis for stability)
        z_values = [
            1.0 + 0.1im,
            5.0 + 0.1im,
            10.0 + 0.5im,
            -1.0 + 0.2im,
            0.0 + 1.0im,
        ]

        # RSCG setup:
        # (zI - H) x = e_β  =>  (σI + A) x = b  where A = -H, σ = z, b = e_β
        A = Hermitian(-Matrix(H))
        b = zeros(ComplexF64, n)
        b[β] = 1.0

        # Reduction matrix: V = e_α' (extracts α-th component)
        V = zeros(ComplexF64, 1, n)
        V[1, α] = 1.0

        # Shifts = z values
        shifts = z_values

        # Solve with RSCG
        Ξ, stats = rscg(A, b, shifts, V; verbose=0)

        @test stats.solved

        # Compare with exact
        for (j, z) in enumerate(z_values)
            G_exact_αβ = G_exact(z)[α, β]
            G_rscg_αβ = Ξ[j][1]

            rel_error = abs(G_rscg_αβ - G_exact_αβ) / abs(G_exact_αβ)
            @test rel_error < 1e-6
        end
    end

    @testset "Multiple elements (row of G)" begin
        Random.seed!(123)  # Fix seed for reproducibility
        # Compute entire row G_α,: for multiple z values
        n = 15
        H_raw = rand(ComplexF64, n, n)
        H = Hermitian(H_raw + H_raw' + 2n * I)

        λ, U = eigen(H)
        G_exact(z) = U * Diagonal(1 ./ (z .- λ)) * U'

        α = 2  # Row index
        z_values = [2.0 + 0.1im, 5.0 + 0.2im, 8.0 + 0.1im]

        A = Hermitian(-Matrix(H))

        # V = e_α' extracts row α
        V = zeros(ComplexF64, 1, n)
        V[1, α] = 1.0

        # For each column β, solve (zI - H) x = e_β
        for β in 1:n
            b = zeros(ComplexF64, n)
            b[β] = 1.0

            Ξ, stats = rscg(A, b, z_values, V; verbose=0)
            @test stats.solved

            for (j, z) in enumerate(z_values)
                G_exact_αβ = G_exact(z)[α, β]
                G_rscg_αβ = Ξ[j][1]

                rel_error = abs(G_rscg_αβ - G_exact_αβ) / (abs(G_exact_αβ) + 1e-15)
                @test rel_error < 5e-6
            end
        end
    end

    @testset "Local density of states (LDOS)" begin
        Random.seed!(456)  # Fix seed for reproducibility
        # LDOS_α(ω) = -1/π * Im[G_αα(ω + iη)]
        n = 20
        H_raw = rand(n, n)
        H = Hermitian(H_raw + H_raw' + 2n * I)

        λ, U = eigen(H)
        G_exact(z) = U * Diagonal(1 ./ (z .- λ)) * U'

        α = 5
        η = 0.1  # Broadening

        # Frequency grid
        ω_values = range(minimum(λ) - 1, maximum(λ) + 1, length=10)
        z_values = [ω + im * η for ω in ω_values]

        A = Hermitian(-Matrix(H))
        b = zeros(ComplexF64, n)
        b[α] = 1.0

        V = zeros(ComplexF64, 1, n)
        V[1, α] = 1.0

        Ξ, stats = rscg(A, b, z_values, V; verbose=0)
        @test stats.solved

        for (j, z) in enumerate(z_values)
            G_exact_αα = G_exact(z)[α, α]
            G_rscg_αα = Ξ[j][1]

            LDOS_exact = -imag(G_exact_αα) / π
            LDOS_rscg = -imag(G_rscg_αα) / π

            rel_error = abs(LDOS_rscg - LDOS_exact) / (abs(LDOS_exact) + 1e-15)
            @test rel_error < 1e-6
        end
    end

    @testset "Full Green's function matrix element" begin
        Random.seed!(789)  # Fix seed for reproducibility
        # Non-reduced mode: compute full x, then extract element
        n = 10
        H_raw = rand(ComplexF64, n, n)
        H = Hermitian(H_raw + H_raw' + 2n * I)

        λ, U = eigen(H)
        G_exact(z) = U * Diagonal(1 ./ (z .- λ)) * U'

        z_values = [3.0 + 0.1im, 7.0 + 0.2im]
        A = Hermitian(-Matrix(H))

        α, β = 2, 4
        b = zeros(ComplexF64, n)
        b[β] = 1.0

        # Non-reduced RSCG
        x, stats = rscg(A, b, z_values; verbose=0)
        @test stats.solved

        for (j, z) in enumerate(z_values)
            G_exact_αβ = G_exact(z)[α, β]
            G_rscg_αβ = x[j][α]

            rel_error = abs(G_rscg_αβ - G_exact_αβ) / abs(G_exact_αβ)
            @test rel_error < 1e-6
        end
    end

end
