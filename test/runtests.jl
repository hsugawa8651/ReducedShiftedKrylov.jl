# Last-Modified: 2025-12-15T09:50:00+09:00

using Test
using LinearAlgebra
using ReducedShiftedKrylov

include("test_greens_function.jl")

@testset "ReducedShiftedKrylov.jl" begin

    @testset "Basic RSCG - Real Hermitian" begin
        # Small positive definite matrix
        n = 10
        A = Hermitian(rand(n, n) + n * I)
        b = rand(n)
        shifts = [0.0, 1.0, 2.0]

        x, stats = rscg(A, b, shifts; verbose=0)

        @test stats.solved
        @test length(x) == 3

        # Check each solution
        for (j, σ) in enumerate(shifts)
            r = (A + σ * I) * x[j] - b
            @test norm(r) < 1e-6
        end
    end

    @testset "Basic RSCG - Complex Hermitian" begin
        n = 10
        H = rand(ComplexF64, n, n)
        A = Hermitian(H + H' + n * I)
        b = rand(ComplexF64, n)
        shifts = [0.0 + 0.1im, 1.0 + 0.0im, 0.0 + 1.0im]

        x, stats = rscg(A, b, shifts; verbose=0)

        @test stats.solved
        @test length(x) == 3

        for (j, σ) in enumerate(shifts)
            r = (A + σ * I) * x[j] - b
            @test norm(r) < 1e-6
        end
    end

    @testset "Reduced RSCG" begin
        n = 20
        m = 3  # reduced dimension
        H = rand(ComplexF64, n, n)
        A = Hermitian(H + H' + n * I)
        b = rand(ComplexF64, n)
        shifts = [0.1im, 0.2im, 0.3im]

        # Reduction matrix (select first m components)
        V = zeros(ComplexF64, m, n)
        for i in 1:m
            V[i, i] = 1.0
        end

        Ξ, stats = rscg(A, b, shifts, V; verbose=0)

        @test stats.solved
        @test length(Ξ) == 3
        @test length(Ξ[1]) == m

        # Compare with direct solution
        for (j, σ) in enumerate(shifts)
            x_direct = (A + σ * I) \ b
            Ξ_direct = V * x_direct
            @test norm(Ξ[j] - Ξ_direct) < 1e-6
        end
    end

    @testset "Workspace reuse" begin
        n = 10
        A = Hermitian(rand(n, n) + n * I)
        b1 = rand(n)
        b2 = rand(n)
        shifts = [0.0, 1.0]

        workspace = krylov_workspace(Val(:rscg), A, b1, length(shifts))

        # First solve
        rscg!(workspace, A, b1, shifts)
        x1_ref, stats1 = results(workspace)
        x1 = deepcopy(x1_ref)  # Copy before next solve
        @test stats1.solved

        # Second solve with same workspace
        rscg!(workspace, A, b2, shifts)
        x2, stats2 = results(workspace)
        @test stats2.solved

        # Verify solutions are different
        @test norm(x1[1] - x2[1]) > 1e-10
    end

    @testset "Error Fallbacks" begin
        @testset "real_type unsupported type" begin
            # String is not a supported numeric type
            @test_throws ArgumentError real_type(String)
            # Symbol is not a supported numeric type
            @test_throws ArgumentError real_type(Symbol)
        end

        @testset "real_type supported types" begin
            # Float types
            @test real_type(Float64) == Float64
            @test real_type(Float32) == Float32
            # Complex types
            @test real_type(ComplexF64) == Float64
            @test real_type(ComplexF32) == Float32
            # Values
            @test real_type(1.0) == Float64
            @test real_type(1.0f0) == Float32
            @test real_type(1.0 + 2.0im) == Float64
            # Vector types
            @test real_type(Vector{Float64}) == Float64
            @test real_type(Vector{ComplexF64}) == Float64
        end

        @testset "krylov_solve unsupported solver" begin
            A = [1.0 0.0; 0.0 1.0]
            b = [1.0, 2.0]
            shifts = [0.0, 1.0]
            @test_throws ArgumentError krylov_solve(Val(:unknown_solver), A, b, shifts)
            V = [1.0 0.0; 0.0 1.0]  # dummy reduction matrix
            @test_throws ArgumentError krylov_solve(Val(:foo), A, b, shifts, V)
        end

        @testset "krylov_workspace unsupported solver" begin
            A = [1.0 0.0; 0.0 1.0]
            b = [1.0, 2.0]
            @test_throws ArgumentError krylov_workspace(Val(:bar), A, b, 2)
            V = [1.0 0.0; 0.0 1.0]
            @test_throws ArgumentError krylov_workspace(Val(:baz), A, b, 2, V)
        end
    end

    @testset "Stagnation Detection" begin
        # Test with well-conditioned problem (no stagnation expected)
        n = 10
        H = rand(ComplexF64, n, n)
        A = Hermitian(H + H' + 2n * I)
        b = rand(ComplexF64, n)
        shifts = [0.1im, 0.2im, 0.3im]

        workspace = RscgWorkspace(A, b, length(shifts))
        rscg!(workspace, A, b, shifts; verbose=0)

        # Well-conditioned problem should not stagnate
        @test !any(workspace.stagnated)
        @test workspace.stats.solved
    end

    @testset "True Residual Computation" begin
        n = 10
        A = Hermitian(rand(n, n) + n * I)
        b = rand(n)
        shifts = [0.0, 1.0, 2.0]

        workspace = RscgWorkspace(A, b, length(shifts))
        rscg!(workspace, A, b, shifts; verbose=0)

        # Compute true residuals
        true_res = compute_true_residuals!(workspace, A, b, shifts)

        @test length(true_res) == length(shifts)

        # True residuals should be close to estimated residuals for converged solution
        for j in eachindex(shifts)
            @test true_res[j] < 1e-6
        end

        # Verify against direct computation
        x, stats = results(workspace)
        for (j, σ) in enumerate(shifts)
            direct_res = norm((A + σ * I) * x[j] - b)
            @test abs(true_res[j] - direct_res) < 1e-12
        end
    end

end
