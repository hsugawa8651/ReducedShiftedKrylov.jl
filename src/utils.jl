# Last-Modified: 2025-12-14T18:55:00+09:00

# Utility functions for ReducedShiftedKrylov
# Following Krylov.jl conventions

export real_type, ktimer

#=
Trait-based type handling for Float/Complex types.
Instead of Union{T, Complex{T}}, use real_type() to extract the underlying real type.
=#

"""
    real_type(::Type{T})
    real_type(x)

Return the real type underlying a float or complex type.

# Examples
```julia
real_type(Float64)        # Float64
real_type(ComplexF64)     # Float64
real_type(1.0 + 2.0im)    # Float64
```
"""
real_type(::Type{T}) where {T <: AbstractFloat} = T
real_type(::Type{Complex{T}}) where {T <: AbstractFloat} = T
real_type(x::Number) = real_type(typeof(x))
real_type(::Type{S}) where {S <: AbstractVector} = real_type(eltype(S))

# Error fallback for unsupported types
real_type(::Type{T}) where {T} = throw(ArgumentError("Unsupported type for real_type: $T"))

"""
    ktimer(start_time::UInt64)

Convert elapsed nanoseconds to seconds.
"""
ktimer(start_time::UInt64) = (time_ns() - start_time) / 1e9

"""
    kdisplay(iter, verbose)

Check if should display at this iteration.
"""
kdisplay(iter::Int, verbose::Int) = verbose > 0 && (iter % verbose == 0)
