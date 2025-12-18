# Last-Modified: 2025-12-10T16:28:20+09:00

"""
    ReducedShiftedKrylov

Reduced-Shifted Krylov solvers for multi-shift linear systems.

Based on: Y. Nagai et al., J. Phys. Soc. Jpn. 86, 014708 (2017)

# Exports
- `rscg`, `rscg!`: Reduced-Shifted Conjugate Gradient method
- `RscgWorkspace`: Workspace for RSCG
- `ReducedShiftStats`: Statistics structure
"""
module ReducedShiftedKrylov

using LinearAlgebra
using Printf

# Type utilities
include("utils.jl")

# Statistics structures
include("stats.jl")

# Workspace structures
include("workspaces.jl")

# RSCG algorithm
include("rscg.jl")

# High-level interface
include("interface.jl")

end # module
