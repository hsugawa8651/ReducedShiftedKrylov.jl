using Documenter
using ReducedShiftedKrylov

makedocs(
    sitename = "ReducedShiftedKrylov.jl",
    modules = [ReducedShiftedKrylov],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home" => "index.md",
        "Workflow" => "workflow.md",
        "API Reference" => "api.md",
    ],
    remotes = nothing,
    warnonly = [:missing_docs],
)

# Deploy to GitHub Pages
deploydocs(
    repo = "github.com/hsugawa8651/ReducedShiftedKrylov.jl.git",
    devbranch = "main",
    push_preview = true,
)
