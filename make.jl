# Generate the executed Jupyter notebook (and the figures used on the slides) from the Literate source.
#
# Run from this folder:  julia --project=. make.jl
#
# Output: ieee9bus_small_signal.ipynb (with all outputs and figures embedded) and figures/*.svg

using Literate

const HERE = @__DIR__
ENV["TUTORIAL_FIGDIR"] = joinpath(HERE, "figures")
mkpath(ENV["TUTORIAL_FIGDIR"])

Literate.notebook(joinpath(HERE, "ieee9bus_small_signal.jl"), HERE; execute=true, credit=false)
