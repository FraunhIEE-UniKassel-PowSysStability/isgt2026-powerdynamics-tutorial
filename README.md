# Small-Signal Stability with PowerDynamics.jl and SciML — Tutorial

Companion tutorial to the talk *"Small-Signal Analysis using Automatic Differentiation in the Open-Source Software
PowerDynamics.jl"* (S. Eberlein, H. Würfel, L. Bittcher), special session *Small-Signal Analysis of Modern Power
Systems: Novel Approaches and Computational Tools*, IEEE ISGT Europe 2026, Budapest.

A modified IEEE 9-bus system in quasi-EMT detail (dq frame, dynamic lines; synchronous generator, grid-forming and grid-following inverter) is
analysed with one nonlinear model and automatic differentiation:

| Section | Analysis | Packages |
|---|---|---|
| 1 | test system, power flow, load-step simulations at different grid strengths | PowerDynamics.jl, NetworkDynamics.jl, OrdinaryDiffEq.jl |
| 2 | eigenvalues, participation factors per device | NetworkDynamics.jl, ForwardDiff.jl |
| 3 | eigenvalue sensitivities (nested AD) | NetworkDynamics.jl, ForwardDiff.jl |
| 4 | eigenvalue tracking by continuation vs. nearest-neighbour matching | BifurcationKit.jl, KrylovKit.jl |
| 5 | stability boundary: grid strength vs. PLL bandwidth | BifurcationKit.jl (PALC) |
| 6 | impedance-based analysis at the GFL bus | NetworkDynamics.jl, ControlSystemsBase.jl |
| 7 | generalized Nyquist criterion and passivity | ControlSystemsBase.jl, RobustAndOptimalControl.jl |
| 8 | gradient-based tuning in the time domain (AD through the ODE solver) | Optimization.jl, Optimisers.jl, ForwardDiff.jl |
| 9 | modal gradient-based tuning (eigenvalue sensitivities) and comparison | ForwardDiff.jl |

The test system is built with the [PoSyDyS 2026 paper companion](https://github.com/pik-icone/PoSyDysPaperCompanion)
(H. Würfel, PIK). PowerDynamics.jl: <https://github.com/JuliaEnergy/PowerDynamics.jl>.

**View the results directly:** open [`ieee9bus_small_signal.ipynb`](ieee9bus_small_signal.ipynb); GitHub renders the
executed notebook with all outputs and figures.

## Files

| File | Content |
|---|---|
| `ieee9bus_small_signal.ipynb` | executed notebook with all outputs and figures |
| `ieee9bus_small_signal.jl` | source (Literate.jl format), runs as a plain Julia script |
| `make.jl` | regenerates the notebook (and the SVG figures in `figures/`) |
| `Project.toml`, `Manifest.toml` | pinned environment (Julia 1.12, PowerDynamics 5.0, NetworkDynamics 1.3) |
| `figures/` | the figures as SVG (as used in the talk) |

## Running

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. ieee9bus_small_signal.jl      # run as a script, or
julia --project=. make.jl                       # regenerate the executed notebook and figures/
```

To work with the notebook interactively, install a Julia kernel (`using Pkg; Pkg.add("IJulia")`) and open
`ieee9bus_small_signal.ipynb` in Jupyter or VS Code. A full run takes about 15–25 minutes on a laptop, most of it
compilation, the stability boundary and the time-domain optimization (set the environment variable `TUTORIAL_TD_ITERS` to fewer iterations
for a quick run).

## License

MIT, see [LICENSE](LICENSE).
