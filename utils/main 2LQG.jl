using FourierFlows
using BenchmarkTools
using CUDA
include("../src/Equation_2LQG.jl")
using .Equation

# load parameters
include("./namelist_2LQG.jl") # not a module

#@btime begin

CUDA.seed!(42)

# Problem initialization
# only squared domains are handled
grid = TwoDGrid(dev; nx = resol, Lx = L, ny = resol, Ly = L, aliased_fraction)
mask = nothing
if dealiasing
    mask  = ((1 .- sign.( sqrt.(grid.Krsq) .- aliased_fraction * (resol-1))) ./ 2)
end

#############################################################################
# Run parameters
params = Equation.Params(resol, ν, nν, λ, Ro, friction_type, μ, κ, add_wn, kf, dkf, ε, dealiasing, mask, CFL, timestepper, flag_2LQG);
# Initial conditions
Fψ, Fτ = Equation.initialize_field(grid, params, IC_type, noise_amplitude, restart_flag, path_to_run, restart_filename)

# Main variables
vars = Equation.Vars(Fψ, Fτ, nothing, nothing, t0, nothing, nothing);
# Forcing initialization
if params.add_wn
    Equation.calcF!(vars, params, grid, dt);
end

#############################################################################
# Main loop 

Equation.run_simulation!(vars, params, grid, Nfield, Nstep, NsaveFlowfield, Nfig,  NsaveEc,  Nspectrum, dt, path_to_run)
   