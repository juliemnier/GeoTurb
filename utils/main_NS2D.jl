using FourierFlows
using BenchmarkTools
using CUDA
include("../../../../../src/Equation_2DNS.jl")
using .Equation

# load parameters
include("namelist_NS2D.jl") # not a module

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
#μ, κ = friction_type == 10 ? (nothing, κ) : (μ, nothing) 
params = Equation.Params(resol, ν, nν, friction_type, μ, κ, η, forcing_type, kf, dkf, ε, dealiasing, mask, add_tracer, CFL, timestepper);
# Initial conditions
Fψ, Fτ, Ffrictionquad, Fscratch, scratch = Equation.initialize_field(grid, params, IC_type, noise_amplitude, restart_flag, path_to_run, restart_filename)
# Main variables
vars = Equation.Vars(Fψ, Fscratch, scratch, Ffrictionquad, Fτ, nothing, t0);
# Forcing initialization
Equation.calcF!(vars, params, grid, dt);
# Linear operators initialization
Lψ, Lτ = Equation.compute_L(params, grid)

#############################################################################
# Main loop 

@btime Equation.run_simulation!(vars, params, Lψ, Lτ, grid, Nfield, NsaveEc, Nstep, NsaveFlowfield, Nfig, Nspectrum, dt, path_to_run)
   

