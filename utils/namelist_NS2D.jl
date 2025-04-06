########################################################################
# choose CPU or GPU below
dev = GPU()
########################################################################
path_to_run = "./" 
restart_filename ="CI.mat"
restart_flag = false
########################################################################
# forcing parameters
########################################################################
forcing_type = 20 
### forcing_type :   10 => White noise forcing 
###                  20 => Kolmogorov forcing
# will vary with forcing_type
kf = 20
dkf = (forcing_type == 10) ? 2 : nothing
ε = 0.1  
########################################################################
# initial condition 
noise_amplitude = 1e-3 ;
IC_type = 10 ;
### IC_type :   10 => zero + perturbations 
###             20 => random gaussian at kf TO DO
###             30 => zero
########################################################################
# advect passive tracer
add_tracer = true ;
########################################################################
# dealiasing
dealiasing = true ; 
aliased_fraction = dealiasing ? 1/3 : nothing
########################################################################
# numerical resolution parameters 
resol, L  = 8192, 2π             # grid resolution and domain length
ν, nν = 2e-14, 4             # hyperviscosity coefficient and hyperviscosity order
CFL = 0.4                   # CFL condition
dt = 0.005                 # timestep, should be larger for kolm forcing (forcing timescale)
t0 = 0.
########################################################################
# fourth-order timesteppers
timestepper = 10
### timestepper :  10 => rk2 exponential formulation
###                20 => explicit_rk4
###                30 => rk4 exponential formulation
###                40 => imex rk4
###                50 => etdrk2
########################################################################
# large-scale dissipation
friction_type = 20 ;
### friction_type :  10 => linear
###                  20 => quadratic
###                  30 => hypofriction
# will vary with friction type
κ = 1e-1                    # linear drag coefficient
μ = 6e-2*kf                 # quadratic drag coefficient
η = 1e-5   
########################################################################
# parameters for main loop
kk = 0               # Counter for visualization
Nstep = 1000       # Total number of state
NsaveFlowfield = 10000 # save solution in a .MAT file
Nfield = 1 # index of saved flowfield
NsaveEc = 100   # save timeseries every NsaveEc timesteps
Nfig = 1000     # save the vorticity and scalar field in a .png file every Nfig timestep
Nspectrum = 1000
########################################################################
