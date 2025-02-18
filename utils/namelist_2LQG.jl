########################################################################
# choose CPU or GPU below
dev = GPU()
########################################################################
path_to_run = "./" 
restart_filename ="CI.mat"
restart_flag = false
flag_2LQG = true
########################################################################
# forcing parameters
########################################################################
# QG parameters
kd = 20
λ = 1/kd
Ro = 1
########################################################################
# Optional white-noise
add_wn = false
### forcing_type :   10 => White noise forcing 
###                  20 => Kolmogorov forcing
# will vary with forcing_type
kf = 20
dkf =  2
ε = 0.1  
########################################################################
# initial condition 
noise_amplitude = 1e-6 ;
IC_type = 10 ;
### IC_type :   10 => zero + perturbations 
###             20 => random gaussian at kf TO DO
###             30 => zero
########################################################################
# dealiasing
dealiasing = true ; 
aliased_fraction = dealiasing ? 1/3 : nothing
########################################################################
# numerical resolution parameters 
resol, L  = 512, 2π             # grid resolution and domain length
ν, nν = 1e-14, 4             # hyperviscosity coefficient and hyperviscosity order
CFL = 0.4                   # CFL condition
dt = 0.0001                 # timestep, should be larger for kolm forcing (forcing timescale)
t0 = 0.
########################################################################
# fourth-order timesteppers
timestepper = 10
### timestepper :  10 => imex_rk4
###                20 => explicit_rk4
###                30 => etdrk4
########################################################################
# large-scale dissipation
friction_type = 20 ;
### friction_type :  10 => linear
###                  20 => quadratic
###                  30 => hypofriction
# will vary with friction type. only modify the type of friction you want
κ = 1e-1                    # linear drag coefficient
μ = 0.3*kf                 # quadratic drag coefficient
########################################################################
# parameters for main loop
kk = 0               # Counter for visualization
Nstep = 200000       # Total number of state
NsaveFlowfield = 10000 # save solution in a .MAT file
Nfield = 1 # index of saved flowfield
NsaveEc = 1000   # save timeseries every NsaveEc timesteps
Nfig = 10000     # save the vorticity and scalar field in a .png file every Nfig timestep
Nspectrum = 10000
########################################################################
