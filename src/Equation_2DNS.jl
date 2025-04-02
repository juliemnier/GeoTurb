__precompile__()
module Equation

    export Params, Vars, calcF!, compute_NL

    include("../utils/SpectralAnalysis.jl")
    using .SpectralAnalysis 
    using FourierFlows
    using CUDA
    using LinearAlgebra: ldiv!
    using LinearAlgebra: mul!
    using Printf
    using Random
    using CairoMakie
    using ColorSchemes
    using MAT
    using Plots: savefig


    """
    Largely inspired from GeophysicalFlows.jl for coding style/structure
    and function definition (but probably uglier)

    """

    struct Params{T} 
        "resolution"
        resol :: Int 
        "small-scale (hyper)-viscosity coefficient"
        ν :: T
        "(hyper)-viscosity order, `nν```≥ 1``"
        nν :: Int
        " friction type: 10 for linear, 20 for quadratic drag"
        friction_type :: Int
        "quadratic friction"
        μ :: Union{Nothing, T}
        "linear friction"
        κ :: Union{Nothing, T}
        "hypofriction"
        η :: Union{Nothing, T}
        " forcing type: 10 for white noise forcing, 20 for kolmogorov forcing"
        forcing_type :: Int 
        " forcing at wavenumber"
        kf :: Int
        " width of annulus in fourier space for wnF "
        dkf :: Union{Nothing, Int}
        " injected energy for white-noise forcing only "
        ε :: Union{Nothing, T}
        "Booleen for dealiasing"
        deal :: Bool
        "Dealiasing mask"
        mask :: Union{Nothing, AbstractArray} 
        "Booleen for passive tracer advection"
        add_tracer :: Bool
        " CFL parameter for adaptative timestep"
        CFL :: T
        " timestepper "
        timestepper :: Int
    end


    """
    Updated variables 

    """
    mutable struct Vars{T <: AbstractArray} 
        "Fourier transform of streamfunction"
        Fψ :: Union{Nothing, T}
        "Fourier transform of streamfunction, previous timestep or scratch variable"
        Fscratch :: Union{Nothing, T}
        "Streamfunction, real space scratch variable"
        scratch :: Union{Nothing, AbstractArray}
        "Friction term in -Δ⟂ψ equation"
        Ffrictionquad :: Union{Nothing, T}
        "Fourier transform of passive tracer τ"
        Fτ :: Union{Nothing, T}
        "Fourier transform of forcing"
        Fh :: Union{Nothing, T}
        "Timestep"
        time :: Float64
    end


    """ 
        initialize_field(grid, params :: Params, IC_type :: Int, noise_amplitude) 
    
    Initialize Vars (IC), handle the passive scalar field 

    """

    function initialize_field(grid, params :: Params, IC_type :: Int, noise_amplitude, restart_flag :: Bool, path_to_run::Union{String, Nothing}, filename::Union{String, Nothing})
        # initialize streamfunction
        
        if restart_flag
            # load .mat file
            file_path = path_to_run * filename

            if isfile(file_path)
                # load CI from .mat file
                CI_data = matread(file_path) 
                if haskey(CI_data, "Fpsi")
                    Fψ = CuArray(CI_data["Fpsi"])*params.resol^2
                else
                    println("Warning: Variable '$Fpsi' not found in '$file_name'.")
                end
                if params.add_tracer 
                    if haskey(CI_data, "Ftau")
                        Fτ = CuArray(CI_data["Ftau"])*params.resol^2
                    else
                        println("Warning: Variable '$Ftau' not found in '$file_name'.")
                    end
                end
            else
                println("Warning: File '$filename' not in current directory.")
            end
        else
            if IC_type == 10
                T = eltype(grid)
                Dev = typeof(grid.device)
                @devzeros Dev Complex{T} (size(grid.Krsq)) Fψ
                @devzeros Dev T (size(grid.rfftplan)) ψ
                @CUDA.allowscalar Fψ[2, 3] += noise_amplitude * (2*CUDA.rand(T) - 1 + 1im*(2*CUDA.rand(T) - 1))*params.resol^2
                @CUDA.allowscalar Fψ[3, 2] += noise_amplitude * (2*CUDA.rand(T)- 1 + 1im*(2*CUDA.rand(T) - 1))*params.resol^2
                @CUDA.allowscalar Fψ[2, 2] += noise_amplitude * (2*CUDA.rand(T) - 1 + 1im*(2*CUDA.rand(T) - 1))*params.resol^2
                @CUDA.allowscalar Fψ[1, 3] += noise_amplitude * (2*CUDA.rand(T) - 1 + 1im*(2*CUDA.rand(T) - 1))*params.resol^2
                # ensure that we got a real initial condition
                ldiv!(ψ, grid.rfftplan, Fψ)
                mul!(Fψ, grid.rfftplan, ψ) 
                # for optional τ
                Fτ = nothing
                if params.add_tracer
                    T = eltype(grid)
                    Dev = typeof(grid.device)
                    @devzeros Dev Complex{T} (size(grid.Krsq)) Fτ
                end
            end
        end
        if params.friction_type == 20
            Ffrictionquad = deepcopy(Fψ)
        else
            Ffrictionquad = nothing
        end

        Dev = typeof(grid.device)
        T = eltype(grid)
        @devzeros Dev T (size(grid.rfftplan)) scratch
        Fscratch = deepcopy(Fψ)

        return Fψ, Fτ, Ffrictionquad, Fscratch, scratch
    end 



    """
        calcF!(vars, params, grid, dt)
        
    Returns the forcing in fourier space. params.forcing_type can be 10 (white-noise-in-time forcing) 
    or 20 (Kolmogorov anisotropic forcing)

    """

    function calcF!(vars :: Vars, params :: Params, grid, dt)

        if params.forcing_type == 10

            # normalize forcing to inject energy at rate ε
            Dev = typeof(grid.device)
            T = eltype(grid)
            @devzeros Dev Complex{T} (grid.nkr, grid.nl) vars.Fh
            
            forcing_bandwidth  =  params.dkf 
            # energy input rate by the forcing
            forcing_spectrum = @. exp(-(sqrt(grid.Krsq) - params.kf)^2 / (2 * forcing_bandwidth^2))
            @CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average
        
            ε0 = inner_prod(Array(forcing_spectrum), Array(grid.invKrsq/2) , Array(grid.Krsq) , 0);
            @. forcing_spectrum *= params.ε/ε0  # normalize forcing to inject energy at rate ε
            
            # Generate random numbers on the device and perform element-wise operations
            rand_values = CUDA.rand(T, grid.nkr, grid.nl) 
        
            @. vars.Fh = sqrt(forcing_spectrum) * cis(2π * rand_values) / sqrt(dt)         
    
        elseif params.forcing_type == 20
            # directly in Fourier space
            Dev = typeof(grid.device)
            T = eltype(grid)
            @devzeros Dev Complex{T} (grid.nkr, grid.nl) vars.Fh
            @CUDA.allowscalar vars.Fh[1, params.kf + 1] = 1/2;
            @CUDA.allowscalar vars.Fh[1, end - params.kf + 1] = 1/2;
            # dft normalization
            @. vars.Fh *= params.resol^2
        end
    end


    """
        compute_L(params, grid)

    Return the linear term in the equation with `params` and `grid`. The linear
    operator ``L`` includes (hyper)-viscosity of order ``n_ν`` with coefficient ``ν`` and 
    hypo-viscocity of order ``n_μ`` with coefficient ``μ``,

    Plain-old viscocity corresponds to ``n_ν = 1`` while ``n_μ = 0`` corresponds to linear drag.

    The nonlinear term is computed via the function `compute_NL`.
    We add the option for quadratic drag in the compute_NL! function.

    """
    function compute_L(params :: Params, grid)
        Lψ = @. -params.ν * grid.Krsq^params.nν
        if params.friction_type == 10
            @. Lψ += - params.κ 
        end
        if params.friction_type == 30 # hypofriction
            @. L += - params.η * grid.invKrsq^2
        end
        CUDA.@allowscalar Lψ[1, 1] = 0
        if params.add_tracer
            # only includes small scale dissipation νh with hyperviscosity
            Lτ = @. - params.ν * grid.Krsq^params.nν
            CUDA.@allowscalar Lτ[1, 1] = 0
        else
            Lτ = nothing
        end
        return Lψ, Lτ
    end

    """
        compute_NL(sol, grid)

    Compute nonlinear term of streamfunction equation for intermediate timestep
    
    """

    function compute_NL(vars :: Vars, Fψn :: AbstractArray, Fτn :: Union{AbstractArray, Nothing}, params :: Params, grid)
     
        Dev = typeof(grid.device)
        T = eltype(grid)
        # initialize physical space arrays
        # initialization could be optimized, bit overkill
        # a minima on garde ces deux arrays qui servent au moins 3 fois chacuns
        @devzeros Dev T (size(grid.rfftplan)) ∂yψ
        @devzeros Dev T (size(grid.rfftplan)) ∂xψ
       # @devzeros Dev T (size(grid.rfftplan)) ζ
        
        # initialize fft arrays
        #@devzeros Dev Complex{T} (grid.nkr, grid.nl) FNL
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) Fuζ
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) Fvζ
    
        # advection 
        ldiv!(∂yψ, grid.rfftplan, @. im * grid.l * Fψn)
        ldiv!(∂xψ, grid.rfftplan, @. im * grid.kr * Fψn)

        ldiv!(vars.scratch, grid.rfftplan, @. grid.Krsq * Fψn) # ζ ifft stored in scratch

       
        mul!(vars.Fscratch, grid.rfftplan,  @. ∂yψ * vars.scratch) # \hat{u*ζ}
        FNL = @. - im * grid.kr * vars.Fscratch # initialize nonlinear term with fft of -dx(uζ) 
        mul!(vars.Fscratch, grid.rfftplan,  @. -∂xψ * vars.scratch ) # \hat{v*ζ}
        @. FNL += - im * grid.l * vars.Fscratch # fft of -dy(vζ) added to nonlinear term

        # add quadratic drag 
        if params.friction_type == 20
            # again, erase Fscratch
            mul!(vars.Fscratch, grid.rfftplan, @. (sqrt( ∂xψ^2 + ∂yψ^2 ) * ∂xψ))
            @. vars.Ffrictionquad = params.μ* im * grid.kr * vars.Fscratch # re-assign for next estimation at substep, add μ*dx(F∇ψ∂xψ)
             # again, erase Fscratch
            mul!(vars.Fscratch, grid.rfftplan, @. (sqrt( ∂xψ^2 + ∂yψ^2 ) * ∂yψ))  
            @. vars.Ffrictionquad += @. params.μ* im * grid.l * vars.Fscratch # add  μ*dy(F∇ψ∂yψ)
            @. FNL += vars.Ffrictionquad
        end
        
        # add forcing
       
        @. FNL += vars.Fh
      
        # switch to ψ equation
        @. FNL *=grid.invKrsq  # we just computed the non-linear term for the vorticity equation : switch to ψ equation
    
        # add passive tracer advection
        if params.add_tracer
            FNLτ = compute_NLτ(vars, Fτn, grid, ∂xψ, ∂yψ)
            Fτn = nothing
        else
            FNLτ = nothing # to avoid definition issue when add_tracer = False
        end
        ∂xψ = nothing
        ∂yψ = nothing
        Fψn = nothing
        return FNL, FNLτ
    end

    """
        compute_NLτ(sol, grid)

    Compute nonlinear term of passive tracer equation for intermediate timestep
    
    """

    function compute_NLτ(vars :: Vars, Fτn :: AbstractArray, grid, ∂xψ :: AbstractArray, ∂yψ :: AbstractArray)    
        
        ldiv!(vars.scratch, grid.rfftplan, deepcopy(Fτn)) # ifft τ stored in scratch
        mul!(vars.Fscratch, grid.rfftplan, @. ∂yψ * vars.scratch) # \hat{u*τ}
        FNLτ = @. - im * grid.kr * vars.Fscratch # add - dx(u*τ) to RHS
        
        mul!(vars.Fscratch, grid.rfftplan, @. -∂xψ  * vars.scratch ) # \hat{v*τ}
        
        @. FNLτ += @. - im * grid.l * vars.Fscratch # add - dy(v*τ) to RHS
 

        return FNLτ
    end


    """  
    compute_RHS

    comptes right-hand-side of ψ equation for explicit timesteppers

    """
    function compute_RHS(vars :: Vars, params :: Params, grid, Fψn :: AbstractArray, Fτn :: Union{Nothing, AbstractArray}, Lψ :: AbstractArray, 
        Lτ :: Union{Nothing, AbstractArray})

        # add linear term
        slope_Fψ = @. Lψ * Fψn
        
        # add non-linear term
        FNL, FNLτ = compute_NL(vars, Fψn, Fτn, params, grid)
        @. slope_Fψ += FNL
            
        if params.add_tracer
            # add linear term and mean gradient of 1 (-v in RHS of τ equation)
            slope_Fτ = @. Lτ * Fτn + im * grid.kr * Fψn
            @. slope_Fτ += FNLτ 
        else
            slope_Fτ = nothing
        end
        return slope_Fψ, slope_Fτ
    end

    """
        rk4_imex_timestepper(vars, param, L, Lτ, grid, dt)

    Solves any divergence-free problem expressed with streamfunction ψ on `grid` and returns the updated variables.
    Optionally includes passive tracer advection.
    Linear terms are always treated implicitly.
    Hoped to be memory efficient for GPU usage; should actually check with FourierFlows timestepper.
    """

    function rk4_imex_timestepper!(vars :: Vars , params :: Params, Lψ :: AbstractArray,
                            Lτ :: Union{Nothing, AbstractArray}, grid, dt :: AbstractFloat)
        """
        See above for function description. Ad-hoc. would require an exponential formulation
        to reach more that 2nd order precision. fails to capture rapid exponential decay if |Lψ|<<1
        """
        
        Dev = typeof(grid.device)
        T = eltype(grid)

        # Initialize relevant arrays for imex rk4
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) FNLf # initialize ponderated non-linear term
        Fψ0 = deepcopy(vars.Fψ) # copy solution at current timestep
        # for passive tracer (optional)
        Fτ0 = params.add_tracer ? deepcopy(vars.Fτ) : nothing # option of advecting a passive tracer, initialization 
        params.add_tracer ? (@devzeros Dev Complex{T} (grid.nkr, grid.nl) FNLτf) : nothing # initialize ponderated non-linear term 
    
        # weights and coefficients for classical explicit rk4 method
        order = [0.5 0.5 1]
        pond = [1/6 1/3 1/3 1/6]
        
        for irk4 in range(1,length(order))
            
            FNL, FNLτ = compute_NL(vars, vars.Fψ, vars.Fτ, params, grid)
            @. FNLf += pond[irk4] * FNL # ponderate
           
            # compute slope estimation for intermediate timestep
            @. vars.Fψ = (Fψ0 + dt*order[irk4]*FNL)/(1 - dt*order[irk4]*Lψ)

            # dealiase intermediate solution
            params.deal ? (@. vars.Fψ *= params.mask) : nothing
            
            if params.add_tracer
                # advecting a passive tracer with mean gradient

                #@. vars.Fτ = (Fτ0+dt*order[irk4]*FNLτ)/(1 - dt*order[irk4]*Lτ)
                @. vars.Fτ = (Fτ0+dt*order[irk4]*(FNLτ + im * grid.kr * vars.Fψ))/(1-dt*order[irk4]*Lτ) 
                # weighted average
                @. FNLτf += pond[irk4] * FNLτ
                # dealiase intermediate solution
                params.deal ?  (@. vars.Fτ *= params.mask) : nothing
                FNLτ = nothing
            end
            FNL = nothing
        end
        
        #irk4 is 4 now for last ponderation step (k4)
        FNL, FNLτ = compute_NL(vars, vars.Fψ, vars.Fτ, params, grid)
           
        #first Fpsi to be able to handle it implicitely in Ftau equation
        @. FNLf +=  pond[end] * FNL;
        FNL = nothing
        # update with ponderated estimation of the slope
        @. vars.Fψ = (Fψ0 +  dt * FNLf) / (1 - dt*Lψ) 
        
        # dealiase
        params.deal ? (@. vars.Fψ *= params.mask) : nothing

        if params.add_tracer
            #then update with total estimation
            @. FNLτf += pond[end] * FNLτ
            @. vars.Fτ = (Fτ0 + dt*(FNLτf + im * grid.kr * vars.Fψ)) / ( 1 - dt*Lτ )
           
            # dealiase
            params.deal ? (@. vars.Fτ *= params.mask) : nothing
        end
        return

    end


    """ IMEX rk2 timestepper

    """

    function etdrk2_timestepper!(vars :: Vars , params :: Params, Lψ :: AbstractArray,
        Lτ :: Union{Nothing, AbstractArray}, grid, dt :: AbstractFloat)
        """
        Memory efficient implementation of etdrk2
        """

        # Initialize relevant arrays 
        Fψ0 = deepcopy(vars.Fψ) # copy solution at current timestep
        # for passive tracer (optional)
        Fτ0 = params.add_tracer ? deepcopy(vars.Fτ) : nothing # option of advecting a passive tracer, initialization 
        
        # first step  
        FNL, FNLτ = compute_NL(vars, vars.Fψ, vars.Fτ, params, grid)   
        @. vars.Fscratch = exp(Lψ*dt/2)
        @. vars.Fψ = @. vars.Fscratch.*Fψ0 + (vars.Fscratch - 1)/Lψ * FNL #intermediate
        params.deal ? (@. vars.Fψ *= params.mask) : nothing
        if params.add_tracer
            @. vars.Fscratch = exp(Lτ*dt/2)
            @. vars.Fτ = @. vars.Fscratch.*Fτ0 + (vars.Fscratch - 1)/Lτ * FNLτ #intermediate
            params.deal ? (@. vars.Fτ *= params.mask) : nothing
        end

        # second step, use Fψ0 as scratch 
        @. Fψ0 =  @. exp(Lψ*dt) *Fψ0 + (exp(Lψ*dt) - 1)/Lψ * FNL - 2*(exp(Lψ*dt) - 1 - Lψ*dt)/(Lψ.^2*dt) * FNL
        if params.add_tracer
            # again, use Fτ0 as scratch
            @. Fτ0 =  @. exp(Lτ*dt) *Fτ0 + (exp(Lτ*dt) - 1)/Lτ* FNLτ - 2*(exp(Lτ*dt) - 1 - Lτ*dt)/(Lτ.^2*dt) * FNLτ
        end
        FNL, FNLτ = compute_NL(vars, vars.Fψ, vars.Fτ, params, grid) # uses Fscratch, that's why we used Fψ0 on the previous line
        # re-assign Fψ
        @. vars.Fψ = @. Fψ0 + 2*(exp(Lψ*dt) - 1 - Lψ*dt)/(Lψ.^2*dt) * FNL
        if params.add_tracer
            #re-assign Fτ
            @. vars.Fτ = @. Fτ0 + 2*(exp(Lτ*dt) - 1 - Lτ*dt)/(Lτ.^2*dt) * FNLτ
        end
   
        Fψ0 = nothing
        Fτ0 = nothing

    return
    end

    """ IMEX ETDRK4 

    """


    function etdrk4_timestepper!(vars :: Vars , params :: Params, Lψ :: AbstractArray,
        Lτ :: Union{Nothing, AbstractArray}, grid, dt :: AbstractFloat)
    """
    See above for function description
    """
   

    return
    end
    """
        explicit rk4 timestepper
        not well suited for stiff ODE but good benchmark
    """
    

    function rk4_explicit_timestepper!(vars :: Vars , params :: Params, Lψ :: AbstractArray, 
        Lτ :: Union{Nothing, AbstractArray}, grid, dt :: AbstractFloat)
        """
        See above for function description
        """
        vars.Fψ0 = deepcopy(vars.Fψ) # copy solution at current timestep
        if params.add_tracer
            Fτ0 = deepcopy(vars.Fτ)
            @devzeros Dev Complex{T} (grid.nkr, grid.nl) final_slope_τ
            @devzeros Dev Complex{T} (grid.nkr, grid.nl) current_slope_τ
        else 
            Fτ0 = nothing
            final_slope_τ = nothing
            current_slope_τ = nothing
        end

        Dev = typeof(grid.device)
        T = eltype(grid)
        # Initialize required slopes for classical rk4 substeps / low storage 
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) final_slope # initialize ponderated slope
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) current_slope # initialize ponderated slope
       
        order = [0 0.5 0.5 1]
        pond = [1/6 1/3 1/3 1/6]
        
        for ii in range(1,4)
            current_slope, current_slope_τ = compute_RHS(vars, params, grid, vars.Fψ0 .+ order[ii]*dt*current_slope, Fτ0 .+ order[ii]*dt*current_slope_τ, Lψ, Lτ)
            
            @. final_slope += current_slope * pond[ii]
            if params.add_tracer
                @. final_slope_τ += current_slope_τ * pond[ii]
            end
        end        

        # update solution with ponderated slope
        @. vars.Fψ = vars.Fψ0 +  dt * final_slope
        if params.add_tracer
            @. vars.Fτ = Fτ0 + dt * final_slope_τ
        end
        # dealiase solution
        params.deal ? (@. vars.Fψ *= params.mask) : nothing
        if params.add_tracer
            params.deal ? (@. vars.Fτ *= params.mask) : nothing
        end
        return
    end



    """ step_forward!(vars :: Vars, params :: Params, L :: AbstractArray, Lτ :: Union{Nothing, AbstractArray}, timestepper :: string, grid, dt)

    stepforwards the equation using the chosen timestepper 

    """

    function step_forward!(vars :: Vars, params :: Params, L :: AbstractArray, Lτ :: Union{Nothing, AbstractArray}, grid, dt)
        start = time_ns()
        # update the forcing if white-noise-in-time
        if params.forcing_type == 10
            vars.Fh = calcF!(vars, params, grid, dt)
        end
        # update solution
        if params.timestepper == 10
            etdrk2_timestepper!(vars, params, L, Lτ, grid, dt)
        end
        if params.timestepper == 20
            rk4_explicit_timestepper!(vars, params, L, Lτ, grid, dt)
        end
        if params.timestepper == 30
            println("ETDRK4 source in writing, choose other timestepper")
            exit()
        end
        if params.timestepper == 40
            rk4_imex_timestepper!(vars, params, L, Lτ, grid, dt)
        end

        stop = time_ns()
        elapsed_time = (stop- start)/1.0e9
        # CFL criteria for next timestep 
        KE, dt = CFL_update_timestep(vars, params, grid, dt)
        # effectively update timestep
        vars.time = vars.time + dt
        return vars, KE, dt, elapsed_time
    end

    """ CFL_update_timestep(vars :: Vars, params :: Params, dt)

    returns adaptative timestep with CFL condition from current solution

    """

    function CFL_update_timestep(vars :: Vars, params :: Params, grid, dt0)
        """ Returns new timestep from CFL conditions and KE 
        """
        # Computing horizontal kinetic energy for CFL condition
        ldiv!(vars.scratch, grid.rfftplan, @. im * grid.l * vars.Fψ) # ifft of u=dyψ stored in scratch
        Ecloc= @. vars.scratch^2 # add u^2 to KE 
        ldiv!(vars.scratch, grid.rfftplan, @. im * grid.kr * vars.Fψ)# ifft of v=-dxψ stored in scratch
        @. Ecloc += vars.scratch^2 # add v^2 to KE 
        dt = minimum([dt0, params.CFL * 2 * π / params.resol / sqrt(maximum(Array(Ecloc)))])
        # updating time step
        KE = sum(Ecloc) / (params.resol * params.resol)
        return KE, dt

    end


    function run_simulation!(vars :: Vars, params :: Params, Lψ :: AbstractArray, 
        Lτ :: Union{Nothing, AbstractArray}, grid, Nfield :: Int, NsaveEc :: Int,
         Nstep :: Int, NsaveFlowfield :: Int, Nfig :: Int, NSpectrum :: Int, dt, path_to_run)
        counterEc = 0
        countSp = 0
        for ii in 1:Nstep
            vars, KE, dt, elapsed_time = step_forward!(vars, params, Lψ, Lτ, grid, dt)
            counterEc+=1
            if counterEc == NsaveEc
                counterEc = 0
                SSD, LSD, inject, D, tau2 = energy_budget_2DNS(vars, params, grid, dt)
                # Open the file for writing or appending
                if round(Int, ii / NsaveEc) == 1
                    open("energy_bal.txt", "w") do file
                        write(file, string(vars.time, " ", KE, " ", dt, " ",inject, " ", SSD, " ", LSD, " ",tau2, " ",D, " ",elapsed_time, "\n"))
                    end
                else
                    open("energy_bal.txt", "a") do file
                        write(file, string(vars.time, " ", KE, " ", dt, " ",inject, " ", SSD, " ", LSD, " ", tau2, " ",D, " ",elapsed_time, "\n"))
                    end
                end
            end
        
            if mod(ii, Nfig) == 0
                save_vort_flowfield_png_2DNS(vars, grid, vars.time, :RdBu_5, path_to_run)
                if params.add_tracer
                    save_tau_flowfield_png_2DNS(vars, grid, vars.time, :RdBu_5, path_to_run)
                end
            end
        
            if mod(ii, NsaveFlowfield) == 0
                save_flowfield(vars, Nfield, path_to_run)
                Nfield+=1
            end
            if mod(ii, NSpectrum) == 0
                save_spectrum(vars, grid, countSp, path_to_run)
                countSp+=1
            end
        
        end
    end

    """     
        save_flowfield(vars :: Vars, Nfield :: Int, path_to_run)
    
    save flowfields as .MAT file in directory "./Fields" 
    """
    function save_flowfield(vars :: Vars, Nfield :: Int, path_to_run)
        filename = path_to_run * "Fields/Flowfield_" * @sprintf("%03d", Nfield) * ".mat"
        matwrite(filename, Dict(
            "Fpsi" => Array(vars.Fψ),
            "Ftau" => Array(vars.Fτ),
            "Nfield" => Nfield,
            "time" => vars.time
        ))
        return
    end

    """     
        save_spectrum(vars :: Vars, countSp :: Int, path_to_run)
    
    save energy spectrum as .MAT file in directory "./Fields" 
    """
    function save_spectrum(vars :: Vars,  grid, countSp :: Int, path_to_run)
        
        rSpecveloc,kmoy = radial_spectrum(Array(vars.Fψ), Array(grid.Krsq), Int(1))
        filename = path_to_run * "Fields/spectrum_" * @sprintf("%03d", countSp) * ".mat"
        matwrite(filename, Dict(
            "rSpecveloc" => Array(rSpecveloc),
            "kmoy" => Array(kmoy),
            "time" => vars.time
        ))
        return
    end


    """
        save_vort_flowfield_png_2DNS(vars :: Vars, grid, timestep)
    
    Save snapshot of vorticity at chosen timestep. Eventually make this more general. 

    """

    function save_vort_flowfield_png_2DNS(vars :: Vars, grid, timestep, choice_colormap, path_to_run)
        """ Save snapshot of vorticity at chosen timestep
        """
        Δψ = deepcopy(grid.Ksq)
        FΔψ = @. grid.Krsq * vars.Fψ
        ldiv!(Δψ, grid.rfftplan, FΔψ)
        fig = Figure()
        ax = Axis(fig[1, 1], 
            title = L"$\Delta \psi$", 
            titlesize = 40,
            xlabel = L"$X$", 
            ylabel = L"$Y$",
            aspect = DataAspect()
        )
        
        hm = heatmap!(ax, 
            grid.x, grid.y, Array(Δψ), 
            colormap = choice_colormap,                   
            colorrange = extrema(Array(Δψ))
            )
        Colorbar(fig[1, 2], hm, ticklabelsize = 18)
        
        save(path_to_run * "Snapshots/Snapshot_vort_" * lpad(string(timestep), 6, '0') * ".png", fig)
        return
    end

    function save_tau_flowfield_png_2DNS(vars :: Vars, grid, timestep, choice_colormap, path_to_run)
        """ Save snapshot of vorticity at chosen timestep
        """
        τ = deepcopy(grid.Ksq)
        ldiv!(τ, grid.rfftplan, deepcopy(vars.Fτ))
        fig = Figure()
        ax = Axis(fig[1, 1], 
            title = L"$\tau$", 
            titlesize = 40,
            xlabel = L"$X$", 
            ylabel = L"$Y$",
            aspect = DataAspect()
        )
        
        hm = heatmap!(ax, 
            grid.x, grid.y, Array(τ), 
            colormap = choice_colormap,                   
            colorrange = extrema(Array(τ))
            )
        Colorbar(fig[1, 2], hm, ticklabelsize = 18)
        
        save(path_to_run * "Snapshots/Snapshot_tau_" * lpad(string(timestep), 6, '0') * ".png", fig)
        return
    end

    """ 
        energy_budget_2DNS(vars :: Vars, params :: Params, grid ::TwoDGrid, dt :: Float64)
    
    Energy budget for 2DNS computed from vorticity equation, multiplied by ψ then averaged.
    
    """

    function energy_budget_2DNS(vars :: Vars, params :: Params, grid ::TwoDGrid, dt :: Float64)
        """ Computes and prints every Nstep timesteps the energy budget of the 2DNS 
            vorticity equation for two types of friction
        """
        k2 = Array(grid.Krsq)
        #Em  = real(inner_prod(Array(vars.Fψ),  Array(grid.Krsq.*vars.Fψ), k2, 0))        
        #Em0 = real(inner_prod(Array(vars.Fψ0),  Array(grid.Krsq.*vars.Fψ0), k2, 0)) 

        SSD = params.ν  * energy(Array(vars.Fψ), k2, 5)
        if params.friction_type == 10
            LSD = params.κ*energy(Array(vars.Fψ),k2,1)
        elseif params.friction_type == 20
            LSD = -real(inner_prod(Array(vars.Fψ),Array(vars.Ffrictionquad), k2, 0)) 
        end
        if params.forcing_type == 10
            inject = real(inner_prod(Array((vars.Fψ + vars.Fψ0)/2), Array(vars.Fh), k2, 0))
        elseif  params.forcing_type == 20
            inject = real(inner_prod(Array(vars.Fψ),  Array(vars.Fh), k2, 0))
        end
        D = nothing
        if params.add_tracer
            D = real(inner_prod(Array(1im * grid.kr .* vars.Fψ), Array(vars.Fτ), k2, 0))
            tau2 = real(inner_prod(Array(vars.Fτ), Array(vars.Fτ), k2, 0))

        end
       
       # LHS = (Em - Em0)/dt/2 # 1/2 (∇ψ)^2 
       # RHS = inject - SSD - LSD
       
        return SSD, LSD, inject, D, tau2
    end

    
end