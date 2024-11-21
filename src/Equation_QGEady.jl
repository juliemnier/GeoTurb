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

    # TODO: decider que faire de inversion_map. comment écrire le bilan QG ? L1 et L0 à initialiser avant d'appeler
    # Params (immutable)
    """
    Largely inspired from GeophysicalFlows.jl for coding style/structure
    and function definition (but probably uglier)

    """

    struct Params{T} <: AbstractParams
        "resolution"
        resol :: Int 
        "small-scale (hyper)-viscosity coefficient"
        ν :: T
        "(hyper)-viscosity order, `nν```≥ 1``"
        nν :: Int
        " rossby deformation radius ~ forcing scale of baroclinic instability"
        λ :: T
        " rossby number "
        Ro :: T
        " friction type: 10 for linear, 20 for quadratic drag"
        friction_type :: Int
        "quadratic friction"
        μ :: Union{Nothing, T}
        "linear friction"
        κ :: Union{Nothing, T}
        " Booleen for optional white-noise forcing "
        add_wn :: Bool
        " forcing at wavenumber"
        kf :: Union{Nothing, Int}
        " size of narrow-band for white-noise forcing "
        dkf :: Union{Nothing, Int}
        " injected energy for white-noise forcing only "
        ε :: Union{Nothing, T}
        "Booleen for dealiasing"
        deal :: Bool
        "Dealiasing mask"
        mask :: Union{Nothing, AbstractArray} #!!! redondant
        "Booleen for passive tracer advection"
        add_tracer :: Bool
        " CFL parameter for adaptative timestep"
        CFL :: T
        " Bottom inversion matrix "
        L0 :: Union{Nothing, AbstractArray}
        " Top inversion matrix "
        L1 :: Union{Nothing, AbstractArray}
        " Bottom inversion matrix "
        
    end



    """
    Updated variables 

    """
    mutable struct Vars{T <: AbstractArray} <: AbstractVars
        "Fourier transform of bottom buoyancy"
        Fb0 :: Union{Nothing, T}
        "Fourier transform of top buoyancy"
        Fb1 :: Union{Nothing, T}
        "Fourier transform of bottom streamfunction"
        Fψ0 :: Union{Nothing, T}
        "Fourier transform of top streamfunction"
        Fψ1 :: Union{Nothing, T}
        "Friction term in -Δ⟂ψ equation"
        Ffrictionquad :: Union{Nothing, T}
        "Fourier transform of forcing"
        Fh :: Union{Nothing, T}
        "Timestep"
        time :: Float64
    end

    
    """ 
        initialize_field(grid, params :: Params, IC_type :: Int, noise_amplitude) 

    Initialize Vars (IC), handle the passive scalar field 

    """

        function initialize_field(grid, params :: Params, IC_type :: Int, noise_amplitude)
            
            # initialize Fb0, Fb1

            if restart_flag
                # Step 1: Load the .MAT file
                file_path = path_to_run * filename
        
                if isfile(file_path)
                    # load CI from .mat file
                    CI_data = matread(file_path) 
                    if haskey(CI_data, "Fb0") && haskey(CI_data, "Fb1")
                        Fb0 = CuArray(CI_data["Fb0"])
                        Fb1 = CuArray(CI_data["Fb1"])
                    else
                        println("Warning: Variable Fb0 or Fb1 not found in '$file_name'.")
                    end
                    if params.add_tracer 
                        if haskey(CI_data, "Ftau")
                            Fτ = CuArray(CI_data["Ftau"])
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

                    @devzeros Dev Complex{T} (size(grid.rfftplan)) b0
                    @devzeros Dev Complex{T} (size(grid.rfftplan)) b1

                    Fb0 = noise_amplitude *(1.0*(2*CUDA.rand(params.resol/2+1,params.resol)-1+im*(2*CUDA.rand(params.resol/2+1,params.resol)-1)))
                    @. Fb0 = Fb0 / grid.Krsq.^2
                    @CUDA.allowscalar Fb0[1,1] = 0
                    Fb1= noise_amplitude *(1.0*(2*CUDA.rand(params.resol/2+1,params.resol)-1+im*(2*CUDA.rand(params.resol/2+1,params.resol)-1)))
                    @. Fb1 = Fb1 / grid.Krsq.^2
                    @CUDA.allowscalar Fb1[1,1] = 0
                    
                    # ensure that we got a real initial condition
                    ldiv!(b0, grid.rfftplan, Fb0)
                    mul!(Fb0, grid.rfftplan, b0) 
                    ldiv!(b1, grid.rfftplan, Fb1)
                    mul!(Fb1, grid.rfftplan, b1) 

                end
                # for optional τ
                Fτ = nothing
                if params.add_tracer
                    T = eltype(grid)
                    Dev = typeof(grid.device)
                    @devzeros Dev Complex{T} (size(grid.Krsq)) Fτ
                end
            end

            return Fb0, Fb1, Fτ
        end 


    """
        calcF!(vars, params, grid, dt)
        
    Returns the optional white-noise-in-time forcing in fourier space

    """

    function calcF!(vars, params, grid, dt)

        if params.add_wn
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
        end
        return
    end


    function initialize_inversion_map(params :: Params, grid)

        L0 = @. (-params.kd / sqrt(grid.Krsq)) / tanh(sqrt(grid.Krsq)/params.kd)
        CUDA.@allowscalar L0[1, 1] = 0
        L1 = @. (-params.kd /sqrt(grid.Krsq)) / sinh(sqrt(grid.Krsq)/params.kd)
        CUDA.@allowscalar L1[1, 1] = 0

        return L0, L1

    end


    """
        inversion_ψb!(vars, params, grid)
    Returns the top and bottom streamfunction given the top and bottom buoyancy Fields

    """

    function initialize_inversion!(Fb0 :: AbstractArray, Fb1 :: AbstractArray, params :: Params)
               
        @. vars.Fψ0 = params.L0 * Fb0 - params.L1 * Fb1
        @. vars.Fψ1 = params.L1 * Fb0 - params.L0 * Fb1

        return 
    end

    """
        compute_L(params, grid)

    Returns the linear term in the equation with `params` and `grid`. The linear
    operator ``L`` includes (hyper)-viscosity of order ``n_ν`` with coefficient ``ν`` and 
    hypo-viscocity of order ``n_μ`` with coefficient ``μ``,

    Plain-old viscocity corresponds to ``n_ν = 1`` while ``n_μ = 0`` corresponds to linear drag.

    The nonlinear term is computed via the function `compute_NL`.
    We add the option for quadratic drag in the compute_NL! function.

    """
    function compute_L(params, grid)

        if params.friction_type == 20
            
            L01 = nothing # here one should define L10 for linear drag
    
            L10 = @. im * grid.kr * params.Ro * params.L1 
            Lb0 = @.  im *params.Ro*grid.kr/2 + params.L0.* (im*grid.kr*params.Ro) -params.ν * grid.Krsq^params.nν
            Lb1 = @. -im*params.Ro*grid.kr/2 - params.L0 * (im*grid.kr*params.Ro) -params.ν * grid.Krsq^params.nν

            CUDA.@allowscalar L10[1, 1] = 0
            CUDA.@allowscalar Lb1[1, 1] = 0
            CUDA.@allowscalar Lb0[1, 1] = 0

        end
        

        CUDA.@allowscalar Lψ[1, 1] = 0
        if params.add_tracer
            # only includes small scale dissipation νh with hyperviscosity
            #TODO : write at some point
            #Lτ = @. - params.ν * grid.Krsq^params.nν
            #CUDA.@allowscalar Lτ[1, 1] = 0
        else
            Lτ = nothing
        end

        return L10, Lb0, Lb1, L01, Lτ

    end   


    """
        compute_NL(sol, grid)

    Compute nonlinear term of streamfunction equation for intermediate timestep
    
    """

    function compute_NL(vars :: Vars, Fb0n :: AbstractArray, Fb1n :: AbstractArray, Fτn :: Union{AbstractArray, Nothing}, params :: Params, grid)
        
        inversion_ψb!(Fb0n, Fb1n, params)

        Dev = typeof(grid.device)
        T = eltype(grid)
        # initialize physical space arrays
        # initialization could be optimized, bit overkill
        @devzeros Dev T (size(grid.rfftplan)) ∂yψ
        @devzeros Dev T (size(grid.rfftplan)) ∂xψ
        @devzeros Dev T (size(grid.rfftplan)) b
        
        # initialize fft arrays
        #@devzeros Dev Complex{T} (grid.nkr, grid.nl) FNL
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) Fub
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) Fvb
    
        # advection for top layer first
        ldiv!(∂yψ, grid.rfftplan, @. im * grid.l * Fψ1n)
        ldiv!(∂xψ, grid.rfftplan, @. im * grid.kr * Fψ1n)
        ldiv!(b, grid.rfftplan, Fb1n)

        ub = @. ∂yψ * b        
        vb = @. -∂xψ * b 

        mul!(Fub, grid.rfftplan, ub) # \hat{u1*b1}
        mul!(Fvb, grid.rfftplan, vb) # \hat{v1*b1}
        
        FNL1 = @. - im * grid.kr * Fub - im * grid.l * Fvb

        # use 1-variables as scratch variables
        # advection for bottom layer 
        ldiv!(∂yψ, grid.rfftplan, @. im * grid.l * Fψ0n)
        ldiv!(∂xψ, grid.rfftplan, @. im * grid.kr * Fψ0n)
        ldiv!(b, grid.rfftplan, Fb0n)
        
        @. ub = ∂yψ * b        
        @. vb = -∂xψ * b 
    
        mul!(Fub, grid.rfftplan, ub) # \hat{u0*b0}
        mul!(Fvb, grid.rfftplan, vb) # \hat{v0*b0}
       
        FNL0 = @. - im * grid.kr * Fub - im * grid.l * Fvb
        

        # add quadratic drag 
        if params.friction_type == 20
            @devzeros Dev Complex{T} (grid.nkr, grid.nl) F∇ψ0∂xψ0 
            @devzeros Dev Complex{T} (grid.nkr, grid.nl) F∇ψ0∂yψ0

            mul!(F∇ψ0∂xψ0, grid.rfftplan, @. (sqrt( ∂xψ^2 + ∂yψ^2 ) * ∂xψ))
            mul!(F∇ψ0∂yψ0, grid.rfftplan, @. (sqrt( ∂xψ^2 + ∂yψ^2 ) * ∂yψ))  
    
            @. vars.Ffrictionquad =  - params.μ*(im * grid.kr * F∇ψ0∂xψ0 + im * grid.l * F∇ψ0∂yψ0)
            @. FNL0 += vars.Ffrictionquad

        end
        
        # add forcing
        if params.add_wn
            # TODO
            @. FNL0 += vars.Fh/sqrt(vars.substep)
            @. FNL1 += vars.Fh/sqrt(vars.substep)
        end
       
        # add passive tracer advection
        if params.add_tracer
            # TODO 
            FNLτ = compute_NLτ(Fτn, grid, ∂xψ, ∂yψ)
        else
            FNLτ = nothing # to avoid definition issue when add_tracer = False
        end

        return FNL0, FNL1, FNLτ
    end

    """
        compute_NLτ(sol, grid)

    Compute nonlinear term of passive tracer equation for intermediate timestep
    
    """

    function compute_NLτ( Fτn :: AbstractArray, grid, ∂yψ :: AbstractArray, ∂xψ :: AbstractArray)    
        # TO DO!!
        Dev = typeof(grid.device)
        T = eltype(grid)
        @devzeros Dev T (size(grid.rfftplan)) τ
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) Fuτ
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) Fvτ

        ldiv!(τ, grid.rfftplan, deepcopy(Fτn))
        # add advection for passive tracer
        τu = @. ∂yψ * τ       # in physical space first  
        τv = @. -∂xψ  * τ     # in physical space first  

        mul!(Fuτ, grid.rfftplan, τu) # \hat{u*τ}
        mul!(Fvτ, grid.rfftplan, τv) # \hat{v*τ}

        FNLτ = @. - im * grid.kr * Fuτ - im * grid.l * Fvτ
        # add mean gradient, here handled explicitely
        @. FNLτ += im * grid.kr * vars.Fψ

        return FNLτ
    end


  

    """
        rk4_imex_timestepper(vars, param, L, Lτ, grid, dt)

    Solves any divergence-free problem expressed with streamfunction ψ on `grid` and returns the updated variables.
    Optionally includes passive tracer advection.
    Linear terms are always treated implicitly.
    Hoped to be memory efficient for GPU usage; should actually check with FourierFlows timestepper.
    """

    function rk4_imex_timestepper!(vars :: Vars , params :: Params, L10 :: AbstractArray, Lb0 :: AbstractArray ,
                             Lb1 :: AbstractArray , L01 :: Union{Nothing, AbstractArray} , 
                             Lτ :: Union{Nothing, AbstractArray}, grid, dt :: AbstractFloat)
        """
        See above for function description
        """
        Dev = typeof(grid.device)
        T = eltype(grid)

        # Initialize relevant arrays for imex rk4
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) FNLf0 # initialize ponderated non-linear term
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) FNLf1 # initialize ponderated non-linear term
        
        vars.Fb0 = deepcopy(vars.Fb0) # copy solution at current timestep
        vars.Fb1 = deepcopy(vars.Fb1) # copy solution at current timestep
        # for passive tracer (optional)
        Fτ0 = params.add_tracer ? deepcopy(vars.Fτ) : nothing # option of advecting a passive tracer, initialization 
        params.add_tracer ? (@devzeros Dev Complex{T} (grid.nkr, grid.nl) FNLτf) : nothing # initialize ponderated non-linear term # TEST
    
        # weights and coefficients for classical explicit rk4 method
        order = [0.5 0.5 1]
        pond = [1/6 1/3 1/3 1/6]
        
        for irk4 in range(1,length(order))
            vars.substep = order[irk4]
            FNL0, FNL1, FNLτ = compute_NL(vars, vars.Fb0, vars.Fb1, vars.Fτ, params, grid)
            @. FNLf0 += pond[irk4] * FNL0 # ponderate
            @. FNLf1 += pond[irk4] * FNL1 # ponderate
           
            # compute slope estimation for intermediate timestep
            if params.friction_type == 20
                # otherwise the term L01 must be computed
                @. vars.Fb0 =((1 - dt*order[irk4]*Lb1) * (vars.Fb0 + dt*order[irk4]*FNL0) 
                    - dt*order[irk4]*L10*(vars.Fb1 + dt*order[irk4]*FNL1))/(1 - dt*order[irk4]*(Lb1+Lb0) + (dt*order[irk4])^2*(Lb0*Lb1 + L10^2))
                @. vars.Fb1 =(dt*order[irk4]*L10 * (vars.Fb0 + dt*order[irk4]*FNL0) 
                    + (1 - dt*order[irk4]*Lb0)*(vars.Fb1 + dt*order[irk4]*FNL1))/(1 - dt*order[irk4]*(Lb1+Lb0) + (dt*order[irk4])^2*(Lb0*Lb1 + L10^2))
            end
            
            # dealiase intermediate solution
            params.deal ? (@. vars.Fb0 *= params.mask) : nothing
            params.deal ? (@. vars.Fb1 *= params.mask) : nothing
            
            if params.add_tracer
                # advecting a passive tracer with mean gradient
                @. vars.Fτ = (Fτ0+dt*order[irk4]*FNLτ)/(1-dt*order[irk4]*Lτ) 
                # weighted average
                @. FNLτf += pond[irk4] * FNLτ
                # dealiase intermediate solution
                params.deal ?  (@. vars.Fτ *= params.mask) : nothing
                FNLτ = nothing
            end
            FNL = nothing
        end
        
        #irk4 is 4 now for last ponderation step (k4)
        FNL0, FNL1, FNLτ = compute_NL(vars, vars.Fb0, vars.Fb1, vars.Fτ, params, grid)
        @. FNLf0 += pond[end] * FNL0 # ponderate
        @. FNLf1 += pond[end] * FNL1 # ponderate
        
        # update with ponderated estimation of the slope
           
        @. vars.Fb0 =((1 - dt*Lb1) * (vars.Fb0 + dt*FNLf0) 
                    - dt*L10*(vars.Fb1 + dt*FNLf1))/(1 - dt*(Lb1+Lb0) + dt^2*(Lb0*Lb1 + L10^2))
        @. vars.Fb1 =(dt*L10 * (vars.Fb0 + dt*FNLf0) 
            + (1 - dt*Lb0)*(vars.Fb1 + dt*FNLf1))/(1 - dt*(Lb1+Lb0) + dt^2*(Lb0*Lb1 + L10^2))


        # dealiase 
        params.deal ? (@. vars.Fb0 *= params.mask) : nothing
        params.deal ? (@. vars.Fb1 *= params.mask) : nothing

        if params.add_tracer
            #then update with total estimation
            @. FNLτf += pond[end] * FNLτ
            @. vars.Fτ = (Fτ0 + dt*(FNLτf + im * grid.kr * vars.Fψ)) / ( 1 - dt*Lτ )
            # dealiase
            params.deal ? (@. vars.Fτ *= params.mask) : nothing
        end
        return

    end


    """ step_forward!(vars :: Vars, params :: Params, L :: AbstractArray, Lτ :: Union{Nothing, AbstractArray}, timestepper :: string, grid, dt)

    stepforwards the equation using the chosen timestepper 

    """

    function step_forward!(vars :: Vars, params :: Params, L10 :: AbstractArray, Lb0 :: AbstractArray ,
        Lb1 :: AbstractArray , L01 :: Union{Nothing, AbstractArray} , 
        Lτ :: Union{Nothing, AbstractArray}, grid, dt :: AbstractFloat)
        # update the forcing if white-noise-in-time
        if params.add_wn
            vars.Fh = calcF!(vars, params, grid, dt)
        end
        # update solution
        if params.timestepper == 10
            rk4_imex_timestepper!(vars, params, L10, Lb0, Lb1, L01, Lτ, grid, dt) 
        end
        if params.timestepper == 20
            println("source has yet to be written")
            exit()
        end
        if params.timestepper == 30
            println("source has yet to be written")
            exit()
        end
        # CFL criteria for next timestep 
        KE, dt = update_timestep(vars, params, grid, dt)
        # effectively update timestep
        vars.time = vars.time + dt
        return vars, KE, dt
    end


    """ update_timestep(vars :: Vars, params :: Params, dt)

    returns adaptative timestep with CFL condition from current solution

    """

    function update_timestep(vars :: Vars, params :: Params, grid, dt0)
        """ Returns new timestep from CFL conditions and KE 
        """
        # Computing horizontal kinetic energy for CFL condition

        inversion_ψb!(vars.Fb0, vars.Fb1, params)

        # TODO: clean below
        ∂yψ = deepcopy(grid.Ksq) # use as scratch variable, for right dim
        ∂xψ= deepcopy(grid.Ksq) # same

        F∂yψ = @. im * grid.l * vars.Fψ1
        F∂xψ = @. im * grid.kr * vars.Fψ1
        ldiv!(∂yψ, grid.rfftplan, F∂yψ)
        ldiv!(∂xψ, grid.rfftplan, F∂xψ)

        Ecloc = @. ( ∂xψ^2 + ∂yψ^2 )

        # use top variables as scratch

        @. F∂yψ = im * grid.l * vars.Fψ0
        @. F∂xψ = im * grid.kr * vars.Fψ0

        ldiv!(∂yψ, grid.rfftplan, F∂yψ)
        ldiv!(∂xψ, grid.rfftplan, F∂xψ)

        @. Ecloc += ( ∂xψ^2 + ∂yψ^2 )
        
        dt = minimum([dt0, params.CFL * 2 * π / params.resol / sqrt(maximum(Array(Ecloc)))])
        # updating time step
        KE = sum(Ecloc) / (params.resol * params.resol)
        return KE, dt

    end


    function run_simulation!(vars :: Vars, params :: Params, inversion_map :: Inversion, L10 :: AbstractArray,
        Lb0 :: AbstractArray , Lb1 :: AbstractArray , L01 :: Union{Nothing, AbstractArray}, 
        Lτ :: Union{Nothing, AbstractArray}, grid, Nfield :: Int, NsaveEc :: Int,
         Nstep :: Int, NsaveFlowfield :: Int, Nfig :: Int, NSpectrum :: Int, dt, path_to_run)
        
        if params.friction_type == 10
            println("source has yet to be written")
            exit()
        end
        counterEc = 0
        countSp = 0
        for ii in 1:Nstep
            vars, KE, dt = step_forward!(vars, params, L10, Lb0, Lb1, L01, Lτ, grid, dt) 
            counterEc+=1
            if counterEc == NsaveEc
                counterEc = 0
                SSD, LSD, inject, D, LHS, RHS = energy_budget_QGEady(vars, params, inversion_map, grid, dt)
                # Open the file for writing or appending
                if round(Int, ii / NsaveEc) == 1
                    open("energy_bal.txt", "w") do file
                        write(file, string(vars.time, " ", KE, " ", dt, " ",inject, " ", SSD, " ", LSD, " ", LHS, " ",RHS, " ",D, "\n"))
                    end
                else
                    open("energy_bal.txt", "a") do file
                        write(file, string(vars.time, " ", KE, " ", dt, " ",inject, " ", SSD, " ", LSD, " ", LHS, " ",RHS, " ",D, "\n"))
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
        save_vort_flowfield_png_QGEady(vars :: Vars, grid, timestep)
    
    Save snapshot of vorticity at chosen timestep. Eventually make this more general. 

    """

    function save_vort_flowfield_png_QGEady(vars :: Vars, grid, timestep, choice_colormap, path_to_run)
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

    function save_tau_flowfield_png_QGEady(vars :: Vars, grid, timestep, choice_colormap, path_to_run)
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

    function energy_budget_QGEady(vars :: Vars, params :: Params, inversion_map :: Inversion, grid ::TwoDGrid, dt :: Float64)
        """ Computes and prints every Nstep timesteps the energy budget of the 2DNS 
            vorticity equation for two types of friction
        """
        k2 = Array(grid.Krsq)
        Em  = real(inner_prod(Array(vars.Fψ),  Array(grid.Krsq.*vars.Fψ), k2, 0))        
        Em0 = real(inner_prod(Array(vars.Fψ0),  Array(grid.Krsq.*vars.Fψ0), k2, 0)) 

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
            D = real(inner_prod(Array(imm * grid.kr .* vars.Fψ), Array(vars.Fτ), k2, 0))
        end
        LHS = (Em - Em0)/dt/2 # 1/2 (∇ψ)^2 
        RHS = inject - SSD - LSD

        
        # Flux=-params.Ro*real(inner_prod(Fpsi0,1i*kx.*gamma1.*Fpsi1));
        #  #Friction=real(kappa_lambda2*sum_prod_symm(k2.*Fpsi0,Fpsi0));
        # Friction=-mu_lambda2*real(sum_prod_symm(Fpsi0,1i*kx.*FGradPsi0dxPsi0(1:resol/2+1,:)+1i*ky.*FGradPsi0dyPsi0(1:resol/2+1,:)));
        # Hyperviscosity=nu*real(sum_prod_symm(k2.^4.*(gamma0-gamma1)/2.*(Fpsi0+Fpsi1),Fpsi0+Fpsi1)+...
        # sum_prod_symm(k2.^4.*(gamma0+gamma1)/2.*(Fpsi0-Fpsi1),Fpsi0-Fpsi1));
        # RHS=Flux-Friction-Hyperviscosity;
        # dt_Em=(Em-Em_0)/dt_0;


        return SSD, LSD, inject, D, LHS, RHS
    end

    
end