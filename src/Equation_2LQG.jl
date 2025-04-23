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
    
    Convention for streamfunction is the standard one in geophysics (u,v) = (-∂yψ, ∂xψ)

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
        " CFL parameter for adaptative timestep"
        CFL :: T
        "Timestepper"
        timestepper :: Int64
        " Booleen to go from 2LQG equations to minimal equation"
        flag_2LQG :: Bool
           
    end



    """
    Updated variables 

    """
    mutable struct Vars{T <: AbstractArray} <: AbstractVars
        "Fourier transform of bottom buoyancy"
        Fψ :: Union{Nothing, T}
        "Fourier transform of top buoyancy"
        Fτ :: Union{Nothing, T}
        "Fourier transform of streamfunction, previous timestep or scratch variable"
        Fscratch :: Union{Nothing, T}
        "Streamfunction, real space scratch variable"
        scratch :: Union{Nothing, AbstractArray}
        "Friction term in -Δ⟂ψ equation"
        Ffrictionquad :: Union{Nothing, T}
        "Substep"
        substep :: Union{Nothing, Float64}
        "Timestep"
        time :: Float64
    end

    
    """ 
        initialize_field(grid, params :: Params, IC_type :: Int, noise_amplitude) 

    Initialize Vars (IC), handle the passive scalar field 

    """

        function initialize_field(grid, params :: Params, IC_type :: Int, noise_amplitude, restart_flag :: Bool,
            path_to_run::Union{String, Nothing}, filename::Union{String, Nothing})
            # initialize Fψ, Fτ

            if restart_flag
                # Step 1: Load the .MAT file
                file_path = path_to_run * filename
        
                if isfile(file_path)
                    # load CI from .mat file
                    CI_data = matread(file_path) 
                    if haskey(CI_data, "Fψ") && haskey(CI_data, "Fτ")
                        Fψ = CuArray(CI_data["Fpsi"])*params.resol^2
                        Fτ = CuArray(CI_data["Ftau"])*params.resol^2
                    else
                        println("Warning: Variable Fψ or Fτ not found in '$file_name'.")
                    end
                else
                    println("Warning: File '$filename' not in current directory.")
                end
            else
                if IC_type == 10
                    T = eltype(grid)
                    Dev = typeof(grid.device)

                    @devzeros Dev Complex{T} (size(grid.rfftplan)) ψ
                    @devzeros Dev Complex{T} (size(grid.rfftplan)) τ

                    Fψ  = noise_amplitude * ( 2*CUDA.rand(T, Int(params.resol/2)+1,params.resol) .- 1 .+ 1im*(2*CUDA.rand(T, Int(params.resol/2)+1,params.resol) .-1))*params.resol^2
                    @. Fψ = Fψ / grid.Krsq.^2
                    @CUDA.allowscalar Fψ[1,1] = 0
                    Fτ = noise_amplitude *(2*CUDA.rand(T, Int(params.resol/2)+1,params.resol).-1+1im*(2*CUDA.rand(T, Int(params.resol/2)+1,params.resol).-1))*params.resol^2
                    @. Fτ = Fτ / grid.Krsq.^2
                    @CUDA.allowscalar Fτ[1,1] = 0
                    
                    # ensure that we got a real initial condition
                    ldiv!(ψ, grid.rfftplan, Fψ)
                    mul!(Fψ, grid.rfftplan, ψ) 
                    ldiv!(τ, grid.rfftplan, Fτ)
                    mul!(Fτ, grid.rfftplan, τ) 

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
    

            return Fψ , Fτ, Ffrictionquad, Fscratch, scratch
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



    """
        compute_L(params, grid, dt)

    Returns the linear term in the equation with `params` `grid`, and `dt`. The linear
    operator ``L`` includes (hyper)-viscosity of order ``n_ν`` with coefficient ``ν`` 

    Plain-old viscocity corresponds to ``n_ν = 1`` while ``n_μ = 0`` corresponds to linear drag.

    The nonlinear term is computed via the function `compute_NL`.
    We add the option for quadratic drag in the compute_NL! function.

    """
    function compute_L(params, grid, dt)

        if params.friction_type == 20
            
            -grid.invKrsq
        @. FNLτ *= - 1 ./(grid.Krsq + 1/params.λ^2)
            Lψψ = @. -params.ν * grid.Krsq^params.nν   # here one should define L10 for linear drag
            
            Lτψ = @.  im * grid.kr * params.U * (grid.Krsq * Int(params.flag_2LQG) - 1/params.λ^2)/(grid.Krsq + 1/params.λ^2)
            #Lτψ = @.  im * grid.kr * params.U * (grid.Krsq * Int(params.flag_2LQG) - 1/params.λ^2)
            Lψτ =  @. im * grid.kr * params.U * Int(params.flag_2LQG)
        
        end
        
        #@. 1/(Lψψ*Lψψ -  Lψτ*Lτψ)
        return Lψψ, Lτψ, Lψτ # Lττ = Lψψ

    end   


    """
        compute_NL(sol, grid)

    Compute nonlinear term of streamfunction equation for intermediate timestep
    
    """

    function compute_NL(vars :: Vars, params :: Params, Fψn :: AbstractArray, Fτn :: AbstractArray, grid)
        
        Dev = typeof(grid.device)
        T = eltype(grid)
        # initialize physical space arrays

        # f is the advecting quantity (scratch between barotropic and baroclinic)
        @devzeros Dev T (size(grid.rfftplan)) ∂yf
        @devzeros Dev T (size(grid.rfftplan)) ∂xf
      

        @devzeros Dev T (size(grid.rfftplan)) Δψ
        @devzeros Dev T (size(grid.rfftplan)) Δτ

        # initialize fft arrays
    
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) FNLψ
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) FNLτ

        ldiv!(vars.scratch, grid.rfftplan, deepcopy(Fτn) )
        ldiv!(Δψ, grid.rfftplan, @. - grid.Krsq * Fψn )
        ldiv!(Δτ, grid.rfftplan, @. - grid.Krsq * Fτn )

        #ldiv!(τ, grid.rfftplan, deepcopy(Fτn))

        ## First advecting with baroclinic velocity f = τ (∂xf.. used as scratch)
        
        ldiv!(∂yf, grid.rfftplan, @. im * grid.l * Fτn) #∂yτ
        ldiv!(∂xf, grid.rfftplan, @. im * grid.kr * Fτn) #∂xτ
 
        # For ψ-equation (barotropic streamnfunction) J(τ, Δτ)
 
        mul!(vars.Fscratch, grid.rfftplan, @. -∂yf * Δτ) # \hat{u*Δτ} 
        @. FNLψ +=   - im * grid.kr * vars.Fscratch
        mul!(vars.Fscratch, grid.rfftplan, ∂xf * Δτ) # \hat{v*Δτ}
        @. FNLψ +=   - im * grid.kl * vars.Fscratch

        # For τ-equation (baroclinic streamnfunction) J(τ, Δψ)
        if params.flag_2LQG
            mul!(vars.Fscratch, grid.rfftplan, @. -∂yf * Δψ) # \hat{u*Δψ} 
            @. FNLτ +=   - im * grid.kr * vars.Fscratch 
            mul!(vars.Fscratch, grid.rfftplan, @. ∂xf * Δψ) # \hat{v*Δτ}
            @. FNLτ +=   - im * grid.kl * vars.Fscratch 
            # saving baroclinic velocities for bottom drag term ψ2 = ψ - τ
            ∂xψ2 = - ∂xf
            ∂yψ2 = - ∂yf
        end   

        ## Now advecting with barotropic velocity f = ψ
        ldiv!(∂yf, grid.rfftplan, @. im * grid.l * Fψn)
        ldiv!(∂xf, grid.rfftplan, @. im * grid.kr * Fψn)

        # For ψ-equation (barotropic streamfunction) J(ψ, Δψ)

        mul!(vars.Fscratch, grid.rfftplan, @. -∂yf * Δψ) # \hat{u*Δψ} 
        @. FNLψ +=   - im * grid.kr * vars.Fscratch
        mul!(vars.Fscratch, grid.rfftplan, @. ∂xf * Δψ) # \hat{v*Δψ}
        @. FNLψ +=   - im * grid.kl * vars.Fscratch

        # For τ-equation (barotropic streamfunction) J(ψ, Δτ)
   
        mul!(vars.Fscratch, grid.rfftplan, @. -∂yf * Δτ) # \hat{u*Δτ} 
        @. FNLτ +=   - im * grid.kr * vars.Fscratch
        mul!(vars.Fscratch, grid.rfftplan, @. ∂xf * Δτ) # \hat{v*Δτ}
        @. FNLτ +=   - im * grid.kl * vars.Fscratch

        # For τ-equation (baroclinic streamfunction) -J(ψ,τ/λ^2)
        Δτ = nothing
        Δψ = nothing

        mul!(vars.Fscratch, grid.rfftplan, @. -∂yf * vars.scratch) # \hat{u*τ} 
        @. FNLτ +=   im * grid.kr * vars.Fscratch / params.λ^2
        mul!(vars.Fscratch, grid.rfftplan, @. ∂xf * vars.scratch) # \hat{v*τ}
        @. FNLτ +=   im * grid.kl * vars.Fscratch / params.λ^2
          
        # add quadratic drag 
        if params.friction_type == 20
            if params.flag_2LQG
                # quadratic drag on ψ2 (bottom layer) only ψ2 = ψ - τ
                @. ∂xψ2 +=  ∂xf
                @. ∂yψ2 +=  ∂yf
                #@devzeros Dev Complex{T} (grid.nkr, grid.nl) F∇ψ2∂xψ2 
                #@devzeros Dev Complex{T} (grid.nkr, grid.nl) F∇ψ2∂yψ2
                mul!(vars.Fscratch, grid.rfftplan, @. (sqrt( ∂xψ2^2 + ∂yψ2^2 ) * ∂xψ2))
                @. vars.Ffrictionquad = - params.μ* im * grid.kr * vars.Fscratch
                mul!(vars.Fscratch, grid.rfftplan, @. (sqrt( ∂xψ2^2 + ∂yψ2^2 ) * ∂yψ2))  
                @. vars.Ffrictionquad = - params.μ* im * grid.l * vars.Fscratch
                
                @. FNLψ += vars.Ffrictionquad/2
                @. FNLτ -= vars.Ffrictionquad/2

            else # drag only on ψ-equation
            
                mul!(vars.Fscratch, grid.rfftplan, @. (sqrt( ∂xψ^2 + ∂yψ^2 ) * ∂xψ))
                @. vars.Ffrictionquad =  - params.μ * im * grid.kr * vars.Fscratch
                mul!(vars.Fscratch, grid.rfftplan, @. (sqrt( ∂xψ^2 + ∂yψ^2 ) * ∂yψ))  
                @. vars.Ffrictionquad =  - params.μ* im * grid.l * vars.Fscratch
                @. FNLψ += vars.Ffrictionquad/2

            end
        end
        
        # add forcing
        #TODO
       
        @. FNLψ *= -grid.invKrsq
        @. FNLτ *= - 1 ./(grid.Krsq + 1/params.λ^2)

        return FNLψ, FNLτ
    end

   

    """
        rk4_imex_timestepper(vars, param, L, Lτ, grid, dt)

    Solves any divergence-free problem expressed with streamfunction ψ on `grid` and returns the updated variables.
    Optionally includes passive tracer advection.
    Linear terms are always treated implicitly.
    
    """

    function rk4_imex_timestepper!(vars :: Vars , params :: Params, grid, dt :: AbstractFloat)
        """
        See above for function description
        """
        Dev = typeof(grid.device)
        T = eltype(grid)

        # Initialize relevant arrays for imex rk4
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) FNLψf # initialize ponderated non-linear term
        @devzeros Dev Complex{T} (grid.nkr, grid.nl) FNLτf # initialize ponderated non-linear term
        
        Fψ0 = deepcopy(vars.Fψ) # copy solution at current timestep
        Fτ0 = deepcopy(vars.Fτ) # copy solution at current timestep
       
        # weights and coefficients for classical explicit rk4 method
        order = [0.5 0.5 1]
        pond = [1/6 1/3 1/3 1/6]
        
        for irk4 in range(1,length(order))
            vars.substep = order[irk4]

            Lψψ, Lτψ, Lψτ, denominator = compute_L(params, grid, dt*vars.substep)   
            FNLψ, FNLτ = compute_NL(vars, params, vars.Fψ, vars.Fτ, grid)
               
            @. FNLψf += pond[irk4] * FNLψ # ponderate
            @. FNLτf += pond[irk4] * FNLτ # ponderate
           
            # compute slope estimation for intermediate timestep
            if params.friction_type == 20
                # otherwise the linear term must be computed in time-dep matrices
                @. vars.Fψ = denominator * (Lψψ*(vars.Fψ0 + dt*vars.substep*FNLψ) - Lψτ*(vars.Fτ0 + dt*vars.substep*FNLτ))
                @. vars.Fτ = denominator * (-Lτψ*(vars.Fψ0 + dt*vars.substep*FNLψ) + Lψψ*(vars.Fτ0 + dt*vars.substep*FNLτ))
            else
                println("Linear drag or other large-scale dissipation not available : source has yet to be written")
                exit()
            end
            
            # dealiase intermediate solution
            params.deal ? (@. vars.Fψ *= params.mask) : nothing
            params.deal ? (@. vars.Fτ *= params.mask) : nothing
            
        end
        
        #irk4 is 4 now for last ponderation step (k4)
        FNLψ, FNLτ = compute_NL(vars, params, vars.Fψ, vars.Fτ, grid)
        
        @. FNLψf += pond[end] * FNLψ # ponderate
        @. FNLτf += pond[end] * FNLτ # ponderate
        
        # update with ponderated estimation of the slope

        @. vars.Fψ = denominator * (Lψψ*(vars.Fψ0 + dt*FNLψf) - Lψτ*(vars.Fτ0 + dt*FNLτf))
        @. vars.Fτ = denominator * (-Lτψ*(vars.Fψ0 + dt*FNLψf) + Lψψ*(vars.Fτ0 + dt*FNLτf))
           
        # dealiase solution
        params.deal ? (@. vars.Fψ *= params.mask) : nothing
        params.deal ? (@. vars.Fτ *= params.mask) : nothing

        return

    end

    """ RK2 with exact integration of the linear terms

    """

    function rk2_timestepper!(vars :: Vars , params :: Params, Lψψ :: AbstractArray, Lψτ :: AbstractArray,
        Lτψ ::AbstractArray, grid, dt :: AbstractFloat)
        """
            rk2 timestepper with exact integration of the linear terms
            Following valadao et al. 2025

        """
        
        Fψ0 = deepcopy(vars.Fψ) # copy solution at current timestep
        Fτ0 = deepcopy(vars.Fτ) 
        FNLψ, FNLτ = compute_NL(vars, params, vars.Fψ, vars.Fτ, grid)
       
        # first step
        @. vars.Fψ = exp(Lψψ*dt/2)*(Fψ0 + (dt/2)*FNLψ) + exp(Lψτ*dt/2)*(Fτ0 + (dt/2)*FNLτ)
        @. vars.Fτ = exp(Lψψ*dt/2)*(Fτ0 + (dt/2)*FNLτ) + exp(Lτψ*dt/2)*(Fτ + (dt/2)*FNLτ)

        FNLψ = nothing
        FNLτ = nothing
        # dealiase
        params.deal ? (@. vars.Fψ *= params.mask) : nothing
        params.deal ? (@. vars.Fτ *= params.mask) : nothing

        # second step
        FNLψ, FNLτ = compute_NL(vars, vars.Fψ, vars.Fτ, params, grid)
        @. vars.Fψ =  exp(Lψψ*dt) * Fψ0 + exp(Lψτ*dt) * Fτ0 + exp(Lψψ*dt/2)*dt*FNLψ + exp(Lψτ*dt/2)*dt*FNLτ
        @. vars.Fτ =  exp(Lψψ*dt) * Fτ0 + exp(Lτψ*dt) * Fψ0 + exp(Lψψ*dt/2)*dt*FNLτ + exp(Lτψ*dt/2)*dt*FNLψ     
        # dealiase
        params.deal ? (@. vars.Fψ *= params.mask) : nothing
        params.deal ? (@. vars.Fτ *= params.mask) : nothing
        
        FNLψ = nothing
        FNLτ = nothing
        
        Fψ0 = nothing
        Fτ0 = nothing
    return
    end



    """ step_forward!(vars :: Vars, params :: Params, L :: AbstractArray, Lτ :: Union{Nothing, AbstractArray}, timestepper :: string, grid, dt)

    stepforwards the equation using the chosen timestepper 

    """

    function step_forward!(vars :: Vars, params :: Params, Lψψ ::AbstractArray, Lψτ ::AbstractArray, Lτψ::AbstractArray, grid, dt :: AbstractFloat)
        # update the forcing if white-noise-in-time
        if params.add_wn
            vars.Fh = calcF!(vars, params, grid, dt)
        end
        # update solution
        if params.timestepper == 10
            rk2_timestepper!(vars, params, Lψψ, Lψτ, Lτψ, grid, dt)
        end
        if params.timestepper == 20
            println("Explicit rk4 not available : source has yet to be written")
            exit()
        end
        if params.timestepper == 30
            println("IMEXrk4 adoc, not available")
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

        # TODO: clean below
        ∂yψ = deepcopy(grid.Ksq) # use as scratch variable, for right dim
        ∂xψ= deepcopy(grid.Ksq) # same

        F∂yψ = @. im * grid.l * (vars.Fψ + vars.Fτ)   
        F∂xψ = @. im * grid.kr * (vars.Fψ + vars.Fτ)
        ldiv!(∂yψ, grid.rfftplan, F∂yψ)
        ldiv!(∂xψ, grid.rfftplan, F∂xψ)

        Ecloc = @. ( ∂xψ^2 + ∂yψ^2 )

        # use top variables as scratch for bottom velocity

        @. F∂yψ = im * grid.l * (vars.Fψ - vars.Fτ)
        @. F∂xψ = im * grid.kr * (vars.Fψ - vars.Fτ)

        ldiv!(∂yψ, grid.rfftplan, F∂yψ)
        ldiv!(∂xψ, grid.rfftplan, F∂xψ)

        @. Ecloc += ( ∂xψ^2 + ∂yψ^2 )
        
        dt = minimum([dt0, params.CFL * 2 * π / params.resol / sqrt(maximum(Array(Ecloc)))])
        # updating time step
        KE = sum(Ecloc) / (params.resol * params.resol)
        return KE, dt

    end


    function run_simulation!(vars :: Vars, params :: Params, Lψψ ::AbstractArray, Lψτ ::AbstractArray, Lτψ::AbstractArray, grid, Nfield :: Int, 
         Nstep :: Int, NsaveFlowfield :: Int, Nfig :: Int, NsaveEc :: Int, NSpectrum :: Int, dt, path_to_run)
        
        if params.friction_type == 10
            println("Linear friction not available : source has yet to be written")
            exit()
        end

        if params.add_wn
            println("White-noise forcing not available: source has yet to be written")
            exit()
        end

        counterEc = 0
        countSp = 0
        for ii in 1:Nstep
            vars, KE, dt = step_forward!(vars, params, Lψψ, Lψτ, Lτψ, grid, dt) 
            counterEc+=1
            
            if counterEc == NsaveEc
                counterEc = 0
                Em_bt, D = energy_budget_QGEady(vars, params, grid, dt)
             
                #SSD, LSD, inject, D, LHS, RHS = energy_budget_QGEady(vars, params, grid, dt)
                # Open the file for writing or appending
                if round(Int, ii / NsaveEc) == 1
                    open("energy_bal.txt", "w") do file
                        write(file, string(vars.time, " ", KE, " ", dt, " ",Em_bt, " ", D,  "\n"))
                    end
                    #open("energy_bal.txt", "w") do file
                    #    write(file, string(vars.time, " ", KE, " ", dt, " ",inject, " ", SSD, " ", LSD, " ", LHS, " ",RHS, " ",D, "\n"))
                    #end
                else
                    open("energy_bal.txt", "a") do file
                        write(file, string(vars.time, " ", KE, " ", dt, " ",Em_bt, " ", D,  "\n"))
                        #write(file, string(vars.time, " ", KE, " ", dt, " ",inject, " ", SSD, " ", LSD, " ", LHS, " ",RHS, " ",D, "\n"))
                    end
                end
            end
                    
            if mod(ii, Nfig) == 0
                save_vort_flowfield_png_2LQG(vars, grid, vars.time, :RdBu_5, path_to_run)
            end
        
            if mod(ii, NsaveFlowfield) == 0
                save_flowfield(vars, Nfield, path_to_run)
                Nfield+=1
            end
           # if mod(ii, NSpectrum) == 0
           #     save_spectrum(vars, grid, countSp, path_to_run)
           #     countSp+=1
           # end
        
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
        
        Fψbt = @. - (vars.Fb0 - vars.Fb1) * grid.invKrsq 
        Fψbc = @. vars.Fψ1 - vars.Fψ0
        rSpecbt,kmoybt = radial_spectrum(Array(Fψbt), Array(grid.Krsq), Int(1))
        rSpecbc,kmoybc = radial_spectrum(Array(Fψbc), Array(grid.Krsq), Int(1))
        filename = path_to_run * "Fields/spectrum_" * @sprintf("%03d", countSp) * ".mat"
        matwrite(filename, Dict(
            "rSpecbt" => Array(rSpecbt),
            "kmoybt" => Array(kmoybt),
            "rSpecbc" => Array(rSpecbc),
            "kmoybc" => Array(kmoybc),
            "time" => vars.time
        ))
        return
    end


    """
        save_vort_flowfield_png_QGEady(vars :: Vars, grid, timestep)
    
    Save snapshot of vorticity at chosen timestep. Eventually make this more general. 

    """

    function save_vort_flowfield_png_2LQG(vars :: Vars, grid, timestep, choice_colormap, path_to_run)
        """ Save snapshot of vorticity at chosen timestep
        """

        Δψ = deepcopy(grid.Ksq)
        FΔψ = @. - grid.Krsq * vars.Fψ
        ldiv!(Δψ, grid.rfftplan, FΔψ)

                
        fig = Figure()
        ax1 = Axis(fig[1, 1], 
            title = L"$\Delta \psi$", 
            titlesize = 40,
            xlabel = L"$X$", 
            ylabel = L"$Y$",
            aspect = DataAspect()
        )
        
        hm = heatmap!(ax1, 
            grid.x, grid.y, Array(Δψ), 
            colormap = choice_colormap,                   
            colorrange = extrema(Array(Δψ))
            )
        Colorbar(fig[1, 2], hm, ticklabelsize = 18)

        
        Δτ= deepcopy(grid.Ksq)
        FΔτ = @. - grid.Krsq * vars.Fτ
        ldiv!(Δτ , grid.rfftplan, FΔτ)

        ax2 = Axis(fig[1, 2], 
            title = L"$\Delta \tau$", 
            titlesize = 40,
            xlabel = L"$X$", 
            ylabel = L"$Y$",
            aspect = DataAspect()
        )
        
        hm = heatmap!(ax2, 
            grid.x, grid.y, Array(Δτ), 
            colormap = choice_colormap,                   
            colorrange = extrema(Array(Δτ))
            )
        Colorbar(fig[1, 2], hm, ticklabelsize = 18)

        τ= deepcopy(grid.Ksq)
        ldiv!(τ , grid.rfftplan, deepcopy(vars.Fτ))

        ax3 = Axis(fig[2, 1], 
            title = L"$\tau$", 
            titlesize = 40,
            xlabel = L"$X$", 
            ylabel = L"$Y$",
            aspect = DataAspect()
        )
        hm = heatmap!(ax3, 
            grid.x, grid.y, Array(τ), 
            colormap = choice_colormap,                   
            colorrange = extrema(Array(τ))
            )
        Colorbar(fig[2, 1], hm, ticklabelsize = 18)

        ψ = deepcopy(grid.Ksq)
        ldiv!(ψ , grid.rfftplan, deepcopy(vars.Fψ))

        ax4 = Axis(fig[2, 2], 
            title = L"$\tau$", 
            titlesize = 40,
            xlabel = L"$X$", 
            ylabel = L"$Y$",
            aspect = DataAspect()
        )
        hm = heatmap!(ax4, 
            grid.x, grid.y, Array(ψ), 
            colormap = choice_colormap,                   
            colorrange = extrema(Array(ψ))
            )
        Colorbar(fig[2, 2], hm, ticklabelsize = 18)

        save(path_to_run * "Snapshots/Snapshot_vort_" * lpad(string(timestep), 6, '0') * ".png", fig)
        return
    end

    
    function energy_budget_2LQG(vars :: Vars, params :: Params, grid ::TwoDGrid, dt :: Float64)
        """ Computes and prints every Nstep timesteps the energy budget of the QG
            Eady energy equation 
        """
        k2 = Array(grid.Krsq)
    
        Em_bt = real(inner_prod(Array(vars.Fψ),  Array(grid.Krsq.*vars.Fψ), k2, 0)) 

        #SSD = params.ν  * energy(Array(vars.Fψ), k2, 5)
    
        #if params.friction_type == 20
        #    LSD = -real(inner_prod(Array(vars.Fψ),Array(vars.Ffrictionquad), k2, 0)) 
        #end

        if params.add_wn
            println("source has yet to be written")
            exit()
        end

    
        D = real(inner_prod(Array(1im * grid.kr .* vars.Fψ), Array(vars.Fτ), k2, 0))
        
        #LHS = (Em - Em0)/dt
        #RHS = inject - SSD - LSD
        #return SSD, LSD, inject, D, LHS, RHS
        return Em_bt, D

    end

    
end