module Quickprocess_2DNS

    export read_timeseries, plot_timeseries, plot_spectrum

    using CUDA, Random, Printf, DelimitedFiles, CairoMakie
    using LinearAlgebra: mul!, ldiv!
    using MAT, JLD2
    include("../utils/Tools.jl")
    using .Toolbox
    using FourierFlows, Statistics
    parsevalsum2 = FourierFlows.parsevalsum2

    """
    function read_timeseries

    """

    function read_timeseries(Dir :: String, Run :: String; savefile :: Bool = false)
       
        S = readdlm(joinpath(Dir, Run, "/energy_bal.txt"))

        time = S[:, 1]'
        D = S[:, 9]'
        KEh = S[:, 2]'
        inject = S[:, 4]'
        SSD = S[:, 5]'
        LSD = S[:, 6]'
        if savefile
            namesave = Dir*"timeseries_"*Run*".jld2"  
            @save namesave time KEh LSD SSD inject D taums
        end

        return time, D, KEh, inject, SSD, LSD
    end

    """
        plot_timeseries(Dir :: String, Run :: String)
    
    plot timeseries for chosen Run in Dir

    """

    function plot_timeseries(Dir :: String, Run :: String)
        " plot timeseries for chosen run "

        time, D, KEh, inject, SSD, LSD = read_timeseries(Dir, Run)
        
        fig = Figure()
        
        ax1 = Axis(fig[1, 1], 
        xlabel = L"$t$", 
        ylabel = L"$KE_h$")
        plot!(ax1, Array(vec(time)), Array(vec(KEh)), color="black")

        ax2 = Axis(fig[2, 1], 
        xlabel = L"$t$", 
        ylabel = L"Energy Budget")
        plot!(ax2, Array(vec(time)), Array(vec(inject)), color="red")
        plot!(ax2, Array(vec(time)), Array(vec(LSD)), color="blue")
        plot!(ax2, Array(vec(time)), Array(vec(SSD)), color="green")
        
        ax3 = Axis(fig[3, 1], 
        xlabel = L"$t$", 
        ylabel = L"$D$")
        plot!(ax3, Array(vec(time)), Array(vec(D)), color="blue")

        display(fig)
        return

    end

    """
        plot_spectrum(Nspectrum :: Int, Dir :: String, Run :: String)

    plot spectrum Nspectrum for chosen Run in Dir

    """

    function plot_spectrum(Nspectrum :: Int, Dir :: String, Run :: String)
        " plot spectrum Nspectrum for chosen Run in Dir"
        file = matread(Dir*Run*"/Fields/spectrum_"* @sprintf("%03d", Nspectrum) * ".mat")

        Sp = file["rSpecveloc"]
        k = file["kmoy"]

        fig = Figure()
        ax = Axis(fig[1,1],
        xscale = log10,
        yscale = log10,
        xlabel = L"$k$", 
        ylabel = L"$E(k)$")
        lines!(ax, k[Sp .> 0], Sp[Sp .> 0], color=:red, linestyle=:dash )
        display(fig)

        return
    end

end