module SpectralAnalysis

   
    export inner_prod, energy, radial_spectrum
    using MAT

    """ 
        inner_prod(ψin :: AbstractArray, Φin :: AbstractArray, k2 :: AbstractArray, ord :: Int)

    for realffts, returns the inner product of two CPU arrays ψin and Φin in fourier space
        
    """
    function inner_prod(ψin :: AbstractArray, Φin :: AbstractArray, k2 :: AbstractArray, ord :: Int)    
        """
        Returns inner product of two realfft CPU array

        """
        innerprod = sum(2 * (k2[2:end-1,:].^ord).*(Φin[2:end-1,:].*conj.(ψin[2:end-1,:]))) +
                sum((k2[1,:].^ord).*(Φin[1,:].*conj(ψin[1,:]))) +
                sum((k2[end,:].^ord).*(Φin[end,:].*conj(ψin[1,:])))
        # dft normalization
        innerprod /= size(ψin, 2)^4 #/resol^4
        return innerprod
    end


    """ 
        energy(ψin :: AbstractArray, k2 :: AbstractArray, ord :: Int)

    for realffts, returns the squared modulus of input realfft CPU array
    
    """
    
    function energy(ψin :: AbstractArray, k2 :: AbstractArray, ord :: Int)
        """ 
        Returns squared modulus of input realfft CPU array
        """
        energy = sum(2 * (k2[2:end-1, :] .^ ord) .* (abs.(ψin[2:end-1, :]) .^ 2)) +
            sum((k2[1, :] .^ ord) .* (abs.(ψin[1, :]) .^ 2)) +
            sum((k2[end, :] .^ ord) .* (abs.(ψin[end, :]) .^ 2))
        # dft renormalization
        energy /= size(ψin, 2)^4 #*resol^4
        return energy
    end


    """ 
        radial_spectrum(input :: AbstractArray, params :: Params, grid, num :: Int, savefile :: Bool, timestep)
    
    """

    function radial_spectrum(input :: AbstractArray, k2 :: AbstractArray, num :: Int)
        """ 
        Returns spectrum of input realfft array using integration in (r, θ) 
        Array must be converted back to CPU
        """
        # Safety: bring back to CPU for easier index handling 
        input = Array(input)
        k2 = Array(k2)
        resol = size(input, 2)
        lkmoy = 1:(round(Int, resol / 2) + 2)
        rSpecsum = zeros(length(lkmoy))
        for ii in 1:(round(Int, resol / 2) + 1)
            fct = 2.0
            if ii == 1
                fct = 1.0
            end
            for jj in 1:resol
                kmn = Int(floor(sqrt(k2[ii, jj]) + 0.5))
                if kmn >= 0 && kmn <= resol / 2 + 1
                    rSpecsum[kmn + 1] += fct * (k2[ii, jj]^num) * abs(input[ii, jj])^2
                end
            end
        end
        
        # renormalization ? CHECK
        rSpecsum /= resol^4
        return rSpecsum, lkmoy
    end

    

end


