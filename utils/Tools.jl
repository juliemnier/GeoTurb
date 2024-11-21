module Toolbox

    export dichotomy

    """ function dichotomy(time, tmin)
    Usual dichotomy function
    """

    function dichotomy(time, tmin)
        # Implement a dichotomy function to find the appropriate index
        return findfirst(x -> x >= tmin, time) - 1
    end

end


