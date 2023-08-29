#=
!***********************************************************
! Sparse modeling approach to analytic continuation
!      of imaginary-time Monte Carlo data
!                            3 Apr 2020  E.Itou and Y. Nagai
!***********************************************************
=#

module Input
    struct Inputdata
        filename::String
        data

        function Inputdata(filename)
            data = readlines(filename)
            return new(filename,data)
        end
    end
    export Inputdata,getFloat64,getInt,getString

    function getvalue(input::Inputdata,type,name::String)
        for i=1:length(input.data)
            u = split(input.data[i])
            len = length(u[name .== u])
            if len != 0
                return parse(type,u[3])
            end
        end        
        return nothing
    end

    function getInt(input::Inputdata,name::String)
        return getvalue(input,Int64,name)
    end

    function getFloat64(input::Inputdata,name::String,initialvalue)
        v = getvalue(input,Float64,name)
        if v === nothing
            return initialvalue
        end
        return v
    end

    function getInt(input::Inputdata,name::String,initialvalue)
        n = getvalue(input,Int64,name)
        if n === nothing
            return initialvalue
        end
        return n
    end

    function getString(input::Inputdata,name::String,initialvalue)
        for i=1:length(input.data)
            u = split(input.data[i])
            len = length(u[name .== u])
            if len != 0
                return u[3]
            end
        end        
        return initialvalue
    end

end