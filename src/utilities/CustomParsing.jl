module CustomParsing
export parse_circuit

using JSON

function typenarrow!(d::Dict{<:Any,<:Any})
    for key in keys(d)
        if typeof(d[key]) == Dict
            typenarrow!(d[key])
        elseif typeof(d[key]) == Vector{Any}
            d[key] = [v for v in d[key]]
        end
    end
    return d
end

function parse_circuit(circuit_path::String)::Vector{Dict{String,Any}}
    circuit = JSON.parsefile(circuit_path)
    return [typenarrow!(gate) for gate in circuit]
end

end; # module
