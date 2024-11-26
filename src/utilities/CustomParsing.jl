module CustomParsing
export parse_circuit, ParsedGate

using JSON

struct ParsedGate
    name::String
    qubits::Vector{Int64}
    params::Vector{Float64}
end

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

function parse_circuit(circuit_path::String)::Vector{ParsedGate}
    circuit = JSON.parsefile(circuit_path)
    parsed_circuit = Vector{ParsedGate}()
    for gate in circuit
        if !(keys(gate) == Set(["Name", "Qubits", "Params"]))
            throw("Key Error: JSON circuit has incorrect keys.")
        end
        qubits = Vector{Int64}()
        params = Vector{Float64}()
        name = gate["Name"]
        if !(name isa String)
            throw("Type Error: Name must be String.")
        end
        for qubit in gate["Qubits"]
            if !(qubit isa Int64)
                throw("Type Error: qubits must be of type Int64.")
            end
            push!(qubits, qubit)
        end
        for param in gate["Params"]
            if !(param isa Float64)
                throw("Type Error: qubits must be of type Float64.")
            end
            push!(params, param)
        end
        push!(parsed_circuit, ParsedGate(name, qubits, params))
    end
    return parsed_circuit
end

end; # module
