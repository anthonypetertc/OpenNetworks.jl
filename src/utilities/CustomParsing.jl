module CustomParsing
export parse_circuit

using JSON
using OpenNetworks: Gates.Gate

#=
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
=#

"""
    parse_circuit(circuit_path::String)::Vector{Gate}

    Arguments
    circuit_path::String
        The path to the JSON file containing the circuit.

    Parses the JSON file at the given path and returns a vector of Gate objects.
    Note that the JSON file must have correct format e.g.:

    
    [
        {
            "Name": "CNOT",
            "Qubits": [0, 1],
            "Params": []
        },
        {
            "Name": "Rz",
            "Qubits": [2],
            "Params": [0.2]
        }
    ]
    
    See example_circuits folder for example circuits.

    # Examples
    ```julia
    using OpenNetworks: CustomParsing

    circuit = CustomParsing.parse_circuit("path/to/circuit.json")
    ```
"""

function parse_circuit(circuit_path::String)::Vector{Gate}
    circuit = JSON.parsefile(circuit_path)
    parsed_circuit = Vector{Gate}()
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
        push!(parsed_circuit, Gate(name, qubits, params))
    end
    return parsed_circuit
end

end; # module
