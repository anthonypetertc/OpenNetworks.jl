module Gates
export Gate

"""
    Gate
A structure to represent a quantum gate.
    name::String
        The name of the gate (e.g., "CNOT", "H", etc.).
    qubits::Vector{Int64}
        The qubits that the gate acts on.
    params::Vector{Float64}
        The parameters of the gate (e.g., angles for rotation gates).
"""

struct Gate
    name::String
    qubits::Vector{Int64}
    params::Vector{Float64}
end

end; #module
