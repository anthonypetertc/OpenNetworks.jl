module Gates
export Gate

struct Gate
    name::String
    qubits::Vector{Int64}
    params::Vector{Float64}
end

end; #module
