module GraphUtils

export extract_adjacency_graph, named_ring_graph

using NamedGraphs: NamedGraphs, add_edges!, NamedGraph
using Graphs
using ITensorNetworks
using JSON
using OpenNetworks: CustomParsing, OpenNetworks
ParsedGate = CustomParsing.ParsedGate
#circ_dict = JSON.parsefile("src/circuits/example_circuits/circ.json")

function extract_adjacency_graph(
    qc::Vector{OpenNetworks.CustomParsing.ParsedGate}
)::NamedGraphs.NamedGraph
    all_qubits = Vector{Int64}()
    for gate in qc
        append!(all_qubits, gate.qubits)
    end
    qubits = Set(all_qubits)
    G = NamedGraphs.NamedGraph(sort!(collect(qubits)))
    for gate in qc
        if length(gate.qubits) == 2
            add_edges!(G, [gate.qubits[1] => gate.qubits[2]])
        end
        if length(gate.qubits) > 2
            throw("More than two qubit gates are not supported.")
        end
    end
    return G
end

#=
function extract_adjacency_graph(qc::Vector{Dict{String,Any}})::NamedGraphs.NamedGraph
    qubits = Set{Tuple{Int}}()
    for gate in qc
        for qubit in gate["Qubits"]
            push!(qubits, (qubit,))
        end
    end
    G = NamedGraphs.NamedGraph(sort!(collect(qubits)))
    for gate in qc
        if length(gate["Qubits"]) == 2
            add_edges!(G, [(gate["Qubits"][1],) => (gate["Qubits"][2],)])
        end
    end
    return G
end
=#

function named_ring_graph(n::Integer)
    G = NamedGraph([i for i in 0:(n - 1)])
    add_edges!(G, [i => i + 1 for i in 0:(n - 2)])
    add_edges!(G, [n - 1 => 0])
    return G
end
end
