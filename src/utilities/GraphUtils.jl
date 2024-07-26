module GraphUtils

export extract_adjacency_graph, named_ring_graph

using NamedGraphs: NamedGraphs, add_edges!, NamedGraph
using Graphs
using ITensorNetworks
using JSON

#circ_dict = JSON.parsefile("src/circuits/example_circuits/circ.json")

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

function extract_adjacency_graph2(qc::Vector{Dict{String,Any}})::NamedGraphs.NamedGraph
    qubits = Set{Int}()
    for gate in qc
        for qubit in gate["Qubits"]
            push!(qubits, qubit)
        end
    end
    G = NamedGraphs.NamedGraph(sort!(collect(qubits)))
    for gate in qc
        if length(gate["Qubits"]) == 2
            add_edges!(G, [gate["Qubits"][1] => gate["Qubits"][2]])
        end
    end
    return G
end

function named_ring_graph(n::Integer)
    G = NamedGraph([(i,) for i in 0:(n - 1)])
    add_edges!(G, [(i,) => (i + 1,) for i in 0:(n - 2)])
    add_edges!(G, [(n - 1,) => (0,)])
    return G
end

function named_ring_graph2(n::Integer)
    G = NamedGraph([i for i in 0:(n - 1)])
    add_edges!(G, [i => i + 1 for i in 0:(n - 2)])
    add_edges!(G, [n - 1 => 0])
    return G
end
end
