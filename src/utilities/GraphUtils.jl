module GraphUtils

export extract_adjacency_graph, named_ring_graph

using NamedGraphs: NamedGraphs, add_edges!, NamedGraph
using Graphs
using ITensorNetworks
using JSON

#circ_dict = JSON.parsefile("src/circuits/example_circuits/circ.json")

function extract_adjacency_graph(qc:: Vector{}, n_qubits::Integer):: NamedGraphs.NamedGraph
    G = NamedGraphs.NamedGraph([(i,) for i in 0:n_qubits-1])
    for gate in qc
        if length(gate["Qubits"]) == 2
            add_edges!(G, [(gate["Qubits"][1],) => (gate["Qubits"][2],)])
        end
    end
    return G
end


function named_ring_graph(n::Integer)
    G = NamedGraph([(i,) for i in 1:n])
    add_edges!(G, [(i,) => (i + 1,) for i in 1:(n - 1)])
    add_edges!(G, [(n,) => (1,)])
    return G
  end
end