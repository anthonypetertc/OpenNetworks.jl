module GraphUtils

export extract_adjacency_graph

using NamedGraphs: NamedGraphs, add_edges!
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

end