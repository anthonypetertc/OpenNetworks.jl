module GraphUtils

export extract_adjacency_graph, named_ring_graph, linenetwork, islinenetwork

using NamedGraphs: NamedGraphs, add_edges!, NamedGraph
using Graphs
using ITensors
using ITensorNetworks: IndsNetwork, vertices
using JSON
using OpenNetworks: Gates.Gate, OpenNetworks, Utils.findindextype

function extract_adjacency_graph(qc::Vector{Gate})::NamedGraphs.NamedGraph
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

function named_ring_graph(n::Integer)
    G = NamedGraph([i for i in 0:(n - 1)])
    add_edges!(G, [i => i + 1 for i in 0:(n - 2)])
    add_edges!(G, [n - 1 => 0])
    return G
end

function linegraph(sites::Vector{<:ITensors.Index{}})::NamedGraphs.NamedGraph
    G = NamedGraphs.NamedGraph([i for i in 1:length(sites)])
    for i in 1:(length(sites) - 1)
        add_edges!(G, [i => i + 1])
    end
    return G
end

function linenetwork(
    sites::Vector{<:ITensors.Index{}}, sitetype::String="Qubit"
)::IndsNetwork
    G = linegraph(sites)
    indsnetwork = siteinds(sitetype, G)
    for (i, site) in enumerate(sites)
        indsnetwork[i] = [site]
    end
    return indsnetwork
end

function islinenetwork(sites::IndsNetwork)::Bool
    T = findindextype(sites)
    siteindices = Vector{T}()
    for v in vertices(sites)
        push!(siteindices, first(sites[v]))
    end
    lg = linegraph(siteindices)
    return sites.data_graph.underlying_graph == lg
end

end
