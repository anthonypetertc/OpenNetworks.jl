module GraphUtils

export extract_adjacency_graph, named_ring_graph, linenetwork, islinenetwork

using NamedGraphs: NamedGraphs, add_edges!, NamedGraph
using ITensors
using ITensorNetworks: IndsNetwork, vertices, neighbors
using OpenNetworks: Gates.Gate

"""
    extract_adjacency_graph(qc::Vector{Gate})::NamedGraphs.NamedGraph

    Arguments
    qc::Vector{Gate}
        The quantum circuit represented as a vector of Gate objects.

    Returns a NamedGraph representing the adjacency graph of the quantum circuit.
"""

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

"""
    named_ring_graph(n::Integer)::NamedGraphs.NamedGraph

    Arguments
    n::Integer
        The number of vertices in the ring graph.

    Returns a NamedGraph representing a ring graph with n vertices.
"""

function named_ring_graph(n::Integer)
    G = NamedGraph([i for i in 0:(n - 1)])
    add_edges!(G, [i => i + 1 for i in 0:(n - 2)])
    add_edges!(G, [n - 1 => 0])
    return G
end

"""
    linegraph(sites::Vector{<:ITensors.Index{}})::NamedGraphs.NamedGraph

    Arguments
    sites::Vector{<:ITensors.Index{}}
        The sites to be connected in a line graph.

    Returns a NamedGraph representing a line graph with the given sites.
"""

function linegraph(sites::Vector{<:ITensors.Index{}})::NamedGraphs.NamedGraph
    G = NamedGraphs.NamedGraph([i for i in 1:length(sites)])
    for i in 1:(length(sites) - 1)
        add_edges!(G, [i => i + 1])
    end
    return G
end

"""
    islinegraph(g::NamedGraphs.NamedGraph)::Bool

    Arguments
    g::NamedGraphs.NamedGraph
        The graph to be checked.

    Returns true if the graph is a line graph, false otherwise.
"""

function islinegraph(g::NamedGraphs.NamedGraph):: Bool
    verts = vertices(g)
    even = count(v -> length(neighbors(g, v)) == 2, verts)
    one = count(v -> length(neighbors(g, v)) == 1, verts)
    size = length(verts)
    if even == size -2 && one == 2
        return true
    else
        return false
    end
end

"""
    linenetwork(sites::Vector{<:ITensors.Index{}}, sitetype::String="Qubit")::IndsNetwork

    Arguments
    sites::Vector{<:ITensors.Index{}}
        The sites to be connected in a line network.
    sitetype::String
        The type of the sites (default is "Qubit").

    Returns an IndsNetwork representing a line network with the given sites.

"""

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

"""
    linenetwork(sites::Vector{<:ITensors.Index{}}, qc::Vector{Gate}, sitetype::String="Qubit")::IndsNetwork

    Arguments
    sites::Vector{<:ITensors.Index{}}
        The sites to be connected in a line network.
    qc::Vector{Gate}
        The quantum circuit represented as a vector of Gate objects.
    sitetype::String
        The type of the sites (default is "Qubit").

    Returns an IndsNetwork representing a line network with the vertices given by qubit numbers
    from the quantum circuit, and the site indices given by the sites.
"""

function linenetwork(
    sites:: Vector{<:ITensors.Index{}},
    qc:: Vector{Gate},
    sitetype:: String="Qubit"
):: IndsNetwork
    G = extract_adjacency_graph(qc)
    if !(islinegraph(G)) throw("Quantum circuit is not acting on a line.") end
    indsnetwork = siteinds(sitetype, G)
    for v in vertices(G)
        sv = filter(s -> hastags(s, "n=$(v)"), sites)
        if length(sv) != 1 
            throw("Sites are not labelled correctly, use siteinds function to create sites for the quantum circuit.")
        end
        indsnetwork[v] = sv
    end
    return indsnetwork
end

"""
    islinenetwork(sites::IndsNetwork)::Bool

    Arguments
    sites::IndsNetwork
        The IndsNetwork to be checked.

    Returns true if the IndsNetwork is a network with 1-d line connectivity, false otherwise.
"""

function islinenetwork(sites::IndsNetwork)::Bool
    return islinegraph(sites.data_graph.underlying_graph)
end


"""
    siteinds(tag::String, qc::Vector{Gate})::Vector{<:Index}

    Arguments
    tag::String
        The tag to be used for the indices.
    qc::Vector{Gate}
        The quantum circuit represented as a vector of Gate objects.

    (Only implemented for circuits with 1-d connectivity on a line)
    Returns a vector of site indices with the given tag, where each index is labelled by the qubit number
    from the quantum circuit.
"""

function ITensors.siteinds(tag::String, qc:: Vector{Gate})::Vector{<:Index}
    g = extract_adjacency_graph(qc)
    if islinegraph(g)
        verts = vertices(g)
        sites = siteinds(tag, length(verts))
        for (i, (s, v)) in enumerate(zip(sites, verts))
            sites[i] = replacetags(s, "n=$i", "n=$(v)")
        end
        return sites
    else
        throw("Graph is not a line, function only implemented for lines.")
    end
end

end
