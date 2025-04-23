module VectorizationNetworks

using NamedGraphs: vertices
using ITensorNetworks:
    ITensorNetworks,
    ITensorNetwork,
    siteinds,
    IndsNetwork,
    inner,
    apply,
    vertices
using ITensors: ITensor, dag, inds, inner, op, delta, ITensors, Index
using ITensorsOpenSystems: Vectorization
using OpenNetworks: VDMNetworks

vectorizer = Vectorization.vectorizer
vectorizer_input = Vectorization.vectorizer_input
vectorizer_output = Vectorization.vectorizer_output
basespace = Vectorization.basespace
fatsiteind = Vectorization.fatsiteind
VDMNetwork = VDMNetworks.VDMNetwork

"""
    fatsiteinds(sites::IndsNetwork{V})::IndsNetwork{V}

    Arguments:
    sites::IndsNetwork{V}
        The input IndsNetwork to be vectorized.
    
    Vectorizers the indices of the input IndsNetwork and returns a new IndsNetwork with vectorized indices.

"""

function Vectorization.fatsiteinds(
    sites::ITensorNetworks.IndsNetwork{V}
)::ITensorNetworks.IndsNetwork{V} where {V}
    vsites = deepcopy(sites)
    for v in vertices(sites)
        if length(sites[v]) != 1
            throw("sites must have exactly one site Index on every vertex.")
        end
        vsites[v] = Vectorization.fatsiteinds(sites[v])
    end
    return vsites
end

function unvectorize_density_matrix!(ρ::VDMNetwork{V})::ITensorNetwork{V} where {V}
    vectorizedinds = siteinds(ρ.network)
    unvectorizedinds = ρ.unvectorizedinds
    ρ = ρ.network

    for vertex in vertices(unvectorizedinds)
        @assert length(vectorizedinds[vertex]) == 1 "vectorized index at site $vertex is not unique"
        @assert length(unvectorizedinds[vertex]) == 1 "site $vertex has wrong number of indices"

        spacename = basespace(vectorizedinds[vertex][1])
        if !ITensors.hastags(unvectorizedinds[vertex][1], spacename)
            throw(
                ArgumentError(
                    "The unvectorised index $(unvectorizedinds[vertex][1]) does not match the vectorised index $(vectorizedinds[vertex][1])",
                ),
            )
        end
        ρ[vertex] *= ITensors.delta(
            ITensors.dag(vectorizedinds[vertex][1]), vectorizer_output(spacename)
        )
        ρ[vertex] *= ITensors.dag(vectorizer(spacename))
        ρ[vertex] *= ITensors.delta(
            ITensors.dag(unvectorizedinds[vertex][1]), vectorizer_input(spacename)
        )
        ρ[vertex] *= ITensors.delta(
            unvectorizedinds[vertex][1]', ITensors.dag(vectorizer_input(spacename))'
        )
    end
    return ρ
end

function unvectorize_density_matrix(ρ::VDMNetwork{V})::ITensorNetwork{V} where {V}
    return unvectorize_density_matrix!(deepcopy(ρ))
end

function idnetwork(ψ::ITensorNetwork{V})::ITensorNetwork{V} where {V}
    # Creates an ITensorNetwork with identity tensors at each site. =#
    data_graph = ITensorNetwork(v -> "0", siteinds(ψ)).data_graph
    sitekeys = vertices(ψ)
    for key in sitekeys
        indices = [
            ind for
            ind in inds(data_graph.vertex_data[key]) if !(ind == siteinds(ψ)[key][1])
        ]
        d = ITensors.dim(siteinds(ψ)[key][1])
        dims = Tuple(append!([d, d], [1 for ind in indices]))
        id_array = Array(delta(siteinds(ψ)[key][1], siteinds(ψ)[key][1]').tensor)
        data_graph.vertex_data[key] = ITensor(
            reshape(id_array, dims), siteinds(ψ)[key][1], siteinds(ψ)[key][1]', indices...
        )
    end
    return ITensorNetwork(data_graph.vertex_data)
end

function vidnetwork(
    ψ::ITensorNetwork{V}, vectorizedinds::IndsNetwork{V,Index}
)::VDMNetwork{V} where {V}
    return VDMNetworks.VDMNetwork(idnetwork(ψ), siteinds(ψ), vectorizedinds)
end

function vectorizedtrace(ρ::VDMNetwork; kwargs...)::Complex
    ψ = ITensorNetwork(v -> "0", ρ.unvectorizedinds)
    idn = vidnetwork(ψ, siteinds(ρ.network))
    return inner(ρ.network, idn.network; kwargs...)
end

function vexpect(ρ::VDMNetwork, op::ITensor; kwargs...)::Complex
    ψ = ITensorNetwork(v -> "0", ρ.unvectorizedinds)
    idn = idnetwork(ψ)
    new_network = apply(op, idn)
    new_op = VDMNetworks.VDMNetwork(new_network, ρ.unvectorizedinds, siteinds(ρ))
    return abs(inner(ρ.network, new_op.network; kwargs...))
end

"""
    vertices(ρ::VDMNetwork)

    Arguments:
    ρ::VDMNetwork
        The VDMNetwork for which to get the vertices.
    
    Returns the vertices of the VDMNetwork.
"""

function ITensorNetworks.vertices(ρ::VDMNetwork)
    return ITensorNetworks.vertices(ρ.network)
end

"""
    edges(ρ::VDMNetwork)

    Arguments:
    ρ::VDMNetwork
        The VDMNetwork for which to get the edges.
    
    Returns the edges of the VDMNetwork.
"""

function ITensorNetworks.edges(ρ::VDMNetwork)
    return ITensorNetworks.edges(ρ.network)
end

"""
    expect(operator::AbstractString, ρ::VDMNetwork{V}; kwargs...)

    Arguments:
    operator::AbstractString
        The operator to be applied.
    ρ::VDMNetwork{V}
        The VDMNetwork for which to compute the expectation value.
    
    Computes the expectation value of the operator with respect to the 
    density matrix correpsonding to the VDMNetwork.
"""

function ITensorNetworks.expect(
    operator::AbstractString, ρ::VDMNetworks.VDMNetwork{V}; kwargs...
) where {V}
    results = Dict()
    for v in vertices(ρ)
        o = op(operator, first(ρ.unvectorizedinds[v]))
        results[v] = vexpect(ρ, o; kwargs...)
    end
    return results
end

end; # module
