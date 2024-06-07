module VectorizationNetworks
export vectorize_density_matrix, innerprod, unvectorize_density_matrix

using NamedGraphs: vertices
using ITensorNetworks:
    AbstractITensorNetwork,
    ITensorNetworks,
    ITensorNetwork,
    ⊗,
    siteinds,
    VidalITensorNetwork,
    IndsNetwork,
    inner,
    apply
using ITensors: ITensor, dag, inds, inner, op, randomITensor, delta, ITensors, Index
import ITensors: outer
using OpenSystemsTools: Vectorization
using OpenNetworks: Utils, VDMNetworks
import Base: show, repr

vectorizer = Vectorization.vectorizer
vectorizer_input = Vectorization.vectorizer_input
vectorizer_output = Vectorization.vectorizer_output
basespace = Vectorization.basespace

#outer = Utils.outer
innerprod = Utils.innerprod
VDMNetwork = VDMNetworks.VDMNetwork

function vectorize_density_matrix!(
    ρ::ITensorNetwork, unvectorizedinds::IndsNetwork, vectorizedinds::IndsNetwork
)::VDMNetwork
    for vertex in vertices(unvectorizedinds)
        @assert length(vectorizedinds[vertex]) == 1 "vectorized index at site $vertex is not unique"
        @assert length(unvectorizedinds[vertex]) == 1 "site $vertex has wrong number of indices"
        spacename = basespace(vectorizedinds[vertex][1])

        if !ITensors.hastags(unvectorizedinds[vertex][1], spacename)
            throw(
                ArgmentError(
                    "The vectorised index at site $i, $(vectorizedinds[vertex][1]), does not match the unvectorized index $(unvectorizedinds[vertex][1])",
                ),
            )
        end
        ρ[vertex] *= ITensors.delta(
            ITensors.dag(unvectorizedinds[vertex][1]),
            ITensors.dag(vectorizer_input(spacename)),
        )
        ρ[vertex] *= ITensors.delta(
            unvectorizedinds[vertex][1]', vectorizer_input(spacename)'
        )
        ρ[vertex] *= vectorizer(spacename)
        ρ[vertex] *= ITensors.delta(
            ITensors.dag(vectorizer_output(spacename)), vectorizedinds[vertex][1]
        )
    end
    return VDMNetwork(ρ, unvectorizedinds)
end

function vectorize_density_matrix(
    ρ::ITensorNetwork, unvectorizedinds::IndsNetwork, vectorizedinds::IndsNetwork
)::VDMNetwork
    return vectorize_density_matrix!(deepcopy(ρ), unvectorizedinds, vectorizedinds)
end

function unvectorize_density_matrix!(ρ::VDMNetwork)::ITensorNetwork
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

function unvectorize_density_matrix(ρ::VDMNetwork)::ITensorNetwork
    return unvectorize_density_matrix!(deepcopy(ρ))
end

function idnetwork(ψ::ITensorNetwork)::ITensorNetwork
    # Purpose: Creates an ITensorNetwork with identity tensors at each site.
    # Inputs: inds (ITensorNetworks.IndsNetwork) - Site indices.
    # Returns: ITensorNetwork - ITensorNetwork with identity tensors at each site.
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

function vidnetwork(ψ::ITensorNetwork, vectorizedinds::IndsNetwork)::VDMNetwork
    return vectorize_density_matrix(idnetwork(ψ), siteinds(ψ), vectorizedinds)
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
    new_op = vectorize_density_matrix(new_network, ρ.unvectorizedinds, siteinds(ρ))
    return inner(ρ.network, new_op.network; kwargs...)
end

end; # module
