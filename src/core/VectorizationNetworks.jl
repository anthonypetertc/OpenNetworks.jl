module VectorizationNetworks

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
    apply,
    vertices
using ITensors: ITensor, dag, inds, inner, op, randomITensor, delta, ITensors, Index, QN
import ITensors: outer
using ITensorsOpenSystems: Vectorization
using OpenNetworks: Utils, VDMNetworks
import Base: show, repr

vectorizer = Vectorization.vectorizer
vectorizer_input = Vectorization.vectorizer_input
vectorizer_output = Vectorization.vectorizer_output
basespace = Vectorization.basespace
fatsiteind = Vectorization.fatsiteind

#outer = Utils.outer
innerprod = Utils.innerprod
VDMNetwork = VDMNetworks.VDMNetwork

function fatsiteinds(sites::ITensorNetworks.IndsNetwork)::ITensorNetworks.IndsNetwork
    vsites = deepcopy(sites)
    for v in vertices(sites)
        if length(sites[v]) != 1
            throw("sites must have exactly one site Index on every vertex.")
        end
        vsites[v] = Vectorization.fatsiteinds(sites[v])
    end
    return vsites
end
#=
function vectorize_density_matrix!(
    ρ::ITensorNetwork{V},
    unvectorizedinds::IndsNetwork{V,Index},
    vectorizedinds::IndsNetwork{V,Index},
) where {V}
    for vertex in vertices(unvectorizedinds)
        @assert length(vectorizedinds[vertex]) == 1 "vectorized index at site $vertex is not unique"
        @assert length(unvectorizedinds[vertex]) == 1 "site $vertex has wrong number of indices"
        spacename = basespace(vectorizedinds[vertex][1])::String
        vin = vectorizer_input(spacename)::Index{Vector{Pair{QN,Int64}}}
        vout = vectorizer_output(spacename)::Index{Vector{Pair{QN,Int64}}}
        vz = vectorizer(spacename)::ITensors.ITensor
        if !ITensors.hastags(unvectorizedinds[vertex][1], spacename)
            throw(
                ArgmentError(
                    "The vectorised index at site $i, $(vectorizedinds[vertex][1]), does not match the unvectorized index $(unvectorizedinds[vertex][1])",
                ),
            )
        end
        ρ[vertex]::ITensors.ITensor *= ITensors.delta(
            ITensors.dag(unvectorizedinds[vertex][1]), ITensors.dag(vin)
        )
        ρ[vertex]::ITensors.ITensor *= ITensors.delta(unvectorizedinds[vertex][1]', vin')
        ρ[vertex]::ITensors.ITensor *= vz
        ρ[vertex]::ITensors.ITensor *= ITensors.delta(
            ITensors.dag(vout), vectorizedinds[vertex][1]
        )
    end
    return ρ
end =#
#=
function vectorize_density_matrix(
    ρ::ITensorNetwork{V},
    unvectorizedinds::IndsNetwork{V,Index},
    vectorizedinds::IndsNetwork{V,Index},
)::VDMNetwork{V} where {V}
    return vectorize_density_matrix!(deepcopy(ρ), unvectorizedinds, vectorizedinds)
end =#

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
    return inner(ρ.network, new_op.network; kwargs...)
end

end; # module
