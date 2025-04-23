module VDMNetworks
export VDMNetwork
using ITensorsOpenSystems:
    Vectorization.basespace,
    Vectorization.vectorizer_input,
    Vectorization.vectorizer_output,
    Vectorization.vectorizer

using ITensorNetworks: ITensorNetwork, IndsNetwork, vertices
using ITensors: Index, ITensors, QN

"""
    VDMNetwork
A structure to represent a vectorized density matrix network.
    network::ITensorNetwork{V}
        The underlying ITensorNetwork representing the density matrix.
    unvectorizedinds::IndsNetwork{V,Index}
        The unvectorized indices of the state.
"""

struct VDMNetwork{V}
    network::ITensorNetwork{V}
    unvectorizedinds::IndsNetwork{V,Index}
end

function Base.show(io::IO, ρ::VDMNetwork)
    return println(io, "VDMNetwork with underlying ITensorNetwork: $(ρ.network)")
end

"""
    VDMNetwork(ρ::ITensorNetwork{V}, sites::IndsNetwork{V,Index}, fatsites::IndsNetwork{V,Index}) where {V}

    Arguments:
    ρ::ITensorNetwork{V}
        The underlying ITensorNetwork representing the density matrix.
    sites::IndsNetwork{V,Index}
        The unvectorized indices of the state.
    fatsites::IndsNetwork{V,Index}
        The vectorized indices of the state.

    Constructs a VDMNetwork from the given ITensorNetwork using fatsites as the vectorized indices.
"""

function VDMNetwork(
    ρ::ITensorNetwork{V}, sites::IndsNetwork{V,Index}, fatsites::IndsNetwork{V,Index}
) where {V}
    new_ρ = vectorize_density_matrix(ρ, sites, fatsites)
    return VDMNetwork{V}(new_ρ, sites)
end

function vectorize_density_matrix(
    ρ::ITensorNetwork{V},
    unvectorizedinds::IndsNetwork{V,Index},
    vectorizedinds::IndsNetwork{V,Index},
)::ITensorNetwork{V} where {V}
    return vectorize_density_matrix!(deepcopy(ρ), unvectorizedinds, vectorizedinds)
end

function vectorize_density_matrix!(
    ρ::ITensorNetwork{V},
    unvectorizedinds::IndsNetwork{V,Index},
    vectorizedinds::IndsNetwork{V,Index},
)::ITensorNetwork{V} where {V}
    for vertex in vertices(unvectorizedinds)
        @assert length(vectorizedinds[vertex]) == 1 "vectorized index at site $vertex is not unique"
        @assert length(unvectorizedinds[vertex]) == 1 "site $vertex has wrong number of indices"
        spacename = basespace(vectorizedinds[vertex][1])::String
        vin = vectorizer_input(spacename)::Index{Vector{Pair{QN,Int64}}}
        vout = vectorizer_output(spacename)::Index{Vector{Pair{QN,Int64}}}
        vz = vectorizer(spacename)::ITensors.ITensor
        ui = unvectorizedinds[vertex][1]
        vi = vectorizedinds[vertex][1]
        if !ITensors.hastags(ui, spacename)
            throw(
                ArgmentError(
                    "The vectorised index at site $i, $(vectorizedinds[vertex][1]), does not match the unvectorized index $(ui)",
                ),
            )
        end
        ρ[vertex]::ITensors.ITensor *= ITensors.delta(ITensors.dag(ui), ITensors.dag(vin))
        ρ[vertex]::ITensors.ITensor *= ITensors.delta(ui', vin')
        ρ[vertex]::ITensors.ITensor *= vz
        ρ[vertex]::ITensors.ITensor *= ITensors.delta(ITensors.dag(vout), vi)
    end
    return ρ
end

function show(io::IO, vdm::VDMNetwork)
    println(io, "VDMNetwork with underlying ITensorNetwork:")
    return show(io, vdm.network)
end


#=
Functions for updating the network of indices of a VDMNetwork. 
Not been tested.

update_unvectorizednetwork!(vdm::VDMNetwork, unvectorizedinds::IndsNetwork)::VDMNetwork =
    VDMNetwork(vdm.network, unvectorizedinds)
update_unvectorizednetwork(vdm::VDMNetwork, unvectorizedinds::IndsNetwork)::VDMNetwork =
    update_unvectorizednetwork!(deepcopy(vdm), unvectorizedinds)
=#

end;