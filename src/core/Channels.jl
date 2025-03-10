module Channels
export Channel, opdouble

using ITensors
using ITensorMPS
import ITensorsOpenSystems: Vectorization, Vectorization.fatsiteinds
using ITensorNetworks:
    AbstractITensorNetwork, ITensorNetwork, ITensorNetworks, IndsNetwork, vertices
using OpenNetworks: VectorizationNetworks, Utils, VDMNetworks
using NamedGraphs: vertices

vectorizer = Vectorization.vectorizer
vectorizer_input = Vectorization.vectorizer_input
vectorizer_output = Vectorization.vectorizer_output
vectorize_density_matrix = Vectorization.VectorizedDensityMatrix
VectorizedDensityMatrix = Vectorization.VectorizedDensityMatrix
vectorize_density_matrix! = Vectorization.vectorize_density_matrix!
basespace = Vectorization.basespace
VDMNetwork = VDMNetworks.VDMNetwork

ITensors.op(::OpName"id", ::SiteType"Qubit") = [
    1 0
    0 1
]

ITensors.op(::OpName"0tens", ::SiteType"Qubit") = [
    0 0
    0 0
]
#=
function vexpect(obs::MPO, rho::MPS)
    @assert false "vexpect has not been tested yet."
    s = siteinds(rho)
    vobs = vectorize_density_matrix(obs, s)
    return inner(vobs, rho)
end
=#

function tagstring(T::ITensors.TagSet)::String
    # Takes a tag set and converts it into a string.
    # Internal function, not covered in tests.

    res = ""
    ts = [tag for tag in T]
    N = length(T)
    for i in 1:(N - 1)
        res *= "$(ts[i]),"
    end
    res *= "$(ts[N])"
    return res
end

function ITensors.findsite(
    sites::Vector{ITensors.Index{Q}}, ind::ITensors.Index{Q}
) where {Q}
    locations = findall(y -> y == ind, sites)
    if length(locations) > 1
        throw("Site indices have the same index in multiple sites!")
    end
    return first(locations)
end

function ITensors.findsite(sites::IndsNetwork, ind::ITensors.Index)
    #= Purpose: Finds the site of an index.=#
    for vertex in vertices(sites)
        if ind in sites[vertex]
            return vertex
        end
    end
end

function ITensors.findsite(ψ::ITensorNetwork{V}, ind::ITensors.Index)::V where {V}
    #= Purpose: Finds the site of an index.=#
    return findsite(siteinds(ψ), ind)
end

function ITensors.findsite(
    ψ::ITensorNetworks.VidalITensorNetwork{V}, ind::ITensors.Index
)::V where {V}
    #=  Finds the site of an index.=#
    return findsite(siteinds(ψ), ind)
end

function ITensors.findsite(ρ::VDMNetwork{V}, ind::ITensors.Index)::V where {V}
    #= Finds the site of an index.=#
    return findsite(ρ.network, ind)
end

function opdouble(o::ITensor)::ITensor
    indices = collect(inds(o; plev=0))
    odag = addtags(dag(o), "dag")
    o *= odag
    fatindices = fatsiteinds(indices)
    for (ind, fatind) in zip(indices, fatindices)
        spacename = basespace(fatind)
        @assert hastags(ind, spacename)
        o *= delta(ITensors.dag(ind), vectorizer_input(spacename)')
        o *= delta(
            addtags(ITensors.dag(ind), "dag"), ITensors.dag(vectorizer_input(spacename))
        )
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), fatind)
        o *= ITensors.delta(ITensors.dag(ind)', vectorizer_input(spacename)')
        o *= ITensors.delta(
            addtags(ITensors.dag(ind)', "dag"), ITensors.dag(vectorizer_input(spacename))
        )
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), fatind')
    end
    return o
end

function opdouble(o::ITensor, fatinds::Vector{<:Index{}})::ITensor
    od = opdouble(o)
    oldfatinds = inds(od; plev=0)
    for (i, ind) in enumerate(oldfatinds)
        newfatind = fatinds[i]
        @assert hastags(newfatind, tags(ind))
        od *= delta(ind, newfatind)
        od *= delta(ind', newfatind')
    end
    return od
end

function opdouble(
    o::ITensor,
    rho::Vectorization.VectorizedDensityMatrix,
    unvectorizedsites::Vector{ITensors.Index{Q}},
)::ITensor where {Q}
    indices = collect(inds(o; plev=0))
    vs = ITensors.siteinds(rho)
    @assert length(vs) == length(unvectorizedsites)
    fatinds = Vector{Index{Q}}()
    for ind in indices
        site = findsite(unvectorizedsites, ind)
        push!(fatinds, vs[site])
    end
    return opdouble(o, fatinds)
end

function opdouble(o::ITensor, ρ::VDMNetwork{V})::ITensor where {V}
    indices = collect(inds(o; plev=0))
    vs = siteinds(ρ)
    firstind = first(vs[first(vertices(vs))])
    ψ = ITensorNetwork(v -> "0", ρ.unvectorizedinds)
    fatinds = Vector{typeof(firstind)}()
    for ind in indices
        site = findsite(ψ, ind)
        push!(fatinds, first(vs[site]))
    end
    return opdouble(o, fatinds)
end

function _krauscheck(kraus_maps_true::Vector{ITensor})::Bool
    #= Purpose: Checks if the Kraus operators are valid. Sum of Kraus operators multiplied by conjugates should be close to identity.
    Inputs: kraus_maps_true (Vector{ITensor}) - Vector of Kraus operators.
    Returns: Bool - True if ΣK†K ≈ I, False otherwise. =#
    kraus_maps = [deepcopy(kr) for kr in kraus_maps_true]
    sites = [ind for ind in ITensors.inds(kraus_maps[1]) if plev(ind) == 0]
    kr_sum = reduce(*, [op("0tens", site) for site in sites])
    id_ops = reduce(*, [op("id", site) for site in sites])
    for kr in kraus_maps
        @assert typeof(kr) == ITensor
        setprime!(kr, 2; plev=0)
        new_kr = kr * id_ops
        setprime!(new_kr, 2; plev=1)
        dag_kr = setprime!(dag(new_kr), 1; plev=0)
        kr_sum += new_kr * dag_kr
    end
    return Array(kr_sum, inds(kr_sum)) ≈ Array(id_ops, inds(kr_sum))
end

function _krausindscheck(kraus_maps::Vector{ITensor})::Nothing
    #= Purpose: Checks if the Kraus operators are acting on the same indices.
    Inputs: kraus_maps (Vector) - Vector of Kraus operators.
    Returns: Nothing =#

    kraus1 = kraus_maps[1]
    for kraus in kraus_maps[2:end]
        @assert isa(noncommonind(kraus1, kraus), Nothing) "Kraus Operators are not acting on the same indices. Use identity maps if required."
    end
end

"""
Channel

A Channel is a struct which is designed to represent a quantum channel.
It contains a name (for debugging purposes), and an ITensor (which 
contains the quantum channel acting on a vectorized density matrix).

# Examples
```julia
julia> s = siteinds("Qubit", 2)
2-element Vector{Index{Int64}}:
 (dim=2|id=286|"Qubit,Site,n=1")
 (dim=2|id=835|"Qubit,Site,n=2")

julia> oZ = op("Z", s[1])
ITensor ord=2 (dim=2|id=286|"Qubit,Site,n=1")' (dim=2|id=286|"Qubit,Site,n=1")
NDTensors.Dense{Float64, Vector{Float64}}

julia> oI = op("I", s[1])
ITensor ord=2 (dim=2|id=286|"Qubit,Site,n=1")' (dim=2|id=286|"Qubit,Site,n=1")
NDTensors.Dense{Float64, Vector{Float64}}

julia> p = 0.01
0.01

julia> kraus_maps = [sqrt(1-p)* oI, sqrt(p) * oZ]
2-element Vector{ITensor}:
 ITensor ord=2
Dim 1: (dim=2|id=286|"Qubit,Site,n=1")'
Dim 2: (dim=2|id=286|"Qubit,Site,n=1")
NDTensors.Dense{Float64, Vector{Float64}}
 2×2
 0.99498743710662  0.0
 0.0               0.99498743710662
 ITensor ord=2
Dim 1: (dim=2|id=286|"Qubit,Site,n=1")'
Dim 2: (dim=2|id=286|"Qubit,Site,n=1")
NDTensors.Dense{Float64, Vector{Float64}}
 2×2
 0.1   0.0
 0.0  -0.1

 # Construct a dephasing Channel from the kraus maps.

julia> dephasing = Channel("dephasing", kraus_maps)
Channel dephasing with indices ((dim=4|id=282|"QubitVec,Site,n=1"), (dim=4|id=282|"QubitVec,Site,n=1")')
```
"""

struct Channel
    name::String
    tensor::ITensor

    function Channel(name::String, tensor::ITensor)::Channel
        return new(name, tensor)
    end

    function Channel(name::String, kraus_maps::Vector{ITensor})::Channel
        @assert _krauscheck(kraus_maps) == true "Kraus operators invalid: ΣK†K ≆ I"
        _krausindscheck(kraus_maps)
        kraus1 = first(kraus_maps)
        fatindices = fatsiteinds(collect(inds(kraus1; plev=0)))
        doubledkraus = Vector{ITensor}()
        for kraus in kraus_maps
            od = opdouble(kraus)
            for fatind in fatindices
                odind = inds(od; tags=tags(fatind), plev=0)
                @assert length(odind) == 1
                odind = first(odind)
                od *= delta(odind, fatind)
                od *= delta(odind', fatind')
            end
            push!(doubledkraus, od)
        end
        tensor = reduce(+, doubledkraus)
        return new(name, tensor)
    end

    function Channel(
        name::String,
        kraus_maps::Vector{ITensor},
        rho::Vectorization.VectorizedDensityMatrix,
        unvectorizedsiteinds::Vector{<:ITensors.Index{}},
    )::Channel
        @assert _krauscheck(kraus_maps) == true "Kraus operators invalid: ΣK†K ≆ I"
        _krausindscheck(kraus_maps)
        tensor = reduce(
            +, [opdouble(kraus, rho, unvectorizedsiteinds) for kraus in kraus_maps]
        )
        return new(name, tensor)
    end

    function Channel(
        name::String, kraus_maps::Vector{ITensor}, ρ::VDMNetwork{V}
    )::Channel where {V}
        @assert _krauscheck(kraus_maps) == true "Kraus operators invalid: ΣK†K ≆ I"
        _krausindscheck(kraus_maps)
        tensor = reduce(+, [opdouble(kraus, ρ) for kraus in kraus_maps])
        return new(name, tensor)
    end
end

function Base.show(io::IO, channel::Channel)
    return println(io, "Channel $(channel.name) with indices $(inds(channel.tensor))")
end

#=
Not tested and should not be used.
function ITensors.apply(channel::Channel, ρ::MPS; kwargs...)::MPS
    channel_tensor = channel.tensor
    return ITensors.apply(channel_tensor, ρ; kwargs...)
end
=#

function ITensors.apply(
    channel::Channel, ρ::VectorizedDensityMatrix; kwargs...
)::VectorizedDensityMatrix
    return ITensors.apply(channel.tensor, ρ; kwargs...)
end

function ITensors.apply(
    channel::Channel, ρ::VDMNetwork{V}; kwargs...
)::VDMNetwork{V} where {V}
    channel_tensor = channel.tensor
    return VDMNetwork(
        ITensorNetworks.apply(channel_tensor, ρ.network; kwargs...), ρ.unvectorizedinds
    )
end

function ITensors.apply(channels::Vector{Channel}, ρ::VDMNetwork{V}; kwargs...) where {V}
    channels_tensor = [channel.tensor for channel in channels]
    return VDMNetwork(
        ITensorNetworks.apply(channels_tensor, ρ.network; kwargs...), ρ.unvectorizedinds
    )
end

function ITensors.apply(channels::Vector{Channel}, ρ::VectorizedDensityMatrix; kwargs...)
    channels_tensor = [channel.tensor for channel in channels]
    return apply(channels_tensor, ρ; kwargs...)
end

function ITensors.apply(
    o::ITensors.ITensor, ρ::VDMNetwork{V}; kwargs...
)::VDMNetwork{V} where {V}
    o2 = opdouble(o, ρ)
    return VDMNetwork(ITensorNetworks.apply(o2, ρ.network; kwargs...), ρ.unvectorizedinds)
end

function compose(post::Channel, pre::Channel)::Channel
    matching = [
        ind for ind in inds(post.tensor) if ind in inds(pre.tensor) && plev(ind) == 0
    ]
    tens_post = deepcopy(post.tensor)
    tens_pre = deepcopy(pre.tensor)
    for ind in matching
        tens_post *= ITensors.delta(ind, ind'')
        tens_pre *= ITensors.delta(ind', ind'')
    end
    new_tensor = tens_post * tens_pre
    return Channel(post.name * "∘" * pre.name, new_tensor)
end

end; # module
