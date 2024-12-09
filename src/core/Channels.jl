module Channels
export Channel, apply, opdouble

using ITensors
using ITensorMPS
import ITensorsOpenSystems: Vectorization, Vectorization.fatsiteinds
using ITensorNetworks: AbstractITensorNetwork, ITensorNetwork, ITensorNetworks, IndsNetwork
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

    res = ""
    ts = [tag for tag in T]
    N = length(T)
    for i in 1:(N - 1)
        res *= "$(ts[i]),"
    end
    res *= "$(ts[N])"
    return res
end
#=
function find_site(ind::ITensors.Index)
    #= Given a site index for an ITensor, this function will return the site it corresponds to.=#

    @assert hastags(ind, "Site") "Can't find site: Index has no site."
    ts = tagstring(tags(ind))
    site = Vector{Int}()
    for s in split(ts, ",")
        if startswith(s, "n=")
            push!(site, parse(Int, split(s, "=")[end]))
        end
    end
    @assert length(site) == 1 "Can't find site: check that index has exactly one site tag."
    return site[1]
end
=#

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
        o *= delta(ITensors.dag(ind), ITensors.dag(vectorizer_input(spacename)))
        o *= delta(addtags(ITensors.dag(ind), "dag"), vectorizer_input(spacename)')
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), fatind)
        o *= ITensors.delta(ITensors.dag(ind)', ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(
            addtags(ITensors.dag(ind)', "dag"), vectorizer_input(spacename)'
        )
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), fatind')
    end
    return o
end

function opdouble(o::ITensor, rho::MPS, sites::Vector{<:ITensors.Index{}})::ITensor
    #=Turns an ITensor, an operator O on the underlying Hilbert space returns an opertor O†⊗O acting on the doubled Hilbert space.=#

    inds = [ind for ind in ITensors.inds(o) if plev(ind) == 0]
    vs = ITensors.siteinds(rho)
    o_dag = addtags(dag(o), "dag")
    o *= o_dag
    site_list = Vector{Int}()
    for ind in inds
        site = findsite(sites, ind)
        push!(site_list, site)
        vinds = [vind for vind in ITensors.inds(rho[site]) if hastags(vind, "Site")]
        @assert length(vinds) == 1 "Tensors of a vectorized MPS should have exactly one physical leg."
        vind = vinds[1]
        spacename = basespace(vind)
        @assert hastags(ind, spacename) "Operator must have the same site-type as vectorized MPS."

        o *= ITensors.delta(ITensors.dag(ind), ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(addtags(ITensors.dag(ind), "dag"), vectorizer_input(spacename)')
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind)

        o *= ITensors.delta(ITensors.dag(ind)', ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(
            addtags(ITensors.dag(ind)', "dag"), vectorizer_input(spacename)'
        )
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind')
    end
    return o
end

function opdouble(
    o::ITensor,
    rho::Vectorization.VectorizedDensityMatrix,
    unvectorizedsites::Vector{<:ITensors.Index{}},
)::ITensor
    #= Turns an ITensor, an operator O on the underlying Hilbert space returns an opertor O†⊗O acting on the doubled Hilbert space.=#

    inds = [ind for ind in ITensors.inds(o) if plev(ind) == 0]
    vs = ITensors.siteinds(rho)
    @assert length(vs) == length(unvectorizedsites)
    o_dag = addtags(dag(o), "dag")
    o *= o_dag
    site_list = Vector{Int}()
    for ind in inds
        site = findsite(unvectorizedsites, ind)
        push!(site_list, site)
        vinds = [vind for vind in ITensors.inds(rho[site]) if hastags(vind, "Site")]
        @assert length(vinds) == 1 "Tensors of a vectorized MPS should have exactly one physical leg."
        vind = vinds[1]
        spacename = basespace(vind)
        @assert hastags(ind, spacename) "Operator must have the same site-type as vectorized MPS."

        o *= ITensors.delta(ITensors.dag(ind), ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(addtags(ITensors.dag(ind), "dag"), vectorizer_input(spacename)')
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind)

        o *= ITensors.delta(ITensors.dag(ind)', ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(
            addtags(ITensors.dag(ind)', "dag"), vectorizer_input(spacename)'
        )
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind')
    end
    return o
end

function opdouble(o::ITensor, ρ::VDMNetwork{V})::ITensor where {V}
    #= Purpose: Turns an ITensor, an operator O on the underlying Hilbert space returns an opertor O†⊗O acting on the doubled Hilbert space.
    Inputs: o (ITensor) - Operator on underlying Hilbert Space.
            ρ (ITensorNetwork) - Density Matrix of the system.
    Returns: ITensor - Operator acting on the doubled Hilbert space. =#
    ψ = ITensorNetwork(v -> "0", ρ.unvectorizedinds)
    ρ = ρ.network

    inds = [ind for ind in ITensors.inds(o) if ITensors.plev(ind) == 0]
    vs = siteinds(ρ)
    o_dag = ITensors.addtags(dag(o), "dag")
    o *= o_dag
    site_list = Vector{V}()
    for ind in inds
        site = findsite(ψ, ind)
        push!(site_list, site)
        vinds = [vind for vind in ITensors.inds(ρ[site]) if ITensors.hastags(vind, "Site")]
        @assert length(vinds) == 1 "Tensors of a vectorized density matrix should have exactly one physical leg."
        vind = vinds[1]
        spacename = basespace(vind)
        @assert ITensors.hastags(ind, spacename) "Operator must have the same site-type as vectorized MPS."

        o *= ITensors.delta(ITensors.dag(ind), ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(
            ITensors.addtags(ITensors.dag(ind), "dag"), vectorizer_input(spacename)'
        )
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind)

        o *= ITensors.delta(ITensors.dag(ind)', ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(
            ITensors.addtags(ITensors.dag(ind)', "dag"), vectorizer_input(spacename)'
        )
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind')
    end
    return o
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

function apply(channel::Channel, ρ::MPS; kwargs...)::MPS
    channel_tensor = channel.tensor
    return ITensors.apply(channel_tensor, ρ; kwargs...)
end

function apply(channel::Channel, ρ::VDMNetwork{V}; kwargs...)::VDMNetwork{V} where {V}
    channel_tensor = channel.tensor
    return VDMNetwork(
        ITensorNetworks.apply(channel_tensor, ρ.network; kwargs...), ρ.unvectorizedinds
    )
end

function apply(o::ITensors.ITensor, ρ::VDMNetwork{V}; kwargs...)::VDMNetwork{V} where {V}
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
