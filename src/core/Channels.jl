module Channels
export Channel, depolarizing_channel, apply, opdouble, find_site

using ITensors
import OpenSystemsTools: Vectorization
using ITensorNetworks: AbstractITensorNetwork, ITensorNetwork, ITensorNetworks
using OpenNetworks: VectorizationNetworks, Utils, VDMNetworks

vectorizer = Vectorization.vectorizer
vectorizer_input = Vectorization.vectorizer_input
vectorizer_output = Vectorization.vectorizer_output
vectorize_density_matrix = Vectorization.vectorize_density_matrix
vectorize_density_matrix! = Vectorization.vectorize_density_matrix!
vectorizedexpect = Vectorization.vectorizedexpect
leftrightapply = Vectorization.leftrightapply
basespace = Vectorization.basespace
VDMNetwork = VDMNetworks.VDMNetwork

#Do I need this?

ITensors.op(::OpName"id",::SiteType"Qubit") =
 [1 0
  0 1]

ITensors.op(::OpName"0tens",::SiteType"Qubit") =
    [0 0
    0 0]




function vexpect(obs::MPO, rho::MPS)
    @assert false "vexpect has not been tested yet."
    s = siteinds(rho)
    vobs = vectorize_density_matrix(obs, s)
    return inner(vobs, rho)
end



function tagstring(T::ITensors.TagSet)::String
    # Purpose: Takes a tag set and converts it into a string.

    res = ""
    ts = [tag for tag in T]
    N = length(T)
    for i in 1:N-1
        res *= "$(ts[i]),"
    end
    res *= "$(ts[N])"
    return res
end

function find_site(ind::ITensors.Index)
    #= Purpose: Given a site index for an ITensor, this function will return the site it corresponds to.=#

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

function find_site(ind::ITensors.Index, ψ::ITensorNetwork)::Tuple
    #= Purpose: Finds the site of an index.=#
    for key in keys(siteinds(ψ).data_graph.vertex_data)
        if ind in siteinds(ψ)[key]
            return key
        end
    end
end

function find_site(ind::ITensors.Index, ρ::VDMNetwork)::Tuple
    #= Purpose: Finds the site of an index.=#
    return find_site(ind, ρ.network)
end

    

function opdouble(o::ITensor, rho::MPS)::ITensor
    #= Purpose: Turns an ITensor, an operator O on the underlying Hilbert space returns an opertor O†⊗O acting on the doubled Hilbert space.
    Inputs: o (ITensor) - Operator on underlying Hilbert Space.
            rho (MPS) - Density Matrix of the system.
    Returns: ITensor - Operator acting on the doubled Hilbert space. =#

    inds = [ind for ind in ITensors.inds(o) if plev(ind)==0]
    vs = ITensors.siteinds(rho)
    o_dag = addtags(dag(o), "dag")
    o *= o_dag
    site_list = Vector{Int}()
    for ind in inds
        site = find_site(ind)
        push!(site_list, site)
        vinds = [vind for vind in ITensors.inds(rho[site]) if hastags(vind, "Site")]
        @assert length(vinds) == 1 "Tensors of a vectorized MPS should have exactly one physical leg."
        vind = vinds[1]
        spacename = basespace(vind)
        @assert hastags(ind, spacename) "Operator must have the same site-type as vectorized MPS."

        o *= ITensors.delta(ITensors.dag(ind), ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(addtags(ITensors.dag(ind), "dag"),vectorizer_input(spacename)')
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind)

        o *= ITensors.delta(ITensors.dag(ind)',ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(addtags(ITensors.dag(ind)', "dag"),vectorizer_input(spacename)')
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind')
    end
    return o
end

function opdouble(o::ITensor, rho::VDMNetwork)::ITensor
    #= Purpose: Turns an ITensor, an operator O on the underlying Hilbert space returns an opertor O†⊗O acting on the doubled Hilbert space.
    Inputs: o (ITensor) - Operator on underlying Hilbert Space.
            rho (ITensorNetwork) - Density Matrix of the system.
    Returns: ITensor - Operator acting on the doubled Hilbert space. =#
    ψ = rho.unvectorizednetwork
    rho = rho.network

    inds = [ind for ind in ITensors.inds(o) if ITensors.plev(ind)==0]
    vs = siteinds(rho)
    o_dag = ITensors.addtags(dag(o), "dag")
    o *= o_dag
    site_list = Vector{Tuple}()
    for ind in inds
        site = find_site(ind, ψ)
        push!(site_list, site)
        vinds = [vind for vind in ITensors.inds(rho[site]) if ITensors.hastags(vind, "Site")]
        @assert length(vinds) == 1 "Tensors of a vectorized density matrix should have exactly one physical leg."
        vind = vinds[1]
        spacename = basespace(vind)
        @assert ITensors.hastags(ind, spacename) "Operator must have the same site-type as vectorized MPS."

        o *= ITensors.delta(ITensors.dag(ind), ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(ITensors.addtags(ITensors.dag(ind), "dag"),vectorizer_input(spacename)')
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind)

        o *= ITensors.delta(ITensors.dag(ind)',ITensors.dag(vectorizer_input(spacename)))
        o *= ITensors.delta(ITensors.addtags(ITensors.dag(ind)', "dag"),vectorizer_input(spacename)')
        o *= vectorizer(spacename)
        o *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vind')
    end
    return o
end

function _krauscheck(kraus_maps_true::Vector{ITensor})::Bool
    #= Purpose: Checks if the Kraus operators are valid. Sum of Kraus operators multiplied by conjugates should be close to identity.
    Inputs: kraus_maps_true (Vector{ITensor}) - Vector of Kraus operators.
    Returns: Bool - True if ΣKK† ≈ I, False otherwise. =#
    kraus_maps = [deepcopy(kr) for kr in kraus_maps_true]
    sites = [ind for ind in ITensors.inds(kraus_maps[1]) if plev(ind)==0]
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

function _krausindscheck(kraus_maps::Vector)
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

    function Channel(name::String, tensor::ITensor)
        return new(name, tensor)
    end

    function Channel(name::String, kraus_maps::Vector{ITensor}, rho::MPS)
            @assert _krauscheck(kraus_maps)==true "Kraus operators invalid: ΣKK† ≆ I"
            _krausindscheck(kraus_maps)
            tensor = reduce(+, [opdouble(kraus, rho) for kraus in kraus_maps])
            return new(name, tensor)
    end

    function Channel(name::String, kraus_maps::Vector{ITensor}, ρ::VDMNetwork)
            @assert _krauscheck(kraus_maps)==true "Kraus operators invalid: ΣKK† ≆ I"
            _krausindscheck(kraus_maps)
            tensor = reduce(+, [opdouble(kraus, ρ) for kraus in kraus_maps])
            return new(name, tensor)
    end
end


function depolarizing_channel(p::Real, sites::Vector, rho:: MPS)::Channel
    #= Purpose: Creates a depolarizing channel for a given density matrix.
    Inputs: p (Real) - Parameter for the depolarizing channel. Must be between 0 and 1.
            sites (Vector) - Vector of sites that the channel acts on.
            rho (MPS) - Density matrix that the channel acts on.
    Returns: Channel - Depolarizing channel with the given parameters. =#

    if !(0 <= p <= 1) throw("parameter p must be between 0 and 1.") end
    for site in sites
        @assert find_site(site) <= length(siteinds(rho)) "All sites must be sites that ρ has."
        @assert hastags(site, "Qubit") "Depolarizing channel only implemented for Qubits."
    end

    k = length(sites)
    if k == 1
        const0 = sqrt(1 - 3*p/4)
        constσ = sqrt(p/4)
        kraus_maps = [constσ*op("σx", sites[1]), constσ*op("σy", sites[1]), constσ*op("σz", sites[1]), const0*op("Id", sites[1])]
        return Channel("depolarizing", kraus_maps, rho)

    elseif k==2
        const0 = sqrt(1 - 15*p/16)
        constσ = sqrt(p/16)
        pauli1 = [op("σx", sites[1]), op("σy", sites[1]), op("σz", sites[1]), op("Id", sites[1])]
        pauli2 = [op("σx", sites[2]), op("σy", sites[2]), op("σz", sites[2]), op("Id", sites[2])]
        kraus_maps = [constσ*x*y for x in pauli1 for y in pauli2]
        replace!(kraus_maps, Pair(constσ*pauli1[end]*pauli2[end], const0*pauli1[end]*pauli2[end]))
        return Channel("depolarizing", kraus_maps, rho)
    
    else
        throw("Depolarizing channel for more than two qubit gates has not been implemented.")
    end
end


function depolarizing_channel(p::Real, sites::Vector, rho:: VDMNetwork)::Channel
    #= Purpose: Creates a depolarizing channel for a given density matrix.
    Inputs: p (Real) - Parameter for the depolarizing channel. Must be between 0 and 1.
            sites (Vector) - Vector of sites that the channel acts on.
            rho (MPS) - Density matrix that the channel acts on.
    Returns: Channel - Depolarizing channel with the given parameters. =#

    if !(0 < p <= 1) throw("parameter p must be between 0 and 1.") end
    for site in sites
        #@assert find_site(site) <= length(siteinds(rho)) "All sites must be sites that ρ has."
        if !(hastags(site, "Qubit")) throw("Depolarizing channel only implemented for Qubits.") end
    end

    k = length(sites)
    if k == 1
        const0 = sqrt(1 - 3*p/4)
        constσ = sqrt(p/4)
        kraus_maps = [constσ*op("σx", sites[1]), constσ*op("σy", sites[1]), constσ*op("σz", sites[1]), const0*op("Id", sites[1])]
        return Channel("depolarizing", kraus_maps, rho)

    elseif k==2
        const0 = sqrt(1 - 15*p/16)
        constσ = sqrt(p/16)
        pauli1 = [op("σx", sites[1]), op("σy", sites[1]), op("σz", sites[1]), op("Id", sites[1])]
        pauli2 = [op("σx", sites[2]), op("σy", sites[2]), op("σz", sites[2]), op("Id", sites[2])]
        kraus_maps = [constσ*x*y for x in pauli1 for y in pauli2]
        replace!(kraus_maps, Pair(constσ*pauli1[end]*pauli2[end], const0*pauli1[end]*pauli2[end]))
        return Channel("depolarizing", kraus_maps, rho)
    
    else
        throw("Depolarizing channel for more than two qubit gates has not been implemented.")
    end
end


function apply(channel::Channel, ρ::MPS; kwargs...)::MPS
    channel_tensor = channel.tensor
    return ITensors.apply(channel_tensor, ρ; kwargs...)
end

function apply(channel::Channel, ρ::VDMNetwork; kwargs...)::VDMNetwork
    channel_tensor = channel.tensor
    return VDMNetwork(ITensorNetworks.apply(channel_tensor, ρ.network; kwargs...), ρ.unvectorizednetwork)
end



function apply(o::ITensors.ITensor, ρ::VDMNetwork)::VDMNetwork
    o2 = opdouble(o, ρ)
    return VDMNetwork(ITensorNetworks.apply(o2, ρ.network), ρ.unvectorizednetwork)
end

function compose(post:: Channel, pre::Channel)::Channel
    matching = [ind for ind in inds(post.tensor) if ind in inds(pre.tensor) && plev(ind)==0]
    tens_post = deepcopy(post.tensor)
    tens_pre = deepcopy(pre.tensor)
    for ind in matching
        tens_post *= ITensors.delta(ind, ind'')
        tens_pre *= ITensors.delta(ind', ind'')
    end
    new_tensor = tens_post * tens_pre
    return Channel(post.name * "∘" * pre.name, new_tensor)
    #=
    if lpost == lpre
        tens2 = prime(post.tensor)
        new_tensor = tens2 * pre.tensor
        swapprime!(new_tensor, 1, 2)
        return Channel(post.name * "∘" * pre.name , new_tensor)
    elseif lpost > lpre
        tens2 = prime(post.tensor)
        new_tensor = tens2 * pre.tensor
        setprime!(new_tensor, 0; plev=1)
        swapprime!(new_tensor, 1, 2)
        return Channel(post.name * "∘" * pre.name , new_tensor)
    elseif lpost < lpre
        tens2 = prime(post.tensor)
        new_tensor = tens2 * pre.tensor
        setprime!(new_tensor, 1; plev=2)
        return Channel(post.name * "∘" * pre.name , new_tensor)
    else
        throw("Compose not implemented for Channels sizes: lpost:$lpost lpre: $lpre")
    end
    =#
end
end;