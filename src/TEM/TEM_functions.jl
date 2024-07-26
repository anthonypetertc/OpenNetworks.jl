module TEM_functions
export itensors_to_ito,
    bondnetwork, bondnetwork!, compile_moment_into_single_tensors, compile_moment_into_tno
using ITensors
using ITensorNetworks
using OpenSystemsTools
using OpenNetworks:
    Utils,
    VectorizationNetworks,
    VDMNetworks,
    Channels,
    GraphUtils,
    NoiseModels,
    NoisyCircuits,
    CustomParsing
using JSON
using Graphs: vertices
using ITensorNetworks:
    BeliefPropagationCache,
    ITensorNetwork,
    VidalITensorNetwork,
    apply,
    environment,
    norm_sqr_network,
    random_tensornetwork,
    siteinds,
    update,
    itensors_to_itensornetwork,
    flatten_networks
using ITensors: ITensors, inner, op
using NamedGraphs: PartitionVertex, named_grid, NamedEdge, NamedGraphs
using Random: Random
using SplitApplyCombine: group
using TimerOutputs

VDMNetwork = VDMNetworks.VDMNetwork

function squeeze(tensor::ITensors.ITensor)::ITensors.ITensor
    #Need to test this and move it somewhere else.
    dim1_legs = [ind for ind in inds(tensor) if dim(ind) == 1]
    other_legs = [ind for ind in inds(tensor) if !(ind in dim1_legs)]
    squeezed_array = dropdims(
        array(tensor, dim1_legs..., other_legs...); dims=Tuple(1:length(dim1_legs))
    )
    squeezed_tensor = ITensor(squeezed_array, other_legs...)
    return squeezed_tensor
end

function itensors_to_ito(
    ts::Vector{ITensor}, ρ::VDMNetwork, bnet::ITensorNetworks.IndsNetwork
)::ITensorNetwork
    g = siteinds(ρ.network).data_graph.underlying_graph
    tn = ITensorNetwork(g)
    for v in vertices(g)
        only_one_tensor_per_vertex = true
        for tensor in ts
            if intersect(inds(tensor), siteinds(ρ.network)[v]) != [] &&
                only_one_tensor_per_vertex
                tn[v] = tensor
                only_one_tensor_per_vertex = false
            elseif intersect(inds(tensor), siteinds(ρ.network)[v]) != [] &&
                !only_one_tensor_per_vertex
                throw("More than one tensor per vertex")
            end
        end
    end
    for v in vertices(g)
        if isempty(tn[v])
            tn[v] = make_id(v, bnet)
        end
    end
    return tn
end

dspace(s::String) = filter(x -> !isspace(x), s)

function bondtag(edge::NamedGraphs.NamedEdge)::ITensors.TagSet
    bondtag = ITensors.TagSet("bond")
    bondtag = ITensors.addtags(bondtag, dspace(string(edge.src)))
    bondtag = ITensors.addtags(bondtag, dspace(string(edge.dst)))
    return bondtag
end

bondtags(edges::Vector{V}) where {V} = [(edge, bondtag(edge)) for edge in edges]

function bondinds(edges::Vector{NamedEdge{V}}; linkdim=1) where {V}
    bts = bondtags(edges)
    bondinds = Vector{Tuple{NamedEdge{V},Index{Int64}}}()
    #QUESTION: do all Indices have type Int64 or can it vary?
    for (edge, tag) in bts
        push!(bondinds, (edge, Index(linkdim, tag)))
    end
    return bondinds
end

function bondnetwork!(s::ITensorNetworks.IndsNetwork; linkdim=1)
    edges = ITensorNetworks.edges(s)
    binds = bondinds(edges; linkdim=linkdim)
    for (edge, bondind) in binds
        src = edge.src
        dst = edge.dst
        push!(s[src], bondind)
        push!(s[dst], bondind)
    end
    return s
end

function bondnetwork(s::ITensorNetworks.IndsNetwork; linkdim=1)
    return bondnetwork!(deepcopy(s); linkdim=linkdim)
end

function make_id(vertex, bnet::ITensorNetworks.IndsNetwork)
    site_inds = [ind for ind in bnet[vertex] if hastags(ind, "Site")]
    bond_inds = [ind for ind in bnet[vertex] if hastags(ind, "bond")]
    @assert length(site_inds) == 1
    site_ind = first(site_inds)
    id = ITensors.delta(site_ind, site_ind')
    for bond in bond_inds
        @assert dim(bond) == 1
        id = ITensor(reshape(array(id), size(id)..., 1), inds(id), bond)
    end
    return id
end

function itensors_to_ito(
    ts::Vector{ITensor}, ψ::ITensorNetwork, bnet::ITensorNetworks.IndsNetwork
)
    g = siteinds(ψ).data_graph.underlying_graph
    tn = ITensorNetwork(g)
    for v in vertices(g)
        only_one_tensor_per_vertex = true
        for tensor in ts
            if intersect(inds(tensor), siteinds(ψ)[v]) != [] && only_one_tensor_per_vertex
                tn[v] = tensor
                only_one_tensor_per_vertex = false
            elseif intersect(inds(tensor), siteinds(ψ)[v]) != [] &&
                !only_one_tensor_per_vertex
                throw("More than one tensor per vertex")
            end
        end
    end
    for v in vertices(g)
        if isempty(tn[v])
            tn[v] = make_id(v, bnet)
        end
    end
    return tn
end

function compile_moment_into_single_tensors(
    tensor_list::Vector{ITensor}, bondnetwork::ITensorNetworks.IndsNetwork
)::Vector{ITensor}
    single_tensor_list = Vector{ITensor}()
    for tensor in tensor_list
        if length(inds(tensor)) != 4
            throw(
                "All channels in moment must be two qubit channels. Please re-compile circuit.",
            )
        end
        s1 = first(inds(tensor))
        one_side = [s for s in inds(tensor) if id(s) == id(s1)]
        site_one_side = Channels.find_site(one_side[1], bondnetwork)
        other_side = [s for s in inds(tensor) if id(s) != id(s1)]
        site_other_side = Channels.find_site(other_side[1], bondnetwork)
        @assert length(one_side) == 2
        @assert length(other_side) == 2
        one_side_legs = bondnetwork[site_one_side]
        other_side_legs = bondnetwork[site_other_side]
        bond_ind = first(intersect(Set(one_side_legs), Set(other_side_legs)))
        q, r = qr(tensor, one_side; tags=tags(bond_ind), positive=true)
        bondnetwork[site_one_side] = append!(
            filter(x -> !hastags(x, tags(bond_ind)), bondnetwork[site_one_side]),
            inds(q; tags=tags(bond_ind)),
        )
        bondnetwork[site_other_side] = append!(
            filter(x -> !hastags(x, tags(bond_ind)), bondnetwork[site_other_side]),
            inds(q; tags=tags(bond_ind)),
        )
        @assert length(bondnetwork[site_one_side]) == length(one_side_legs) "Bond legs should be replaced, not multiplied."
        @assert length(bondnetwork[site_other_side]) == length(other_side_legs) "Bond legs should be replaced, not multiplied."
        push!(single_tensor_list, q)
        push!(single_tensor_list, r)
    end
    return single_tensor_list
end

function compile_moment_into_single_tensors(
    moment::Vector{Channels.Channel}, bondnetwork::ITensorNetworks.IndsNetwork
)::Vector{ITensor}
    return compile_moment_into_single_tensors(
        [channel.tensor for channel in moment], bondnetwork
    )
end

function compile_moment_into_tno(
    single_tensor_list::Vector{ITensor},
    bondnetwork::ITensorNetworks.IndsNetwork,
    ρ::Union{VDMNetwork,ITensorNetworks.ITensorNetwork},
)
    list_of_compiled_tensors = Vector{ITensors.ITensor}()
    for tensor in single_tensor_list
        site_inds = inds(tensor; plev=0, tags="Site")
        @assert length(site_inds) == 1 "Tensors in single_tensor_list must act on one site only."
        loc = Channels.find_site(first(site_inds), bondnetwork)
        site_bonds = bondnetwork[loc]
        for index in site_bonds
            if isempty(inds(tensor; tags=tags(index)))
                tensor = ITensor(
                    reshape(Array(tensor.tensor), size(tensor.tensor)..., 1),
                    inds(tensor)...,
                    index,
                )
            elseif length(inds(tensor; tags=tags(index), plev=0)) != 1 ||
                !(index in inds(tensor; tags=tags(index)))
                throw("Bond-Network and list of tensors are not compatible.")
            end
        end
        push!(list_of_compiled_tensors, tensor)
    end
    return itensors_to_ito(list_of_compiled_tensors, ρ, bondnetwork)
end

function compile_circuit_into_tnos(
    ns::NoisyCircuits.NoisyCircuit, ρ::Union{VDMNetwork,ITensorNetworks.ITensorNetwork}
)::Vector{ITensorNetwork}
    tno_list = Vector{ITensorNetwork}()
    for moment in ns.moments_list
        sites = siteinds(ρ.network)
        bnet = bondnetwork(sites)
        @show sites
        @show bnet
        single_tensors_list = compile_moment_into_single_tensors(moment, bnet)
        tno = compile_moment_into_tno(single_tensors_list, bnet, ρ)
        push!(tno_list, tno)
    end
    return tno_list
end

function removetags!(ψ::ITensorNetworks.ITensorNetwork, tag)
    for v in vertices(ψ)
        ψ[v] = ITensors.removetags(ψ[v], tag)
    end
    return ψ
end

removetags(ψ, tag) = removetags!(deepcopy(ψ), tag)

function bp_contract_tnos(
    o1::ITensorNetwork, o2::ITensorNetwork; maxdim::Int, cache_update_kwargs
)
    o1p = prime(o1; tags="Site")
    ocomb = flatten_networks(o1p, o2; flatten=true, combine_linkinds=true)
    ocomb = ITensorNetworks.setprime(ocomb, 1; plev=2, tags="Site")
    ocomb = ITensorNetworks.addtags(ocomb, "Out"; plev=1, tags="Site")

    oo = norm_sqr_network(ocomb)
    bp_cache = BeliefPropagationCache(oo, group(v -> v[1], vertices(oo)))
    bp_cache = update(bp_cache; cache_update_kwargs...)
    cache_ref = Ref{BeliefPropagationCache}(bp_cache)

    vo = VidalITensorNetwork(ocomb; (cache!)=cache_ref, maxdim=maxdim)
    new_ocomb = ITensorNetworks.setprime(
        ITensorNetwork(vo; (cache!)=cache_ref), 1; tags="Out"
    )
    new_ocomb = removetags(new_ocomb, "Out")
    return new_ocomb
end

function bp_apply_tno(
    tno::ITensorNetwork, ψ::ITensorNetwork; maxdim::Int, cache_update_kwargs
)
    oψ = flatten_networks(tno, ψ; flatten=true, combine_linkinds=true)
    ψooψ = norm_sqr_network(oψ)

    bp_cache = BeliefPropagationCache(ψooψ, group(v -> v[1], vertices(ψooψ)))
    bp_cache = update(bp_cache; cache_update_kwargs...)
    cache_ref = Ref{BeliefPropagationCache}(bp_cache)

    voψ = VidalITensorNetwork(oψ; (cache!)=cache_ref, maxdim=maxdim)
    oψ = ITensorNetwork(voψ; (cache!)=cache_ref)
    return oψ
end

end
