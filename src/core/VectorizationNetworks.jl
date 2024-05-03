module VectorizationNetworks
export vectorize_density_matrix, VDMNetwork, innerprod, unvectorize_density_matrix

using ITensorNetworks: AbstractITensorNetwork, ITensorNetworks, ITensorNetwork, ⊗, siteinds
using ITensors: ITensor, prime, dag, combiner, inds, inner, op, randomITensor, delta, ITensors, Index
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


function vectorize_density_matrix!(ρ::AbstractITensorNetwork, ψ::AbstractITensorNetwork, vectorizedinds::ITensorNetworks.IndsNetwork)::VDMNetwork
    #ρ = operator_to_state(ρ, ψ)
    #inds = ITensors.siteinds(all,o,plev=0)
    # Then I want to get the site indices in a list that I can iterate through. I can do this by unvectorizedinds.data_graph.vertex_data
    unvectorizedinds = siteinds(ψ)
    inds = unvectorizedinds.data_graph.vertex_data
    
    for key in keys(inds)
        @assert length(vectorizedinds[key]) == 1 "vectorized index at site $key is not unique"
        @assert length(unvectorizedinds[key]) == 1 "site $key has wrong number of indices"
        spacename = basespace(vectorizedinds[key][1])
        
        if !ITensors.hastags(unvectorizedinds[key][1],spacename)
            throw(ArgmentError("The vectorised index at site $i, $(vectorizedinds[key][1]), does not match the unvectorized index $(unvectorizedinds[key][1])"))
        end
        ρ[key] *= ITensors.delta(ITensors.dag(unvectorizedinds[key][1]),ITensors.dag(vectorizer_input(spacename)))
        ρ[key] *= ITensors.delta(unvectorizedinds[key][1]',vectorizer_input(spacename)')
        ρ[key] *= vectorizer(spacename)
        ρ[key] *= ITensors.delta(ITensors.dag(vectorizer_output(spacename)), vectorizedinds[key][1])
    end
    return VDMNetwork(ρ, ψ)
end


function vectorize_density_matrix(ρ::AbstractITensorNetwork, ψ::AbstractITensorNetwork, vectorizedinds::ITensorNetworks.IndsNetwork)::VDMNetwork
    return vectorize_density_matrix!(deepcopy(ρ), ψ, vectorizedinds)
end



function unvectorize_density_matrix!(ρ::VDMNetwork)::AbstractITensorNetwork
    vectorizedinds = siteinds(ρ.network)
    unvectorizedinds = siteinds(ρ.unvectorizednetwork)
    ρ = ρ.network
    
    for key in keys(unvectorizedinds.data_graph.vertex_data)
        @assert length(vectorizedinds[key]) == 1 "vectorized index at site $key is not unique"
        @assert length(unvectorizedinds[key]) == 1 "site $key has wrong number of indices"

        spacename = basespace(vectorizedinds[key][1])
        if !ITensors.hastags(unvectorizedinds[key][1],spacename)
            throw(ArgumentError("The unvectorised index $(unvectorizedinds[key][1]) does not match the vectorised index $(vectorizedinds[key][1])"))
        end
        ρ[key] *= ITensors.delta(ITensors.dag(vectorizedinds[key][1]), vectorizer_output(spacename))
        ρ[key] *= ITensors.dag(vectorizer(spacename))
        ρ[key] *= ITensors.delta(ITensors.dag(unvectorizedinds[key][1]), vectorizer_input(spacename))
        ρ[key] *= ITensors.delta(unvectorizedinds[key][1]', ITensors.dag(vectorizer_input(spacename))')
    end
    return ρ
end


function unvectorize_density_matrix(ρ::VDMNetwork)::AbstractITensorNetwork
    return unvectorize_density_matrix!(deepcopy(ρ))
end



function idnetwork(ψ::ITensorNetwork)::ITensorNetwork
    # Purpose: Creates an ITensorNetwork with identity tensors at each site.
    # Inputs: inds (ITensorNetworks.IndsNetwork) - Site indices.
    # Returns: ITensorNetwork - ITensorNetwork with identity tensors at each site.
    data_graph = ITensorNetworks.ITensorNetwork(v->"0", siteinds(ψ)).data_graph
    sitekeys = keys(siteinds(ψ).data_graph.vertex_data)
    for key in sitekeys
        indices = [ind for ind in inds(data_graph.vertex_data[key]) if !(ind==siteinds(ψ)[key][1])]
        d = ITensors.dim(siteinds(ψ)[key][1])
        dims = Tuple(append!([d, d], [1 for ind in indices]))
        id_array = Array(delta(siteinds(ψ)[key][1], siteinds(ψ)[key][1]').tensor)
        data_graph.vertex_data[key] = ITensor(reshape(id_array, dims), siteinds(ψ)[key][1], siteinds(ψ)[key][1]', indices...)
    end
    return ITensorNetwork(data_graph.vertex_data)
end


function vidnetwork(ψ::ITensorNetwork, vectorizedinds::ITensorNetworks.IndsNetwork)::VDMNetwork
    return vectorize_density_matrix(idnetwork(ψ),ψ, vectorizedinds)
end



function vectorizedtrace(ρ::VDMNetwork; kwargs...)::Complex
    idn = vidnetwork(ρ.unvectorizednetwork, siteinds(ρ.network))
    return ITensorNetworks.inner(ρ.network, idn.network; kwargs...)
end


function vexpect(ρ::VDMNetwork, op::ITensor; kwargs...)::Complex
    idn = idnetwork(ρ.unvectorizednetwork)
    new_network = ITensorNetworks.apply(op, idn; kwargs...)
    new_op = vectorize_density_matrix(new_network, ρ.unvectorizednetwork, siteinds(ρ))
    return ITensorNetworks.inner(ρ.network, new_op.network; kwargs...)
end 

#=
function find_site(ind::ITensors.Index, ψ::ITensorNetwork)::Tuple
    #= Purpose: Finds the site of an index.
    Inputs: ind (ITensorIndex) - Index of an ITensor.
    Returns: Int - Site of the index. =#
    for key in keys(siteinds(ψ).data_graph.vertex_data)
        if ind in siteinds(ψ)[key]
            return key
        end
    end
end


function opdouble(o::ITensor, rho::AbstractITensorNetwork, ψ::ITensorNetwork)::ITensor
    #= Purpose: Turns an ITensor, an operator O on the underlying Hilbert space returns an opertor O†⊗O acting on the doubled Hilbert space.
    Inputs: o (ITensor) - Operator on underlying Hilbert Space.
            rho (ITensorNetwork) - Density Matrix of the system.
    Returns: ITensor - Operator acting on the doubled Hilbert space. =#

    inds = [ind for ind in ITensors.inds(o) if ITensors.plev(ind)==0]
    vs = ITensors.siteinds(rho)
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

function apply(o::ITensors.ITensor, ρ::ITensorNetwork, ψ::ITensorNetwork)::ITensorNetwork
    #= Purpose: Applies an operator to a density matrix.
    Inputs: o (ITensor) - Operator to apply.
            ρ (ITensorNetwork) - Density matrix.
    Returns: ITensorNetwork - New density matrix. =#
    o2 = opdouble(o, ρ, ψ)
    return ITensorNetworks.apply(o2, ρ)
end
=#

end;