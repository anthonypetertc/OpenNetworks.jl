module Utils
export swapprime, swapprime!, innerprod, outer, trace

using ITensors
using ITensorNetworks: AbstractITensorNetwork, siteinds, ⊗, prime, dag, contract, ITensorNetwork, ITensorNetworks, edges
import ITensors: swapprime, swapprime!, outer, siteinds
using OpenNetworks: VDMNetworks

VDMNetwork = VDMNetworks.VDMNetwork

function swapprime!(ψ::AbstractITensorNetwork, pl1::Int, pl2::Int; kwargs...)::AbstractITensorNetwork
    #= Purpose: Swaps the prime level of the ITensorNetwork.
    Inputs: ψ (AbstractITensorNetwork) - ITensorNetwork to swap the prime level of.
    Returns: AbstractITensorNetwork - ITensorNetwork with the prime level swapped. =#
    vd = ψ.data_graph.vertex_data
    for key in keys(vd)
        vd[key] = swapprime(vd[key], pl1, pl2; kwargs...)
    end
    return ψ
end

function swapprime(ψ::AbstractITensorNetwork, pl1::Int, pl2::Int; kwargs...)::AbstractITensorNetwork
    return swapprime!(deepcopy(ψ), pl1, pl2; kwargs...)
end

function innerprod(ψ::AbstractITensorNetwork, ϕ::AbstractITensorNetwork)::Complex
    #= Purpose: Computes the inner product of two ITensorNetworks.
    Inputs: ψ (AbstractITensorNetwork) - First ITensorNetwork.
            ϕ (AbstractITensorNetwork) - Second ITensorNetwork.
    Returns: Complex - Inner product of the two ITensorNetworks. =#
    @assert ψ.data_graph.underlying_graph.vertices == ϕ.data_graph.underlying_graph.vertices "The two ITensorNetworks must have the same underlying graph."
    return Array(contract(ψ ⊗ dag(ϕ)).tensor)[1]
end


function innerprod(ρ::VDMNetwork, ϕ::VDMNetwork)::Complex
    return innerprod(ρ.network, ϕ.network)
end

function siteinds(ρ::VDMNetwork)::ITensorNetworks.IndsNetwork
    return siteinds(ρ.network)
end



function compress_underlying_graph(o::AbstractITensorNetwork, ψ::AbstractITensorNetwork)::AbstractITensorNetwork
    # Might want to merge the bond legs so that I don't end up doubling them up.
    # I can do this later.
    vd = copy(ψ.data_graph.vertex_data)
    for key in keys(vd)
        vd[key] = o[(key, 1)] * o[(key, 2)]
    end
    return ITensorNetwork(vd)
end


function merge_bond_legs(ψ::AbstractITensorNetwork)::AbstractITensorNetwork
    dg = ψ.data_graph
    for edge in edges(ψ)
        o1 = dg.vertex_data[edge.src]
        o2 = dg.vertex_data[edge.dst]
        connecting_bonds = [bond for bond in intersect(inds(o1), inds(o2)) if plev(bond) == 0]
        if length(connecting_bonds) > 0
            for bond in connecting_bonds
                C = combiner(bond, bond'; tags=tags(bond))
                o1 *= C
                o2 *= C
            end
        end
        dg.vertex_data[edge.src] = o1
        dg.vertex_data[edge.dst] = o2
    end
    return ITensorNetworks._ITensorNetwork(dg)
end
        


function outer(ψ::AbstractITensorNetwork, ϕ::AbstractITensorNetwork)::AbstractITensorNetwork
    #= Purpose: Computes the outer product of two ITensorNetworks.
    Inputs: ψ (AbstractITensorNetwork) - First ITensorNetwork.
            ϕ (AbstractITensorNetwork) - Second ITensorNetwork.
    Returns: ITensorNetwork - Outer product of the two ITensorNetworks. =#
    @assert ψ.data_graph.underlying_graph == ϕ.data_graph.underlying_graph "The two ITensorNetworks must have the same underlying graph."
    ψϕ =  ψ ⊗ prime(dag(ϕ))
    return merge_bond_legs(compress_underlying_graph(ψϕ, ψ))
end

function trace(ρ::AbstractITensorNetwork)::Complex
    #= Purpose: Computes the trace of an ITensorNetwork.
    Inputs: ρ (AbstractITensorNetwork) - ITensorNetwork to compute the trace of.
    Returns: Complex - Trace of the ITensorNetwork. =#
    ρ = deepcopy(ρ)
    for key in keys(ρ.data_graph.vertex_data)
        s = [ind for ind in inds(ρ[key]) if hastags(ind, "Site") && plev(ind) == 0]
        @assert length(s) == 1 "Should only be one physical leg per site."
        ρ[key] *= delta(s[1], prime(s[1]))
    end
    return Array(contract(ρ).tensor)[1]
end

function relabel!(ψ::AbstractITensorNetwork, ind_net::ITensorNetworks.IndsNetwork)
    #= Purpose: Relabels the sites of an ITensorNetwork.
    Inputs: ψ (AbstractITensorNetwork) - ITensorNetwork to relabel.
            f (Function) - Function to relabel the sites.
    Returns: AbstractITensorNetwork - Relabeled ITensorNetwork. =#
    for key in keys(ψ.data_graph.vertex_data)
        println(siteinds(ψ)[key])
        println(ind_net[key])
        dd = ITensors.delta(siteinds(ψ)[key][1], ind_net[key][1])
        ψ[key] *= dd
        println(ψ[key])
    end
end

end;