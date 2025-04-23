module Utils
export swapprime, swapprime!, innerprod, trace

using ITensors
using ITensorNetworks:
    AbstractITensorNetwork,
    siteinds,
    ⊗,
    prime,
    dag,
    contract,
    ITensorNetwork,
    ITensorNetworks,
    edges
import ITensors: swapprime, swapprime!, outer, siteinds
using OpenNetworks: VDMNetworks

VDMNetwork = VDMNetworks.VDMNetwork

function findindextype(i::ITensors.Index)::Type
    return typeof(i)
end

function findindextype(fatsites::ITensorNetworks.IndsNetwork)::Type
    firstind = first(fatsites[first(ITensorNetworks.vertices(fatsites))])
    return findindextype(firstind)
end

"""
    swapprime!(ψ::AbstractITensorNetwork, pl1::Int, pl2::Int; kwargs...)

    Arguments:
    ψ::AbstractITensorNetwork
        The ITensorNetwork to swap the prime level of.
    pl1::Int
        The first prime level to swap.
    pl2::Int
        The second prime level to swap.
    kwargs...
        Additional keyword arguments to pass to the swapprime function.

    Swaps the prime level of all indices of all ITensors in 
    the ITensorNetwork ψ from pl1 to pl2.

"""

function swapprime!(
    ψ::AbstractITensorNetwork, pl1::Int, pl2::Int; kwargs...
)::AbstractITensorNetwork
    vd = ψ.data_graph.vertex_data
    for key in keys(vd)
        vd[key] = swapprime(vd[key], pl1, pl2; kwargs...)
    end
    return ψ
end

"""
    swapprime(ψ::AbstractITensorNetwork, pl1::Int, pl2::Int; kwargs...)

    Arguments:
    ψ::AbstractITensorNetwork
        The ITensorNetwork to swap the prime level of.
    pl1::Int
        The first prime level to swap.
    pl2::Int
        The second prime level to swap.
    kwargs...
        Additional keyword arguments to pass to the swapprime function.

    Swaps the prime level of all indices of all ITensors in 
    the ITensorNetwork ψ from pl1 to pl2.

"""

function swapprime(
    ψ::AbstractITensorNetwork, pl1::Int, pl2::Int; kwargs...
)::AbstractITensorNetwork
    return swapprime!(deepcopy(ψ), pl1, pl2; kwargs...)
end

"""
    swapprime!(ρ::VDMNetwork, pl1::Int, pl2::Int; kwargs...)

    Arguments:
    ρ::VDMNetwork
        The VDMNetwork to swap the prime level of.
    pl1::Int
        The first prime level to swap.
    pl2::Int
        The second prime level to swap.
    kwargs...
        Additional keyword arguments to pass to the swapprime function.

    Swaps the prime level of all indices of all ITensors in 
    the VDMNetwork ρ from pl1 to pl2.

"""

function swapprime!(ρ::VDMNetwork, pl1::Int, pl2::Int; kwargs...)::VDMNetwork
    swapprime!(ρ.network, pl1, pl2; kwargs...)
    return ρ
end

"""
    swapprime(ρ::VDMNetwork, pl1::Int, pl2::Int; kwargs...)

    Arguments:
    ρ::VDMNetwork
        The VDMNetwork to swap the prime level of.
    pl1::Int
        The first prime level to swap.
    pl2::Int
        The second prime level to swap.
    kwargs...
        Additional keyword arguments to pass to the swapprime function.

    Swaps the prime level of all indices of all ITensors in 
    the VDMNetwork ρ from pl1 to pl2.

"""


function swapprime(ρ::VDMNetwork, pl1::Int, pl2::Int; kwargs...)::VDMNetwork
    return swapprime!(deepcopy(ρ), pl1, pl2; kwargs...)
end


function innerprod(ρ::VDMNetwork, ϕ::VDMNetwork)::Complex
    return ITensorNetworks.inner(ρ.network, ϕ.network; alg="exact")
end

function siteinds(ρ::VDMNetwork)::ITensorNetworks.IndsNetwork
    return siteinds(ρ.network)
end

function compress_underlying_graph(
    o::AbstractITensorNetwork, ψ::AbstractITensorNetwork
)::AbstractITensorNetwork
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
        connecting_bonds = [
            bond for bond in intersect(inds(o1), inds(o2)) if plev(bond) == 0
        ]
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

function ITensors.outer(
    ψ::AbstractITensorNetwork, ϕ::AbstractITensorNetwork
)::AbstractITensorNetwork
    if !(ψ.data_graph.underlying_graph == ϕ.data_graph.underlying_graph)
        throw("The two ITensorNetworks must have the same underlying graph.")
    end
    ψϕ = ψ ⊗ dag(ϕ)
    return merge_bond_legs(compress_underlying_graph(ψϕ, ψ))
end

"""
    trace(ρ::AbstractITensorNetwork)::Complex

    Arguments:
    ρ::AbstractITensorNetwork
        The ITensorNetwork to compute the trace of.

    Computes the trace of the ITensorNetwork ρ by contracting input and output legs
    of every tensor.

"""

function trace(ρ::AbstractITensorNetwork)::Complex
    ρ = deepcopy(ρ)
    for key in keys(ρ.data_graph.vertex_data)
        s = [ind for ind in inds(ρ[key]) if hastags(ind, "Site") && plev(ind) == 0]
        sp = inds(ρ[key]; tags="Site", plev=1)
        if !(length(s) == 1) || !(length(sp) == 1)
            throw("Should only be one physical leg per site.")
        end
        ρ[key] *= delta(s[1], prime(s[1]))
    end
    return Array(contract(ρ).tensor)[1]
end


#=
function relabel!(ψ::AbstractITensorNetwork, ind_net::ITensorNetworks.IndsNetwork)
    #= Purpose: Relabels the sites of an ITensorNetwork.
    Inputs: ψ (AbstractITensorNetwork) - ITensorNetwork to relabel.
            f (Function) - Function to relabel the sites.
    Returns: AbstractITensorNetwork - Relabeled ITensorNetwork. =#
    throw("Not implemented yet.")
    for key in keys(ψ.data_graph.vertex_data)
        dd = ITensors.delta(siteinds(ψ)[key][1], ind_net[key][1])
        ψ[key] *= dd
    end
end
=#

end;
