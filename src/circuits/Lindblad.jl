module Lindblad
export lindbladevolve, trotterize

using ITensors
using ITensorsOpenSystems: Vectorization 
using ITensorNetworks: ITensorNetworks, edges, src, dst, vertices
using OpenNetworks:
    Channels,
    NoisyCircuits,
    Utils,
    GraphUtils.linenetwork,
    Channels.Channel


"""
    lindbladevolve(H::Sum, A::Vector{<:Sum}, Δt::Float64, fatsites::Vector{<:Index}, name::String="L")

    Arguments
    H::Sum
        The Hamiltonian of the system.
    A::Vector{<:Sum}
        The Lindblad jump operators.
    Δt::Float64
        The time step for the evolution.
    fatsites::Vector{<:Index}
        The sites on which the Lindblad operator acts.
    name::String="L"
        The name of the channel.

    Returns a Channel representing the Lindblad evolution operator, 
    obtained from Hamiltonian H and jump operators A.

"""

function lindbladevolve(
    H::Sum, A::Vector{<:Sum}, Δt::Float64, fatsites::Vector{<:Index}, name::String="L"
)
    #= Exact Lindblad evolution, from Hamiltonian and Jump operators presented as
    OpSum's =#
    if length(fatsites) > 8
        throw("System is too large! Try Trotterizing the evolution operator.")
    end
    if isempty(H) && isempty(A)
        throw("Attempting to evolve by an empty Lindbladian.")
    elseif isempty(H) && !isempty(A)
        LD = Vectorization.dissipator(A, fatsites)
        L = ITensors.contract(LD)
    elseif isempty(A) && !isempty(H)
        LH = -im * Vectorization.commutator(H, fatsites)
        L = ITensors.contract(LH)
    elseif !isempty(A) && !isempty(H)
        LH = -im * Vectorization.commutator(H, fatsites)
        LD = Vectorization.dissipator(A, fatsites)
        L = ITensors.contract(LH + LD)
    end
    return Channel(name, exp(Δt * L))
end

function lindbladevolvecache!(
    H::Sum,
    A::Vector{<:Sum},
    Δt::Float64,
    sites::Vector{ITensors.Index{V}},
    cache::Dict{Vector{ITensors.Index{V}},Channels.Channel},
    name::String="L",
)::Channels.Channel where {V}
    #= Checks cache to see if there is already a channel on those sites stored in cache, and if so uses it.
        Otherwise, it generates a channel by Lindblad evolution, and stores it in cache.

        Important note: this function should only be used in situations where there is only ever one channel that
        will act on the same individual qubits. =#
    if sites in keys(cache)
        return cache[sites]
    else
        P = lindbladevolve(H, A, Δt, sites, name)
        cache[sites] = P
        return P
    end
end

function convertprodop(prodop::Prod{Op}, e::Dict{V,Int64}) where {V}
    new_gs = Vector{Op}()
    for g in prodop
        new_sites = Vector{Int64}()
        for site in g.sites
            push!(new_sites, e[site])
        end
        push!(new_gs, Op(g.which_op, new_sites..., g.params...))
    end
    return reduce(*, new_gs)
end

function localhamiltonian(
    H::Sum, e::Dict{V,Int64}, size::Union{Int64,Nothing}=nothing
) where {V}
    locH = OpSum()
    for h in H
        acts_on = Set()
        for g in h.args[2]
            for site in g.sites
                push!(acts_on, site)
            end
        end
        if length(acts_on) > 2
            throw("Trotterized evolution only implemented for two-site Hamiltonians.")
        end
        if issubset(acts_on, keys(e)) && (isnothing(size) || length(acts_on) == size)
            prodop = convertprodop(h.args[2], e)
            locH += h.args[1] * prodop
        end
    end
    return locH
end

function localjumps(
    A::Vector{<:Sum}, e::Dict{V,Int64}, size::Union{Int64,Nothing}=nothing
) where {V}
    locA = fill(OpSum(), length(A))
    for (i, jump) in enumerate(A)
        for a in jump
            acts_on = Set()
            for g in a.args[2]
                for site in g.sites
                    push!(acts_on, site)
                end
            end
            if length(acts_on) > 2
                throw("Trotterized evolution only implemented for two-site jump operators.")
            end
            if issubset(acts_on, keys(e)) && (isnothing(size) || length(acts_on) == size)
                prodop = convertprodop(a.args[2], e)
                locA[i] += a.args[1] * prodop
            end
        end
    end
    for i in length(locA):-1:1
        if locA[i] == OpSum()
            deleteat!(locA, i)
        end
    end
    return locA
end

function firstordertrotter(
    H::Sum,
    A::Vector{<:Sum},
    steps::Int64,
    Δt::Float64,
    sites::ITensorNetworks.IndsNetwork{V},
)::NoisyCircuits.NoisyCircuit where {V}
    channel_list = Vector{Channels.Channel}()
    Q = Utils.findindextype(sites)
    cache = Dict{Vector{Q},Channels.Channel}()
    for _ in 1:steps
        for v in vertices(sites)
            ev = Dict{V,Int64}(v => 1)
            locH = localhamiltonian(H, ev, 1)
            locA = localjumps(A, ev, 1)
            push!(
                channel_list, lindbladevolvecache!(locH, locA, Δt, [first(sites[v])], cache)
            )
        end
        for e in edges(sites)
            esites = Dict{V,Int64}(src(e) => 1, dst(e) => 2)
            locH = localhamiltonian(H, esites, 2)
            locA = localjumps(A, esites, 2)
            push!(
                channel_list,
                lindbladevolvecache!(
                    locH, locA, Δt, [first(sites[src(e)]), first(sites[dst(e)])], cache
                ),
            )
        end
    end
    return NoisyCircuits.NoisyCircuit(channel_list, sites)
end

function secondordertrotter(
    H::Sum,
    A::Vector{<:Sum},
    steps::Int64,
    Δt::Float64,
    sites::ITensorNetworks.IndsNetwork{V},
)::NoisyCircuits.NoisyCircuit where {V}
    channel_list = Vector{Channels.Channel}()
    Q = Utils.findindextype(sites)
    cache = Dict{Vector{Q},Channels.Channel}()
    for i in 1:steps
        for v in vertices(sites)
            ev = Dict{V,Int64}(v => 1)
            locH = localhamiltonian(H, ev, 1)
            locA = localjumps(A, ev, 1)
            push!(
                channel_list,
                lindbladevolvecache!(locH, locA, 0.5 * Δt, [first(sites[v])], cache),
            )
        end
        for e in edges(sites)
            esites = Dict{V,Int64}(src(e) => 1, dst(e) => 2)
            locH = localhamiltonian(H, esites, 2)
            locA = localjumps(A, esites, 2)
            push!(
                channel_list,
                lindbladevolvecache!(
                    locH,
                    locA,
                    0.5 * Δt,
                    [first(sites[src(e)]), first(sites[dst(e)])],
                    cache,
                ),
            )
        end
        for e in reverse(edges(sites))
            esites = Dict{V,Int64}(src(e) => 1, dst(e) => 2)
            locH = localhamiltonian(H, esites, 2)
            locA = localjumps(A, esites, 2)
            push!(
                channel_list,
                lindbladevolvecache!(
                    locH,
                    locA,
                    0.5 * Δt,
                    [first(sites[src(e)]), first(sites[dst(e)])],
                    cache,
                ),
            )
        end
        for v in reverse(vertices(sites))
            ev = Dict{V,Int64}(v => 1)
            locH = localhamiltonian(H, ev, 1)
            locA = localjumps(A, ev, 1)
            push!(
                channel_list,
                lindbladevolvecache!(locH, locA, 0.5 * Δt, [first(sites[v])], cache),
            )
        end
    end
    return NoisyCircuits.NoisyCircuit(channel_list, sites)
end

"""
    trotterize(H::Sum, A::Vector{<:Sum}, steps::Int64, Δt::Float64, sites::ITensorNetworks.IndsNetwork{V}; order::Int64=2)

    Arguments
    H::Sum
        The Hamiltonian of the system.
    A::Vector{<:Sum}
        The Lindblad jump operators.
    steps::Int64
        The number of Trotter steps.
    Δt::Float64
        The time step for the evolution.
    sites::ITensorNetworks.IndsNetwork{V} OR Vector{<:ITensors.Index{}}
        The sites on which the Lindblad operator acts.
    order::Int64=2
        The order of the Trotter decomposition (1 or 2).

    Returns a NoisyCircuit representing the Trotterized evolution operator, 
    obtained from Hamiltonian H and jump operators A, acting on the sites specified.
    If the sites is an IndsNetwork, the underlying graph can have arbitrary connectivity,
    if it is a vector of indices, the underlying graph must be a chain.

"""


function trotterize(
    H::Sum,
    A::Vector{<:Sum},
    steps::Int64,
    Δt::Float64,
    sites::ITensorNetworks.IndsNetwork{V};
    order::Int64=2,
)::NoisyCircuits.NoisyCircuit where {V}
    if order == 1
        return firstordertrotter(H, A, steps, Δt, sites)
    elseif order == 2
        return secondordertrotter(H, A, steps, Δt, sites)
    else
        throw("Higher order Suzuki-Trotter not implemented! order must be 2 or 1.")
    end
end

function trotterize(
    H::Sum,
    A::Vector{<:Sum},
    steps::Int64,
    Δt::Float64,
    sites::Vector{<:ITensors.Index{}};
    order::Int64=2,
)::NoisyCircuits.NoisyCircuit
    return trotterize(H, A, steps, Δt, linenetwork(sites); order=order)
end

end #module
