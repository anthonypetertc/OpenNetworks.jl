module Lindblad
export lindbladevolve, trotterizedlindblad
using ITensors
using ITensorMPS
using ITensorsOpenSystems: ITensorsOpenSystems, Vectorization
using ITensorNetworks: ITensorNetworks, edges, src, dst, vertices
using NamedGraphs
using OpenNetworks: OpenNetworks, Channels, VectorizationNetworks, NoisyCircuits
Channel = Channels.Channel
VectorizedDensityMatrix = Vectorization.VectorizedDensityMatrix
named_grid = NamedGraphs.NamedGraphGenerators.named_grid

function lindbladevolve(
    H::Sum, A::Vector{<:Sum}, Δt::Float64, fatsites::Vector{<:Index}, name::String="L"
)
    #=if length(fatsites) > 2
        throw("Two site or one site evolution, only!")
    end=#
    LH = -im * Vectorization.commutator(H, fatsites)
    LD = Vectorization.dissipator(A, fatsites)
    L = ITensors.contract(LH + LD)
    return Channel(name, exp(Δt * L))
end

function convertprodop(prodop::Prod{Op}, e::Dict{<:Any,Int64})
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

function localhamiltonian(H::Sum, e::Dict{<:Any,Int64})
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
        if issubset(acts_on, keys(e))
            prodop = convertprodop(h.args[2], e)
            locH += h.args[1] * prodop
        end
    end
    return locH
end

function localjumps(A::Vector{<:Sum}, e::Dict{<:Any,Int64})
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
            if issubset(acts_on, keys(e))
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

function trotterizedlindblad(
    H::Sum, A::Vector{<:Sum}, steps::Int64, Δt::Float64, sites::ITensorNetworks.IndsNetwork
)
    channel_list = Vector{Channels.Channel}()
    for _ in steps
        for e in edges(sites)
            esites = Dict(src(e) => 1, dst(e) => 2)
            locH = localhamiltonian(H, esites)
            locA = localjumps(A, esites)
            push!(
                channel_list,
                lindbladevolve(
                    locH, locA, Δt, [first(sites[src(e)]), first(sites[dst(e)])]
                ),
            )
        end
    end
    return NoisyCircuits.NoisyCircuit(channel_list, sites)
end

end #module
