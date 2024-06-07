using Test
using OpenSystemsTools
using ITensors
using OpenNetworks: VDMNetworks, Utils, Channels, VectorizationNetworks
using NamedGraphs: named_grid
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks

#=
depolarizing_channel = Channels.depolarizing_channel
opdouble = Channels.opdouble
apply = Channels.apply
Channel = Channels.Channel
find_site = Channels.find_site

vectorize_density_matrix = VectorizationNetworks.vectorize_density_matrix
#opdouble = VectorizationBP.opdouble
swapprime = Utils.swapprime
=#
apply = Channels.apply
swapprime! = Utils.swapprime!
swapprime = Utils.swapprime
vectorize_density_matrix = VectorizationNetworks.vectorize_density_matrix

g_dims = (2, 2)
g = named_grid(g_dims)
sites = siteinds("Qubit", g)
vsites = siteinds("QubitVec", g)
χ = 4
#Random.seed!(1564)
ψ = ITensorNetworks.random_tensornetwork(sites; link_space=χ)
ρ = Utils.outer(ψ, ψ)
vρ = vectorize_density_matrix(ρ, ψ, vsites)

@testset "test swapprime" begin
    @testset "swapprime!" begin
        swapprime!(ψ, 0, 1)
        for ind in inds(ψ[(1, 1)])
            @test plev(ind) == 1
        end
    end

    @testset "swapprime" begin
        ϕ = swapprime(ψ, 1, 2)
        for ind in inds(ϕ[(1, 1)])
            @test plev(ind) == 2
        end
    end

    @testset "swapprime! VDMNetwork" begin
        swapprime!(vρ, 0, 1)
        for ind in inds(vρ.network[(1, 1)])
            @test plev(ind) == 1
        end
    end

    @testset "swapprime VDMNetwork" begin
        ϕ = swapprime(vρ, 1, 2)
        for ind in inds(ϕ.network[(1, 1)])
            @test plev(ind) == 2
        end
    end
end;

#Prepare state in all zero state.
ψ = ITensorNetwork(v -> "0", sites);
ρ = vectorize_density_matrix(Utils.outer(ψ, ψ), ψ, vsites)
#Apply X gate to first qubit.
o = op("X", sites[(1, 1)])
ϕ = ITensorNetworks.apply(o, ψ)
σ = vectorize_density_matrix(Utils.outer(ϕ, ϕ), ϕ, vsites)

g2 = named_grid((3, 3))
ψ2 = ITensorNetwork(v -> "0", siteinds("Qubit", g2))

@testset "innerprod" begin
    @test Utils.innerprod(ψ, ψ) ≈ 1.0
    @test Utils.innerprod(ψ, ϕ) ≈ 0.0
    @test Utils.innerprod(ρ, ρ) ≈ 1.0
    @test Utils.innerprod(ρ, σ) ≈ 0.0
end;

@testset "siteinds" begin
    @test siteinds(ρ).data_graph.vertex_data == vsites.data_graph.vertex_data
end;

@testset "outer" begin
    o = Utils.outer(ψ, ϕ)
    @test Utils.innerprod(o, o) ≈ 1.0
    @test Utils.innerprod(o, swapprime(o, 0, 1)) ≈ 0.0
    for tens in o.data_graph.vertex_data
        @test length([ind for ind in inds(tens) if !hastags(ind, "Qubit")]) == 2
    end
    @test_throws "The two ITensorNetworks must have the same underlying graph." Utils.outer(
        ψ, ψ2
    )
    ϕ1 = ITensorNetworks.random_tensornetwork(sites; link_space=χ)
    ϕ2 = ITensorNetworks.random_tensornetwork(sites; link_space=χ)
    @test_throws "The two ITensorNetworks must have the bond indices labelled in the same way." Utils.outer(
        ϕ1, ϕ2
    )
end;

ψ = ITensorNetwork(v -> "0", sites)
ρ = vectorize_density_matrix(Utils.outer(ψ, ψ), ψ, vsites)

@testset "trace_unitary" begin
    unvectorized = deepcopy(ψ)
    evolved = deepcopy(ρ)
    for i in 1:6
        vertex = rand(keys(unvectorized.data_graph.vertex_data))
        qubit1 = sites[vertex]
        qubit2 = sites[rand(
            Graphs.neighbors(unvectorized.data_graph.underlying_graph, vertex)
        )]
        qubits = [qubit1, qubit2]
        append!(qubits, qubits')

        Q, _ = qr(randn(ComplexF64, 4, 4))
        U = ITensor(Array(Q), qubits)
        evolved = Channels.apply(U, evolved)
        unvectorized = ITensorNetworks.apply(U, unvectorized)
        dm = outer(unvectorized, unvectorized)
        @test Utils.innerprod(evolved, evolved) ≈ 1.0
    end
end;
