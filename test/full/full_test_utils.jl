using Test
using ITensorsOpenSystems
using ITensors
using OpenNetworks: VDMNetworks, Utils, Channels, VectorizationNetworks
using NamedGraphs: NamedGraphGenerators.named_grid
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks

apply = Channels.apply
swapprime! = Utils.swapprime!
swapprime = Utils.swapprime

g_dims = square_g_dims
g = square_g
sites = square_sites
vsites = square_vsites
χ = 4
#Random.seed!(1564)
ψ = square_rand_ψ
ρ = square_rand_ρ
vρ = square_rand_vρ

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

@testset "trace" begin
   @test Utils.trace(vρ) == Utils.trace(ρ) 
end

#Prepare state in all zero state.
ψ = ITensorNetwork(v -> "0", sites);
ρ = VDMNetworks.VDMNetwork(outer(ψ', ψ), sites, vsites)
#Apply X gate to first qubit.
o = op("X", sites[(1, 1)])
ϕ = ITensorNetworks.apply(o, ψ)
σ = VDMNetworks.VDMNetwork(outer(ϕ', ϕ), sites, vsites)

g2 = named_grid((3, 3))
ψ2 = ITensorNetwork(v -> "0", siteinds("Qubit", g2))

@testset "siteinds" begin
    @test siteinds(ρ).data_graph.vertex_data == vsites.data_graph.vertex_data
end;

@testset "outer" begin
    o = outer(ψ', ϕ)
    @test first(contract(o ⊗ dag(o))) ≈ 1.0
    o2 = swapprime(o, 1, 0)
    @test first(contract(o ⊗ dag(o2))) ≈ 0.0
    #@test Utils.trace(Utils.swapprime(o ⊗ dag(Utils.swapprime(o, 0, 1)'), 2, 1)) ≈ 0.0
    for tens in o.data_graph.vertex_data
        @test length([ind for ind in inds(tens) if !hastags(ind, "Qubit")]) == 2
    end
    @test_throws "The two ITensorNetworks must have the same underlying graph." outer(
        ψ', ψ2
    )
end;

ψ = ITensorNetwork(v -> "0", sites)
ρ = VDMNetworks.VDMNetwork(outer(ψ', ψ), sites, vsites)

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
        dm = outer(unvectorized', unvectorized)
        @test Utils.innerprod(evolved, evolved) ≈ 1.0
    end
end;


