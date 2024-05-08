using Test
using OpenSystemsTools
using ITensors
using OpenNetworks: VectorizationNetworks, Utils, Channels, GraphUtils, NoisyCircuits
using NamedGraphs: named_grid
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks


N = 12
G = GraphUtils.named_ring_graph(N)

sites = ITensorNetworks.siteinds("Qubit", G)
vsites = ITensorNetworks.siteinds("QubitVec", G)
#I should re-write some of my functions so that they don't require a reference state, only the site inds.
ψ = ITensorNetwork(v -> "0", sites);
ρ = VectorizationNetworks.vectorize_density_matrix(Utils.outer(ψ, ψ),ψ, vsites)

@testset "Make Gate" begin
    @testset "Single Qubit Gate" begin
        tensor = NoisyCircuits.make_gate("X", [1], Dict(), sites)
        @test tensor == op("X", sites[(1,)])
    end

    @testset "Two Qubit Gate" begin
        tensor = NoisyCircuits.make_gate("CX", [1, 2], Dict(), sites)
        @test tensor == op("CX", sites[(1,)][1], sites[(2,)][1])
    end

    @testset "Three Qubit Gate" begin
        tensor = NoisyCircuits.make_gate("CCX", [1, 2, 3], Dict(), sites)
        @test tensor == op("CCX", sites[(1,)][1], sites[(2,)][1], sites[(3,)][1])
    end

    @testset "Gate with Parameters" begin
        tensor = NoisyCircuits.make_gate("Rx", [1], Dict(:θ => π/2), sites)
        @test tensor == op("Rx", sites[(1,)]; θ = π/2)
    end

    @testset "Five Qubit Gate" begin
        @test_throws "Only 3 qubit gates or less." tensor = NoisyCircuits.make_gate("CCCCX", [1, 2, 3, 4, 5], Dict(), sites)
    end
end;


