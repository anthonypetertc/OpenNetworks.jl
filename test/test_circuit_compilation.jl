using Test
using OpenSystemsTools
using ITensors
using OpenNetworks: VectorizationNetworks, Utils, Channels, GraphUtils, NoisyCircuits, NoiseModels
using NamedGraphs: named_grid
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks
using JSON

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



@testset "Load circuit and add noise" begin
    bell_pair_circuit = JSON.parsefile("example_circuits/bell_pair.json")
    g = GraphUtils.extract_adjacency_graph(bell_pair_circuit, 2)
    sites = ITensorNetworks.siteinds("Qubit", g)
    vsites = ITensorNetworks.siteinds("QubitVec", g)
    ψ = ITensorNetwork(v -> "0", sites);
    ρ = VectorizationNetworks.vectorize_density_matrix(Utils.outer(ψ, ψ),ψ, vsites)
    p = 0.1
    depol_channel = Channels.depolarizing_channel(p, [sites[(0,)][1], sites[(1,)][1]], ρ);
    noise_instruction = NoiseModels.NoiseInstruction("depolarizing", depol_channel, [vsites[(0,)][1], vsites[(1,)][1]], Set(["CX"]), Set([vsites[(0,)][1], vsites[(1,)][1]]));
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), sites, vsites);

    noisy_circuit = NoisyCircuits.add_noise_to_circuit(bell_pair_circuit, noise_model, 2)
    for channel in noisy_circuit
        ρ = Channels.apply(channel, ρ)
    end

    expected_dm = Array([(1-p)/2 + p/4 0 0 (1-p)/2 ; 0 p/4 0 0 ; 0 0 p/4 0 ; (1-p)/2 0 0 (1-p)/2 + p/4])
    reshaped_dm = reshape(permutedims(reshape(expected_dm, (2,2,2,2)), [1, 3, 2, 4]), (4,4))
    @test Array(ITensorNetworks.contract(ρ.network).tensor) ≈ reshaped_dm
end;
