using Test
using ITensorsOpenSystems
using ITensors
using OpenNetworks:
    VectorizationNetworks,
    Utils,
    Channels,
    GraphUtils,
    NoisyCircuits,
    NoiseModels,
    CustomParsing,
    VDMNetworks
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks

N = 12
G = GraphUtils.named_ring_graph(N)

sites = ITensorNetworks.siteinds("Qubit", G)
vsites = ITensorNetworks.siteinds("QubitVec", G)
ψ = ITensorNetwork(v -> "0", sites);
ρ = VDMNetworks.VDMNetwork(Utils.outer(ψ, ψ), sites, vsites)

@testset "Prepare Parameters" begin
    @test NoisyCircuits.prepare_params([π / 2], "U") == Dict(:θ => π / 2)
    @test NoisyCircuits.prepare_params([π / 2, π / 4], "U") ==
        Dict(:θ => π / 2, :ϕ => π / 4)
    @test NoisyCircuits.prepare_params([π / 2, π / 4, π / 8], "U") ==
        Dict(:θ => π / 2, :ϕ => π / 4, :λ => π / 8)
    @test NoisyCircuits.prepare_params([π / 4], "Rzz") == Dict(:ϕ => π / 8)
    @test_throws "Incorrect number of params for gate Rzz." NoisyCircuits.prepare_params(
        [π / 2, π / 4], "Rzz"
    )
    @test_throws "Only 3 parameters or less." NoisyCircuits.prepare_params(
        [π / 2, π / 4, π / 8, π / 16], "U"
    )
end;

@testset "Prepare noise for gate" begin
    p = 0.1
    depol_channel = Channels.depolarizing_channel(p, [sites[0][1], sites[1][1]], ρ)
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        [vsites[0][1], vsites[1][1]],
        Set(["CX"]),
        Set([vsites[i][1] for i in 0:7]),
    )
    noise_tensor = NoiseModels.prepare_noise_for_gate(
        noise_instruction, [vsites[4][1], vsites[5][1]]
    )
    @test Set(inds(noise_tensor.tensor)) == Set{ITensors.Index{Int64}}([
        vsites[4][1]', vsites[5][1]', vsites[4][1], vsites[5][1]
    ])
    @test_throws "Noise instruction does not apply to this qubit." NoiseModels.prepare_noise_for_gate(
        noise_instruction, [vsites[0][1], vsites[11][1]]
    )
end;

@testset "Make Gate" begin
    @testset "Single Qubit Gate" begin
        tensor = NoisyCircuits.make_gate("X", [1], Dict{Symbol,Float64}(), sites)
        @test tensor == op("X", sites[1])
    end

    @testset "Two Qubit Gate" begin
        tensor = NoisyCircuits.make_gate("CX", [1, 2], Dict{Symbol,Float64}(), sites)
        @test tensor == op("CX", sites[1][1], sites[2][1])
    end

    @testset "Three Qubit Gate" begin
        tensor = NoisyCircuits.make_gate("CCX", [1, 2, 3], Dict{Symbol,Float64}(), sites)
        @test tensor == op("CCX", sites[1][1], sites[2][1], sites[3][1])
    end

    @testset "Gate with Parameters" begin
        tensor = NoisyCircuits.make_gate("Rx", [1], Dict(:θ => π / 2), sites)
        @test tensor == op("Rx", sites[1]; θ=π / 2)
    end
    @testset "Five Qubit Gate" begin
        @test_throws "Only 3 qubit gates or less." tensor = NoisyCircuits.make_gate(
            "CCCCX", [1, 2, 3, 4, 5], Dict{Symbol,Float64}(), sites
        )
    end
end;

bell_pair_circuit = CustomParsing.parse_circuit("example_circuits/bell_pair.json")
bell_g = GraphUtils.extract_adjacency_graph(bell_pair_circuit)
bell_sites = ITensorNetworks.siteinds("Qubit", bell_g)
bell_vsites = ITensorNetworks.siteinds("QubitVec", bell_g)
bell_ψ = ITensorNetwork(v -> "0", bell_sites);
bell_ρ = VDMNetworks.VDMNetwork(Utils.outer(bell_ψ, bell_ψ), bell_sites, bell_vsites)

@testset "Add noise to Bell pair." begin
    ρ = bell_ρ
    p = 0.1
    depol_channel = Channels.depolarizing_channel(
        p, [bell_sites[0][1], bell_sites[1][1]], bell_ρ
    )
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        [bell_vsites[0][1], bell_vsites[1][1]],
        Set(["CX"]),
        Set([bell_vsites[0][1], bell_vsites[1][1]]),
    )
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), bell_sites, bell_vsites)
    noisy_circuit = NoisyCircuits.add_noise_to_circuit(bell_pair_circuit, noise_model)

    for (i, channel) in enumerate(noisy_circuit)
        ρ = Channels.apply(channel, ρ)
    end

    expected_dm = Array(
        [
            (1 - p) / 2+p / 4 0 0 (1 - p)/2
            0 p/4 0 0
            0 0 p/4 0
            (1 - p)/2 0 0 (1 - p) / 2+p / 4
        ]
    )
    reshaped_dm = reshape(
        permutedims(reshape(expected_dm, (2, 2, 2, 2)), [1, 3, 2, 4]), (4, 4)
    )
    @test Array(ITensorNetworks.contract(ρ.network).tensor) ≈ reshaped_dm
end;

ring_circuit = CustomParsing.parse_circuit("example_circuits/circ.json")
ring_g = GraphUtils.extract_adjacency_graph(ring_circuit)
ring_sites = ITensorNetworks.siteinds("Qubit", ring_g)
ring_vsites = ITensorNetworks.siteinds("QubitVec", ring_g)
ring_ψ = ITensorNetwork(v -> "0", ring_sites);
ring_ρ = VDMNetworks.VDMNetwork(Utils.outer(ring_ψ, ring_ψ), ring_sites, ring_vsites)

@testset "Tests on ring circuit" begin
    sites = ring_sites
    vsites = ring_vsites
    ψ = ring_ψ
    ρ = ring_ρ
    circ = ring_circuit

    p = 0.1
    depol_channel = Channels.depolarizing_channel(p, [sites[0][1], sites[1][1]], ρ)
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        [vsites[0][1], vsites[1][1]],
        Set(["CX"]),
        Set([vsites[i][1] for i in 0:11]),
    )
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), sites, vsites)
    noisy_circuit = NoisyCircuits.add_noise_to_circuit(circ, noise_model)
    compressed_noisy_circuit = NoisyCircuits.absorb_single_qubit_gates(noisy_circuit)
    #=   compiled_noisy_circuit, n_gates = NoisyCircuits.compile_into_moments(
           compressed_noisy_circuit, vsites
       ) =#
    circuit_object = NoisyCircuits.NoisyCircuit(circ, noise_model)

    @testset "Noise applied to correct qubits." begin
        for gate in noisy_circuit
            if length(inds(gate.tensor)) == 4
                @test gate.name == "depolarizing∘CX"
            end
        end
    end

    @testset "Compression of noisy circuit." begin
        @test length(compressed_noisy_circuit) == 12
        for gate in compressed_noisy_circuit
            @test length(inds(gate.tensor)) == 4
        end
    end
    #=
        @testset "Compilation into moments" begin
            @test length(compiled_noisy_circuit) == 12
            for (i, moment) in enumerate(compiled_noisy_circuit)
                @test length(moment) == 1
                @test moment[1].tensor == compressed_noisy_circuit[i].tensor
                @test moment[1].name == compressed_noisy_circuit[i].name
            end
        end
    =#
    @testset "Noisy Circuit object" begin
        @test circuit_object.fatsites == noise_model.vectorizedsiteinds
        for (j, gate) in enumerate(circuit_object.channel_list)
            @test gate.tensor == compressed_noisy_circuit[j].tensor
            @test gate.name == compressed_noisy_circuit[j].name
        end
    end
end;

#=
@testset "Compilation into moments, small circuit." begin
    circuit = CustomParsing.parse_circuit("example_circuits/test_compile_circuit.json")

    g = GraphUtils.extract_adjacency_graph(circuit)
    s = ITensorNetworks.siteinds("Qubit", g)
    vs = ITensorNetworks.siteinds("QubitVec", g)

    ψ = ITensorNetwork(v -> "0", s)
    ρ = VDMNetworks.VDMNetwork(Utils.outer(ψ, ψ), s, vs)

    p = 0.05
    depol_channel = Channels.depolarizing_channel(p, [s[0][1], s[1][1]], ρ)
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        [vs[0][1], vs[1][1]],
        Set(["Rzz"]),
        Set([vs[i][1] for i in 0:1]),
    )
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), s, vs)

    noisy_circuit = NoisyCircuits.add_noise_to_circuit(circuit, noise_model)
    compressed_circuit = NoisyCircuits.absorb_single_qubit_gates(noisy_circuit)
    moments_list1, _ = NoisyCircuits.compile_into_moments(
        compressed_circuit, noise_model.vectorizedsiteinds
    )

    @test [length(moment) for moment in moments_list1] == [2, 3, 3, 4, 3, 3]

    # I have checked by hand that this circuit should have moments of length (2, 3, 3, 4, 3, 3)
end;
=#
