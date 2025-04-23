using Test
using OpenNetworks:
    VectorizationNetworks,
    Utils,
    Channels,
    GraphUtils,
    NoisyCircuits,
    NoiseModels,
    Circuits,
    CustomParsing,
    VDMNetworks,
    PreBuiltChannels,
    Evolution
using ITensorNetworks

N = 12
qc = CustomParsing.parse_circuit("example_circuits/circ_inverse.json")

g = GraphUtils.extract_adjacency_graph(qc)
sites = ITensorNetworks.siteinds("Qubit", g)
vsites = ITensorNetworks.siteinds("QubitVec", g)
ψ = ITensorNetwork(v -> "0", sites);
ρ = VDMNetworks.VDMNetwork(outer(ψ', ψ), sites, vsites)

@testset "Test circuit apply" begin
    p = 0.01
    depol_channel = PreBuiltChannels.depolarizing(p, [sites[0][1], sites[1][1]], ρ)
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        [vsites[0][1], vsites[1][1]],
        Set(["RZZ"]),
        Set([vsites[i][1] for i in 0:11]),
    )
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), sites, vsites)
    noisy_circuit = NoisyCircuits.NoisyCircuit(qc, noise_model)
    evolved_ρ = ITensorNetworks.apply(ρ, noisy_circuit; maxdim=128, cutoff=1e-16)
    @test ITensorNetworks.inner(evolved_ρ.network, ρ.network) ≈ 1.0
end;

@testset "Test evolution noiseless evolution" begin
    circuit = Circuits.prepare_noiseless_circuit(qc, sites)
    apply_kwargs = Dict{Symbol,Real}(:maxdim => 50, :cutoff => 1e-16)
    cache_update_kwargs = Dict{Symbol,Any}(:maxiter => 16, :tol => 1e-6, :verbose => true)
    progress_kwargs = Dict{Symbol,Any}(:enabled => false)
    evolved_ψ = Circuits.run_compiled_circuit(
        ψ, circuit; progress_kwargs, cache_update_kwargs, apply_kwargs
    )
    @test abs(ITensorNetworks.inner(ψ, evolved_ψ)) ≈ 1.0
end;

@testset "no noise run_circuit" begin
    qc = CustomParsing.parse_circuit("example_circuits/bell_pair.json")
    g = GraphUtils.extract_adjacency_graph(qc)
    sites = ITensorNetworks.siteinds("Qubit", g)
    vsites = ITensorNetworks.siteinds("QubitVec", g)
    ψ = ITensorNetwork(v -> "0", sites);
    ρ = VDMNetworks.VDMNetwork(outer(ψ', ψ), sites, vsites)

    ψ2 = ITensorNetwork(v -> "1", sites);
    ρ2 = VDMNetworks.VDMNetwork(outer(ψ2', ψ2), sites, vsites)
    
    
    p = 0.01
    depol_channel = PreBuiltChannels.depolarizing(p, [sites[0][1], sites[1][1]], ρ)
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        [vsites[0][1], vsites[1][1]],
        Set(["RZZ"]),
        Set([vsites[i][1] for i in 0:1]),
    )
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), sites, vsites)
    noisy_circuit = NoisyCircuits.NoisyCircuit(qc, noise_model)

    cache_update_kwargs = Dict(:maxiter => 16, :tol => 1e-6, :verbose => false)
    apply_kwargs = Dict(:maxdim => 20, :cutoff => 1e-14)
    
    
    evolved_ρ = Evolution.run_circuit(ρ, noisy_circuit; cache_update_kwargs, apply_kwargs)
    @test Utils.innerprod(evolved_ρ, ρ) ≈ 0.5
    @test Utils.innerprod(evolved_ρ, ρ2) ≈ 0.5
end;

@testset "noise run_circuit" begin
    qc = CustomParsing.parse_circuit("example_circuits/bell_pair.json")
    g = GraphUtils.extract_adjacency_graph(qc)
    sites = ITensorNetworks.siteinds("Qubit", g)
    vsites = ITensorNetworks.siteinds("QubitVec", g)
    ψ = ITensorNetwork(v -> "0", sites);
    ρ = VDMNetworks.VDMNetwork(outer(ψ', ψ), sites, vsites)

    ψ2 = ITensorNetwork(v -> "1", sites);
    ρ2 = VDMNetworks.VDMNetwork(outer(ψ2', ψ2), sites, vsites)
    
    
    p = 0.01
    depol_channel = PreBuiltChannels.depolarizing(p, [sites[0][1], sites[1][1]], ρ)
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        [vsites[0][1], vsites[1][1]],
        Set(["CX"]),
        Set([vsites[i][1] for i in 0:1]),
    )
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), sites, vsites)
    noisy_circuit = NoisyCircuits.NoisyCircuit(qc, noise_model)
    
    cache_update_kwargs = Dict(:maxiter => 16, :tol => 1e-6, :verbose => false)
    apply_kwargs = Dict(:maxdim => 20, :cutoff => 1e-14)
    
    
    evolved_ρ = Evolution.run_circuit(ρ, noisy_circuit; cache_update_kwargs, apply_kwargs)
    @test Utils.innerprod(evolved_ρ, ρ) ≈ 0.5 * (1 - 8 * p/16)
    @test Utils.innerprod(evolved_ρ, ρ2) ≈ 0.5 * (1 - 8 * p/16)
end;
