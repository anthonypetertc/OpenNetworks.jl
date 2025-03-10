using Test
using ITensorsOpenSystems:
    Vectorization.fatsiteinds,
    Vectorization.VectorizedDensityMatrix
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
    Evolution,
    VDMNetworks.VDMNetwork,
    NoisyCircuits.NoisyCircuit
using ITensorNetworks
using ITensors
using ITensorMPS



qc = CustomParsing.parse_circuit("example_circuits/bell_pair.json")

sites = siteinds("Qubit", qc)
fatsites = fatsiteinds(sites)
ψ = productMPS(sites, "0")
ρ = VectorizedDensityMatrix(outer(ψ', ψ), fatsites)

ψ2 = productMPS(sites, "1")
ρ2 = VectorizedDensityMatrix(outer(ψ2', ψ2), fatsites)


@testset "Test circuit tebd no noise" begin
    p = 0.01
    depol_channel = PreBuiltChannels.depolarizing(p, sites, ρ, sites)
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        fatsites,
        Set(["RZZ"]),
        Set(fatsites),
    )
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), sites, fatsites, qc)
    noisy_circuit = NoisyCircuits.NoisyCircuit(qc, noise_model)
    evolved_ρ = Evolution.run_circuit(ρ, noisy_circuit; maxdim=128, cutoff=1e-16)
    @test inner(evolved_ρ, ρ) ≈ 0.5
end; 


@testset "Test circuit tebd noise" begin
    p = 0.01
    depol_channel = PreBuiltChannels.depolarizing(p, sites, ρ, sites)
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        fatsites,
        Set(["CX"]),
        Set(fatsites),
    )
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), sites, fatsites, qc)
    noisy_circuit = NoisyCircuits.NoisyCircuit(qc, noise_model)
    evolved_ρ = Evolution.run_circuit(ρ, noisy_circuit; maxdim=128, cutoff=1e-16)
    @test inner(evolved_ρ, ρ) ≈ 0.5 * (1 - 8 * p/16)
    @test inner(evolved_ρ, ρ2) ≈ 0.5 * (1 - 8 * p/16)
end; 
