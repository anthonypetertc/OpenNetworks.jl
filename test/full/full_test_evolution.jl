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
    VDMNetworks
using ITensorNetworks

N = 12
qc = CustomParsing.parse_circuit("example_circuits/circ_inverse.json")

g = GraphUtils.extract_adjacency_graph(qc)
sites = ITensorNetworks.siteinds("Qubit", g)
vsites = ITensorNetworks.siteinds("QubitVec", g)
ψ = ITensorNetwork(v -> "0", sites);
ρ = VDMNetworks.VDMNetwork(Utils.outer(ψ, ψ), sites, vsites)

@testset "Test circuit evolution" begin
    p = 0.01
    depol_channel = Channels.depolarizing_channel(p, [sites[0][1], sites[1][1]], ρ)
    noise_instruction = NoiseModels.NoiseInstruction(
        "depolarizing",
        depol_channel,
        [vsites[0][1], vsites[1][1]],
        Set(["RZZ"]),
        Set([vsites[i][1] for i in 0:11]),
    )
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), sites, vsites)
    noisy_circuit = NoisyCircuits.NoisyCircuit(qc, noise_model)
    evolved_ρ = NoisyCircuits.apply(ρ, noisy_circuit; maxdim=128, cutoff=1e-16)
    @test ITensorNetworks.inner(evolved_ρ.network, ρ.network) ≈ 1.0
end;
#=
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
=#
