using Test
using OpenNetworks: VectorizationNetworks, Utils, Channels, GraphUtils, NoisyCircuits, NoiseModels
using ITensorNetworks
using JSON

N =12
qc = JSON.parsefile("example_circuits/circ_inverse.json")
qc = [Utils.typenarrow!(gate) for gate in qc]

g = GraphUtils.extract_adjacency_graph(qc, N)
sites = ITensorNetworks.siteinds("Qubit", g)
vsites = ITensorNetworks.siteinds("QubitVec", g)
ψ = ITensorNetwork(v -> "0", sites);
ρ = VectorizationNetworks.vectorize_density_matrix(Utils.outer(ψ, ψ),ψ, vsites)

@testset "Test circuit evolution" begin
    p = 0.01
    depol_channel = Channels.depolarizing_channel(p, [sites[(0,)][1], sites[(1,)][1]], ρ);
    noise_instruction = NoiseModels.NoiseInstruction("depolarizing", depol_channel, [vsites[(0,)][1], vsites[(1,)][1]], Set(["RZZ"]), Set([vsites[(i,)][1] for i in 0:11]));
    noise_model = NoiseModels.NoiseModel(Set([noise_instruction]), sites, vsites);
    noisy_circuit = NoisyCircuits.NoisyCircuit(qc, noise_model)
    evolved_ρ = NoisyCircuits.apply(ρ, noisy_circuit; maxdim=128, cutoff=1e-16)
    @test ITensorNetworks.inner(evolved_ρ.network, ρ.network) ≈ 1.0
end;

