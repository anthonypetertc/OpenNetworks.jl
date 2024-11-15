using Test
using ITensorsOpenSystems: Vectorization
using ITensorNetworks: ⊗, prime, dag, ITensorNetworks, siteinds, contract
using ITensors: ITensors, op
using OpenNetworks: VectorizationNetworks, Utils
using Random
using LinearAlgebra
using Graphs

vectorize_density_matrix = VectorizationNetworks.vectorize_density_matrix
swapprime = Utils.swapprime

g_dims = square_g_dims
g = square_g
sites = square_sites
vsites = square_vsites

ψ = square_rand_ψ
ρ = square_rand_ρ
vρ = square_rand_vρ

@testset "site inds" begin
    for index in VectorizationNetworks.siteinds(vρ).data_graph.vertex_data
        @test ITensors.hastags(index, "QubitVec")
    end
end;

@testset "unvectorize" begin
    ρ2 = VectorizationNetworks.unvectorize_density_matrix(vρ)
    exact_dm_unvectorized = Array(contract(ρ ⊗ swapprime(prime(ρ2), 0, 2)).tensor)[1]
    original_unvectorized = Array(contract(ρ ⊗ swapprime(prime(dag(ρ)), 0, 2)).tensor)[1]
    @test exact_dm_unvectorized ≈ original_unvectorized
end;

@testset "has tags" begin
    for ind in Utils.siteinds(vρ).data_graph.vertex_data
        @test ITensors.hastags(ind, "QubitVec")
    end
end;
