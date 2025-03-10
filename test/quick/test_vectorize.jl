using Test
using ITensorsOpenSystems: Vectorization
using ITensorNetworks: ⊗, prime, dag, ITensorNetworks, siteinds, contract
using ITensors: ITensors, op
using OpenNetworks: VectorizationNetworks, Utils
using Random
using LinearAlgebra
using Graphs

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

@testset "vertices" begin
    @test vertices(vρ) == vertices(ρ)
end

@testset "edges" begin
    @test edges(vρ) == edges(ρ)
end


@testset "expectation" begin
    o = op("Z", sites[(1, 1)])
    @test VectorizationNetworks.vexpect(vρ, o; alg="exact") ≈
        Utils.trace(ITensorNetworks.apply(o, ρ))

    @test VectorizationNetworks.vexpect(vρ, o; alg="exact") ≈
        expect("Z", vρ; alg="exact")[(1,1)]
end;

