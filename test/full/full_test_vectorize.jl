using Test
using ITensorsOpenSystems: Vectorization
using ITensorNetworks: ⊗, prime, dag, ITensorNetwork, ITensorNetworks, siteinds, contract
using ITensors: ITensors, op
using OpenNetworks: VectorizationNetworks, Utils

swapprime = Utils.swapprime

g_dims = square_g_dims
g = square_g
sites = square_sites
vsites = square_vsites
χ = 4

ψ = square_rand_ψ
ρ = square_rand_ρ
vρ = square_rand_vρ

X = op("X", sites[(1, 1)])
Y = op("Y", sites[(1, 2)])
Z = op("Z", sites[(2, 2)])
Id1 = op("Id", sites[(1, 1)])
Xt = [0 1; 1 0]
Yt = [0 -im; im 0]
Zt = [1 0; 0 -1]
Idt = [1 0; 0 1]

T = op("T", sites[(2, 1)])
Tt = [1 0; 0 exp(im * π / 4)]

@testset "site inds" begin
    for index in VectorizationNetworks.siteinds(vρ).data_graph.vertex_data
        @test ITensors.hastags(index, "QubitVec")
    end
end;

@testset "purity" begin
    @test Utils.innerprod(vρ, vρ) ≈
        Array(contract(ρ ⊗ swapprime(prime(dag(ρ)), 0, 2)).tensor)[1]
end;

@testset "unvectorize" begin
    ρ2 = VectorizationNetworks.unvectorize_density_matrix(vρ)
    @test Array(contract(ρ ⊗ swapprime(prime(ρ2), 0, 2)).tensor)[1] ≈ sqrt(
        Array(contract(ρ ⊗ swapprime(prime(dag(ρ)), 0, 2)).tensor)[1] *
        Array(contract(ρ2 ⊗ swapprime(prime(dag(ρ2)), 0, 2)).tensor)[1],
    )
end;

@testset "trace" begin
    @test VectorizationNetworks.vectorizedtrace(vρ; alg="exact") ≈ Utils.trace(ρ)
end;

@testset "expectation" begin
    o = op("Z", sites[(1, 1)])
    @test VectorizationNetworks.vexpect(vρ, o; alg="exact") ≈
        Utils.trace(ITensorNetworks.apply(o, ρ))
end;

@testset "vectorize dm" begin
    for ind in Utils.siteinds(vρ).data_graph.vertex_data
        @test ITensors.hastags(ind, "QubitVec")
    end
end;



@testset "vertices" begin
    @test vertices(vρ) == vertices(ρ)
end;

@testset "edges" begin
    @test edges(vρ) == edges(ρ)
end;


@testset "expectation" begin
    o = op("Z", sites[(1, 1)])
    @test VectorizationNetworks.vexpect(vρ, o; alg="exact") ≈
        Utils.trace(ITensorNetworks.apply(o, ρ))

    @test VectorizationNetworks.vexpect(vρ, o; alg="exact") ≈
        expect("Z", vρ; alg="exact")[(1,1)]
end;

