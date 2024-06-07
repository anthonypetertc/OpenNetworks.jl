using Test
using OpenSystemsTools: Vectorization
using ITensorNetworks: ⊗, prime, dag, ITensorNetwork, ITensorNetworks, siteinds, contract
using ITensors: ITensors, op
using OpenNetworks: VectorizationNetworks, Utils
using NamedGraphs: named_grid
using Random
using LinearAlgebra
using Graphs

vectorize_density_matrix = VectorizationNetworks.vectorize_density_matrix
#opdouble = Channels.opdouble
swapprime = Utils.swapprime

#=Vectorization.@build_vectorized_space("Qubit",["Id","X","Y","Z","CX",
                                           "H","S","T","Rx","Ry","Rz"])=#

g_dims = (2, 2)
g = named_grid(g_dims)
sites = siteinds("Qubit", g)
vsites = siteinds("QubitVec", g)
χ = 4
#Random.seed!(1564)
ψ = ITensorNetworks.random_tensornetwork(sites; link_space=χ)
ρ = Utils.outer(ψ, ψ)
vρ = vectorize_density_matrix(ρ, ψ, vsites)

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
