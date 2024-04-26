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

X = op("X", sites[(1,1)])
Y = op("Y", sites[(1,2)])
Z = op("Z", sites[(2,2)])
Id1 = op("Id", sites[(1,1)])
Xt = [0 1; 1 0]
Yt = [0 -im; im 0]
Zt = [1 0; 0 -1]
Idt = [1 0; 0 1]

T = op("T", sites[(2, 1)])
Tt = [1 0; 0 exp(im*π/4)]

@testset "site inds" begin
    for index in VectorizationNetworks.siteinds(vρ).data_graph.vertex_data
        @test ITensors.hastags(index, "QubitVec")
    end
end;

@testset "purity" begin
    @test Utils.innerprod(vρ, vρ) ≈ Array(contract(ρ ⊗ swapprime(prime(dag(ρ)), 0, 2)).tensor)[1]
end;

@testset "unvectorize" begin
    ρ2 = VectorizationNetworks.unvectorize_density_matrix(vρ)
    @test Array(contract(ρ ⊗ swapprime(prime(ρ2), 0, 2)).tensor)[1] ≈ sqrt(Array(contract(ρ ⊗ swapprime(prime(dag(ρ)), 0, 2)).tensor)[1]
     * Array(contract(ρ2 ⊗ swapprime(prime(dag(ρ2)), 0, 2)).tensor)[1])
end;

@testset "trace" begin
    @test VectorizationNetworks.vectorizedtrace(vρ) ≈ Utils.trace(ρ)
end;

@testset "expectation" begin
    o = op("Z", sites[(1,1)])
    @test VectorizationNetworks.vexpect(vρ, o) ≈ Utils.trace(ITensorNetworks.apply(o, ρ))
end;

#=
@testset "opdouble" begin
    vX1 = opdouble(X, vρ, ψ)
    vY1 = opdouble(Y, vρ, ψ)
    vZ1 = opdouble(Z, vρ, ψ)
    vT1 = opdouble(T, vρ, ψ)
    @test all(vX1.tensor ≈ kron(Xt, Xt))
    @test Channels.find_site(inds(vX1)[1], vρ) == (1,1) 
    @test all(vY1.tensor ≈ kron(Yt, conj(Yt)))
    @test Channels.find_site(inds(vY1)[1], vρ) == (1,2)
    @test all(vZ1.tensor ≈ kron(Zt, Zt))
    @test Channels.find_site(inds(vZ1)[1], vρ) == (2,2)
    @test all(vT1.tensor ≈ kron(Tt, conj(Tt)))
    @test Channels.find_site(inds(vT1)[1], vρ) == (2,1)
end;




@testset "apply" begin
    # 1. Check that Unitary evolution is trace preserving.
    Q, _ = qr(randn(ComplexF64, 4, 4))
    qubits = [sites[(1,1)], sites[(1,2)]]
    append!(qubits, qubits')
    U = ITensors.ITensor(Array(Q), qubits)
    evolved = Channels.apply(U, vρ, ψ)
    @test VectorizationNetworks.vectorizedtrace(evolved, ψ) ≈ Utils.trace(ρ)
end;

@testset "apply_random" begin
    unvectorized = deepcopy(ψ)
    evolved = deepcopy(vρ)
    for i in 1:6
        vertex = rand(keys(unvectorized.data_graph.vertex_data))
        qubit1 = sites[vertex]
        qubit2 = sites[rand(Graphs.neighbors(unvectorized.data_graph.underlying_graph, vertex))]
        qubits = [qubit1, qubit2]
        append!(qubits, qubits')
    
        Q, _ = qr(randn(ComplexF64, 4, 4))
        U = ITensor(Array(Q), qubits)
        evolved = Channels.apply(U, evolved, unvectorized)
        unvectorized = ITensorNetworks.apply(U, unvectorized)
    
        v2 = VectorizationNetworks.vectorize_density_matrix(outer(unvectorized, unvectorized), ψ, vsites)
    end
    
    v2 = VectorizationNetworks.vectorize_density_matrix(outer(unvectorized, unvectorized), ψ, vsites)
    norm_const = sqrt(Utils.innerprod(v2, v2) * Utils.innerprod(evolved, evolved))
    @test Utils.innerprod(v2, evolved)/norm_const≈ 1
end;
=#