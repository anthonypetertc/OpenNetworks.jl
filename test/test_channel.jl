using Test
using OpenSystemsTools
using ITensors
using OpenNetworks: VectorizationNetworks, Utils, Channels
using NamedGraphs: named_grid
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks

depolarizing_channel = Channels.depolarizing_channel
opdouble = Channels.opdouble
apply = Channels.apply
Channel = Channels.Channel
find_site = Channels.find_site

vectorize_density_matrix = VectorizationNetworks.vectorize_density_matrix
#opdouble = VectorizationBP.opdouble
swapprime = Utils.swapprime


#include("/home/tony/OpenNetworks.jl/src/utils/channel.jl")


ITensors.op(::OpName"Id",::SiteType"Qubit") = [1 0 
                                             0 1 ]

Vectorization.@build_vectorized_space("Qubit",["Id","X","Y","Z","CX",
                                           "H","S","T","Rx","Ry","Rz"])


sites = siteinds("Qubit", 16)
psi = productMPS(sites, "0")
rho = outer(psi', psi)

vs = siteinds("QubitVec", 16)
vrho = Vectorization.vectorize_density_matrix(rho, vs)

X1 = ITensor(Op("σx", 1), sites)
Y1 = ITensor(Op("σy", 1), sites)
Z1 = ITensor(Op("σz", 1), sites)
Id1 = op("Id", sites[1])
Y3 = ITensor(Op("σy", 3), sites)
Z12 = ITensor(Op("σz", 12), sites)
Xt = [0 1; 1 0]
Yt = [0 -im; im 0]
Zt = [1 0; 0 -1]
Idt = [1 0; 0 1]

T1 = op("T", sites[1])
Tt = [1 0; 0 exp(im*π/4)]

#@assert all(opdouble(X1, vrho).tensor ≈ kron(Xt, Xt))

@testset "opdouble" begin
    vX1 = opdouble(X1, vrho)
    vY3 = opdouble(Y3, vrho)
    vZ12 = opdouble(Z12, vrho)
    vT1 = opdouble(T1, vrho)
    @test all(vX1.tensor ≈ kron(Xt, Xt))
    @test find_site(inds(vX1)[1]) == 1 
    @test all(vY3.tensor ≈ kron(Yt, conj(Yt)))
    @test find_site(inds(vY3)[1]) == 3
    @test all(vZ12.tensor ≈ kron(Zt, Zt))
    @test find_site(inds(vZ12)[1]) == 12
    @test all(vT1.tensor ≈ kron(Tt, conj(Tt)))
end;

@testset "Channel" begin
    kraus_maps = sqrt(1/4)*[Id1, X1, Y1, Z1]
    max_depol = depolarizing_channel(1, [sites[4]], vrho)
    max_depol_t = (1/4)*(kron(Idt, conj(Idt)) + kron(Xt, conj(Xt)) + kron(Yt, conj(Yt))+kron(Zt, conj(Zt)))
    @test all(max_depol_t ≈ max_depol.tensor.tensor)
    #Would be good to test the two qubit version as well. 
end;

@testset "Channel evolution" begin
    #Single qubit maximally depolarizing.
    ψ = productMPS([sites[1]], "0")
    ψ1 = productMPS([sites[1]], "1")
    ρmax = Vectorization.vectorize_density_matrix(0.5*outer(ψ', ψ) + 0.5*outer(ψ1', ψ1), [vs[1]])
    ρ = Vectorization.vectorize_density_matrix(outer(ψ', ψ), [vs[1]])
    max_depol1 = depolarizing_channel(1, [sites[1]], ρ)
    ρ2 = apply(max_depol1, ρ)
    @test all(ρ2[1].tensor ≈ ρmax[1].tensor)

    #Two qubit maximum depolarizing.
    ψ00 = productMPS([sites[1], sites[2]], ["0", "0"])
    ψ01 = productMPS([sites[1], sites[2]], ["0", "1"])
    ψ10 = productMPS([sites[1], sites[2]], ["1", "0"])
    ψ11 = productMPS([sites[1], sites[2]], ["1", "1"])
    ρmax = Vectorization.vectorize_density_matrix(0.25*outer(ψ00', ψ00)+0.25*outer(ψ01', ψ01)+0.25*outer(ψ10', ψ10)+0.25*outer(ψ11', ψ11), [vs[1], vs[2]])
    ρ = Vectorization.vectorize_density_matrix(outer(ψ00',ψ00), [vs[1], vs[2]])
    max_depol2 = depolarizing_channel(1, [sites[1], sites[2]], ρ)
    ρ2 = apply(max_depol2, ρ)
    @test inner(ρ2, ρmax)/(sqrt(inner(ρ2, ρ2)*inner(ρmax, ρmax))) ≈ 1

    #Random Kraus channels.
    n_ops = 6
    ρ1 = ρ
    ρ2 = ρ
    dm = [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]
    for _ in 1:2
        qubits = [sites[1], sites[2]]
        append!(qubits, qubits')
        kraus_channels = Vector{ITensor}()
        kraus_matrices = Vector{Matrix}()
        for _ in 1:n_ops
            unscaled_kraus, _, _ = svd(rand(Complex{Float64}, (4 ,4)))
            push!(kraus_matrices, ((1/sqrt(n_ops))*unscaled_kraus))
            push!(kraus_channels, (1/sqrt(n_ops))*ITensor(unscaled_kraus, qubits))
        end
        channel = Channel("random_channel", kraus_channels, ρ)
        ρ2 = apply(channel, ρ2)
        dm = reduce(+, [transpose(conj(K))*dm*K for K in kraus_matrices])
    end
    reshaped_dm = reshape(permutedims(reshape(dm, (2,2,2,2)), [1, 3, 2, 4]), (4,4))
    @test reshape(ITensors.contract(ρ2).tensor, (4,4))≈ reshaped_dm
end;

# Next we test these functions applied to Networks.

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


@testset "opdouble" begin
    vX1 = opdouble(X, vρ)
    vY1 = opdouble(Y, vρ)
    vZ1 = opdouble(Z, vρ)
    vT1 = opdouble(T, vρ)
    @test all(vX1.tensor ≈ kron(Xt, Xt))
    @test find_site(inds(vX1)[1], vρ) == (1,1) 
    @test all(vY1.tensor ≈ kron(Yt, conj(Yt)))
    @test find_site(inds(vY1)[1], vρ) == (1,2)
    @test all(vZ1.tensor ≈ kron(Zt, Zt))
    @test find_site(inds(vZ1)[1], vρ) == (2,2)
    @test all(vT1.tensor ≈ kron(Tt, conj(Tt)))
    @test find_site(inds(vT1)[1], vρ) == (2,1)
end;




@testset "apply" begin
    # 1. Check that Unitary evolution is trace preserving.
    Q, _ = qr(randn(ComplexF64, 4, 4))
    qubits = [sites[(1,1)], sites[(1,2)]]
    append!(qubits, qubits')
    U = ITensors.ITensor(Array(Q), qubits)
    evolved = Channels.apply(U, vρ)
    @test VectorizationNetworks.vectorizedtrace(evolved) ≈ Utils.trace(ρ)
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
        evolved = Channels.apply(U, evolved)
        unvectorized = ITensorNetworks.apply(U, unvectorized)
    end
    
    v2 = VectorizationNetworks.vectorize_density_matrix(outer(unvectorized, unvectorized), evolved.unvectorizednetwork, vsites)
    norm_const = sqrt(Utils.innerprod(v2, v2) * Utils.innerprod(evolved, evolved))
    @test Utils.innerprod(v2, evolved)/norm_const≈ 1
end;

#=
@testset "Channel" begin
    kraus_maps = sqrt(1/4)*[Id1, X1, Y1, Z1]
    max_depol = depolarizing_channel(1, [sites[4]], vrho)
    max_depol_t = (1/4)*(kron(Idt, conj(Idt)) + kron(Xt, conj(Xt)) + kron(Yt, conj(Yt))+kron(Zt, conj(Zt)))
    @test all(max_depol_t ≈ max_depol.tensor.tensor)
    #Would be good to test the two qubit version as well. 
end;
=#

@testset "Channel evolution" begin
    ρ0 = deepcopy(ρ)
    vρ0 = vectorize_density_matrix(ρ0, ψ, vsites)
    n_ops = 3
    vertex = rand(keys(ψ.data_graph.vertex_data))
    qubit1 = sites[vertex]
    qubit2 = sites[rand(Graphs.neighbors(ψ.data_graph.underlying_graph, vertex))]
    qubits = [qubit1, qubit2]
    append!(qubits, qubits')
    for ii in 1:3
        kraus_channels = Vector{ITensor}()
        kraus_matrices = Vector{Matrix}()
        for _ in 1:n_ops
            unscaled_kraus, _, _ = svd(rand(Complex{Float64}, (4 ,4)))
            push!(kraus_matrices, ((1/sqrt(n_ops))*unscaled_kraus))
            push!(kraus_channels, (1/sqrt(n_ops))*ITensor(unscaled_kraus, qubits))
        end
        channel = Channel("random_channel", kraus_channels, vρ0)
        vρ0 = apply(channel, vρ0)
        ρ0 = reduce(+, [swapprime(ITensorNetworks.apply(K, swapprime(ITensorNetworks.apply(conj(K), ρ0), 0, 1)), 0, 1) for K in kraus_channels])
    end
    vρ1 = vectorize_density_matrix(ρ0, ψ, vsites)
    @test Utils.innerprod(vρ1, vρ0)/sqrt(Utils.innerprod(vρ1, vρ1)*Utils.innerprod(vρ0, vρ0)) ≈ 1
end;