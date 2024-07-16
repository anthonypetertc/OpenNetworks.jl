using Test
using OpenSystemsTools
using ITensors
using OpenNetworks: VectorizationNetworks, Utils, Channels
using NamedGraphs: named_grid, vertices
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks: ITensorNetworks, siteinds, ITensorNetwork

depolarizing_channel = Channels.depolarizing_channel
opdouble = Channels.opdouble
apply = Channels.apply
Channel = Channels.Channel
find_site = Channels.find_site

vectorize_density_matrix = VectorizationNetworks.vectorize_density_matrix
swapprime = Utils.swapprime

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
Tt = [1 0; 0 exp(im * π / 4)]

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
    kraus_maps = sqrt(1 / 4) * [Id1, X1, Y1, Z1]
    max_depol = depolarizing_channel(1, [sites[4]], vrho)
    max_depol_t =
        (1 / 4) * (
            kron(Idt, conj(Idt)) +
            kron(Xt, conj(Xt)) +
            kron(Yt, conj(Yt)) +
            kron(Zt, conj(Zt))
        )
    @test all(max_depol_t ≈ max_depol.tensor.tensor)
    #Would be good to test the two qubit version as well.
end;

# Next we test these functions applied to Networks.

X = op("X", square_sites[(1, 1)])
Y = op("Y", square_sites[(1, 2)])
Z = op("Z", square_sites[(2, 2)])
Id1 = op("Id", square_sites[(1, 1)])
Xt = [0 1; 1 0]
Yt = [0 -im; im 0]
Zt = [1 0; 0 -1]
Idt = [1 0; 0 1]
CXt = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]

T = op("T", square_sites[(2, 1)])
Tt = [1 0; 0 exp(im * π / 4)]

@testset "opdouble" begin
    vX1 = opdouble(X, square_rand_vρ)
    vY1 = opdouble(Y, square_rand_vρ)
    vZ1 = opdouble(Z, square_rand_vρ)
    vT1 = opdouble(T, square_rand_vρ)
    @test all(vX1.tensor ≈ kron(Xt, Xt))
    @test find_site(inds(vX1)[1], square_rand_vρ) == (1, 1)
    @test all(vY1.tensor ≈ kron(Yt, conj(Yt)))
    @test find_site(inds(vY1)[1], square_rand_vρ) == (1, 2)
    @test all(vZ1.tensor ≈ kron(Zt, Zt))
    @test find_site(inds(vZ1)[1], square_rand_vρ) == (2, 2)
    @test all(vT1.tensor ≈ kron(Tt, conj(Tt)))
    @test find_site(inds(vT1)[1], square_rand_vρ) == (2, 1)
end;

@testset "apply" begin
    # 1. Check that Unitary evolution is trace preserving.
    Q, _ = qr(randn(ComplexF64, 4, 4))
    qubits = [square_sites[(1, 1)], square_sites[(1, 2)]]
    append!(qubits, qubits')
    U = ITensors.ITensor(Array(Q), qubits)
    evolved = Channels.apply(U, square_rand_vρ)
    @test VectorizationNetworks.vectorizedtrace(evolved; alg="exact") ≈
        Utils.trace(square_rand_ρ)
end;

@testset "apply_random" begin
    unvectorized = deepcopy(square_rand_ψ)
    evolved = deepcopy(square_rand_vρ)
    for i in 1:6
        vertex = rand(keys(unvectorized.data_graph.vertex_data))
        qubit1 = square_sites[vertex]
        qubit2 = square_sites[rand(
            Graphs.neighbors(unvectorized.data_graph.underlying_graph, vertex)
        )]
        qubits = [qubit1, qubit2]
        append!(qubits, qubits')

        Q, _ = qr(randn(ComplexF64, 4, 4))
        U = ITensor(Array(Q), qubits)
        evolved = Channels.apply(U, evolved)
        unvectorized = ITensorNetworks.apply(U, unvectorized)
    end

    v2 = VectorizationNetworks.vectorize_density_matrix(
        outer(unvectorized, unvectorized), evolved.unvectorizedinds, square_vsites
    )
    norm_const = sqrt(Utils.innerprod(v2, v2) * Utils.innerprod(evolved, evolved))
    @test Utils.innerprod(v2, evolved) / norm_const ≈ 1
end;

@testset "Channel" begin
    kraus_maps = sqrt(1 / 4) * [Id1, X1, Y1, Z1]
    max_depol = depolarizing_channel(1, [square_sites[(2, 2)]], square_rand_vρ)
    max_depol_t =
        (1 / 4) * (
            kron(Idt, conj(Idt)) +
            kron(Xt, conj(Xt)) +
            kron(Yt, conj(Yt)) +
            kron(Zt, conj(Zt))
        )
    @test all(max_depol_t ≈ max_depol.tensor.tensor)
    #Would be good to test the two qubit version as well.
end;

@testset "Channel evolution" begin
    ρ0 = deepcopy(square_rand_ρ)
    vρ0 = vectorize_density_matrix(ρ0, square_sites, square_vsites)
    n_ops = 3
    vertex = rand(keys(square_rand_ψ.data_graph.vertex_data))
    qubit1 = square_sites[vertex]
    qubit2 = square_sites[rand(
        Graphs.neighbors(square_rand_ψ.data_graph.underlying_graph, vertex)
    )]
    qubits = [qubit1, qubit2]
    append!(qubits, qubits')
    for ii in 1:3
        kraus_channels = Vector{ITensor}()
        kraus_matrices = Vector{Matrix}()
        for _ in 1:n_ops
            unscaled_kraus, _, _ = svd(rand(Complex{Float64}, (4, 4)))
            push!(kraus_matrices, ((1 / sqrt(n_ops)) * unscaled_kraus))
            push!(kraus_channels, (1 / sqrt(n_ops)) * ITensor(unscaled_kraus, qubits))
        end
        channel = Channel("random_channel", kraus_channels, vρ0)
        vρ0 = apply(channel, vρ0)
        ρ0 = reduce(
            +,
            [
                swapprime(
                    ITensorNetworks.apply(
                        K, swapprime(ITensorNetworks.apply(conj(K), ρ0), 0, 1)
                    ),
                    0,
                    1,
                ) for K in kraus_channels
            ],
        )
    end
    vρ1 = vectorize_density_matrix(ρ0, square_sites, square_vsites)
    @test Utils.innerprod(vρ1, vρ0) /
          sqrt(Utils.innerprod(vρ1, vρ1) * Utils.innerprod(vρ0, vρ0)) ≈ 1
end;

@testset "Compose" begin
    tensor = op("X", square_sites[(1, 1)])
    gate_channel = Channel("X", [tensor], square_rand_vρ)
    new_chan = Channels.compose(gate_channel, gate_channel)
    @test new_chan.tensor.tensor ≈ kron(Idt, Idt)

    y_channel = Channel("Y", [op("Y", square_sites[(1, 1)])], square_rand_vρ)
    XY_channel = Channels.compose(gate_channel, y_channel)
    @test XY_channel.tensor.tensor ≈ kron(Zt, Zt)
end;

index_list = Vector{ITensors.Index}()
append!(index_list, square_sites[(1, 1)])
append!(index_list, square_sites[(1, 2)])
CX_channel = Channel("CX", [op("CX", index_list)], square_rand_vρ)
X_channel = Channel("X", [op("X", square_sites[(1, 1)])], square_rand_vρ)
composed_channel = Channels.compose(CX_channel, X_channel)
ψ00 = ITensorNetwork(v -> "0", square_sites)
ψ01 = ITensorNetworks.apply(op("X", square_sites[(1, 2)]), ψ00)
ψ10 = ITensorNetworks.apply(op("X", square_sites[(1, 1)]), ψ00)
ψ11 = ITensorNetworks.apply(op("X", square_sites[(1, 1)]), ψ01)
ρ00 = VectorizationNetworks.vectorize_density_matrix(
    Utils.outer(ψ00, ψ00), square_sites, square_vsites
)
ρ01 = VectorizationNetworks.vectorize_density_matrix(
    Utils.outer(ψ01, ψ01), square_sites, square_vsites
)
ρ10 = VectorizationNetworks.vectorize_density_matrix(
    Utils.outer(ψ10, ψ10), square_sites, square_vsites
)
ρ11 = VectorizationNetworks.vectorize_density_matrix(
    Utils.outer(ψ11, ψ11), square_sites, square_vsites
)

@testset "Compose two qubit" begin
    @test Utils.innerprod(Channels.apply(composed_channel, ρ00), ρ11) ≈ 1
    @test Utils.innerprod(Channels.apply(composed_channel, ρ01), ρ10) ≈ 1
    @test Utils.innerprod(Channels.apply(composed_channel, ρ10), ρ00) ≈ 1
    @test Utils.innerprod(Channels.apply(composed_channel, ρ11), ρ01) ≈ 1
end;
