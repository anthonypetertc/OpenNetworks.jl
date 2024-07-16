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
    vT1 = opdouble(T, square_rand_vρ)
    @test all(vX1.tensor ≈ kron(Xt, Xt))
    @test find_site(inds(vX1)[1], square_rand_vρ) == (1, 1)
    @test all(vT1.tensor ≈ kron(Tt, conj(Tt)))
    @test find_site(inds(vT1)[1], square_rand_vρ) == (2, 1)
end;

@testset "Channel" begin
    #kraus_maps = sqrt(1 / 4) * [Id1, X1, Y1, Z1]
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

@testset "Compose" begin
    tensor = op("X", square_sites[(1, 1)])
    gate_channel = Channel("X", [tensor], square_rand_vρ)
    new_chan = Channels.compose(gate_channel, gate_channel)
    @test new_chan.tensor.tensor ≈ kron(Idt, Idt)

    y_channel = Channel("Y", [op("Y", square_sites[(1, 1)])], square_rand_vρ)
    XY_channel = Channels.compose(gate_channel, y_channel)
    @test XY_channel.tensor.tensor ≈ kron(Zt, Zt)
end;
