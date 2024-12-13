using Test
using ITensorsOpenSystems
using ITensors
using OpenNetworks: Utils, Channels, PreBuiltChannels.depolarizing
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks: ITensorNetworks, siteinds, ITensorNetwork

opdouble = Channels.opdouble
apply = Channels.apply
Channel = Channels.Channel

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
    vX1 = opdouble(X)
    vT1 = opdouble(T)
    @test all(vX1.tensor ≈ kron(Xt, Xt))
    @test all(vT1.tensor ≈ kron(conj(Tt), Tt))
end;

@testset "opdouble" begin
    vX1 = opdouble(X, square_rand_vρ)
    vT1 = opdouble(T, square_rand_vρ)
    @test all(vX1.tensor ≈ kron(Xt, Xt))
    @test findsite(square_rand_vρ, inds(vX1)[1]) == (1, 1)
    @test all(vT1.tensor ≈ kron(conj(Tt), Tt))
    @test findsite(square_rand_vρ, inds(vT1)[1]) == (2, 1)
end;

@testset "Channel" begin
    #kraus_maps = sqrt(1 / 4) * [Id1, X1, Y1, Z1]
    max_depol = depolarizing(1, [square_sites[(2, 2)]], square_rand_vρ)
    max_depol2 = depolarizing(1, square_sites[(2, 2)])
    max_depol_t =
        (1 / 4) * (
            kron(Idt, conj(Idt)) +
            kron(Xt, conj(Xt)) +
            kron(Yt, conj(Yt)) +
            kron(Zt, conj(Zt))
        )
    @test all(max_depol_t ≈ max_depol.tensor.tensor)
    @test all(max_depol_t ≈ max_depol2.tensor.tensor)
    #Would be good to test the two qubit version as well.
end;

@testset "Channel" begin
    Id1 = op("Id", square_sites[(1, 1)])
    X1 = op("X", square_sites[(1, 1)])
    Z1 = op("Z", square_sites[(1, 1)])
    Y1 = op("Y", square_sites[(1, 1)])
    kraus_maps = sqrt(1 / 4) * [Id1, X1, Y1, Z1]
    max_depol = Channels.Channel("depol", kraus_maps)
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
