using Test
using OpenSystemsTools
using ITensors
import OpenBP: Channels

depolarizing_channel = Channels.depolarizing_channel
opdouble = Channels.opdouble
apply = Channels.apply
Channel = Channels.Channel
find_site = Channels.find_site


#include("/home/tony/OpenBP.jl/src/utils/channel.jl")


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
    @test reshape(contract(ρ2).tensor, (4,4))≈ reshaped_dm
end;