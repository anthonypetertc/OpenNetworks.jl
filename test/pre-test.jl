using ITensorsOpenSystems: Vectorization
using ITensors
using OpenNetworks: VectorizationNetworks, Utils, Channels, VDMNetworks
using NamedGraphs: vertices, NamedGraphGenerators.named_grid
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks: ITensorNetworks, siteinds, ITensorNetwork

opdouble = Channels.opdouble
apply = Channels.apply
Channel = Channels.Channel
swapprime = Utils.swapprime
#=
ITensors.op(::OpName"Id", ::SiteType"Qubit") = [
    1 0
    0 1
]

Vectorization.@build_vectorized_space(
    "Qubit", ["Id", "X", "Y", "Z", "CX", "H", "S", "T", "Rx", "Ry", "Rz"]
)
=#

square_g_dims = (2, 2)
square_g = named_grid(square_g_dims)
square_sites = siteinds("Qubit", square_g)
square_vsites = Vectorization.fatsiteinds(square_sites)
χ = 4
Random.seed!(1564)
square_rand_ψ = ITensorNetworks.random_tensornetwork(square_sites; link_space=χ)
square_rand_ρ = Utils.outer(square_rand_ψ', square_rand_ψ)
square_rand_vρ = VDMNetworks.VDMNetwork(square_rand_ρ, square_sites, square_vsites)
