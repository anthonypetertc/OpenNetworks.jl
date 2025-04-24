using ITensorsOpenSystems: Vectorization
using ITensors
using OpenNetworks: Utils, VDMNetworks
using NamedGraphs: NamedGraphGenerators.named_grid
using Random
using ITensorNetworks: ITensorNetworks, siteinds, ITensorNetwork


square_g_dims = (2, 2)
square_g = named_grid(square_g_dims)
square_sites = siteinds("Qubit", square_g)
square_vsites = Vectorization.fatsiteinds(square_sites)
χ = 4
Random.seed!(1564)
square_rand_ψ = ITensorNetworks.random_tensornetwork(square_sites; link_space=χ)
square_rand_ρ = Utils.outer(square_rand_ψ', square_rand_ψ)
square_rand_vρ = VDMNetworks.VDMNetwork(square_rand_ρ, square_sites, square_vsites)
