
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

ITensors.op(::OpName"Id", ::SiteType"Qubit") = [
    1 0
    0 1
]

Vectorization.@build_vectorized_space(
    "Qubit", ["Id", "X", "Y", "Z", "CX", "H", "S", "T", "Rx", "Ry", "Rz"]
)
