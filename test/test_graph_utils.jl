using Test
using OpenSystemsTools
using ITensors
using OpenNetworks: VDMNetworks, Utils, Channels, VectorizationNetworks, GraphUtils
using NamedGraphs: named_grid
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks
using JSON

circ = [Utils.typenarrow!(elm) for elm in JSON.parsefile("example_circuits/circ.json")]
G = GraphUtils.extract_adjacency_graph(circ, 12)
G2 = GraphUtils.named_ring_graph(12)

@testset "GraphUtils" begin
    @test G == G2
end;