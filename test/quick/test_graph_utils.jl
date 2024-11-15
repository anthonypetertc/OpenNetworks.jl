using Test
using ITensorsOpenSystems
using ITensors
using OpenNetworks:
    VDMNetworks, Utils, Channels, VectorizationNetworks, GraphUtils, CustomParsing
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks

circ = CustomParsing.parse_circuit("example_circuits/circ.json")
G = GraphUtils.extract_adjacency_graph(circ)
G2 = GraphUtils.named_ring_graph(12)

@testset "GraphUtils" begin
    @test G == G2
end;
