using Test
using ITensorsOpenSystems
using ITensors
using OpenNetworks:
    VDMNetworks, Utils, Channels, VectorizationNetworks, GraphUtils, CustomParsing
using Random
using LinearAlgebra
using Graphs
using ITensorNetworks: vertices, edges, src, dst, add_edge!

circ = CustomParsing.parse_circuit("example_circuits/circ.json")
G = GraphUtils.extract_adjacency_graph(circ)
G2 = GraphUtils.named_ring_graph(12)

@testset "GraphUtils" begin
    @test G == G2
end;

@testset "linegraph" begin
    s = siteinds("Qubit", 10)
    G = GraphUtils.linegraph(s)
    @test length(vertices(G)) == 10
    for i in 1:9
        @test src(edges(G)[i]) == i
        @test dst(edges(G)[i]) == i + 1
    end
end

@testset "linenetwork" begin
    s = siteinds("Qubit", 10)
    G = GraphUtils.linegraph(s)
    s2 = GraphUtils.linenetwork(s)
    for i in 1:10
        @test s2[i] == [s[i]]
    end
    @test G == s2.data_graph.underlying_graph
end

@testset "islinegraph" begin
    s = siteinds("Qubit", 10)
    G = GraphUtils.linegraph(s)
    @test GraphUtils.islinegraph(G)
    
    add_edge!(G, 1=>10)
    @test !(GraphUtils.islinegraph(G))
end