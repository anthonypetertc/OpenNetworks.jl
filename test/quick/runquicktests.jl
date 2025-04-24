

@testset "Channel tests" begin
    include("test_channel.jl")
end

@testset "Vectorization tests" begin
    include("test_vectorize.jl")
end

@testset "Utils tests" begin
    include("test_utils.jl")
end

@testset "GraphUtils tests" begin
    include("test_graph_utils.jl")
end
@testset "Circuit Compilation" begin
    include("test_circuit_compilation.jl")
end

@testset "Lindblad" begin
    include("test_lindblad.jl")
end
