@testset "Channel tests" begin
    include("full_test_channel.jl")
end

@testset "Vectorization tests" begin
    include("full_test_vectorize.jl")
end

@testset "Utils tests" begin
    include("full_test_utils.jl")
end

@testset "GraphUtils tests" begin
    include("full_test_graph_utils.jl")
end

@testset "Circuit Compilation" begin
    include("full_test_compilation.jl")
end

@testset "Circuit Evolution" begin
    include("full_test_evolution.jl")
end

@testset "Lindblad Evolution" begin
    include("full_test_lindblad.jl")
end

@testset "TEBD evolution" begin
    include("full_test_tebd.jl")
end

#=
@testset "Tutorials" begin
    include("full_test_tutorials.jl")
end =#
