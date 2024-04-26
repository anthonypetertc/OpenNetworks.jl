
using Test
using OpenBP


@testset "Channel tests" begin
    include("test_channel.jl")
end

@testset "Vectorization tests" begin
    include("test_vectorize_density_matrix.jl")
end




