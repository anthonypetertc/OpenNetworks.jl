
using Test
using LibGit2
using FilePathsBase

mktempdir() do temp_dir
    println("Cloning the repository into temporary directory: $temp_dir")

    # Define the repository URL
    repo_url = "git@github.com:anthonypetertc/OpenNetworks.jl.git"

    # Clone the repository to the temporary directory
    LibGit2.clone(repo_url, temp_dir)

    # Path to tutorial scripts
    tutorial1_path = joinpath(joinpath(temp_dir, "tutorials"), "tutorial1.jl")
    tutorial2_path = joinpath(joinpath(temp_dir, "tutorials"), "tutorial2.jl")
    tutorial3_path = joinpath(joinpath(temp_dir, "tutorials"), "tutorial3.jl")
    tutorial4_path = joinpath(joinpath(temp_dir, "tutorials"), "tutorial4.jl")
    tutorial5_path = joinpath(joinpath(temp_dir, "tutorials"), "tutorial5.jl")
    tutorial6_path = joinpath(joinpath(temp_dir, "tutorials"), "tutorial6.jl")

    # Test set for Tutorial 1
    @testset "Run Tutorial 1" begin
        include(tutorial1_path)  # Run the script for tutorial1
        test_results = Dict{Any,Any}(
            (1, 2) => 0.871970464257344 + 0.0im,
            (1, 1) => 0.876951707349454 + 0.0im,
            (2, 2) => 0.7991453026994829 + 0.0im,
            (2, 1) => 0.872221969099563 + 0.0im,
        )
        @test keys(results) == keys(test_results)
        for key in keys(results)
            @test results[key] ≈ test_results[key]
        end
        println("Tutorial 1 ran successfully.")
    end

    #Test set for Tutorial 2
    @testset "Run Tutorial 2" begin
        include(tutorial2_path)  # Run the script for tutorial2
        println("Tutorial 2 ran successfully.")
    end

    #Test set for Tutorial 3
    @testset "Run Tutorial 3" begin
        include(tutorial3_path)  # Run the script for tutorial2
        println("Tutorial 3 ran successfully.")
    end

    #Test set for Tutorial 4
    @testset "Run Tutorial 4" begin
        include(tutorial4_path)  # Run the script for tutorial2
        @test isapprox(overlap, 1.0; rtol=1e-5)
        println("Tutorial 2 ran successfully.")
    end

    #Test set for Tutorial 5
    @testset "Run Tutorial 5" begin
        include(tutorial5_path)  # Run the script for tutorial2
        test_results = Dict{Any,Any}(
            (1, 2) => 0.864221047830895 + 0.0im,
            (3, 1) => 0.8583918479554361 + 0.0im,
            (1, 3) => 0.8584138414167275 + 0.0im,
            (1, 4) => 0.9258432724371759 + 0.0im,
            (3, 2) => 0.864221048465909 + 0.0im,
            (3, 3) => 0.8584138414151525 + 0.0im,
            (2, 1) => 0.8593689445599892 + 0.0im,
            (3, 4) => 0.9258432724371519 + 0.0im,
            (2, 2) => 0.525989389916663 + 0.0im,
            (2, 3) => 0.8593909258704917 + 0.0im,
            (0, 0) => 0.7941335848658215 + 0.0im,
            (2, 4) => 0.9258434250648341 + 0.0im,
            (1, 1) => 0.858391847933819 + 0.0im,
        )
        @test keys(results) == keys(test_results)
        for key in keys(results)
            @test results[key] ≈ test_results[key]
        end
        println("Tutorial 5 ran successfully.")
    end

    #Test set for Tutorial 6
    @testset "Run Tutorial 6" begin
        include(tutorial6_path)  # Run the script for tutorial2
        test_results = Dict{Any,Any}(
            5 => 0.8573544832576087 + 0.0im,
            16 => 0.7941320660423742 + 0.0im,
            20 => 0.8573545060508532 + 0.0im,
            12 => 0.8573547878770773 + 0.0im,
            8 => 0.7941320664102999 + 0.0im,
            17 => 0.8573545059021814 + 0.0im,
            1 => 0.7959862049463845 + 0.0im,
            19 => 0.8573544834772303 + 0.0im,
            22 => 0.8573545065805296 + 0.0im,
            6 => 0.8573654911019087 + 0.0im,
            11 => 0.8573653772192178 + 0.0im,
            9 => 0.8573545059267764 + 0.0im,
            14 => 0.8573654653778019 + 0.0im,
            3 => 0.8593681143005779 + 0.0im,
            7 => 0.8573545061600791 + 0.0im,
            4 => 0.7941323583939681 + 0.0im,
            13 => 0.8573547880730681 + 0.0im,
            15 => 0.8573545056434441 + 0.0im,
            2 => 0.5259790300149363 + 0.0im,
            10 => 0.8573654028214092 + 0.0im,
            18 => 0.8573654918388969 + 0.0im,
            21 => 0.8573654922748036 + 0.0im,
        )
        @test keys(results) == keys(test_results)
        for key in keys(results)
            @test results[key] ≈ test_results[key]
        end
        println("Tutorial 6 ran successfully.")
    end

    # The temporary directory is automatically deleted at the end of the `mktempdir` block
end
