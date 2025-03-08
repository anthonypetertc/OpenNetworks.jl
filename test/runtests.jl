
using Test
using OpenNetworks

include("pre-test.jl")

#To run tests with ARGS in the REPL:
#using Pkg;Pkg.activate(".");Pkg.test(test_args=["full"])

if "quick" in ARGS
    include("quick/runquicktests.jl")
else
    include("full/runfulltests.jl")
end
