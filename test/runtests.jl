
using Test
using OpenNetworks

include("pre-test.jl")

#To run tests with ARGS in the REPL:
#using Pkg;Pkg.activate(".");Pkg.test(test_args=["full"])

if "full" in ARGS
    include("full/runfulltests.jl")
else
    include("quick/runquicktests.jl")
end
