module OpenNetworks

export VectorizationNetworks

include("core/VDMNetworks.jl")
include("core/Gates.jl")
include("utilities/Utils.jl")
include("core/VectorizationNetworks.jl")
include("core/Channels.jl")
include("utilities/CustomParsing.jl")
include("utilities/GraphUtils.jl")
include("circuits/PreBuiltChannels.jl")
include("circuits/NoiseModels.jl")
include("circuits/ProgressSettings.jl")
include("circuits/NoisyCircuits.jl")
include("circuits/Circuits.jl")
include("circuits/Lindblad.jl")
include("circuits/Evolution.jl")

using .VDMNetworks
export VDMNetwork

using .Gates
export Gate

using .GraphUtils
export extract_adjacency_graph

using .NoisyCircuits
export NoisyCircuit

using .Lindblad
export trotterize

end
