module NoiseModels


using ITensors
using ITensorNetworks
using OpenSystemsTools
using OpenNetworks: Channels, Utils, VectorizationNetworks, VDMNetworks

VDMNetwork = VDMNetworks.VDMNetwork

struct NoiseInstruction
    name_of_instruction:: String
    channel:: Channels.Channel
    name_of_gates:: Set{String}
    qubits_noise_applies_to:: Set{Index{}}
end

struct NoiseModel
    noise_instructions:: Vector{NoiseInstruction}
    siteinds:: ITensorNetworks.IndsNetwork
end





end