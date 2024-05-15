module NoiseModels


using ITensors
using ITensorNetworks
using OpenSystemsTools
using OpenNetworks: Channels, Utils, VectorizationNetworks, VDMNetworks

VDMNetwork = VDMNetworks.VDMNetwork

struct NoiseInstruction
    name_of_instruction:: String
    channel:: Channels.Channel
    index_ordering_of_channel:: Vector{<:ITensors.Index{}}
    name_of_gates:: Set{<:AbstractString}
    qubits_noise_applies_to:: Set{<:ITensors.Index{}}
    
    function NoiseInstruction(
        name_of_instruction:: String, 
        channel:: Channels.Channel,
        index_ordering_of_channel:: Vector{<:ITensors.Index{}}, 
        name_of_gates:: Set{<:AbstractString}, 
        qubits_noise_applies_to:: Set{<:ITensors.Index{}})

        new_tensor = copy(channel.tensor)
        for (i, index) in enumerate(index_ordering_of_channel)
            if !(index in inds(channel.tensor))
                throw("Index not in channel tensor.")
            elseif plev(index) != 0
                throw("Index has plev != 0.")
            else
                ref_index = ITensors.addtags(index, "position_$i")
                new_tensor *= ITensors.delta(index, ref_index)
                new_tensor *= ITensors.delta(index', ref_index')
                index_ordering_of_channel[i] = ref_index
            end
        end
        new(name_of_instruction, Channels.Channel(channel.name, new_tensor), index_ordering_of_channel, name_of_gates, qubits_noise_applies_to)
    end
end

function prepare_noise_for_gate(
    noise_instruction:: NoiseInstruction,
    Qubits:: Vector{<:ITensors.Index}
):: Channels.Channel
    new_tensor = deepcopy(noise_instruction.channel.tensor)
    for (i, qubit) in enumerate(Qubits)
        if !(qubit in noise_instruction.qubits_noise_applies_to)
            throw("Noise instruction does not apply to this qubit.")
        else
            noise_index = noise_instruction.index_ordering_of_channel[i]
            new_tensor *= ITensors.delta(noise_index, qubit)
            new_tensor *= ITensors.delta(noise_index', qubit')
        end
    end
    return Channels.Channel(noise_instruction.channel.name, new_tensor)
    end

struct NoiseModel
    noise_instructions:: Set{NoiseInstruction}
    siteinds:: ITensorNetworks.IndsNetwork
    vectorizedsiteinds:: ITensorNetworks.IndsNetwork
end





end