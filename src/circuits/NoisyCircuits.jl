module NoisyCircuits


using ITensors
using ITensorNetworks
using OpenSystemsTools
using OpenNetworks: Channels, Utils, VectorizationNetworks, VDMNetworks, NoiseModels, GraphUtils

VDMNetwork = VDMNetworks.VDMNetwork
NoiseModel = NoiseModels.NoiseModel
Channel = Channels.Channel

struct NoisyCircuit
    moments_list:: Vector{Vector{Channel}}
    sites:: ITensorNetworks.IndsNetwork
    noise_model:: NoiseModel #Not sure that I actually need this.
end

#noise_model::NoiseModel, 
function add_noise_to_circuit(qc::Vector{}, noise_model::NoiseModel, n_qubits:: Integer)::Vector{Channel}
    G = GraphUtils.extract_adjacency_graph(qc, n_qubits)
    sites = ITensorNetworks.siteinds("Qubit", G)
    vsites = ITensorNetworks.siteinds("QubitVec", G)
    #I should re-write some of my functions so that they don't require a reference state, only the site inds.
    ψ = ITensorNetwork(v -> "0", sites);
    ρ = VectorizationNetworks.vectorize_density_matrix(Utils.outer(ψ, ψ),ψ, vsites)
    channel_list = Vector{Channel}()
    for gate in qc
        qubits = gate["Qubits"]
        name = gate["Name"]
        params = gate["Params"]
        tensor = make_gate(name, qubits, params, sites)
        gate_channel = Channel(name, [tensor], ρ)
        count = 0
        for instruction in noise_model.noise_instructions
            if issubset(Set(qubits), instruction.qubits_noise_applies_to) && name in instruction.name_of_gates
                if count !=0 
                    throw("Multiple instructions for the same gate.")
                end
                count += 1
                noisy_gate_channel = Channels.compose(post = instruction.channel, pre = gate_channel)
                push!(channel_list, noisy_gate_channel)
            end
        end
    end
    return channel_list
end


function make_gate(name:: String, qubits:: Vector{}, params:: Dict, sites::ITensorNetworks.IndsNetwork):: ITensors.ITensor
    #if params != [] throw("Not implemented yet.") end
    ss = [sites[(qubit,)] for qubit in qubits] 
    if length(qubits) == 1
        tensor = op(name, ss[1][1]; params...) 
    elseif length(qubits) == 2
        tensor = op(name, ss[1][1], ss[2][1]; params...)
    elseif length(qubits) == 3
        tensor = op(name, ss[1][1], ss[2][1], ss[3][1]; params...)
    else
        throw("Only 4 qubit gates or less.")
    end
    return tensor
end

end
