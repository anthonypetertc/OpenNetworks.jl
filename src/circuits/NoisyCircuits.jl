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
function add_noise_to_circuit(qc::Vector{<:Any}, noise_model::NoiseModels.NoiseModel, n_qubits:: Integer)::Vector{Channel}
    sites = noise_model.siteinds
    vsites = noise_model.vectorizedsiteinds
    #I should re-write some of my functions so that they don't require a reference state, only the site inds.
    ψ = ITensorNetwork(v -> "0", sites);
    ρ = VectorizationNetworks.vectorize_density_matrix(Utils.outer(ψ, ψ),ψ, vsites)
    channel_list = Vector{Channel}()
    for gate in qc
        qubits = gate["Qubits"]
        name = gate["Name"]
        params = gate["Params"]
        params = prepare_params(params)
        tensor = make_gate(name, qubits, params, sites)
        gate_channel = Channel(name, [tensor], ρ)
        count = 0
        for instruction in noise_model.noise_instructions
            if issubset(Set([vsites[(i,)][1] for i in qubits]), instruction.qubits_noise_applies_to) && name in instruction.name_of_gates
                #Need to introduce logging: println("Adding noise to gate.")
                if count !=0 throw("Multiple instructions for the same gate.") end
                count += 1
                index_ordering_of_gate = [vsites[(i,)][1] for i in qubits]
                noise_channel = NoiseModels.prepare_noise_for_gate(instruction, index_ordering_of_gate)
                gate_channel = Channels.compose(noise_channel, gate_channel)
            end
        end
        push!(channel_list, gate_channel)
    end
    return channel_list
end


function make_gate(name:: String, qubits:: Vector{<:Any}, params:: Dict, sites::ITensorNetworks.IndsNetwork):: ITensors.ITensor
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

function prepare_params(params:: Vector{<:Any}):: Dict
    if length(params) > 3 throw("Only 3 parameters or less.") end
    possible_keywords = [:θ, :ϕ, :λ]
    keywords = possible_keywords[1:length(params)]
    return Dict(zip(keywords, params))
end

end # module