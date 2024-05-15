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
    noise_model:: NoiseModel

    function NoisyCircuit(moments_list:: Vector{Vector{Channel}}, noise_model:: NoiseModel)
        return new(moments_list, noise_model)
    end

    function NoisyCircuit(list_of_dicts:: Vector{Dict{String, Any}}, noise_model:: NoiseModel)
        noisy_circuit = add_noise_to_circuit(list_of_dicts, noise_model)
        compressed_circuit = absorb_single_qubit_gates(noisy_circuit)
        moments_list = compile_into_moments(compressed_circuit, noise_model.vectorizedsiteinds)
        return new(moments_list, noise_model)
    end
end


function compile_into_moments!(channel_list::Vector{Channel}, siteinds::ITensorNetworks.IndsNetwork)::Vector{Vector{Channel}}
    sites = Set{ITensors.Index{<:Any}}()
    for key in keys(siteinds.data_graph.vertex_data)
        push!(sites, siteinds[key][1])
    end
    moments_list = Vector{Vector{Channel}}()
    current_moment = Vector{Channel}()
    current_moment_inds = Set{ITensors.Index{<:Any}}()
    while channel_list != []
        current_channel = pop!(channel_list)
        current_inds = Set([ind for ind in inds(current_channel.tensor) if plev(ind) == 0])
        if current_inds ∩ current_moment_inds == Set()
            pushfirst!(current_moment, current_channel)
            current_moment_inds = current_moment_inds ∪ current_inds
        else
            push!(channel_list, current_channel)
            pushfirst!(moments_list, current_moment)
            current_moment = Vector{Channel}()
            current_moment_inds = Set{ITensors.Index{<:Any}}()
        end
    end
    pushfirst!(moments_list, current_moment)
    return moments_list
end
    
compile_into_moments(channel_list::Vector{Channel}, siteinds::ITensorNetworks.IndsNetwork) = compile_into_moments!(deepcopy(channel_list), siteinds)

function absorb_single_qubit_gates(channel_list:: Vector{Channel}):: Vector{Channel}
    new_channel_list = Vector{Channel}()
    single_qubit_list = Vector{Channel}()
    index_list = Vector{Any}()
    for channel in channel_list
        if length(inds(channel.tensor)) == 2
            prepend!(single_qubit_list, [channel])
            gate_index = [ind for ind in inds(channel.tensor) if plev(ind) == 0]
            @assert length(gate_index) == 1
            prepend!(index_list, gate_index)
        else
            new_channel = deepcopy(channel)
            locations_to_remove = []
            for (i, ind) in enumerate(index_list)
                if ind in inds(channel.tensor)
                    new_channel = Channels.compose(new_channel, single_qubit_list[i])
                    push!(locations_to_remove, i)
                end
            end
            for i in reverse(locations_to_remove)
                deleteat!(single_qubit_list, i)
                deleteat!(index_list, i)
            end
            push!(new_channel_list, new_channel)
        end
    end
    squeezed_single_qubits, new_index_list = squeeze_single_qubit_gates(single_qubit_list, index_list)
    for j in 0:(length(new_channel_list)-1)
        locations_to_remove = []
        for (i, index) in enumerate(new_index_list)
            if index in inds(new_channel_list[end-j].tensor)
                new_channel_list[end-j] = Channels.compose(squeezed_single_qubits[i], new_channel_list[end-j])
                push!(locations_to_remove, i)
            end
        end
        for i in reverse(locations_to_remove)
            deleteat!(squeezed_single_qubits, i)
            deleteat!(new_index_list, i)
        end
    end
    append!(new_channel_list, squeezed_single_qubits)
    return new_channel_list
end



function squeeze_single_qubit_gates(channel_list:: Vector{Channel}, index_list::Vector{<:Any}):: Tuple{Vector{Channel}, Vector{<:Any}}
    new_channel_list = Vector{Channel}()
    new_index_list = Vector{ITensors.Any}()
    index_set = Set(index_list)
    for ind in index_set
        locations = [i for i in 1:length(index_list) if index_list[i] == ind]
        new_channel = deepcopy(channel_list[locations[1]])
        for i in locations[2:end]
            new_channel = Channels.compose(new_channel, channel_list[i])
        end
        push!(new_channel_list, new_channel)
        push!(new_index_list, ind)
    end
    @assert length(new_channel_list) == length(new_index_list)
    return new_channel_list, new_index_list
end

#Would also be good to write another function that adds noise to a circuit represented as a list of channels or ITensors instead as 
#the list of dicts that I am using at the moment. 

function add_noise_to_circuit(qc::Vector{Dict{String, Any}}, noise_model::NoiseModels.NoiseModel)::Vector{Channel}
    n_qubits = length(keys(noise_model.siteinds.data_graph.vertex_data))
    if (GraphUtils.extract_adjacency_graph(qc, n_qubits) != noise_model.siteinds.data_graph.underlying_graph)
         throw("The circuit and the noiseNoisyCircuits model do not have the same sites.") end
    sites = noise_model.siteinds
    vsites = noise_model.vectorizedsiteinds
    #I should re-write some of my functions so that they don't require a reference state, only the site inds.
    ψ = ITensorNetwork(v -> "0", sites);
    ρ = VectorizationNetworks.vectorize_density_matrix(Utils.outer(ψ, ψ),ψ, vsites)
    channel_list = Vector{Channel}()
    for gate in qc
        if !haskey(gate, "Qubits") || !haskey(gate, "Name") || !haskey(gate, "Params")
            throw("Gate does not have the correct keys.")
        end
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
    ss = [sites[(qubit,)] for qubit in qubits] 
    if length(qubits) == 1
        tensor = op(name, ss[1][1]; params...) 
    elseif length(qubits) == 2
        tensor = op(name, ss[1][1], ss[2][1]; params...)
    elseif length(qubits) == 3
        tensor = op(name, ss[1][1], ss[2][1], ss[3][1]; params...)
    else
        throw("Only 3 qubit gates or less.")
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