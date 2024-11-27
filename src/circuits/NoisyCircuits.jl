module NoisyCircuits
export NoisyCircuit

using ITensors: ITensors, inds, op, plev, ITensor, Index
using ITensorNetworks:
    ITensorNetworks,
    apply,
    environment,
    ITensorNetwork,
    BeliefPropagationCache,
    norm_sqr_network,
    VidalITensorNetwork,
    siteinds,
    update,
    gauge_error,
    vertices

using ITensorsOpenSystems
using OpenNetworks:
    Channels,
    Utils,
    VectorizationNetworks,
    VDMNetworks,
    NoiseModels,
    GraphUtils,
    CustomParsing,
    ProgressSettings
using SplitApplyCombine: group
using ProgressMeter

VDMNetwork = VDMNetworks.VDMNetwork
NoiseModel = NoiseModels.NoiseModel
Channel = Channels.Channel
default_progress_kwargs = ProgressSettings.default_progress_kwargs

struct NoisyCircuit{V}
    channel_list::Vector{Channel}
    fatsites::ITensorNetworks.IndsNetwork{V}
    n_gates::Int
end

#=
function NoisyCircuit(
    moments_list::Vector{Vector{Channel}},
    fatsites::ITensorNetworks.IndsNetwork,
    n_gates::Int,
)
    return new(moments_list, fatsites, n_gates)
end=#

function NoisyCircuit(
    parsedcircuit::Vector{CustomParsing.ParsedGate}, noise_model::NoiseModel
)
    noisy_circuit = add_noise_to_circuit(parsedcircuit, noise_model)
    channel = first(noisycircuit)
    firstind = first(inds(channel.tensor))
    compressedcircuit = absorb_single_qubit_gates(noisy_circuit, firstind)
    #=moments_list, n_gates = compile_into_moments(
         compressed_circuit, noise_model.vectorizedsiteinds
     )=#
    n_gates = length(compressedcircuit)
    return NoisyCircuit(compressedcircuit, noise_model.vectorizedsiteinds, n_gates)
end

function NoisyCircuit(channel_list::Vector{Channel}, fatsites::ITensorNetworks.IndsNetwork)
    channel = first(noisycircuit)
    firstind = first(inds(channel.tensor))
    compressedcircuit = absorb_single_qubit_gates(channel_list, firstind)
    #moments_list, n_gates = compile_into_moments(compressed_circuit, fatsites)
    n_gates = length(compressedcircuit)
    return NoisyCircuit(compressedcircuit, fatsites, n_gates)
end

function ITensors.apply(
    ρ::VDMNetwork, noisy_circuit::NoisyCircuit; apply_kwargs...
)::VDMNetwork
    for channel in noisy_circuit.channel_list
        ρ = Channels.apply(channel, ρ; apply_kwargs...)
    end
    return ρ
end

function run_circuit(
    ρ::VDMNetwork,
    noisy_circuit::NoisyCircuit,
    regauge_frequency::Integer=50;
    progress_kwargs=default_progress_kwargs,
    cache_update_kwargs,
    apply_kwargs,
)::VDMNetwork
    norm_sqr = norm_sqr_network(ρ.network)
    #Simple Belief Propagation Grouping
    bp_cache = BeliefPropagationCache(norm_sqr, group(v -> v[1], vertices(norm_sqr)))
    bp_cache = update(bp_cache; maxiter=20)
    evolved_ψ = VidalITensorNetwork(ρ.network)

    p = Progress(noisy_circuit.n_gates; progress_kwargs...)
    for (j, gate) in enumerate(noisy_circuit.channel_list)
        #println("Applying gate $j from moment $i")
        indices = [ind for ind in inds(gate.tensor) if plev(ind) == 0]
        channel_sites = [Channels.find_site(ind, evolved_ψ) for ind in indices]
        if length(channel_sites) == 1
            #println("Applying single qubit gate")
            evolved_ψ[channel_sites[1]] = ITensors.apply(gate, evolved_ψ[channel_sites[1]])
        elseif length(channel_sites) == 2
            #println("Applying a two qubit gate.")
            evolved_ψ = ITensorNetworks.apply(gate.tensor, evolved_ψ; apply_kwargs...)
        else
            throw("Invalid gate: Only two qubit and one qubit gates are supported.")
        end

        ProgressMeter.next!(p)
        if i % regauge_frequency == 0
            ge = ITensorNetworks.gauge_error(evolved_ψ)
            println("Gauge error is $ge")
            if ge > 1e-6
                cache_ref = Ref{BeliefPropagationCache}(bp_cache)
                ψ_symm = ITensorNetwork(evolved_ψ; (cache!)=cache_ref)
                evolved_ψ = VidalITensorNetwork(
                    ψ_symm;
                    (cache!)=cache_ref,
                    cache_update_kwargs=(; cache_update_kwargs...),
                )
            end
        end
    end

    cache_ref = Ref{BeliefPropagationCache}(bp_cache)
    global ψ_symm = ITensorNetwork(evolved_ψ; (cache!)=cache_ref)
    evolved_ρ = VDMNetworks.VDMNetwork(ψ_symm, ρ.unvectorizedinds)
    return evolved_ρ
end

function compile_into_moments!(
    channel_list::Vector{Channel}, siteinds::ITensorNetworks.IndsNetwork
)::Tuple{Vector{Vector{Channel}},Int}
    sites = Set{ITensors.Index{<:Any}}()
    for key in keys(siteinds.data_graph.vertex_data)
        push!(sites, siteinds[key][1])
    end
    moments_list = Vector{Vector{Channel}}()
    current_moment = Vector{Channel}()
    current_moment_inds = Set{ITensors.Index{<:Any}}()
    n_gates = length(channel_list)
    while channel_list != []
        current_moment, current_moment_inds, channel_list = single_pass_compile_into_moments!(
            channel_list, current_moment, current_moment_inds
        )
        pushfirst!(moments_list, current_moment)
        current_moment = Vector{Channel}()
        current_moment_inds = Set{ITensors.Index{<:Any}}()
    end
    return moments_list, n_gates
end

function single_pass_compile_into_moments!(
    channel_list::Vector{Channel},
    current_moment::Vector{Channel},
    current_moment_inds::Set{ITensors.Index{<:Any}},
)::Tuple{Vector{Channel},Set{ITensors.Index{<:Any}},Vector{Channel}}
    if channel_list == []
        throw("Empty channel list.")
    end
    for (i, channel) in enumerate(reverse(channel_list))
        current_inds = inds(channel.tensor; plev=0)
        if isempty(current_inds ∩ current_moment_inds)
            pushfirst!(current_moment, channel)
            current_moment_inds = current_moment_inds ∪ current_inds
            deleteat!(channel_list, findlast(==(channel), channel_list))
        else
            current_moment_inds = current_moment_inds ∪ current_inds
        end
    end
    return current_moment, current_moment_inds, channel_list
end

function compile_into_moments(
    channel_list::Vector{Channel}, siteinds::ITensorNetworks.IndsNetwork
)
    return compile_into_moments!(deepcopy(channel_list), siteinds)
end
#=
function absorb_single_qubit_gates(channel_list::Vector{Channel})::Vector{Channel}
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
    squeezed_single_qubits, new_index_list = squeeze_single_qubit_gates(
        single_qubit_list, index_list
    )
    for j in 0:(length(new_channel_list) - 1)
        locations_to_remove = []
        for (i, index) in enumerate(new_index_list)
            if index in inds(new_channel_list[end - j].tensor)
                new_channel_list[end - j] = Channels.compose(
                    squeezed_single_qubits[i], new_channel_list[end - j]
                )
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

function squeeze_single_qubit_gates(
    channel_list::Vector{Channel}, index_list::Vector{<:Any}
)::Tuple{Vector{Channel},Vector{<:Any}}
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
=#

function absorb_single_qubit_gates(
    channel::Channels.Channel,
    index_list::Vector{ITensors.Index{V}},
    single_qubit_list::Vector{Channel},
    new_channel_list::Vector{Channel},
)::Vector{Channel} where {V}
    #= Takes a multi-qubit channel, a list of single qubit gates & indices on which they act.
    It then absorbs any single qubit gates that act on one of the sites that the multiqubit gates
    act on, and then adds the combined channel into new_channel_list. =#

    new_channel = deepcopy(channel)
    locations_to_remove = Vector{Int64}()
    for (i, ind) in enumerate(index_list)
        if ind in indices
            new_channel = Channels.compose(new_channel, single_qubit_list[i])
            push!(locations_to_remove, i)
        end
    end
    for i in reverse(locations_to_remove)
        deleteat!(single_qubit_list, i)
        deleteat!(index_list, i)
    end
    push!(new_channel_list, new_channel)
    return new_channel_list
end

function reverse_absorb_single_qubit_gates(
    new_channel_list::Vector{Channels.Channel},
    new_index_list::Vector{ITensors.Index{V}},
    squeezed_single_qubits::Vector{Channels.Channel},
)::Vector{Channels.Channel} where {V}
    #= Takes a list of channels and a list of single qubit gates acting after those channels,
    absorbs any that can be absorbed into into the channels, and then appends the rest to the
    new_channel_list =#

    for j in 0:(length(new_channel_list) - 1)
        locations_to_remove = Vector{Int64}()
        for (i, index) in enumerate(new_index_list)
            if index in inds(new_channel_list[end - j].tensor)
                new_channel_list[end - j] = Channels.compose(
                    squeezed_single_qubits[i], new_channel_list[end - j]
                )
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

function absorb_single_qubit_gates(
    channel_list::Vector{Channels.Channel}, index::ITensors.Index{V}
)::Vector{Channels.Channel} where {V}
    new_channel_list = Vector{Channels.Channel}()
    single_qubit_list = Vector{Channels.Channel}()
    index_list = Vector{ITensors.Index{V}}()
    for channel in channel_list
        indices = inds(channel.tensor)
        sites = filter(indices; plev=0)
        if length(indices) == 2
            #If a single site channel, add it to the single_qubit_list.
            prepend!(single_qubit_list, [channel])
            prepend!(index_list, sites)
        else
            #If a multi-site channel, absorb as many single qubit channels into it as possible.
            new_channel_list = absorb_single_qubits(
                channel, index_list, single_qubit_list, new_channel_list
            )
        end
    end
    # Once we've run out of multi-qubit channels, combine all single qubit channels
    # acting on the same sites.
    squeezed_single_qubits, new_index_list = squeeze_single_qubit_gates(
        single_qubit_list, index_list
    )
    # Now reverse-absorb these into the list of new channels,
    # to get the final list of channels.
    new_channel_list = reverse_absorb_single_qubit_gates(
        new_channel_list, new_index_list, squeeze_single_qubit_gates
    )
    return new_channel_list
end

function squeeze_single_qubit_gates(
    channel_list::Vector{Channel}, index_list::Vector{ITensors.Index{V}}
)::Tuple{Vector{Channel},Vector{ITensors.Index{V}}} where {V}
    #= Takes a list of single qubit channels and a list of indices corresponding to their sites,
    composes all the single qubit channels that act on the same site, and returns the list of composed
    single qubit gates, and new list of corresponding site indices. =#
    new_channel_list = Vector{Channel}()
    new_index_list = Vector{ITensors.Index{V}}()
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

function add_noise_to_circuit(
    qc::Vector{CustomParsing.ParsedGate}, noise_model::NoiseModel{V}
)::Vector{Channel} where {V}
    if (
        GraphUtils.extract_adjacency_graph(qc) !=
        noise_model.siteinds.data_graph.underlying_graph
    )
        throw("The circuit and the noiseNoisyCircuits model do not have the same sites.")
    end
    sites = noise_model.siteinds
    vsites = noise_model.vectorizedsiteinds
    ψ = ITensorNetwork(v -> "0", sites)::ITensorNetwork{V}
    ρ = VDMNetworks.VDMNetwork(Utils.outer(ψ, ψ), sites, vsites)::VDMNetwork{V}
    channel_list = Vector{Channels.Channel}()
    for gate in qc
        qubits = gate.qubits
        name = gate.name
        params_dict = gate.params
        params = prepare_params(params_dict, name)
        tensor = make_gate(name, qubits, params, sites)
        gate_channel = Channels.Channel(name, [tensor], ρ)::Channels.Channel
        count = 0
        for instruction in noise_model.noise_instructions
            if issubset(
                Set([vsites[i][1] for i in qubits]), instruction.qubits_noise_applies_to
            ) && name in instruction.name_of_gates
                if count != 0
                    throw("Multiple instructions for the same gate.")
                end
                count += 1
                index_ordering_of_gate = [vsites[i][1] for i in qubits]
                noise_channel = NoiseModels.prepare_noise_for_gate(
                    instruction, index_ordering_of_gate
                )
                gate_channel = Channels.compose(noise_channel, gate_channel)
            end
        end
        push!(channel_list, gate_channel)
    end
    return channel_list
end

#=
function add_noise_to_circuit(
    qc::Vector{Dict{String,Any}}, noise_model::NoiseModel
)::Vector{Channel}
    if (
        GraphUtils.extract_adjacency_graph(qc) !=
        noise_model.siteinds.data_graph.underlying_graph
    )
        throw("The circuit and the noiseNoisyCircuits model do not have the same sites.")
    end
    sites = noise_model.siteinds
    vsites = noise_model.vectorizedsiteinds
    ψ = ITensorNetwork(v -> "0", sites)
    ρ = VectorizationNetworks.vectorize_density_matrix(Utils.outer(ψ, ψ), sites, vsites)
    channel_list = Vector{Channel}()
    for gate in qc
        if !haskey(gate, "Qubits") || !haskey(gate, "Name") || !haskey(gate, "Params")
            throw("Gate does not have the correct keys.")
        end
        qubits = gate["Qubits"]
        name = gate["Name"]
        params = gate["Params"]
        params = prepare_params(params, name)
        tensor = make_gate(name, qubits, params, sites)
        gate_channel = Channel(name, [tensor], ρ)
        count = 0
        for instruction in noise_model.noise_instructions
            if issubset(
                Set([vsites[(i,)][1] for i in qubits]), instruction.qubits_noise_applies_to
            ) && name in instruction.name_of_gates
                #Need to introduce logging: println("Adding noise to gate.")
                if count != 0
                    throw("Multiple instructions for the same gate.")
                end
                count += 1
                index_ordering_of_gate = [vsites[(i,)][1] for i in qubits]
                noise_channel = NoiseModels.prepare_noise_for_gate(
                    instruction, index_ordering_of_gate
                )
                gate_channel = Channels.compose(noise_channel, gate_channel)
            end
        end
        push!(channel_list, gate_channel)
    end
    return channel_list
end
=#

function make_gate(
    name::String,
    qubits::Vector{Int64},
    params::Dict{Symbol,Float64},
    sites::ITensorNetworks.IndsNetwork,
)::ITensors.ITensor
    ss = [sites[qubit] for qubit in qubits]
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

function prepare_params(params::Vector{Float64}, name::String)::Dict{Symbol,Float64}
    if length(params) > 3
        throw("Only 3 parameters or less.")
    end
    if (name == "Rzz") || (name == "Rxx") || (name == "Ryy") || (name == "Phase")
        if length(params) != 1
            throw("Incorrect number of parameters for gate $name.")
        end
        return Dict(:ϕ => params[1] / 2)
    end
    possible_keywords = [:θ, :ϕ, :λ]
    keywords = possible_keywords[1:length(params)]
    return Dict(zip(keywords, params))
end

end; # module
