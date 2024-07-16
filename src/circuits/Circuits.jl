module Circuits
export prepare_noiseless_circuit, run_compiled_circuit

using ITensorNetworks:
    ITensorNetworks,
    ITensorNetwork,
    norm_sqr_network,
    BeliefPropagationCache,
    update,
    VidalITensorNetwork,
    vertices,
    gauge_error,
    apply
using ITensors: ITensors, ITensor, inds, plev
using OpenNetworks: Channels, NoisyCircuits, ProgressSettings
using ProgressMeter
using SplitApplyCombine: group

default_progress_kwargs = ProgressSettings.default_progress_kwargs

function prepare_noiseless_circuit(
    qc::Vector{Dict{String,Any}}, sites::ITensorNetworks.IndsNetwork
)
    ψ = ITensorNetwork(v -> "0", sites)
    gate_list = Vector{ITensor}()
    for gate in qc
        if !haskey(gate, "Qubits") || !haskey(gate, "Name") || !haskey(gate, "Params")
            throw("Gate does not have the correct keys.")
        end
        qubits = gate["Qubits"]
        name = gate["Name"]
        params = gate["Params"]
        params = NoisyCircuits.prepare_params(params, name)
        tensor = NoisyCircuits.make_gate(name, qubits, params, sites)
        push!(gate_list, tensor)
    end
    return gate_list
end

function run_compiled_circuit(
    ψ::ITensorNetworks.ITensorNetwork,
    circuit::Vector{ITensor},
    regauge_frequency::Integer=50;
    progress_kwargs=default_progress_kwargs,
    cache_update_kwargs,
    apply_kwargs,
)::ITensorNetworks.ITensorNetwork
    norm_sqr = ITensorNetworks.norm_sqr_network(ψ)
    bp_cache = BeliefPropagationCache(norm_sqr, group(v -> v[1], vertices(norm_sqr)))
    bp_cache = update(bp_cache; maxiter=20)
    evolved_ψ = VidalITensorNetwork(ψ)

    p = Progress(length(circuit); progress_kwargs...)

    for (i, gate) in enumerate(circuit)
        #println("Applying gate $j from moment $i")
        indices = [ind for ind in inds(gate) if plev(ind) == 0]
        channel_sites = [Channels.find_site(ind, evolved_ψ) for ind in indices]
        #env = ITensorNetworks.environment(bp_cache, PartitionVertex.(channel_sites))
        if length(channel_sites) == 1
            #println("Applying single qubit gate")
            evolved_ψ[channel_sites[1]] = ITensors.apply(gate, evolved_ψ[channel_sites[1]])
        elseif length(channel_sites) == 2
            #println("Applying a two qubit gate.")
            evolved_ψ = ITensorNetworks.apply(gate, evolved_ψ; apply_kwargs...)
        else
            throw("Invalid gate: Only two qubit and one qubit gates are supported.")
        end
        ProgressMeter.next!(p)

        if i % regauge_frequency == 0
            ge = gauge_error(evolved_ψ)
            #println("Gauge error is $ge")
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
    return ψ_symm
end

end; # module
