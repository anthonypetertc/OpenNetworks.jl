module Circuits
export prepare_noiseless_circuit, run_circuit

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
using OpenNetworks: Channels, NoisyCircuits
using ProgressMeter
using SplitApplyCombine: group

run_circuit = NoisyCircuits.run_circuit

function prepare_noiseless_circuit(
    qc::Vector{Dict{String,Any}}, sites::ITensorNetworks.IndsNetwork
)
    #g = GraphUtils.extract_adjacency_graph(qc, n_qubits)
    #vsites = ITensorNetworks.siteinds("QubitVec", g)
    #I should re-write some of my functions so that they don't require a reference state, only the site inds.
    ψ = ITensorNetwork(v -> "0", sites)
    #ρ = VectorizationNetworks.vectorize_density_matrix(Utils.outer(ψ, ψ),ψ, vsites)
    #channel_list = Vector{Channel}()
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
        #gate_channel = Channel(name, [tensor], ρ)
        #push!(channel_list, gate_channel)
        push!(gate_list, tensor)
    end
    return gate_list
end

function run_compiled_circuit(
    ψ::ITensorNetworks.ITensorNetwork,
    circuit::Vector{ITensor},
    regauge_frequency::Integer=50;
    cache_update_kwargs,
    apply_kwargs,
)::ITensorNetworks.ITensorNetwork
    norm_sqr = ITensorNetworks.norm_sqr_network(ψ)
    bp_cache = BeliefPropagationCache(norm_sqr, group(v -> v[1], vertices(norm_sqr)))
    bp_cache = update(bp_cache; maxiter=20)
    evolved_ψ = VidalITensorNetwork(ψ)

    @showprogress dt = 1 desc = "Applying circuit..." for (i, gate) in enumerate(circuit)
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
        ge = gauge_error(evolved_ψ)
        #println("Gauge error is $ge")
        if ge > 1e-6 && i % regauge_frequency == 0
            cache_ref = Ref{BeliefPropagationCache}(bp_cache)
            ψ_symm = ITensorNetwork(evolved_ψ; (cache!)=cache_ref)
            evolved_ψ = VidalITensorNetwork(
                ψ_symm; (cache!)=cache_ref, cache_update_kwargs=(; cache_update_kwargs...)
            )
        end
    end

    cache_ref = Ref{BeliefPropagationCache}(bp_cache)
    global ψ_symm = ITensorNetwork(evolved_ψ; (cache!)=cache_ref)
    return ψ_symm
end

end; # module
