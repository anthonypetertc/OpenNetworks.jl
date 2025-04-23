module Evolution
export run_circuit

using ITensors: ITensors, inds, op, plev, ITensor
using ITensorsOpenSystems: Vectorization.VectorizedDensityMatrix
using ITensorNetworks:
    ITensorNetworks,
    apply,
    ITensorNetwork,
    BeliefPropagationCache,
    norm_sqr_network,
    VidalITensorNetwork,
    update,
    gauge_error,
    vertices
using OpenNetworks:
    Channels,
    Utils,
    NoiseModels,
    GraphUtils,
    Gates.Gate,
    ProgressSettings.default_progress_kwargs,
    Utils.findindextype,
    VDMNetworks.VDMNetwork,
    Channels.Channel,
    GraphUtils.islinenetwork,
    NoisyCircuits.NoisyCircuit,
    NoisyCircuits.compile_into_moments,
    Utils.findsite
using SplitApplyCombine: group
using ProgressMeter

"""
    apply(ρ::VDMNetwork, noisy_circuit::NoisyCircuit; apply_kwargs...)
    
    Arguments
    ρ::VDMNetwork
        The vectorized density matrix network to which the circuit is applied.
    noisy_circuit::NoisyCircuit
        The noisy circuit to be applied to the density matrix network.
    apply_kwargs...
        Additional keyword arguments to pass to the apply function.

    Applies the Channels in the noisy circuit to the vectorized density matrix network, 
    one by one, in the order they are defined in the circuit.

"""


function ITensors.apply(
    ρ::VDMNetwork, noisy_circuit::NoisyCircuit; apply_kwargs...
)::VDMNetwork
    for channel in noisy_circuit.channel_list
        ρ = apply(channel, ρ; apply_kwargs...)
    end
    return ρ
end

"""
    run_circuit(ρ::VDMNetwork, noisy_circuit::NoisyCircuit, regauge_frequency::Integer=50; progress_kwargs=default_progress_kwargs, cache_update_kwargs, apply_kwargs)

    Arguments
    ρ::VDMNetwork
        The vectorized density matrix network to which the circuit is applied.
    noisy_circuit::NoisyCircuit
        The noisy circuit to be applied to the density matrix network.
    regauge_frequency::Integer
        The frequency at which to regauge the network.
    progress_kwargs::Dict{Symbol, Any}
        Additional arguments for the progress bar.
    cache_update_kwargs::Dict{Symbol, Any}
        Additional arguments for the cache update.
    apply_kwargs...
        Additional keyword arguments to pass to the apply function.

   Evolves the vectorized density matrix network by the given noisy circuit,
   using the Simple Update algorithm, and BP regauing at regular intervals.

"""

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
    bp_cache = update(bp_cache; cache_update_kwargs...)
    evolved_ψ = VidalITensorNetwork(ρ.network)

    p = Progress(noisy_circuit.n_gates; progress_kwargs...)
    for (j, gate) in enumerate(noisy_circuit.channel_list)
        #println("Applying gate $j from moment $i")
        indices = collect(inds(gate.tensor; plev=0))
        channel_sites = [findsite(evolved_ψ, ind) for ind in indices]
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
        if j % regauge_frequency == 0
            ge = ITensorNetworks.gauge_error(evolved_ψ)
            #println("Gauge error is $ge")
            if ge > cache_update_kwargs[:tol]
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
    ψ_symm = ITensorNetwork(evolved_ψ; (cache!)=cache_ref)
    evolved_ρ = VDMNetwork(ψ_symm, ρ.unvectorizedinds)
    return evolved_ρ
end

"""
    run_circuit(ρ::VectorizedDensityMatrix, noisy_circuit::NoisyCircuit; apply_kwargs...)

    Arguments
    ρ::VectorizedDensityMatrix
        The vectorized density matrix to which the circuit is applied.
    noisy_circuit::NoisyCircuit
        The noisy circuit to be applied to the density matrix.
    apply_kwargs...
        Additional keyword arguments to pass to the apply function.

    Evolves the vectorized density matrix by the given noisy circuit,
    using TEBD. (Only works for nearest neighbour circuits on a line).

"""

function run_circuit(
    ρ::VectorizedDensityMatrix, noisy_circuit::NoisyCircuit; apply_kwargs...
)
    if !(islinenetwork(noisy_circuit.fatsites))
        throw(
            "Circuit is not nearest neighbour on a line! Recompile circuit or use belief propagation.",
        )
    end
    ρev = deepcopy(ρ)
    moments_list, _ = compile_into_moments(noisy_circuit)
    for moment in moments_list
        #moment_tensor = [channel.tensor for channel in moment] #overload apply so that I can apply a sequence of channels in one go.
        ρev = apply(moment, ρev; apply_kwargs...)
    end
    return ρev
end

end
