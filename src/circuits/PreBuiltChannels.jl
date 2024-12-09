module PreBuiltChannels
export depolarizing, dephasing
using ITensors
using ITensorMPS
using ITensorsOpenSystems:
    Vectorization, ITensorsOpenSystems, Vectorization.VectorizedDensityMatrix
using ITensorNetworks
using OpenNetworks: Channels.Channel, VDMNetworks.VDMNetwork

function depolarizing(p::Real, sites::Vector{<:ITensors.Index{}})::Channel
    #= Creates a depolarizing channel for a given density matrix. =#

    if !(0 <= p <= 1)
        throw("parameter p must be between 0 and 1.")
    end
    for site in sites
        @assert hastags(site, "Qubit") "Depolarizing channel only implemented for Qubits."
    end

    k = length(sites)
    if k == 1
        const0 = sqrt(1 - 3 * p / 4)
        constσ = sqrt(p / 4)
        kraus_maps = [
            constσ * op("σx", sites[1]),
            constσ * op("σy", sites[1]),
            constσ * op("σz", sites[1]),
            const0 * op("Id", sites[1]),
        ]
        return Channel("depolarizing", kraus_maps)

    elseif k == 2
        const0 = sqrt(1 - 15 * p / 16)
        constσ = sqrt(p / 16)
        pauli1 = [
            op("σx", sites[1]), op("σy", sites[1]), op("σz", sites[1]), op("Id", sites[1])
        ]
        pauli2 = [
            op("σx", sites[2]), op("σy", sites[2]), op("σz", sites[2]), op("Id", sites[2])
        ]
        kraus_maps = [constσ * x * y for x in pauli1 for y in pauli2]
        replace!(
            kraus_maps,
            Pair(constσ * pauli1[end] * pauli2[end], const0 * pauli1[end] * pauli2[end]),
        )
        return Channel("depolarizing", kraus_maps)

    else
        throw(
            "Depolarizing channel for more than two qubit gates has not been implemented."
        )
    end
end

function depolarizing(
    p::Real,
    sites::Vector,
    rho::Vectorization.VectorizedDensityMatrix,
    unvectorizedsiteinds::Vector{<:ITensors.Index{}},
)::Channel
    #= Creates a depolarizing channel for a given density matrix. =#

    if !(0 <= p <= 1)
        throw("parameter p must be between 0 and 1.")
    end
    for site in sites
        @assert hastags(site, "Qubit") "Depolarizing channel only implemented for Qubits."
    end

    k = length(sites)
    if k == 1
        const0 = sqrt(1 - 3 * p / 4)
        constσ = sqrt(p / 4)
        kraus_maps = [
            constσ * op("σx", sites[1]),
            constσ * op("σy", sites[1]),
            constσ * op("σz", sites[1]),
            const0 * op("Id", sites[1]),
        ]
        return Channel("depolarizing", kraus_maps, rho, unvectorizedsiteinds)

    elseif k == 2
        const0 = sqrt(1 - 15 * p / 16)
        constσ = sqrt(p / 16)
        pauli1 = [
            op("σx", sites[1]), op("σy", sites[1]), op("σz", sites[1]), op("Id", sites[1])
        ]
        pauli2 = [
            op("σx", sites[2]), op("σy", sites[2]), op("σz", sites[2]), op("Id", sites[2])
        ]
        kraus_maps = [constσ * x * y for x in pauli1 for y in pauli2]
        replace!(
            kraus_maps,
            Pair(constσ * pauli1[end] * pauli2[end], const0 * pauli1[end] * pauli2[end]),
        )
        return Channel("depolarizing", kraus_maps, rho, unvectorizedsiteinds)

    else
        throw(
            "Depolarizing channel for more than two qubit gates has not been implemented."
        )
    end
end

function depolarizing(
    p::Real, sites::Vector, rho::MPS, unvectorizedsites::Vector{<:ITensors.Index{}}
)::Channel
    #= Creates a depolarizing channel for a given density matrix.=#

    if !(0 <= p <= 1)
        throw("parameter p must be between 0 and 1.")
    end
    for site in sites
        @assert hastags(site, "Qubit") "Depolarizing channel only implemented for Qubits."
    end

    k = length(sites)
    if k == 1
        const0 = sqrt(1 - 3 * p / 4)
        constσ = sqrt(p / 4)
        kraus_maps = [
            constσ * op("σx", sites[1]),
            constσ * op("σy", sites[1]),
            constσ * op("σz", sites[1]),
            const0 * op("Id", sites[1]),
        ]
        return Channel("depolarizing", kraus_maps, rho, unvectorizedsites)

    elseif k == 2
        const0 = sqrt(1 - 15 * p / 16)
        constσ = sqrt(p / 16)
        pauli1 = [
            op("σx", sites[1]), op("σy", sites[1]), op("σz", sites[1]), op("Id", sites[1])
        ]
        pauli2 = [
            op("σx", sites[2]), op("σy", sites[2]), op("σz", sites[2]), op("Id", sites[2])
        ]
        kraus_maps = [constσ * x * y for x in pauli1 for y in pauli2]
        replace!(
            kraus_maps,
            Pair(constσ * pauli1[end] * pauli2[end], const0 * pauli1[end] * pauli2[end]),
        )
        return Channel("depolarizing", kraus_maps, rho, unvectorizedsites)

    else
        throw(
            "Depolarizing channel for more than two qubit gates has not been implemented."
        )
    end
end

function depolarizing(p::Real, sites::Vector, rho::VDMNetwork{V})::Channel where {V}
    #= Creates a depolarizing channel for a given density matrix. =#

    if !(0 < p <= 1)
        throw("parameter p must be between 0 and 1.")
    end
    for site in sites
        if !(hastags(site, "Qubit"))
            throw("Depolarizing channel only implemented for Qubits.")
        end
    end

    k = length(sites)
    if k == 1
        const0 = sqrt(1 - 3 * p / 4)
        constσ = sqrt(p / 4)
        kraus_maps = [
            constσ * op("σx", sites[1]),
            constσ * op("σy", sites[1]),
            constσ * op("σz", sites[1]),
            const0 * op("Id", sites[1]),
        ]
        return Channel("depolarizing", kraus_maps, rho)

    elseif k == 2
        const0 = sqrt(1 - 15 * p / 16)
        constσ = sqrt(p / 16)
        pauli1 = [
            op("σx", sites[1]), op("σy", sites[1]), op("σz", sites[1]), op("Id", sites[1])
        ]
        pauli2 = [
            op("σx", sites[2]), op("σy", sites[2]), op("σz", sites[2]), op("Id", sites[2])
        ]
        kraus_maps = [constσ * x * y for x in pauli1 for y in pauli2]
        replace!(
            kraus_maps,
            Pair(constσ * pauli1[end] * pauli2[end], const0 * pauli1[end] * pauli2[end]),
        )
        return Channel("depolarizing", kraus_maps, rho)

    else
        throw(
            "Depolarizing channel for more than two qubit gates has not been implemented."
        )
    end
end

function dephasing(p::Real, site::ITensors.Index)::Channel
    M0 = sqrt(1 - p) * delta(site', site)
    M1 = sqrt(p) * op("Z", site)
    return Channel("dephasing", [M0, M1])
end

function dephasing(p::Real, site::ITensors.Index, ρ::VDMNetwork{V})::Channel where {V}
    M0 = sqrt(1 - p) * delta(site', site)
    M1 = sqrt(p) * op("Z", site)
    return Channel("dephasing", [M0, M1], ρ)
end

function dephasing(
    p::Real,
    site::ITensors.Index,
    ρ::VectorizedDensityMatrix,
    unvectorizedsites::Vector{<:ITensors.Index{}},
)::Channel
    M0 = sqrt(1 - p) * delta(site', site)
    M1 = sqrt(p) * op("Z", site)
    return Channel("dephasing", [M0, M1], ρ, unvectorizedsites)
end

end; #module
