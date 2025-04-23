module PreBuiltChannels
export depolarizing, dephasing

using ITensors
using ITensorsOpenSystems:
    Vectorization.VectorizedDensityMatrix
using OpenNetworks: Channels.Channel, VDMNetworks.VDMNetwork


function _kraus_depolarizing1(p:: Real, sites::Vector)::Vector{ITensors.ITensor}
    const0 = sqrt(1 - 3 * p / 4)
    constσ = sqrt(p / 4)
    kraus_maps = [
        constσ * op("σx", sites[1]),
        constσ * op("σy", sites[1]),
        constσ * op("σz", sites[1]),
        const0 * op("Id", sites[1]),
    ]
    return kraus_maps
end

function _kraus_depolarizing2(p:: Real, sites:: Vector{<: ITensors.Index{}}):: Vector{ITensors.ITensor}
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
    return kraus_maps
end

"""
    depolarizing(p::Real, sites::Vector{<:ITensors.Index{}})::Channel

    Arguments
    p::Real
        The depolarizing probability.
    sites::Vector{<:ITensors.Index{}}
        The sites on which the depolarizing channel acts.
    
    Returns a Channel representing the depolarizing channel on the given sites.
    Only implemented for the one or two qubit case.

"""

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
        kraus_maps = _kraus_depolarizing1(p, sites)
        return Channel("depolarizing", kraus_maps)

    elseif k == 2
        kraus_maps = _kraus_depolarizing2(p, sites)
        return Channel("depolarizing", kraus_maps)

    else
        throw(
            "Depolarizing channel for more than two qubit gates has not been implemented."
        )
    end
end

"""
    depolarizing(p::Real, sites::Vector{<:ITensors.Index{}}, rho::VectorizedDensityMatrix,
        unvectorizedsiteinds::Vector{<:ITensors.Index{}})::Channel

    Arguments
    p::Real
        The depolarizing probability.
    sites::Vector{<:ITensors.Index{}}
        The sites on which the depolarizing channel acts.
    rho::VectorizedDensityMatrix
        The density matrix to which the depolarizing channel is applied.
    unvectorizedsiteinds::Vector{<:ITensors.Index{}}
        The unvectorized site indices of the density matrix.

    Returns a Channel representing the depolarizing channel on the given sites.
    The sites must be from the unvectorisedsiteinds, and the Channel will act
    on the corresponding vectorized sites.

"""

function depolarizing(
    p::Real,
    sites::Vector,
    rho::VectorizedDensityMatrix,
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
        kraus_maps = _kraus_depolarizing1(p, sites)
        return Channel("depolarizing", kraus_maps, rho, unvectorizedsiteinds)

    elseif k == 2
        kraus_maps = _kraus_depolarizing2(p, sites)
        return Channel("depolarizing", kraus_maps, rho, unvectorizedsiteinds)

    else
        throw(
            "Depolarizing channel for more than two qubit gates has not been implemented."
        )
    end
end

"""
    depolarizing(p::Real, sites::Vector{<:ITensors.Index{}}, rho::VDMNetwork{V})::Channel where {V}

    Arguments
    p::Real
        The depolarizing probability.
    sites::Vector{<:ITensors.Index{}}
        The sites on which the depolarizing channel acts.
    rho::VDMNetwork{V}
        The density matrix to which the depolarizing channel is applied.

    Returns a Channel representing the depolarizing channel on the given sites,
    the sites must be from the unvectorisedsiteinds, and the Channel will act
    on the corresponding vectorized sites.

"""


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
        kraus_maps = _kraus_depolarizing1(p, sites)
        return Channel("depolarizing", kraus_maps, rho)

    elseif k == 2
        kraus_maps = _kraus_depolarizing2(p, sites)
        return Channel("depolarizing", kraus_maps, rho)

    else
        throw(
            "Depolarizing channel for more than two qubit gates has not been implemented."
        )
    end
end


function _kraus_dephasing(p:: Real, site:: ITensors.Index):: Vector{ITensor}
    M0 = sqrt(1 - p) * delta(site', site)
    M1 = sqrt(p) * op("Z", site)
    return [M0, M1]
end

"""
    dephasing(p::Real, site::ITensors.Index)::Channel

    Arguments
    p::Real
        The dephasing probability.
    site::ITensors.Index{}
        The site on which the dephasing channel acts.

    Returns a Channel representing the dephasing channel on the given site.
"""

function dephasing(p::Real, site::ITensors.Index)::Channel
    kraus_maps = _kraus_dephasing(p, site)
    return Channel("dephasing", kraus_maps)
end

"""
    dephasing(p::Real, site::ITensors.Index, ρ::VDMNetwork{V})::Channel where {V}

    Arguments
    p::Real
        The dephasing probability.
    site::ITensors.Index{}
        The site on which the dephasing channel acts.
    ρ::VDMNetwork{V}
        The density matrix to which the dephasing channel is applied.

    Returns a Channel representing the dephasing channel on the given site.
    The site must be from the unvectorizedsiteinds, and the Channel will act
    on the corresponding vectorized sites.
"""

function dephasing(p::Real, site::ITensors.Index, ρ::VDMNetwork{V})::Channel where {V}
    kraus_maps = _kraus_dephasing(p ,site)
    return Channel("dephasing", kraus_maps, ρ)
end

"""
    dephasing(p::Real, site::ITensors.Index, ρ::VectorizedDensityMatrix,
        unvectorizedsites::Vector{<:ITensors.Index{}},)::Channel

    Arguments
    p::Real
        The dephasing probability.
    site::ITensors.Index{}
        The site on which the dephasing channel acts.
    ρ::VectorizedDensityMatrix
        The density matrix to which the dephasing channel is applied.
    unvectorizedsites::Vector{<:ITensors.Index{}}
        The unvectorized site indices of the density matrix.

    Returns a Channel representing the dephasing channel on the given site.
    The site must be from the unvectorizedsiteinds, and the Channel will act
    on the corresponding vectorized sites.
"""

function dephasing(
    p::Real,
    site::ITensors.Index,
    ρ::VectorizedDensityMatrix,
    unvectorizedsites::Vector{<:ITensors.Index{}},
)::Channel
    kraus_maps = _kraus_dephasing(p, site)
    return Channel("dephasing", kraus_maps, ρ, unvectorizedsites)
end

end; # module