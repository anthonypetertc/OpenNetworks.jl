module VDMNetworks
export VDMNetwork, update_unvectorizednetwork!, update_unvectorizednetwork
using ITensorNetworks: ITensorNetwork, ITensorNetworks

struct VDMNetwork
    network::ITensorNetwork
    unvectorizednetwork::ITensorNetwork

    function VDMNetwork(ρ::ITensorNetwork, unvectorizednetwork::ITensorNetwork)
        new(ρ, unvectorizednetwork)
    end
end

function show(io::IO, vdm:: VDMNetwork)
    println(io, "VDMNetwork with underlying ITensorNetwork:")
    show(io, vdm.network)
end

update_unvectorizednetwork!(vdm::VDMNetwork, unvectorizednetwork::ITensorNetwork)::VDMNetwork = VDMNetwork(vdm.network, unvectorizednetwork)
update_unvectorizednetwork(vdm::VDMNetwork, unvectorizednetwork::ITensorNetworks.IndsNetwork)::VDMNetwork = update_unvectorizednetwork!(deepcopy(vdm), unvectorizednetwork)
end;