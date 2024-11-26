module VDMNetworks
export VDMNetwork, update_unvectorizednetwork!, update_unvectorizednetwork
using ITensorNetworks: ITensorNetwork, ITensorNetworks
using ITensors: Index

struct VDMNetwork{V}
    network::ITensorNetwork{V}
    unvectorizedinds::ITensorNetworks.IndsNetwork{V,Index}
end

function show(io::IO, vdm::VDMNetwork)
    println(io, "VDMNetwork with underlying ITensorNetwork:")
    return show(io, vdm.network)
end

update_unvectorizednetwork!(
    vdm::VDMNetwork, unvectorizedinds::ITensorNetworks.IndsNetwork
)::VDMNetwork = VDMNetwork(vdm.network, unvectorizedinds)
update_unvectorizednetwork(
    vdm::VDMNetwork, unvectorizedinds::ITensorNetworks.IndsNetwork
)::VDMNetwork = update_unvectorizednetwork!(deepcopy(vdm), unvectorizedinds)
end;
