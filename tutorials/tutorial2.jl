### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ f41d2771-05ba-43d6-97b5-129d3a3932c0
begin
    using Pkg
    cd("/home/tony/OpenNetworks.jl")
    Pkg.activate(".")
    using ITensors
    using ITensorNetworks: vertices
    using ITensorsOpenSystems: Vectorization.fatsiteinds
    using NamedGraphs
    using OpenNetworks:
        OpenNetworks,
        CustomParsing.parse_circuit,
        Gates.Gate,
        GraphUtils.extract_adjacency_graph,
        PreBuiltChannels.depolarizing,
        PreBuiltChannels.dephasing,
        NoiseModels.NoiseInstruction,
        NoiseModels.NoiseModel,
        NoisyCircuits.NoisyCircuit,
        VDMNetworks.VDMNetwork
end

# ╔═╡ 535bb6b0-b26e-11ef-3744-6dc3e92c9261
md"""
In this tutorial we will see how to import a circuit from a JSON file, or build a circuit, and create a NoisyCircuit struct by adding a well defined noise model to the circuit.

Firstly, for importing a circuit stored in a JSON file, the format of the JSON file should follow the example in example_circuit.json

It should be stored as a list of dictionaries, one corresponding to each gate and with three keys under each dictionary:

Name (String) - the Name of the gate following the ITensors format.
Qubits (list[Int]) - the Qubits on which the qubits acts, should be a list of integers.
Params (list[Float]) - the parameters of for the gate (if any, this can be an empty list if none are needed)

Given a JSON file stored with this format, it is possible to load the circuit & build the adjacency graph that corresponds to the connectivity of the circuit in a simple way. This creates a simple interface for importing circuits from qiskit, cirq, or any other source.

For an example, see below:
"""

# ╔═╡ 7c7c9a90-9d5f-42de-8e58-b95d552f0027
begin
    circuit = parse_circuit("tutorials/example_circuit.json")
    graph = extract_adjacency_graph(circuit)
    nothing
end

# ╔═╡ 18403f2a-4907-4b3e-b049-7bf2328d4318
md"""
Alternatively, it is also possible to build the circuit directly, by specifying a list of ParsedGate objects explicitly. For instance:
"""

# ╔═╡ 33f07f5f-01b7-420a-9d6b-65c67ae5a6cc
begin
    H1 = Gate("H", [1], [])
    CX = Gate("CX", [1, 2], [])
    qc = [H1, CX]
    two_qubit_graph = extract_adjacency_graph(qc)
    sites = siteinds("Qubit", two_qubit_graph)
    fatsites = fatsiteinds(sites)
    nothing
end

# ╔═╡ 90861417-2b1f-440c-a3a4-7b52810cb7b7
md"""
Next, one can specify a noise model for the circuit. The way this is done is by building a seeries of NoiseInstruction structs. Each NoiseInstruction struct has the following data:

- name\_of\_instruction (String)
- channel (Channels.Channel): a Channel object that corresponds to the noise that will perturb the system. This can be indexed in any way.
- index\_ordering\_of\_channel (Vector{ITensors.Index}): should be a vector with the same indices that appear in the channel, but in the specific order that they should appear in.
- name\_of\_gates (Set{AbstractString}): names of the gates that this channel should be applied to (it will always be applied after gates with this name).
- qubits\_noise\_applies\_to (Set{ITensors.Index}): The set of site indices corresponding to the sites on which this noise channel should be applied.

Once NoiseInstruction structs have been built they can be packaged into a NoiseModel struct which then provides a simple way to add noise to a circuit.

To proceed we will make use of two pre-defined quantum channels that can be obtained from the package: depolarizing channel, and dephasing channel. In a future tutorial we will see how to build custom channels for other noise models.
"""

# ╔═╡ 36894750-345c-4a3b-8192-c4b8938f9f50
begin
    p = 0.005 #set parameter for depolarizing channel.
    dummyinds = ITensors.siteinds("Qubit", 2) #dummy indices for defining the channel.
    depol = depolarizing(p, dummyinds) #use pre-built channel.
    fatdummyinds = collect(inds(depol.tensor; plev=0)) #find the fat indices, depolarizing channel is symmetric so order of the indices doesn't matter.
    noiseinstruction = NoiseInstruction(
        "depolarizing",
        depol,
        fatdummyinds,
        Set(["CX"]), #name of gates the noise applies to.
        Set([first(sites[v]) for v in vertices(sites)]),
    )
    @show noiseinstruction
end

# ╔═╡ 59805c09-efa4-4b1e-a163-398513a2032b
md"""
Then we can take this NoiseInstruction (or a set of them) and build a NoiseModel struct from them, which should include all noise instructions that you wish to apply to your circuit.
"""

# ╔═╡ 1b7ba257-5dff-4b8a-81ea-429a3b7ea9cf
begin
    noisemodel = NoiseModel(
        Set([noiseinstruction]), #Set of all noise instructions.
        sites, # sites for the circuit the noise model will be applied to.
        fatsites, #doubled site indices for the density matrix evolution.
    )
    nothing
    @show noisemodel
end

# ╔═╡ 11966a96-0c25-45b7-9f27-d9ce0e45dc0f
md"""
It is then straightforward to build a NoisyCircuit struct from the the sequence of gates and the noise model, which can then be used to evolve a density matrix.
"""

# ╔═╡ 5007d349-4395-4f76-b99a-910f232b89a6
begin
    noisycircuit = NoisyCircuit(
        qc, #Vector of Gates
        noisemodel, #NoiseModel struct
    )
end

# ╔═╡ a2b116d3-dc56-4545-8e4e-eb8eba7ffebb
md"""
Once a NoisyCircuit object has been built it can be run using the run_circuit function by following tutorial 1.
"""

# ╔═╡ Cell order:
# ╟─f41d2771-05ba-43d6-97b5-129d3a3932c0
# ╟─535bb6b0-b26e-11ef-3744-6dc3e92c9261
# ╠═7c7c9a90-9d5f-42de-8e58-b95d552f0027
# ╟─18403f2a-4907-4b3e-b049-7bf2328d4318
# ╠═33f07f5f-01b7-420a-9d6b-65c67ae5a6cc
# ╟─90861417-2b1f-440c-a3a4-7b52810cb7b7
# ╠═36894750-345c-4a3b-8192-c4b8938f9f50
# ╟─59805c09-efa4-4b1e-a163-398513a2032b
# ╠═1b7ba257-5dff-4b8a-81ea-429a3b7ea9cf
# ╟─11966a96-0c25-45b7-9f27-d9ce0e45dc0f
# ╠═5007d349-4395-4f76-b99a-910f232b89a6
# ╟─a2b116d3-dc56-4545-8e4e-eb8eba7ffebb
