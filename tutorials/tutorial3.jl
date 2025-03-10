### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ d7f95a07-7404-49a8-8431-74ab1ccd8388
begin
    using Pkg
    cd("/home/tony/MEGA/git/OpenNetworks.jl")
    Pkg.activate(".")
    using ITensors
    using ITensorMPS
    using ITensorNetworks
    using ITensorsOpenSystems: Vectorization.fatsiteinds
    using OpenNetworks
    using OpenNetworks: Lindblad.lindbladevolve, Channels.Channel
end

# ╔═╡ a0967194-b64d-11ef-37e7-fb65a9385867
md"""
In this notebook we will show how to construct custom channels, and use them either as noise models for circuits, or to build NoisyCircuit objects from such channels and then use them for time evolution.

There are two approaches that can be used to construct quantum channels:

1. Provide a set of valid Kraus Operators as ITensors.
2. Provide a Hamiltonian and jump operators for a local (e.g. 2-qubit) evolution.

Firstly, we will see how to define a channel using Kraus operators:
"""

# ╔═╡ 4a42c869-3151-47ea-9562-94881f4614f6
begin
    sites = siteinds("Qubit", 2)
    fatsites = fatsiteinds(sites)

    p = 0.02 # dephasing parameter p.
    Mdephasing1 = sqrt(1 - p) * delta(sites[1]', sites[1]) #First Kraus map for dephasing channel.
    Mdephasing2 = sqrt(p) * op("Z", sites[1]) # Second Kraus map.
    dephasing_channel = Channel("dephasing", [Mdephasing1, Mdephasing2])#Channel can be constructed by giving the channel a name and a list of Kraus maps.
end

# ╔═╡ 0fb8e20f-4a30-4c4b-b7d2-c48dde4ece9a
md"""
Secondly, we can also construct Channels using Hamiltonian and jump operators.
"""

# ╔═╡ e23a3e82-7005-4903-8ce9-f7f7b51242e9
begin #Define the Hamiltonian (Note, that this can be an empty OpSum() if there is no coherent term.)
    h = 0.3
    H = OpSum()
    H += 1, "X", 1, "X", 2 #XX term.
    H += 1, "Y", 1, "Y", 2 ##YY term.
    H += h, "X", 1
    H += h, "X", 2
end

# ╔═╡ b3be2979-004a-480d-ad07-681b1c880b02
begin
    A = fill(OpSum(), 1)
    A[1] += 0.05, "Z", 1
    A[1] += 0.05, "Z", 2
end

# ╔═╡ e525e56c-7506-43b3-9c43-83e9be303903
begin
    Δt = 0.1 #Length of time for the linblad evolution.
    evolution_channel = lindbladevolve(H, A, Δt, fatsites) #Construct the desired channel.
end

# ╔═╡ 93c695e0-f6c1-4951-a22b-d68875077287
md"""
Once your channel has been built (by either method) it is possible to use this channel to build a NoiseInstruction struct, by following in the steps of tutorial2, but using your custom built channel instead of the depolarizing channel.

Alternatively, one can take a sequence of channels and build a NoisyChannel struct from them and then use this to evolve a density matrix.
"""

# ╔═╡ 07a6181e-89ec-45ce-a575-dbdd16f7ffcb
begin
    channels = [dephasing_channel, evolution_channel] #Vector of channels, in the order that you want them to be applied.
    noisycircuit = NoisyCircuit(channels, fatsites) #Build a NoisyCircuit object from these channels.
end

# ╔═╡ Cell order:
# ╠═d7f95a07-7404-49a8-8431-74ab1ccd8388
# ╠═a0967194-b64d-11ef-37e7-fb65a9385867
# ╠═4a42c869-3151-47ea-9562-94881f4614f6
# ╠═0fb8e20f-4a30-4c4b-b7d2-c48dde4ece9a
# ╠═e23a3e82-7005-4903-8ce9-f7f7b51242e9
# ╠═b3be2979-004a-480d-ad07-681b1c880b02
# ╠═e525e56c-7506-43b3-9c43-83e9be303903
# ╠═93c695e0-f6c1-4951-a22b-d68875077287
# ╠═07a6181e-89ec-45ce-a575-dbdd16f7ffcb
