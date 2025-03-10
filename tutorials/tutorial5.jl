### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 42b0508f-5716-465e-b5e2-078b41e4599c
begin
    using Pkg
    cd("/home/tony/MEGA/git/OpenNetworks.jl")
    Pkg.activate(".")
    using ITensors
    using ITensorNetworks: ITensorNetwork, vertices, edges, src, dst
    using ITensorsOpenSystems: Vectorization.fatsiteinds
    using NamedGraphs: NamedGraph, add_edges!
    using OpenNetworks
    using OpenNetworks: Evolution.run_circuit
    using ITensorUnicodePlots: @visualize
end

# ╔═╡ 3316be4e-b87f-11ef-3c99-67a35443edd5
md"""
In this tutorial we will see how to use the Belief Propagation algorithm to regauge and evolve tensor network states with tree connectivity (on which BP is exact).

To do so, we will make use of the NamedGraphs package to construct a star and then build a Lindbladian with appropriate connectivity.
"""

# ╔═╡ 8362dbe1-ed06-4276-94af-dd98e31b11d6
begin #Make a tree with 3 legs, and 5 qubits per leg, joined together at a central node.
    legs = 3
    length_of_leg = 5
    qubits = Vector{Tuple{Int64,Int64}}()
    for leg in 1:legs
        for qubit in 1:(length_of_leg - 1)
            push!(qubits, (leg, qubit))
        end
    end
    push!(qubits, (0, 0)) #Add center of star.
    G = NamedGraph(sort!(collect(qubits)))
    for pair1 in qubits
        for pair2 in qubits
            if first(pair1) == first(pair2) && last(pair2) == last(pair1) + 1
                add_edges!(G, [pair1 => pair2]) #Add edges connecting the sites on the legs.
            end
        end
    end

    for pair in qubits
        if last(pair) == 1
            add_edges!(G, [(0, 0) => pair]) #Add edges connecting the center to the legs.
        end
    end
    @visualize G #Make a visualization of the Graph.
    nothing
end

# ╔═╡ b16d6ac2-e7b9-45ac-85cc-e374429fd5ef
begin
    s = siteinds("S=1/2", G)
    fs = fatsiteinds(s)
    ψ = ITensorNetwork("0", s) #Initialize in the all 0 state.
    ρ = VDMNetwork(outer(ψ', ψ), s, fs) #Initial density matrix.
    nothing
end

# ╔═╡ 0e9a1555-3d72-4cbb-913c-795b70466cee
md"""
We will study the Transverse Field Ising Hamiltonian, with dephasing on every spin,
and spin injections into the last spin of every odd leg, and spin ejections from the last spin of every even leg.
"""

# ╔═╡ 878ec2a9-e619-4f9d-9a41-4bae83d91aaa
begin #Parameters for the Transverse field Ising Hamiltonian & dissipation.
    J = 0.4
    hx = 0.2
    γin = 0.2
    γout = 0.4
    γdp = 0.08
end

# ╔═╡ 0878b7fa-1a7b-43e5-bb00-0a0f4a0ce5a7
begin
    H = OpSum()
    for edge in edges(fs)
        global H += -J, "X", src(edge), "X", dst(edge) #two qubit terms.
    end
    for v in vertices(fs)
        global H += -hx, "Z", v #Transverse field.
    end
end

# ╔═╡ 215b0f0a-d362-461f-a46a-ae823d79fd22
begin
    A = fill(OpSum(), 6)
    for v in vertices(fs)
        if last(v) == 2 && isodd(first(v))
            A[first(v)] += sqrt(γin), "S+", v #Spin injections into the odd legs.
        elseif last(v) == 2 && iseven(first(v))
            A[first(v)] += sqrt(γout), "S-", v #Spin ejections from the even legs.
        end
        A[6] += sqrt(γdp), "Z", v #dephasing on every spin
    end
end

# ╔═╡ 49fd7a8c-1f9d-4afa-bfe3-3f211c16638f
begin
    Dt = 0.1
    steps = 5
    circuit = trotterize(H, A, steps, Dt, fs; order=2) #Second order trotter decompositon of the circuit.
    nothing
end

# ╔═╡ 688b5f99-3084-412d-8a79-9a40cddf3bab
begin
    cache_update_kwargs = Dict(:maxiter => 8, :tol => 1e-6)
    apply_kwargs = Dict(:maxdim => 16, :cutoff => 1e-14)
    ρevolved = run_circuit(ρ, circuit; cache_update_kwargs, apply_kwargs)
    results = expect("Z", ρevolved; alg="bp")
    @show results
    nothing
end

# ╔═╡ Cell order:
# ╠═42b0508f-5716-465e-b5e2-078b41e4599c
# ╠═3316be4e-b87f-11ef-3c99-67a35443edd5
# ╠═8362dbe1-ed06-4276-94af-dd98e31b11d6
# ╠═b16d6ac2-e7b9-45ac-85cc-e374429fd5ef
# ╠═0e9a1555-3d72-4cbb-913c-795b70466cee
# ╠═878ec2a9-e619-4f9d-9a41-4bae83d91aaa
# ╠═0878b7fa-1a7b-43e5-bb00-0a0f4a0ce5a7
# ╠═215b0f0a-d362-461f-a46a-ae823d79fd22
# ╠═49fd7a8c-1f9d-4afa-bfe3-3f211c16638f
# ╠═688b5f99-3084-412d-8a79-9a40cddf3bab
