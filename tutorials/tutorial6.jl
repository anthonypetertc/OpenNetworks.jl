### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 353855f6-be2d-11ef-2540-652abdaf5532
begin
    using Pkg
    cd("/home/tony/OpenNetworks.jl")
    Pkg.activate(".")
    using ITensors
    using ITensorNetworks: ITensorNetwork, vertices, edges, src, dst
    using ITensorsOpenSystems: Vectorization.fatsiteinds
    using NamedGraphs: NamedGraph, add_edges!
    using OpenNetworks
    using OpenNetworks: Evolution.run_circuit
    using ITensorUnicodePlots: @visualize
end

# ╔═╡ 7071072f-40af-4d42-8e18-604b37d600e1
md"""
In this tutorial we will see how to create a graph with arbitrary 2d connectivity and study Lindblad evolution on this graph.
"""

# ╔═╡ d166a769-82f4-4670-b806-4c1d3ac009c9
begin
    G = NamedGraph(sort!(collect(1:22))) #Make Graph with 22 sites.
    couplings = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 1),
        (1, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 4),
        (16, 20),
        (20, 21),
        (21, 22),
        (22, 8),
    ] #Arbitrary 2d connectivity.

    for coupling in couplings
        add_edges!(G, [first(coupling) => last(coupling)])
    end

    @visualize G #Make a visualization of the Graph.
    nothing
end

# ╔═╡ 8bb11071-19da-4cd4-afae-abbfdf363327
begin
    s = siteinds("S=1/2", G)
    fs = fatsiteinds(s)
    ψ = ITensorNetwork("0", s) #Initialize in the all 0 state.
    ρ = VDMNetwork(outer(ψ', ψ), s, fs) #Initial density matrix.
    nothing
end

# ╔═╡ 775c17f5-ee6a-413a-84d0-eba67caa537a
begin #Parameters for the Transverse field Ising Hamiltonian & dissipation.
    J = 0.4
    hx = 0.2
    γin = 0.2
    γout = 0.4
    γdp = 0.08
end

# ╔═╡ 81a95cd5-5a1a-485c-a43d-563f867390eb
begin
    H = OpSum()
    for edge in edges(fs)
        global H += -J, "X", src(edge), "X", dst(edge) #two qubit terms.
    end
    for v in vertices(fs)
        global H += -hx, "Z", v #Transverse field.
    end
end

# ╔═╡ 0cb53376-1144-4319-a996-b71cb4120363
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

# ╔═╡ 4db9f1c9-3f7c-45de-8f06-aacab67785ec
begin
    Dt = 0.1
    steps = 5
    circuit = trotterize(H, A, steps, Dt, fs; order=2) #Second order trotter decompositon of the circuit.
    nothing
end

# ╔═╡ e6402e63-085d-4383-b25d-5c116d57e7c7
begin
    cache_update_kwargs = Dict(:maxiter => 8, :tol => 1e-6)
    apply_kwargs = Dict(:maxdim => 16, :cutoff => 1e-14)
    ρevolved = run_circuit(ρ, circuit; cache_update_kwargs, apply_kwargs)
    results = expect("Z", ρevolved; alg="bp")
    @show results
    nothing
end

# ╔═╡ Cell order:
# ╟─353855f6-be2d-11ef-2540-652abdaf5532
# ╠═7071072f-40af-4d42-8e18-604b37d600e1
# ╠═d166a769-82f4-4670-b806-4c1d3ac009c9
# ╠═8bb11071-19da-4cd4-afae-abbfdf363327
# ╠═775c17f5-ee6a-413a-84d0-eba67caa537a
# ╠═81a95cd5-5a1a-485c-a43d-563f867390eb
# ╠═0cb53376-1144-4319-a996-b71cb4120363
# ╠═4db9f1c9-3f7c-45de-8f06-aacab67785ec
# ╠═e6402e63-085d-4383-b25d-5c116d57e7c7
