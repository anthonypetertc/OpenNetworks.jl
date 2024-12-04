### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 12040980-0562-4128-84fe-cf3487aa23d6
begin
    using Pkg
    cd("/home/tony/OpenNetworks.jl")
    Pkg.activate(".")
    using ITensors
    using ITensorNetworks: ITensorNetworks, edges, vertices, src, dst, ITensorNetwork
    using ITensorsOpenSystems
    using NamedGraphs
    using OpenNetworks:
        OpenNetworks,
        VDMNetworks.VDMNetwork,
        Lindblad.trotterize,
        NoisyCircuits.run_circuit,
        VectorizationNetworks.expect
    #using ITensorsOpenSystems
    #using OpenNetworks
end

# ╔═╡ 1363eaa0-357c-4fc6-946b-4e1fd196bb5f
md"""
In this tutorial we will consider a small system on a 2x2 grid, with Hamiltonian:

```math
H = \sum_{i, j} \sigma^x_i \sigma^x_j + \sigma^y_i \sigma^y_j + \Delta \sigma^z_i\sigma^z_j + h\sum_i \sigma_i ^x
```

(Where the first sum is over (i, j) adjacent on the lattice.)

And jump operators:

```math
A_1 = \sqrt{\gamma_{in}} \sigma^+_{(1,1)}
```

```math
A_2 = \sqrt{\gamma_{out}} \sigma^-_{(2,2)}
```

```math
A_3 = \sqrt{\gamma_{dp}} \sum_i \sigma_i^z
```
"""

# ╔═╡ 34202135-fe32-42fc-a899-de6ee257bf23
md"""
The first step, is to build a named graph with the appropriate connectivity, and to define the site indices and fat site indices that we will be working with:
"""

# ╔═╡ e068b586-d6e8-41c4-957c-a0acf1bda1e4
begin
    g = NamedGraphs.NamedGraphGenerators.named_grid((2, 2))
    sites = siteinds("Qubit", g)
    fatsites = OpenNetworks.VectorizationNetworks.fatsiteinds(sites)
end

# ╔═╡ c1fa6831-2895-42e3-ad63-07785c885b72
md"""
The next step is to define an appropriate Hamiltonian and Jump operators for the system of interest:
"""

# ╔═╡ 88c3bfaf-efb6-40c9-adcc-716c97395c01
begin #Set parameters
    Δ = 0.2
    h = 0.1
    γin = 0.3
    γout = 0.3
    γdp = 0.05
end

# ╔═╡ 23f920e1-f95b-4325-ac35-2a19a2d9df86
begin
    H = OpSum()
    for e in edges(fatsites)
        global H
        s = src(e)
        d = dst(e)
        H += 1, "X", s, "X", d #XX term
        H += 1, "Y", s, "Y", d #YY term
        H += Δ, "Z", s, "Z", d #ZZ term
    end
    for v in vertices(fatsites)
        global H += h, "X", v #Transverse field
    end
end

# ╔═╡ af311d34-b5d0-4198-ba9f-442957abab1c
begin
    A = fill(OpSum(), 3)
    A[1] += sqrt(γin), "S+", (1, 1) #Spin injection
    A[2] += sqrt(γout), "S-", (2, 2) #Spin ejection
    for v in vertices(fatsites)
        A[3] += sqrt(γdp), "Z", v
    end
end

# ╔═╡ cf540f36-eee7-499a-bbc0-7d64df3e9f5f
md"""
With this defined, we want to construct an initial state (density matrix) to evolve under this dynamics.
"""

# ╔═╡ 246f44bd-1712-4434-bf06-909e6d243cf9
begin
    ψ = ITensorNetwork(v -> "0", sites) #Prepare the all 0 state
    ρ = VDMNetwork(OpenNetworks.Utils.outer(ψ, ψ), sites, fatsites) #Vectorized density matrix network.
end

# ╔═╡ 96add787-4157-4d59-8ed1-42565acaf16e
md"""
Next we need to trotterize our evolution operators to define a NoisyCircuit struct made up of a sequence of discrete channels.
"""

# ╔═╡ 6fd4861d-d607-4f5c-b6bf-7d914cef436c
begin
    steps = 10 #Number of Trotter steps.
    δt = 0.1 #Step size.
    circuit = trotterize(H, A, steps, δt, fatsites; order=2)
    # Second order Trotter decomposition for the evolution operator.
    @show circuit
end

# ╔═╡ ca9c1968-f7d4-4e01-9f07-d8a2da6b1bdf
md"""
Finally, to run the circuit, we can use the run_circuit function. This requires two sets of configurations which are passed in as seperate key word arguments.

cache_update_kwargs provides parameters for the belief approximation regauging which includes:
- :maxiter (Int64) maximum number of iterations to run the bp algorithm.
- :tol (Float64) tolerance for bp algorithm.
- :verbose (Bool) print intermediate outputs or not.

apply_kwargs provides parameters for the application of individual gates/channels:
- :maxdim (Int64) maximum bond dimension.
- :cutoff (Float64) cutoff for truncation of singular values.
"""

# ╔═╡ 0d75a46a-ff97-4f4b-b8a7-4e12492c13a3
begin
    cache_update_kwargs = Dict(:maxiter => 16, :tol => 1e-6, :verbose => false)

    apply_kwargs = Dict(:maxdim => 20, :cutoff => 1e-14)

    ρevolved = run_circuit(ρ, circuit; cache_update_kwargs, apply_kwargs)
    results = expect("Z", ρevolved; alg="bp") #Compute Z expectation values.
end

# ╔═╡ Cell order:
# ╠═12040980-0562-4128-84fe-cf3487aa23d6
# ╠═1363eaa0-357c-4fc6-946b-4e1fd196bb5f
# ╠═34202135-fe32-42fc-a899-de6ee257bf23
# ╠═e068b586-d6e8-41c4-957c-a0acf1bda1e4
# ╠═c1fa6831-2895-42e3-ad63-07785c885b72
# ╠═88c3bfaf-efb6-40c9-adcc-716c97395c01
# ╠═23f920e1-f95b-4325-ac35-2a19a2d9df86
# ╠═af311d34-b5d0-4198-ba9f-442957abab1c
# ╠═cf540f36-eee7-499a-bbc0-7d64df3e9f5f
# ╠═246f44bd-1712-4434-bf06-909e6d243cf9
# ╠═96add787-4157-4d59-8ed1-42565acaf16e
# ╠═6fd4861d-d607-4f5c-b6bf-7d914cef436c
# ╠═ca9c1968-f7d4-4e01-9f07-d8a2da6b1bdf
# ╠═0d75a46a-ff97-4f4b-b8a7-4e12492c13a3
