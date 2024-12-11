### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 71c97f22-c4e3-46fc-85a2-7a5b8feb035b
begin
    using Pkg
    cd("/home/tony/OpenNetworks.jl")
    Pkg.activate(".")
    using ITensors: siteinds, outer, OpSum, apply, inner
    using ITensorMPS: productMPS
    using ITensorsOpenSystems:
        Vectorization.VectorizedDensityMatrix, Vectorization.fatsiteinds
    using OpenNetworks: Lindblad.trotterize, Evolution.run_circuit, Lindblad.lindbladevolve
end

# ╔═╡ fc6f06c2-b7ea-11ef-23e5-bba2ad58eea9
md"""
In this notebook we will see how to study evolution by a trotterized Lindbladian, with nearest neighbour interactions on a line, using TEBD. The approach is very similar to what is in tutorial1, but using the fact that the system is on a line we do not have to use the belief propagation algorithm to regauge the tensor network.
"""

# ╔═╡ 566c5100-c363-486f-a5f7-577efad107ae
begin
    #Define sites and initial density matrix.
    n_sites = 4
    s = siteinds("Qubit", n_sites)
    fs = fatsiteinds(s)
    ψ = productMPS(s, "0")
    ρ = VectorizedDensityMatrix(outer(ψ', ψ), fs)
end

# ╔═╡ 322636e5-6c90-43bb-aa7f-e229f9325eaf
begin
    #Parameters.
    h = 0.1
    D = 0.2
    yin = 0.3
    yout = 0.3
    ydp = 0.04
end

# ╔═╡ c2fbf528-69fd-414e-97a6-33dffef03e12
begin
    #Define Hamiltonian and Jump Operators.
    H = OpSum() #Hamiltonian.
    for i in 1:(n_sites - 1)
        global H
        H += 1, "X", i, "X", i + 1 # XX term
        H += 1, "Y", i, "Y", i + 1 # YY term
        H += D, "Z", i, "Z", i + 1 # ZZ term
    end
    for i in 1:n_sites
        global H
        H += h, "X", i
    end

    A = fill(OpSum(), 1) #Jump Operators.
    for i in 1:n_sites
        global A
        A[1] += sqrt(ydp), "Z", i
    end
end

# ╔═╡ 04d433dd-5f48-4f75-b4a8-92bf737a8fbb
md"""
To Trotterize the Hamiltonian and perform the evolution, we can simply use the trotterize function followed by run_circuit with the desired parameters for the tensor network evolution.
"""

# ╔═╡ 228d73eb-9567-4683-90c0-5128847f9c44
begin
    steps = 100 #Number of Trotter steps.
    Dt = 0.01 #Trotter step size.

    noisycircuit = trotterize(H, A, steps, Dt, fs; order=2) #Second order Trotter decomposition of the Lindblad evolution.
    ρ2 = run_circuit(ρ, noisycircuit; maxdim=16) #Run the circuit, with maximum bond dimension of 16.
end

# ╔═╡ 29e0de23-481c-4d1f-81dd-d198cbcdfd4d
md"""
To check that this is giving correct results, we can check this against an exact evolution of the circuit with no Trotter approximation and no MPS truncations.
"""

# ╔═╡ 9c8bcb9b-6daa-47e7-9e8c-cd16a24fc267
begin
    L = lindbladevolve(H, A, Dt * steps, fs)#Exact evolution operator.
    ρ3 = apply(L, ρ) # Exact evolution.

    overlap = inner(ρ2, ρ3) / sqrt(inner(ρ2, ρ2) * inner(ρ3, ρ3)) #normalized inner product between the vectorized density matrices.
    overlap = overlap.re
    @show overlap #overlap should be close to 1.
end

# ╔═╡ Cell order:
# ╠═71c97f22-c4e3-46fc-85a2-7a5b8feb035b
# ╠═fc6f06c2-b7ea-11ef-23e5-bba2ad58eea9
# ╠═566c5100-c363-486f-a5f7-577efad107ae
# ╠═322636e5-6c90-43bb-aa7f-e229f9325eaf
# ╠═c2fbf528-69fd-414e-97a6-33dffef03e12
# ╠═04d433dd-5f48-4f75-b4a8-92bf737a8fbb
# ╠═228d73eb-9567-4683-90c0-5128847f9c44
# ╠═29e0de23-481c-4d1f-81dd-d198cbcdfd4d
# ╠═9c8bcb9b-6daa-47e7-9e8c-cd16a24fc267
