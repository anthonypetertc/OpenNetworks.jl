using ITensors
using ITensorMPS
using ITensorsOpenSystems: Vectorization.VectorizedDensityMatrix, Vectorization.fatsiteinds
using OpenNetworks:
    Lindblad, Channels, PreBuiltChannels, Evolution.run_circuit, Lindblad.trotterize
using Test

@testset "test convertprodop" begin
    H = OpSum()
    H += 1, "X", (1, 1), "X", (1, 2)
    newop = Lindblad.convertprodop(H[1].args[2], Dict((1, 1) => 1, (1, 2) => 2))
    H2 = OpSum()
    H2 += 1, "X", 1, "X", 2

    @test H2[1].args[2] == newop
end;

@testset "local hamiltonian" begin
    N = 3
    h = 0.1
    D = 0.2
    yin = 0.3
    yout = 0.3
    ydp = 0.04
    H = OpSum()
    # Open boundary chain:
    for i in 1:(N - 1)
        H
        H += h, "X", i # Transverse field

        H += 1, "X", i, "X", i + 1 # XX term
        H += 1, "Y", i, "Y", i + 1 # YY term
        H += D, "Z", i, "Z", i + 1 # ZZ term
    end
    H += h, "X", N

    H12 = Lindblad.localhamiltonian(H, Dict(1 => 1, 2 => 2))
    H23 = Lindblad.localhamiltonian(H, Dict(2 => 2, 3 => 3))

    G12 = OpSum()
    G12 += h, "X", 1
    G12 += 1, "X", 1, "X", 2
    G12 += 1, "Y", 1, "Y", 2
    G12 += D, "Z", 1, "Z", 2
    G12 += h, "X", 2

    @test G12 == H12

    G23 = OpSum()
    G23 += h, "X", 2
    G23 += 1, "X", 2, "X", 3
    G23 += 1, "Y", 2, "Y", 3
    G23 += D, "Z", 2, "Z", 3
    G23 += h, "X", 3

    @test G23 == H23
end;

@testset "local jumps" begin
    N = 3
    h = 0.1
    D = 0.2
    yin = 0.3
    yout = 0.3
    ydp = 0.04
    A = fill(OpSum(), 3)         # An array of OpSum's for jump operators
    A[1] += sqrt(yin), "S+", 1    # Spin injection
    A[2] += sqrt(yout), "S-", N   # Spin ejection

    for i in 1:N
        A[3] += sqrt(ydp), "Z", i # Collective dephasing
    end

    A23 = Lindblad.localjumps(A, Dict(2 => 2, 3 => 3))
    B23 = fill(OpSum(), 2)
    B23[1] += sqrt(yout), "S-", N
    B23[2] += sqrt(ydp), "Z", 2
    B23[2] += sqrt(ydp), "Z", 3
    @test A23 == B23
end

@testset "Continuous dephasing" begin
    system = siteinds("S=1/2", 2)
    fatsys = Vectorization.fatsiteinds(system)
    ψ = productMPS(system, "1")
    ψ[1] = ITensors.apply(op("H", system[1]), ψ[1])
    ρ = VectorizedDensityMatrix(outer(ψ', ψ), fatsys)
    H = OpSum()
    H += 1, "I", 1
    A = fill(OpSum(), 1)
    A[1] += 1, "Z", 1
    t = 0.5
    lindbladdephasing = Lindblad.lindbladevolve(H, A, t, [fatsys[1]])

    p = (1 - exp(-2 * t)) / 2
    krausdephasing = PreBuiltChannels.dephasing(p, system[1], ρ, system)

    @test krausdephasing.tensor ≈ lindbladdephasing.tensor
    rtol = 1e-8
end

@testset "Trotterized XXZ" begin
    h = 0.1
    D = 0.2
    yin = 0.3
    yout = 0.3
    ydp = 0.04

    H = OpSum()
    for e in edges(square_sites)
        es = src(e)
        ds = dst(e)
        H += 1, "X", es, "X", ds # XX term
        H += 1, "Y", es, "Y", ds # YY term
        H += D, "Z", es, "Z", ds # ZZ term
    end
    for v in vertices(square_sites)
        H += h, "X", v
    end

    A = fill(OpSum(), 1)
    for v in vertices(square_sites)
        A[1] += sqrt(ydp), "Z", v
    end

    e = Dict((1, 1) => 1, (1, 2) => 2, (2, 1) => 3, (2, 2) => 4)
    H2 = Lindblad.localhamiltonian(H, e)
    A2 = Lindblad.localjumps(A, e)

    sites = [first(square_vsites[v]) for v in vertices(square_vsites)]

    dt = 0.01
    steps = 20
    t = steps * dt
    ψ = ITensorNetwork(v -> "0", square_sites)
    ψ[(1, 1)] = ITensors.apply(op("H", square_sites[(1, 1)]), ψ[(1, 1)])
    ρ = OpenNetworks.VDMNetworks.VDMNetwork(outer(ψ', ψ), square_sites, square_vsites)

    L = OpenNetworks.Lindblad.lindbladevolve(H2, A2, t, sites) # Time evolution operator for the exact lindbladian evolution.
    trottercircuit = OpenNetworks.Lindblad.firstordertrotter(H, A, steps, dt, square_vsites) #Trotterized evolution.
    trottercircuit2 = OpenNetworks.Lindblad.secondordertrotter(
        H, A, steps, dt, square_vsites
    )

    rho2 = ITensors.apply(L.tensor, ITensorNetworks.contract(ρ.network)) # Exact evolution.

    gates_list1 = trottercircuit.channel_list
    gate = gates_list1[1].tensor
    for g in gates_list1
        gate = ITensors.apply(g.tensor, gate)
    end
    rho3 = ITensors.apply(gate, ITensorNetworks.contract(ρ.network)) # Evolution by first order Trotter circuit.

    gates_list2 = trottercircuit2.channel_list
    gate2 = gates_list2[1].tensor
    for g in gates_list1
        gate2 = ITensors.apply(g.tensor, gate2)
    end
    rho4 = ITensors.apply(gate2, ITensorNetworks.contract(ρ.network)) #Evolution by second order Trotter circuit.

    overlap =
        first(ITensorNetworks.contract(dag(rho2), rho3)) / sqrt(
            first(ITensorNetworks.contract(dag(rho2), rho2)) *
            first(ITensorNetworks.contract(dag(rho3), rho3)),
        )
    overlap = overlap.re
    @test isapprox(overlap, 1; rtol=dt) #Normalized overlap should be approximately 1.

    overlap2 =
        first(ITensorNetworks.contract(dag(rho2), rho4)) / sqrt(
            first(ITensorNetworks.contract(dag(rho2), rho2)) *
            first(ITensorNetworks.contract(dag(rho4), rho4)),
        )
    @test isapprox(overlap, 1; rtol=dt)
end

@testset "tebd" begin
    n_sites = 4
    s = siteinds("Qubit", n_sites)
    fs = fatsiteinds(s)
    ψ = productMPS(s, "0")
    ρ = VectorizedDensityMatrix(outer(ψ', ψ), fs)

    h = 0.1
    D = 0.2
    yin = 0.3
    yout = 0.3
    ydp = 0.04

    H = OpSum()
    for i in 1:(n_sites - 1)
        H += 1, "X", i, "X", i + 1 # XX term
        H += 1, "Y", i, "Y", i + 1 # YY term
        H += D, "Z", i, "Z", i + 1 # ZZ term
    end
    for i in 1:n_sites
        H += h, "X", i
    end

    A = fill(OpSum(), 1)
    for i in 1:n_sites
        A[1] += sqrt(ydp), "Z", i
    end

    steps = 100
    Dt = 0.01

    noisycircuit = trotterize(H, A, steps, Dt, fs; order=2)
    ρ2 = run_circuit(ρ, noisycircuit)

    # Now by exact diagonalization.
    L = OpenNetworks.Lindblad.lindbladevolve(H, A, Dt * steps, fs)
    ρ3 = ITensors.apply(L, ρ) # Exact evolution.

    overlap = inner(ρ2, ρ3) / sqrt(inner(ρ2, ρ2) * inner(ρ3, ρ3))
    overlap = overlap.re
    @test isapprox(overlap, 1; rtol=0.0001) #should be correct up to Trotter errors.
end
