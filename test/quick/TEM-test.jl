using OpenNetworks: OpenNetworks, TEM_functions
using NamedGraphs: NamedEdge

@testset "bonds" begin
    b12 = TEM_functions.bondtag(NamedEdge(1 => 2))
    @test hastags(b12, "bond")
    @test hastags(b12, "1")
    edges = [NamedEdge(1 => 2), NamedEdge(3 => 4), NamedEdge(32 => 23)]
    bts = TEM_functions.bondtags(edges)
    for (i, edge) in enumerate(edges)
        @test bts[i][1] == edge
        @test hastags(bts[i][2], "bond")
    end
    @test hastags(bts[3][2], "32")
    @test hastags(bts[3][2], "23")
    binds = TEM_functions.bondinds(edges)
    @test binds[1][1] == edges[1]
    @test hastags(binds[1][2], "bond")
    @test hastags(binds[1][2], "1")
    @test hastags(binds[1][2], "2")
end

@testset "pair bonds" begin
    s = copy(square_sites)
    e = first(edges(s))
    b = TEM_functions.bondtag(e)
    @test hastags(b, "bond")
    @test hastags(b, TEM_functions.dspace(string(e.src)))
end

@testset "bondinds" begin
    s = copy(square_sites)
    bnet = TEM_functions.bondnetwork(s)
    for e in ITensorNetworks.edges(bnet)
        @test !isempty((
            ind for
            ind in bnet[e.src] if hastags(ind, string(e.src)) && hastags(ind, string(e.dst))
        ))
        @test !isempty((
            ind for
            ind in bnet[e.dst] if hastags(ind, string(e.src)) && hastags(ind, string(e.dst))
        ))
    end
end

@testset "itensors to ito" begin
    ρ = square_rand_vρ
    bnet = TEM_functions.bondnetwork(siteinds(ρ); linkdim=2)
    tensor_list = Vector{ITensor}()
    for v in vertices(bnet)
        tensor = ITensors.randomITensor(bnet[v]...)
        push!(tensor_list, tensor)
    end
    tno = TEM_functions.itensors_to_ito(tensor_list, ρ)
    v1 = first(vertices(bnet))
    @test tno[v1] == first(tensor_list)

    ψ = square_rand_ψ
    bnet = TEM_functions.bondnetwork(siteinds(ψ); linkdim=2)
    tensor_list = Vector{ITensor}()
    for v in vertices(bnet)
        tensor = ITensors.randomITensor(bnet[v]...)
        push!(tensor_list, tensor)
    end
    tno = TEM_functions.itensors_to_ito(tensor_list, ψ)
    v1 = first(vertices(bnet))
    @test tno[v1] == first(tensor_list)
end

@testset "compile into tno" begin
    ψ = square_rand_ψ
    s = siteinds(ψ)
    gate1 = randomITensor(s[(1, 1)], s[(1, 2)], s[(1, 1)]', s[(1, 2)]')
    gate2 = randomITensor(s[(2, 1)], s[(2, 2)], s[(2, 1)]', s[(2, 2)]')
    bnet = TEM_functions.bondnetwork(s)
    single_tensor_list = TEM_functions.compile_moment_into_single_tensors(
        [gate1, gate2], bnet
    )
    combined_tensor1 = reduce(*, single_tensor_list)
    combined_tensor2 = gate1 * gate2
    @test Array(combined_tensor1, inds(combined_tensor1)) ≈
        Array(combined_tensor2, inds(combined_tensor1))
    tno = TEM_functions.compile_moment_into_tno(single_tensor_list, bnet, ψ)
    @test Array(gate1, inds(gate1)) ≈
        Array(TEM_functions.squeeze(tno[(1, 1)] * tno[(1, 2)]), inds(gate1))
end
