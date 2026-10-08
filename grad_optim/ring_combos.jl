# ring_combos.jl  —  pure combinatorics for the ring search (no packages, no globals)
#
# Included by ring_search.jl; kept separate so the enumeration can be unit-tested
# without CUDA / the operator cache.

"""
    for_each_combination(fn, N, k)

Call `fn(idx)` for every k-subset `idx` (ascending, 1-based) of 1:N, in lexicographic
order. `fn` first so `do` blocks work. The same vector is reused between calls —
copy it if you keep it.
"""
function for_each_combination(fn, N::Int, k::Int)
    idx = collect(1:k)
    while true
        fn(idx)
        i = k
        while i >= 1 && idx[i] == N - k + i; i -= 1; end
        i == 0 && break
        idx[i] += 1
        for j in i+1:k; idx[j] = idx[j-1] + 1; end
    end
end

"""
    paired_magnitudes(candidates) -> Vector{Int}

The tray magnitudes `a > 0` for which BOTH `+a` and `-a` are candidates, ascending.
A "paired ring" set is a choice of such magnitudes, each contributing the two rings
`-a` and `+a` (same tray NUMBER either side of z = 0).
"""
paired_magnitudes(candidates) =
    sort([a for a in unique(abs.(candidates)) if a > 0 && (a in candidates) && (-a in candidates)])

"""
    for_each_paired_set(fn, candidates, m)

Call `fn(idx)` for every choice of `m` pair magnitudes, where `idx` is the ascending
vector of CANDIDATE INDICES of the resulting `2m` rings (`-a` and `+a` for each chosen
`a`), i.e. `sort(candidates[idx])` is e.g. `[-10, -5, 5, 10]` for m = 2, a ∈ {5, 10}.
Number of sets is `binomial(length(paired_magnitudes(candidates)), m)`.
"""
function for_each_paired_set(fn, candidates, m::Int)
    mags = paired_magnitudes(candidates)
    cidx = Dict(t => i for (i, t) in enumerate(candidates))
    for_each_combination(length(mags), m) do pidx
        idx = sort!([cidx[s * mags[j]] for j in pidx for s in (-1, 1)])
        fn(idx)
    end
end
