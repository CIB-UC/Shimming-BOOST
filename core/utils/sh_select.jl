# utils/sh_select.jl — choose WHICH spherical-harmonic coefficients define the field map
#
# Shared by the two SH-based Stage-0 adapters (Field_data_SH_interpolator.jl,
# Field_data_shell.jl). Pure: only LinearAlgebra / Statistics / Printf (stdlib),
# no globals, no file reads — so it can be unit-tested on its own.
#
# The adapters fit  B ≈ Σ_{l=0}^{L} Σ_{m=-l}^{l} c_lm (r/Rn)^l Y_lm  (columns ordered
# l = 0..L, m = -l..l; column index of (l,m) is  l² + l + m + 1). This file decides which
# of those columns the field map is built from:
#
#   :all      every column, degrees 0..L          (the original behaviour — untouched fit)
#   :first_n  degrees 0..use_degree only          (column block 1:(use_degree+1)²)
#   :top_k    the mean (l=0) plus the k largest |c_lm| with l ≥ 1 of the full fit
#
# For :first_n / :top_k the kept columns are RE-FITTED by least squares against the data
# (not just the full fit's coefficients truncated), so the result is the best field of
# that restricted form — identical to truncation on a single sphere, better on volume scans.

using LinearAlgebra, Statistics, Printf

sh_col(l::Int, m::Int) = l^2 + l + m + 1
sh_lm(L::Int) = [(l, m) for l in 0:L for m in -l:l]        # matches the column order

const SH_MODES = (:all, :first_n, :top_k)

"""
    sh_select_columns(c_full, L, mode; use_degree = L, k = 5) -> Vector{Int}

Ascending column indices to keep. The mean (l = 0, column 1) is always kept; for
`:top_k` the ranking ignores it, exactly like the verifier's pyramid.
"""
function sh_select_columns(c_full::AbstractVector{<:Real}, L::Int, mode::Symbol;
                           use_degree::Int = L, k::Int = 5)
    ncoef = (L + 1)^2
    length(c_full) == ncoef || error("sh_select_columns: expected $(ncoef) coefficients for L=$(L), got $(length(c_full)).")
    mode in SH_MODES || error("sh_select must be one of $(SH_MODES); got :$(mode).")
    mode === :all && return collect(1:ncoef)
    if mode === :first_n
        1 <= use_degree <= L ||
            error("sh_use_degree = $(use_degree) must satisfy 1 ≤ sh_use_degree ≤ sh_degree ($(L)).")
        return collect(1:(use_degree + 1)^2)
    end
    1 <= k <= ncoef - 1 || error("sh_top_k = $(k) must satisfy 1 ≤ sh_top_k ≤ $(ncoef - 1) (all coefficients with l ≥ 1).")
    top = sortperm(abs.(c_full[2:end]); rev = true)[1:k] .+ 1
    return sort!(vcat(1, top))
end

"""
    sh_refit(A, b, cols) -> Vector{Float64}

Least-squares coefficients restricted to columns `cols` of the design matrix `A`
(column-scaled for conditioning, like the adapters' own fit); every other entry is 0.
"""
function sh_refit(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real}, cols::AbstractVector{Int})
    As = A[:, cols]
    cn = [norm(view(As, :, j)) for j in axes(As, 2)]
    cn[cn .== 0] .= 1.0
    cs = (As ./ cn') \ b
    c = zeros(Float64, size(A, 2))
    c[cols] = cs ./ cn
    return c
end

_ppm(v) = 1e6 * (maximum(v) - minimum(v)) / mean(v)

"""
    sh_summary(io, c_full, c_used, cols, L; mode, use_degree, k, A, b)

Human-readable decomposition of the measured field: per-degree strength, the `k`
largest coefficients (l ≥ 1), what was kept, and how much of the field that keeps.
Coefficients are in the units of `b` (mT).
"""
function sh_summary(io::IO, c_full, c_used, cols, L::Int; mode::Symbol, use_degree::Int, k::Int, A, b)
    lm = sh_lm(L)
    println(io, "  ── spherical-harmonic decomposition (fit degree L=$(L), coefficients a_{n,m} in mT) ──")
    @printf(io, "  mean field a_{0,0} = %.4f mT\n", c_full[1])
    tot = sum(abs2, c_full[2:end])
    println(io, "  degree   energy Σa²   share    largest |a_{n,m}|")
    for n in 1:L
        idx = sh_col(n, -n):sh_col(n, n)
        e = sum(abs2, c_full[idx]); j = idx[argmax(abs.(c_full[idx]))]
        @printf(io, "   n=%-3d  %10.3e  %5.1f%%    %9.4f  (m=%+d)\n", n, e, 100 * e / tot, c_full[j], lm[j][2])
    end
    kk = min(max(k, 1), length(c_full) - 1)
    order = sortperm(abs.(c_full[2:end]); rev = true)[1:kk] .+ 1
    println(io, "  largest $(kk) coefficients (n ≥ 1):")
    cum = 0.0
    for (r, j) in enumerate(order)
        cum += c_full[j]^2
        @printf(io, "   #%-2d  a_{%d,%+d} = %+9.4f mT   (%.1f%% of n≥1 energy, cumulative %.1f%%)\n",
                r, lm[j][1], lm[j][2], c_full[j], 100 * c_full[j]^2 / tot, 100 * cum / tot)
    end
    if mode === :all
        println(io, "  field map built from: ALL coefficients, degrees 0..$(L)  (sh_select = \"all\")")
    else
        what = mode === :first_n ? "degrees 0..$(use_degree)  ($(length(cols)) of $(length(c_full)) coefficients)" :
               "mean + the $(length(cols) - 1) largest coefficients above  ($(length(cols)) of $(length(c_full)))"
        println(io, "  field map built from: ", what, "  (sh_select = \"", mode, "\", refitted to the data)")
        bf = A * c_full; bu = A * c_used
        @printf(io, "  vs measured, all fitted points:  full fit RMS %.4f mT / %.0f ppm   →   selection RMS %.4f mT / %.0f ppm   (measured %.0f ppm)\n",
                sqrt(mean(abs2, bf .- b)), _ppm(bf), sqrt(mean(abs2, bu .- b)), _ppm(bu), _ppm(b))
    end
    return nothing
end

"""
    sh_write_csv(path, c_full, c_used, L)

One row per coefficient: n, m, a_full, a_used (0 if dropped), rank by |a_full| among
n ≥ 1 (0 for the mean), kept (1/0).
"""
function sh_write_csv(path::AbstractString, c_full, c_used, cols, L::Int)
    lm = sh_lm(L)
    rank = zeros(Int, length(c_full))
    rank[sortperm(abs.(c_full[2:end]); rev = true) .+ 1] = 1:(length(c_full) - 1)
    kept = falses(length(c_full)); kept[cols] .= true
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "n,m,a_full_mT,a_used_mT,rank_abs_n_ge_1,kept")
        for j in eachindex(c_full)
            @printf(io, "%d,%d,%.10g,%.10g,%d,%d\n", lm[j][1], lm[j][2], c_full[j], c_used[j], rank[j], kept[j])
        end
    end
    return path
end
