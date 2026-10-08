# ring_search.jl  —  find the best set of n shim rings to place
#
#   julia stages/stage1_optimize/grad_optim/ring_search.jl
#
# Instead of hand-setting positions_in_tray_new_wished, this searches candidate
# ring (tray) positions and returns the best n-ring combination. The field is
# linear in the moments, so each candidate ring's field response is a fixed block
# of columns — we build them ONCE for all candidates, then scoring any ring subset
# is just selecting its columns and solving a small variance problem (ms each).
#
#   method = :greedy      forward selection, n·N solves — scales to any n.
#   method = :exhaustive  all C(N,n) subsets (guarded by ring_search_max_combos).
#   method = :paired      only SYMMETRIC sets: n = 2m rings made of m pairs {−a, +a}
#                         (same tray number either side of z = 0, e.g. [-10,-5,5,10]);
#                         tries every choice of m pair magnitudes, C(#pairs, m) sets.
#                         n is the TOTAL ring count and must be even.
#
# Config knobs: ring_search_* in config.toml. Objective = grad_data_term (variance
# or softrange) over the shell → ppm. All CPU except the one-time GPU operator build.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))          # project lives in the repo root
include(joinpath(@__DIR__, "..", "..", "..", "pipeline_config.jl"))
if eval_domain === :shell                                  # score on a spherical shell at Rmax …
    include(joinpath(@__DIR__, "..", "setup_shell.jl"))
else                                                        # … or the dense grid (default)
    include(joinpath(@__DIR__, "..", "setup.jl"))          # grid, fld_field, msk, N, threads, blocks, positions_from_rings_mm
end
include(joinpath(@__DIR__, "..", "..", "..", "core", "kernels", "f_kernel.jl")) # _Btot!
include(joinpath(@__DIR__, "grad_math.jl"))                # make_data_cost, shell_metrics
include(joinpath(@__DIR__, "ring_combos.jl"))              # for_each_combination, for_each_paired_set
using Optim, Random, JLD2, Statistics, Printf, DataFrames, CSV

const MU_MAG   = magnet_moment_Am2                          # per-magnet moment; config magnet_Br_T, magnet_side_mm (matches setup.jl μ_base)
const OUTDIR   = joinpath(optimizer_iter_dir, "RingSearch"); mkpath(OUTDIR)
const OPCACHE  = joinpath(optimizer_iter_dir, "GradOpt", "ring_operator.jld2")
mkpath(dirname(OPCACHE))                                    # GradOpt/ may not exist yet (ring search is standalone)
const CONFIGTOML = joinpath(ROOT, "config.toml")

# --- candidate rings (integer tray numbers), 0 excluded ----------------------
const lo, hi   = ring_search_range[1], ring_search_range[2]
const candidates = [t for t in lo:ring_search_step:hi if t != 0]
const Ncand    = length(candidates)
const cand_z   = [ringpos_from_tray_mm([t]; tray_slot_spacing_mm = tray_slot_spacing_mm,
                                            front_tray_shift_mm  = front_tray_shift_mm,
                                            back_tray_shift_mm   = back_tray_shift_mm)[1]
                  for t in candidates]                       # axial mm

# --- shell bookkeeping (rows of the operator) --------------------------------
const shell_idx = findall(>(0f0), vec(Array(msk)))
const Ns        = length(shell_idx)
const b_shell   = Float64.(vec(Array(fld_field))[shell_idx])      # base field on shell (mT)
const Ngrid     = Int(length(grid.X))

# geometry/field signature — rebuild the cached operator only if something changed
const SIG = string((candidates, shim_radius_mm, mags_per_segment, num_trays,
                    angle_per_segment_deg, angular_offset_deg,
                    tray_slot_spacing_mm, front_tray_shift_mm, back_tray_shift_mm,
                    Rmin, Rmax, interpolated_fieldmap_name, MU_MAG, String(eval_domain)))

# ---------------------------------------------------------------------------
# Build one candidate ring's operator columns (Ns × mpr) for unit x/y moments.
# Uses only that ring's magnets (small m) → cost O(Ngrid·mpr) per column.
# ---------------------------------------------------------------------------
function ring_columns(t)
    pos = positions_from_rings_mm([t]; occupied_trays = positions_in_tray_occupied,
              shim_radius_mm = shim_radius_mm, mags_per_segment = mags_per_segment,
              num_segments = num_trays, angle_per_segment_deg = angle_per_segment_deg,
              angular_offset_deg = angular_offset_deg,
              tray_slot_spacing_mm = tray_slot_spacing_mm,
              front_tray_shift_mm = front_tray_shift_mm,
              back_tray_shift_mm = back_tray_shift_mm)
    Pr = Float32.(CuArray(hcat(pos...) .* 0.001))          # 3×mpr, mm→m
    mpr = size(Pr, 2); mr = Int32(mpr)
    Munit = CUDA.zeros(Float32, 3, mpr); zerofld = CUDA.zeros(Float32, Ngrid); Btmp = similar(grid.X)
    Gx = zeros(Float32, Ns, mpr); Gy = zeros(Float32, Ns, mpr)
    for i in 1:mpr
        Munit .= 0f0; CUDA.@allowscalar Munit[1, i] = 1f0
        @cuda threads=threads blocks=blocks _Btot!(zerofld, Btmp, grid.X, grid.Y, grid.Z, Pr, Munit, mr, N)
        Gx[:, i] .= Array(Btmp)[shell_idx]
        Munit .= 0f0; CUDA.@allowscalar Munit[2, i] = 1f0
        @cuda threads=threads blocks=blocks _Btot!(zerofld, Btmp, grid.X, grid.Y, grid.Z, Pr, Munit, mr, N)
        Gy[:, i] .= Array(Btmp)[shell_idx]
    end
    return Gx, Gy
end

# Build (or load) the per-candidate operator: Gx_all, Gy_all  (Ns × mpr × Ncand)
function get_operator()
    if isfile(OPCACHE)
        d = jldopen(OPCACHE, "r"); sig = haskey(d, "SIG") ? d["SIG"] : ""; close(d)
        if sig == SIG
            @load OPCACHE Gx_all Gy_all mpr
            println("Loaded cached ring operator ($(Ncand) rings × $(mpr) mags).")
            return Gx_all, Gy_all, mpr
        end
    end
    println("Building ring operator for $(Ncand) candidate rings … (one-time GPU pass)")
    Gx1, Gy1 = ring_columns(candidates[1]); mpr = size(Gx1, 2)
    Gx_all = zeros(Float32, Ns, mpr, Ncand); Gy_all = zeros(Float32, Ns, mpr, Ncand)
    Gx_all[:, :, 1] .= Gx1; Gy_all[:, :, 1] .= Gy1
    for c in 2:Ncand
        Gx_all[:, :, c], Gy_all[:, :, c] = ring_columns(candidates[c])
        (c % 10 == 0 || c == Ncand) && @printf("  built %d/%d rings\r", c, Ncand)
    end
    println()
    @save OPCACHE Gx_all Gy_all mpr SIG candidates
    mem = round(2 * length(Gx_all) * sizeof(Float32) / 1e6, digits = 1)
    println("Cached ring operator → ", OPCACHE, "  ($(mem) MB)")
    return Gx_all, Gy_all, mpr
end

const Gx_all, Gy_all, MPR = get_operator()

# Solve in Float64. The cached operator is Float32 (compact on disk / straight from the
# GPU kernel), but feeding a Float32 operator into the Float64 L-BFGS did two bad things:
#   (1) GX*u, GX'*adj, … mixed Float32×Float64 → Julia's SLOW generic matmul (no BLAS);
#   (2) the gradient sat at the Float32 noise floor, so g_tol=1e-10 was never met and
#       EVERY solve ran to the 2000-iteration cap (~100 ms each) instead of ~30 iters.
# Float64 copies fix both. Cost: 2×(1742×7×50)×8 B ≈ 5 MB — negligible.
const GXF = Float64.(Gx_all)
const GYF = Float64.(Gy_all)

# Ranking sub-shell. Each subset solve is an (MPR·k)-variable L-BFGS (e.g. 84·4 = 336)
# whose cost is dominated by the Ns×(MPR·k) operator matmul, so it scales with the
# number of shell ROWS. The bore field is smooth, so a few hundred well-spread rows
# rank subsets essentially identically to all Ns (ppm differs <0.1%). We therefore
# RANK on this sub-shell and re-solve the WINNER on the full shell for the reported
# ppm — ~10–35× faster per ranking solve (Ns 1742→250). Raise/lower RANK_NS to trade
# ranking fidelity vs speed.
const RANK_NS   = min(Ns, 250)
const rank_rows = unique(round.(Int, range(1, Ns; length = RANK_NS)))
const GXR       = GXF[rank_rows, :, :]
const GYR       = GYF[rank_rows, :, :]
const b_rank    = b_shell[rank_rows]

# gradient self-check on a small subset (analytic vs finite difference)
let idxs = collect(1:min(2, Ncand))
    GX = reshape(GXF[:, :, idxs], Ns, MPR * length(idxs))
    GY = reshape(GYF[:, :, idxs], Ns, MPR * length(idxs))
    mu = fill(MU_MAG, MPR * length(idxs))
    fg! = make_data_cost(GX, GY, b_shell, mu; data_term = grad_data_term, beta = grad_softrange_beta)
    Random.seed!(0)
    @printf("gradient self-check (FD) : max rel err = %.2e\n", fd_gradcheck(fg!, 2π .* rand(MPR * length(idxs))))
end

# ---------------------------------------------------------------------------
# Solve the variance problem for a subset of candidate indices → (ppm, θ).
# idxs are sorted so the magnet order matches setup.jl (rings ascending).
# ---------------------------------------------------------------------------
function solve_subset(idxs; seeds, g_tol = 1e-10, iterations = 2000,
                      GXop = GXF, GYop = GYF, bvec = b_shell)
    k  = length(idxs)
    GX = reshape(GXop[:, :, idxs], size(GXop, 1), MPR * k)
    GY = reshape(GYop[:, :, idxs], size(GYop, 1), MPR * k)
    mu = fill(MU_MAG, MPR * k)
    fg! = make_data_cost(GX, GY, bvec, mu; data_term = grad_data_term, beta = grad_softrange_beta)
    f(θ)     = fg!(true, nothing, θ)
    g!(G, θ) = (fg!(nothing, G, θ); G)
    bestθ = nothing; bestp = Inf
    for s in seeds
        Random.seed!(s)
        θ = Optim.minimizer(optimize(f, g!, 2π .* rand(MPR * k), LBFGS(),
                                     Optim.Options(g_tol = g_tol, iterations = iterations)))
        p = shell_metrics(GX, GY, bvec, mu, θ).ppm
        p < bestp && (bestp = p; bestθ = θ)
    end
    return bestp, bestθ
end

# two candidate rings (by index) are compatible only if they are far enough apart:
#   • at least ring_search_min_sep mm axially, AND
#   • more than ring_search_min_spots_between empty tray spots between them
#     (spots between trays a,b = |a−b|−1), so "3 or fewer spots" ⇒ rejected.
pair_ok(i, j) = abs(cand_z[i] - cand_z[j]) >= ring_search_min_sep &&
                (abs(candidates[i] - candidates[j]) - 1) >= ring_search_min_spots_between
set_ok(idxs) = all(p -> all(q -> p >= q || pair_ok(idxs[p], idxs[q]), eachindex(idxs)), eachindex(idxs))

const SEEDS_SEARCH = first(grad_seeds):first(grad_seeds)     # 1 seed while ranking (unimodal)
const SEEDS_FINAL  = grad_seeds                              # multistart for the reported winner
# Ranking only needs to ORDER subsets, not nail the angles, so a loose tol + low iter
# cap lets each ranking solve stop in ~tens of iters. The winner is re-solved tight below.
const RANK_GTOL    = 1e-7
const RANK_ITERS   = 400

# ---------------------------------------------------------------------------
# n bounded by n_max and (optionally) the magnet budget
# ---------------------------------------------------------------------------
n_req = ring_search_n
budget_cap = ring_search_magnet_budget > 0 ? fld(ring_search_magnet_budget, MPR) : n_req
# :paired places rings two at a time (−a and +a), so n is the TOTAL ring count and must be even.
const PAIRED = ring_search_method === :paired
PAIRED && isodd(n_req) &&
    error("ring_search_method = \"paired\" needs an EVEN ring_search_n (rings come in ±pairs); got $(n_req).")
const PAIR_MAGS = PAIRED ? paired_magnitudes(candidates) : Int[]
PAIRED && isempty(PAIR_MAGS) &&
    error("paired ring search: no tray has BOTH +a and −a in ring_search_range $(ring_search_range) — use a range symmetric about 0.")
n_cap = min(n_req, ring_search_n_max, budget_cap, Ncand, PAIRED ? 2 * length(PAIR_MAGS) : typemax(Int))
PAIRED && isodd(n_cap) && (n_cap -= 1)                     # a cap can leave an odd count; drop the unpaired ring
const n = n_cap
n < n_req && @warn "n reduced" requested=n_req using n_max=ring_search_n_max budget_cap=budget_cap Ncand=Ncand
n >= 1 || error("ring search: n reduced to 0 — check ring_search_n, ring_search_n_max, the magnet budget and the range.")

# baseline (no shims) for reference
const ppm0 = 1e6 * (maximum(b_shell) - minimum(b_shell)) / mean(b_shell)

println("\n", "="^64)
println("RING SEARCH   method=$(ring_search_method)   n=$(n)   candidates=$(Ncand)  (trays $(candidates[1])…$(candidates[end]))")
PAIRED && println("  paired: $(n ÷ 2) pair(s) of {−a, +a} chosen from $(length(PAIR_MAGS)) magnitudes (|tray| $(PAIR_MAGS[1])…$(PAIR_MAGS[end]))")
println("  objective=$(grad_data_term)   min_sep=$(ring_search_min_sep) mm   min_spots=$(ring_search_min_spots_between)   baseline ppm=$(round(ppm0,digits=1))")
println("="^64)

# ---------------------------------------------------------------------------
# GREEDY forward selection
# ---------------------------------------------------------------------------
function run_greedy()
    selected = Int[]; remaining = collect(1:Ncand); trace = NamedTuple[]
    for r in 1:n
        bestc = 0; bestp = Inf; bestθ = nothing
        for c in remaining
            all(s -> pair_ok(c, s), selected) || continue
            p, θ = solve_subset(sort(vcat(selected, c)); seeds = SEEDS_SEARCH, g_tol = RANK_GTOL,
                                iterations = RANK_ITERS, GXop = GXR, GYop = GYR, bvec = b_rank)
            p < bestp && (bestp = p; bestc = c; bestθ = θ)
        end
        bestc == 0 && (@warn "no candidate satisfies min_sep at round $r — stopping"; break)
        push!(selected, bestc); filter!(!=(bestc), remaining)
        rings = sort(candidates[selected])
        push!(trace, (round = r, added = candidates[bestc], rings = rings, ppm = bestp))
        @printf("  round %d: +tray %-4d → rings %-28s ppm = %.1f\n", r, candidates[bestc], string(rings), bestp)
    end
    return sort(selected), trace
end

# ---------------------------------------------------------------------------
# EXHAUSTIVE over all C(Ncand, n) subsets (guarded)
# (for_each_combination lives in ring_combos.jl)
# ---------------------------------------------------------------------------

function run_exhaustive()
    nC = binomial(Ncand, n)
    nC > ring_search_max_combos &&
        error("C($Ncand,$n) = $nC exceeds ring_search_max_combos = $(ring_search_max_combos).\n" *
              "Use method=\"greedy\", lower ring_search_n, or narrow ring_search_range.")
    println("  scoring $(nC) subsets …")
    best_idx = Int[]; bestp = Inf; results = Tuple{Vector{Int},Float64}[]; cnt = 0
    for_each_combination(Ncand, n) do idx
        set_ok(idx) || return
        p, _ = solve_subset(collect(idx); seeds = SEEDS_SEARCH, g_tol = RANK_GTOL,
                            iterations = RANK_ITERS, GXop = GXR, GYop = GYR, bvec = b_rank)
        push!(results, (candidates[idx], p))
        p < bestp && (bestp = p; best_idx = collect(idx))
        cnt += 1; cnt % 1000 == 0 && (@printf("    %d scored\n", cnt); flush(stdout))
    end
    println()
    return best_idx, results
end

# ---------------------------------------------------------------------------
# PAIRED: every choice of n/2 pair magnitudes → the symmetric set {−a…, +a…}.
# Only the POSITIONS are paired; each magnet's angle is still free (no mirror
# constraint on the solution), so this is a restriction of the ring search's
# space, not of the physics. Pairing is by tray NUMBER; the axial mirror is exact
# only when front_tray_shift_mm == back_tray_shift_mm.
# ---------------------------------------------------------------------------
function run_paired()
    m  = n ÷ 2
    nC = binomial(length(PAIR_MAGS), m)
    nC > ring_search_max_combos &&
        error("paired: C($(length(PAIR_MAGS)),$m) = $nC exceeds ring_search_max_combos = $(ring_search_max_combos).\n" *
              "Lower ring_search_n, narrow ring_search_range, or raise ring_search_max_combos.")
    println("  scoring up to $(nC) paired sets …")
    best_idx = Int[]; bestp = Inf; results = Tuple{Vector{Int},Float64}[]; cnt = 0; skipped = 0
    for_each_paired_set(candidates, m) do idx
        set_ok(idx) || (skipped += 1; return)
        p, _ = solve_subset(copy(idx); seeds = SEEDS_SEARCH, g_tol = RANK_GTOL,
                            iterations = RANK_ITERS, GXop = GXR, GYop = GYR, bvec = b_rank)
        push!(results, (candidates[idx], p))
        p < bestp && (bestp = p; best_idx = copy(idx))
        cnt += 1; cnt % 250 == 0 && (@printf("    %d scored\n", cnt); flush(stdout))
    end
    println("  scored $(cnt) sets; $(skipped) rejected by min_sep / min_spots_between")
    return best_idx, results
end

# ---------------------------------------------------------------------------
best_idx, extra = ring_search_method === :greedy ? run_greedy() :
                  ring_search_method === :exhaustive ? run_exhaustive() :
                  PAIRED ? run_paired() :
                  error("ring_search_method = $(ring_search_method); use :greedy, :exhaustive or :paired")
isempty(best_idx) && error("ring search produced no valid ring set (check min_sep / range).")

# re-solve the winner with the full multistart for a robust reported ppm
best_rings = sort(candidates[best_idx])
ppm_best, θ_best = solve_subset(sort(best_idx); seeds = SEEDS_FINAL)
@printf("\nBEST %d rings: %s   ppm = %.1f   (baseline %.1f → %.1f%% of baseline)\n",
        n, string(best_rings), ppm_best, ppm0, 100 * ppm_best / ppm0)

# --- outputs: CSV + plot + solution jld2 ------------------------------------
if ring_search_method === :greedy
    df = DataFrame(round = [t.round for t in extra], added = [t.added for t in extra],
                   rings = [string(t.rings) for t in extra], ppm = [t.ppm for t in extra])
else
    sort!(extra, by = x -> x[2])
    top = extra[1:min(50, length(extra))]
    df = DataFrame(rank = 1:length(top), rings = [string(r) for (r, _) in top], ppm = [p for (_, p) in top])
end
csv = joinpath(OUTDIR, "ring_search.csv"); CSV.write(csv, df)
println("Results → ", csv)

try
    using GLMakie
    fig = Figure(size = (720, 470))
    if ring_search_method === :greedy
        ax = Axis(fig[1, 1]; title = "Greedy ring selection", xlabel = "# rings", ylabel = "ppm")
        scatterlines!(ax, [t.round for t in extra], [t.ppm for t in extra]; color = :dodgerblue, markersize = 10)
    elseif PAIRED
        srt = sort(extra, by = x -> x[2])
        if n == 2   # one pair: ppm vs the pair's tray magnitude
            ax = Axis(fig[1, 1]; title = "Single ±pair ppm vs |tray|", xlabel = "|tray|  (rings −a, +a)", ylabel = "ppm")
            ord = sort(extra, by = x -> maximum(x[1]))
            scatterlines!(ax, [maximum(r) for (r, _) in ord], [p for (_, p) in ord]; color = :dodgerblue, markersize = 9)
        else        # all sets ranked best → worst
            ax = Axis(fig[1, 1]; title = "Paired sets, ranked (n=$(n) rings)", xlabel = "rank", ylabel = "ppm")
            scatter!(ax, 1:length(srt), [p for (_, p) in srt]; color = :dodgerblue, markersize = 5)
        end
    elseif n == 2
        zs = sort(unique(cand_z))
        Mheat = fill(NaN, length(zs), length(zs)); zi = Dict(z => i for (i, z) in enumerate(zs))
        for (rings, p) in extra
            i = zi[cand_z[findfirst(==(rings[1]), candidates)]]; j = zi[cand_z[findfirst(==(rings[2]), candidates)]]
            Mheat[i, j] = p; Mheat[j, i] = p
        end
        ax = Axis(fig[1, 1]; title = "Pair ppm (z₁ vs z₂, mm)", xlabel = "ring 1 z", ylabel = "ring 2 z")
        heatmap!(ax, zs, zs, Mheat; colormap = :viridis)
        Colorbar(fig[1, 2])
    end
    png = joinpath(OUTDIR, "ring_search.png"); save(png, fig); println("Plot → ", png)
catch e
    @warn "plot skipped (CSV still written)" exception = e
end

θdeg = Float32.(mod.(rad2deg.(θ_best), 360.0)); final_state = ones(Float32, length(θ_best))
@save joinpath(OUTDIR, "ring_search_best.jld2") best_rings θdeg ppm_best method=String(ring_search_method) n
println("Best solution → ", joinpath(OUTDIR, "ring_search_best.jld2"))

# --- optionally apply the winner into config + save the standard Stage-1 result
if ring_search_apply
    lines = readlines(CONFIGTOML)
    for (i, ln) in enumerate(lines)
        if occursin(r"^\s*positions_in_tray_new_wished\s*=", ln)
            h = findfirst('#', ln); cmt = h === nothing ? "" : "   " * rstrip(ln[h:end])
            lines[i] = "positions_in_tray_new_wished = [" * join(best_rings, ", ") * "]" * cmt
        end
    end
    open(CONFIGTOML, "w") do io; for ln in lines; println(io, ln); end; end
    λ = 0.0; bestθ = θdeg; ppm = ppm_best
    mkpath(dirname(optimizer_result_path))
    @save optimizer_result_path λ bestθ final_state ppm
    println("Applied: positions_in_tray_new_wished = ", best_rings)
    println("Saved standard Stage-1 result → ", optimizer_result_path, "  (export/STL can run now)")
end
