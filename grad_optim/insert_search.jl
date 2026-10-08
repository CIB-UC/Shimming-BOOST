# insert_search.jl  —  sequential PER-INSERT placement (where + rotations)
#
#   julia grad_optim/insert_search.jl
#
# A different granularity from ring_search.jl. Instead of asking "which RINGS
# should I fill" (84 magnets at a time), this asks "which single INSERT should I
# place next" — one tray at one axial ring, `mags_per_segment` (7) magnets — and
# then builds on top of it.
#
# THE ALGORITHM (frozen sequential placement / matching pursuit):
#
#   b ← measured base field on the shell
#   repeat n times:
#       for every free, legal insert slot (ring c, tray t):
#           solve its 7 angles against the CURRENT b     (7-variable L-BFGS)
#           score the resulting ppm
#       keep the best slot, re-solve it on the full shell with multistart,
#       FREEZE its angles and FOLD its field into b
#
# So each round searches against a background that already contains everything
# placed so far — "build on from the best insert we have put". Placed angles are
# never revisited, which is what makes this different from ring_search's greedy
# (that re-solves every placed ring jointly at each round). Each solve here is
# only 7 variables regardless of how many inserts are already down, so the cost
# per round is flat.
#
# HOW MANY INSERTS: from the magnet budget. inserts = magnets_available ÷
# mags_per_segment (integer division — every insert is assumed FULL; we never
# plan a partially-filled insert).
#
# SPACING RULE: two inserts in the SAME TRAY must have AT LEAST
# insert_search_min_spots_between empty ring slots between them. Different trays
# are physically separate printed pieces, so they are unconstrained.
#
# IMPORTANT — with fixed-magnitude magnets an insert cannot be "switched off",
# only rotated, so ppm is NOT guaranteed to improve monotonically. This script
# places the whole budget, logs ppm after every insert, and reports which PREFIX
# scored best, so you can choose to print fewer than you budgeted.
#
# OUTPUT: a sparse shim CSV (only the placed inserts) that Stage 2 consumes
# directly, PLUS a standard Stage-1 result jld2 whose `final_state` carries the
# sparsity (1 = insert placed, 0 = empty slot). That mask is what lets the rest of
# the pipeline treat this like any other result: eval_metrics scores it, and
# export_csv.jl regenerates the same sparse CSV instead of a dense one.
#
# RingNumber in the CSV, and every printed/logged "ring", is the REAL, physical
# InsertPos (candidates[...], e.g. -7) — never a 0-based sequential index. The
# only place a sequential position is still used is internal array addressing
# for final_state/θ_all (best_rings_slot below), which is never written to the
# CSV or printed anywhere.
#
# Config knobs: insert_search_* + magnets_available in config.toml.

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))          # project lives in the repo root
include(joinpath(@__DIR__, "..", "pipeline_config.jl"))
if eval_domain === :shell                                  # score on a spherical shell at Rmax …
    include(joinpath(@__DIR__, "..", "setup_shell.jl"))
else                                                        # … or the dense grid (default)
    include(joinpath(@__DIR__, "..", "setup.jl"))
end
include(joinpath(@__DIR__, "..", "kernels", "f_kernel.jl")) # _Btot!
include(joinpath(@__DIR__, "grad_math.jl"))                # make_data_cost, shell_metrics
using Optim, Random, JLD2, Statistics, Printf, DataFrames, CSV, Dates

const MU_MAG     = magnet_moment_Am2                        # per-magnet moment (config magnet_Br_T, magnet_side_mm)
const MPS        = mags_per_segment                         # magnets per insert (7)
const OUTDIR     = joinpath(optimizer_iter_dir, "InsertSearch"); mkpath(OUTDIR)
const CONFIGTOML = joinpath(ROOT, "config.toml")

# The per-ring operator is SHARED with ring_search.jl: a ring block already holds
# all num_trays × MPS columns, and one insert is just the MPS-column slice for its
# tray. Same cache file, same signature ⇒ whichever search runs first pays the
# one-time GPU build and the other loads it. Keep SIG identical to ring_search.jl.
const OPCACHE = joinpath(optimizer_iter_dir, "GradOpt", "ring_operator.jld2")
mkpath(dirname(OPCACHE))

# --- candidate rings (integer tray numbers), 0 excluded ----------------------
const lo, hi     = insert_search_range[1], insert_search_range[2]
const candidates = [t for t in lo:insert_search_step:hi if t != 0]
const Ncand      = length(candidates)
isempty(candidates) && error("insert_search_range = $(insert_search_range) yields no candidate rings.")

# --- shell bookkeeping (rows of the operator) --------------------------------
const shell_idx = findall(>(0f0), vec(Array(msk)))
const Ns        = length(shell_idx)
const b_shell   = Float64.(vec(Array(fld_field))[shell_idx])      # base field on shell (mT)
const Ngrid     = Int(length(grid.X))

# geometry/field signature — MUST match ring_search.jl so the cache is shared
const SIG = string((candidates, shim_radius_mm, mags_per_segment, num_trays,
                    angle_per_segment_deg, angular_offset_deg,
                    tray_slot_spacing_mm, front_tray_shift_mm, back_tray_shift_mm,
                    Rmin, Rmax, interpolated_fieldmap_name, MU_MAG, String(eval_domain)))

# ---------------------------------------------------------------------------
# Magnet positions for one ring, in the SAME column order the operator uses:
# positions_from_rings_mm loops segments outer, magnets inner, so segment s owns
# columns (s-1)*MPS+1 … s*MPS.
# ---------------------------------------------------------------------------
ring_positions(t) = positions_from_rings_mm([t]; occupied_trays = positions_in_tray_occupied,
          shim_radius_mm = shim_radius_mm, mags_per_segment = mags_per_segment,
          num_segments = num_trays, angle_per_segment_deg = angle_per_segment_deg,
          angular_offset_deg = angular_offset_deg,
          tray_slot_spacing_mm = tray_slot_spacing_mm,
          front_tray_shift_mm = front_tray_shift_mm,
          back_tray_shift_mm = back_tray_shift_mm)

seg_cols(s) = ((s - 1) * MPS + 1):(s * MPS)      # operator columns owned by segment s

function ring_columns(t)
    pos = ring_positions(t)
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

function get_operator()
    if isfile(OPCACHE)
        d = jldopen(OPCACHE, "r"); sig = haskey(d, "SIG") ? d["SIG"] : ""; close(d)
        if sig == SIG
            @load OPCACHE Gx_all Gy_all mpr
            println("Loaded cached ring operator ($(Ncand) rings × $(mpr) mags) — shared with ring_search.")
            return Gx_all, Gy_all, mpr
        end
    end
    println("Building ring operator for $(Ncand) candidate rings … (one-time GPU pass)")
    Gx1, Gy1 = ring_columns(candidates[1]); mpr = size(Gx1, 2)
    Gx_all = zeros(Float32, Ns, mpr, Ncand); Gy_all = zeros(Float32, Ns, mpr, Ncand)
    Gx_all[:, :, 1] .= Gx1; Gy_all[:, :, 1] .= Gy1
    for c in 2:Ncand
        Gx_all[:, :, c], Gy_all[:, :, c] = ring_columns(candidates[c])
        (c % 10 == 0 || c == Ncand) && (@printf("  built %d/%d rings\n", c, Ncand); flush(stdout))
    end
    @save OPCACHE Gx_all Gy_all mpr SIG candidates
    mem = round(2 * length(Gx_all) * sizeof(Float32) / 1e6, digits = 1)
    println("Cached ring operator → ", OPCACHE, "  ($(mem) MB)")
    return Gx_all, Gy_all, mpr
end

const Gx_all, Gy_all, MPR = get_operator()
MPR == num_trays * MPS || error(
    "operator has $(MPR) columns per ring but num_trays×mags_per_segment = $(num_trays*MPS). " *
    "Delete $(OPCACHE) and re-run so it rebuilds for the current geometry.")

# Float64 copies: mixing Float32 operator with Float64 L-BFGS drops out of BLAS and
# strands the gradient at the Float32 noise floor (see the note in ring_search.jl).
const GXF = Float64.(Gx_all)
const GYF = Float64.(Gy_all)

# Ranking sub-shell — normally rank candidates CHEAPLY on a subsample, then
# re-solve only the winner on all Ns rows. Set RANK_FULL_SHELL = true to instead
# rank every candidate slot on the FULL shell (all Ns points) — an accuracy/speed
# experiment: does the ~250-point subsample ever pick a different "best slot" than
# scoring against everything? Cost scales with Ns/RANK_NS per round (e.g. Ns=1742
# vs 250 ⇒ ~7x slower ranking), so expect this to take noticeably longer per insert.
const RANK_FULL_SHELL = true
const RANK_NS   = RANK_FULL_SHELL ? Ns : min(Ns, 250)
const rank_rows = RANK_FULL_SHELL ? collect(1:Ns) : unique(round.(Int, range(1, Ns; length = RANK_NS)))
const GXR       = GXF[rank_rows, :, :]
const GYR       = GYF[rank_rows, :, :]

const SEEDS_SEARCH = first(grad_seeds):first(grad_seeds)    # 1 seed while ranking
const SEEDS_FINAL  = grad_seeds                             # multistart for the chosen insert
const RANK_GTOL    = 1e-7
const RANK_ITERS   = 400
const MU7          = fill(MU_MAG, MPS)

# gradient self-check on one insert-sized block (analytic vs finite difference)
let GX = GXF[:, seg_cols(1), 1], GY = GYF[:, seg_cols(1), 1]
    fg! = make_data_cost(GX, GY, b_shell, MU7; data_term = grad_data_term, beta = grad_softrange_beta)
    Random.seed!(0)
    @printf("gradient self-check (FD) : max rel err = %.2e\n", fd_gradcheck(fg!, 2π .* rand(MPS)))
end

# ---------------------------------------------------------------------------
# Segment index → physical tray number. positions_from_rings_mm builds segments
# at fixed angles; CSV_to_STL later re-derives the tray from (x,y). We map it the
# SAME way here so the spacing rule and the printed folder agree.
# (assign_tray replicated from utils/helping_functions_for_JIG.jl — kept inline so
#  this script does not have to pull in the Gmsh-dependent geometry helpers.)
# ---------------------------------------------------------------------------
assign_tray_local(x, y) =
    mod(round(Int, mod(atan(y, x) * 180 / pi, 360.0) / (360.0 / num_trays)) + 8, num_trays) + 1

const SEG_TRAY = let pos = ring_positions(candidates[1]), tr = Int[]
    for s in 1:num_trays
        ts = [assign_tray_local(pos[i][1], pos[i][2]) for i in seg_cols(s)]
        length(unique(ts)) > 1 &&
            @warn "segment $s spans more than one tray — the middle magnet decides" trays = unique(ts)
        push!(tr, ts[cld(length(ts), 2)])          # the middle magnet decides
    end
    tr
end
length(unique(SEG_TRAY)) == num_trays ||
    @warn "segment→tray map is not a bijection — the same-tray spacing rule may be wrong" map = SEG_TRAY

# ---------------------------------------------------------------------------
# Spacing rule — SAME TRAY only.
# Empty ring slots strictly between trays a and b. Tray 0 does not exist, so a
# pair straddling the centre has one fewer real slot between them than |a−b|−1.
# (ring_search.jl uses the simpler |a−b|−1 and so is very slightly stricter
#  across the origin; this one counts physical slots.)
# ---------------------------------------------------------------------------
function spots_between(a::Int, b::Int)
    l, h = minmax(a, b)
    n = h - l - 1
    (l < 0 < h) && (n -= 1)
    return n
end
# slots are (candidate-ring index, segment index); only SAME-tray pairs are constrained
function compatible(ci, s, placed)
    for p in placed
        SEG_TRAY[s] == SEG_TRAY[p.seg] || continue                    # different tray ⇒ free
        spots_between(candidates[ci], candidates[p.ci]) >= insert_search_min_spots_between || return false
    end
    return true
end

# InsertPos -> the printed P/N sign token, matching make_label's engraved N/P glyphs
# (utils/helping_functions_for_JIG.jl) — e.g. -7 -> "N07", +12 -> "P12".
insertpos_label(pos::Int) = (pos < 0 ? "N" : "P") * lpad(abs(pos), 2, '0')

# ---------------------------------------------------------------------------
# Budget → number of inserts. Every insert is assumed FULL (no partial inserts).
# ---------------------------------------------------------------------------
const n_budget = fld(magnets_available, MPS)
const n_slots  = Ncand * num_trays
const n_cap    = insert_search_max_inserts > 0 ? insert_search_max_inserts : typemax(Int)
const n_place  = min(n_budget, n_cap, n_slots)
n_place < 1 && error("magnets_available = $(magnets_available) gives $(n_budget) full inserts " *
                     "($(MPS) magnets each) — need at least $(MPS) magnets.")
leftover = magnets_available - n_place * MPS

ppm_of(v) = 1e6 * (maximum(v) - minimum(v)) / mean(v)
const ppm0 = ppm_of(b_shell)

println("\n", "="^70)
println("INSERT SEARCH   (sequential per-insert placement, frozen angles)")
@printf("  magnets available : %d  →  %d full inserts of %d", magnets_available, n_budget, MPS)
leftover > 0 && @printf("   (%d magnets left over, not placed)", leftover)
println()
println("  placing           : $(n_place) insert(s)" *
        (n_place < n_budget ? "   (capped by insert_search_max_inserts / slots)" : ""))
println("  candidate slots   : $(Ncand) rings × $(num_trays) trays = $(n_slots)" *
        "   (trays $(candidates[1])…$(candidates[end]))")
println("  spacing rule      : at least $(insert_search_min_spots_between) empty ring slots between inserts IN THE SAME TRAY")
println("  ranking pass      : " * (RANK_FULL_SHELL ?
        "FULL shell ($(Ns) pts/candidate — no subsampling)" :
        "subsampled ($(RANK_NS) of $(Ns) pts/candidate)"))
@printf("  objective         : %s   ·   baseline ppm = %.1f\n", grad_data_term, ppm0)
println("="^70)

# ---------------------------------------------------------------------------
# Solve one insert's MPS angles against a given base field.
# ---------------------------------------------------------------------------
function solve_insert(GX, GY, bvec; seeds, g_tol, iterations)
    fg! = make_data_cost(GX, GY, bvec, MU7; data_term = grad_data_term, beta = grad_softrange_beta)
    f(θ)     = fg!(true, nothing, θ)
    g!(G, θ) = (fg!(nothing, G, θ); G)
    bestθ = nothing; bestp = Inf
    for s in seeds
        Random.seed!(s)
        θ = Optim.minimizer(optimize(f, g!, 2π .* rand(MPS), LBFGS(),
                                     Optim.Options(g_tol = g_tol, iterations = iterations)))
        p = shell_metrics(GX, GY, bvec, MU7, θ).ppm
        p < bestp && (bestp = p; bestθ = θ)
    end
    return bestp, bestθ
end

# ---------------------------------------------------------------------------
# MAIN LOOP — place one insert per round, freezing and folding as we go.
# ---------------------------------------------------------------------------
# Wrapped in a function (not a bare top-level loop) so the inner solves see typed
# locals instead of globals — this is the hot path: n_place × Ncand × num_trays solves.
function run_insert_search()
    b_cur  = copy(b_shell)                  # base + everything placed so far
    placed = NamedTuple[]
    for step in 1:n_place
        b_rank = b_cur[rank_rows]
        bestp = Inf; best_ci = 0; best_s = 0; scanned = 0
        for ci in 1:Ncand, s in 1:num_trays
            any(p -> p.ci == ci && p.seg == s, placed) && continue      # slot already used
            compatible(ci, s, placed) || continue
            p, _ = solve_insert(GXR[:, seg_cols(s), ci], GYR[:, seg_cols(s), ci], b_rank;
                                seeds = SEEDS_SEARCH, g_tol = RANK_GTOL, iterations = RANK_ITERS)
            scanned += 1
            p < bestp && (bestp = p; best_ci = ci; best_s = s)
        end
        if best_ci == 0
            @warn "no legal slot left at round $step — stopping early" placed = length(placed)
            break
        end

        # re-solve the winner on the FULL shell with multistart, then freeze + fold
        GXf = GXF[:, seg_cols(best_s), best_ci]
        GYf = GYF[:, seg_cols(best_s), best_ci]
        _, θ = solve_insert(GXf, GYf, b_cur; seeds = SEEDS_FINAL, g_tol = 1e-10, iterations = 2000)
        b_cur .+= GXf * (MU7 .* cos.(θ)) .+ GYf * (MU7 .* sin.(θ))
        ppm_now = ppm_of(b_cur)

        push!(placed, (step = step, ci = best_ci, seg = best_s,
                       ring = candidates[best_ci], tray = SEG_TRAY[best_s],
                       θ = θ, ppm = ppm_now, magnets = step * MPS))
        @printf("  insert %2d/%d: ring %-4d tray %-3d  (%d slots scored)   ppm = %9.1f   (%.1f%% of baseline)\n",
                step, n_place, candidates[best_ci], SEG_TRAY[best_s], scanned, ppm_now, 100 * ppm_now / ppm0)
        flush(stdout)
    end
    return placed
end

const placed = run_insert_search()
isempty(placed) && error("insert search placed nothing (check insert_search_range / spacing rule).")

# ---------------------------------------------------------------------------
# Best PREFIX — fixed-magnitude magnets mean more inserts is not always better.
# ---------------------------------------------------------------------------
const ppm_trace = [p.ppm for p in placed]
const k_best    = argmin(ppm_trace)
const ppm_best  = ppm_trace[k_best]

println("\n", "-"^70)
@printf("placed %d insert(s); BEST PREFIX = first %d  →  ppm = %.1f  (baseline %.1f, %.1f%%)\n",
        length(placed), k_best, ppm_best, ppm0, 100 * ppm_best / ppm0)
if k_best < length(placed)
    @printf("NOTE: the last %d insert(s) made it WORSE (%.1f → %.1f). Printing only the first %d\n      uses %d magnets instead of %d and scores better.\n",
            length(placed) - k_best, ppm_best, ppm_trace[end], k_best,
            k_best * MPS, length(placed) * MPS)
end
println("-"^70)

# Everything below is written for the BEST PREFIX, not the full budget.
const chosen = placed[1:k_best]

# --- outputs: trace CSV + plot + solution jld2 -------------------------------
df = DataFrame(step = [p.step for p in placed], ring = [p.ring for p in placed],
               tray = [p.tray for p in placed], magnets_used = [p.magnets for p in placed],
               ppm = [p.ppm for p in placed],
               kept = [p.step <= k_best for p in placed])
trace_csv = joinpath(OUTDIR, "insert_search.csv"); CSV.write(trace_csv, df)
println("Trace → ", trace_csv)

try
    using GLMakie
    fig = Figure(size = (760, 480))
    ax = Axis(fig[1, 1]; title = "Sequential insert placement", xlabel = "# inserts placed", ylabel = "ppm")
    scatterlines!(ax, 1:length(ppm_trace), ppm_trace; color = :dodgerblue, markersize = 10)
    hlines!(ax, [ppm0]; color = :grey, linestyle = :dash)
    scatter!(ax, [k_best], [ppm_best]; color = :orangered, markersize = 16)
    # annotate separately: a Makie signature change here must not cost us the whole plot
    try
        text!(ax, Point2f(k_best, ppm_best); text = "  best ($(k_best) inserts, $(k_best*MPS) magnets)",
              align = (:left, :center), fontsize = 13)
    catch; end
    png = joinpath(OUTDIR, "insert_search.png"); save(png, fig); println("Plot → ", png)
catch e
    @warn "plot skipped (CSV still written)" exception = e
end

best_rings = sort(unique(p.ring for p in chosen))
@save joinpath(OUTDIR, "insert_search_best.jld2") best_rings ppm_best ppm0 k_best ppm_trace
println("Best solution → ", joinpath(OUTDIR, "insert_search_best.jld2"))

# ---------------------------------------------------------------------------
# Sparse shim CSV — X, Y, RingNumber, Angle. Only the placed inserts, so
# export_csv.jl (which writes EVERY magnet of every wished ring) cannot be used.
# RingNumber is the REAL InsertPos (e.g. -7), matching export_csv.jl's convention
# and what CSV_to_STL.jl/make_label print on the physical part.
# ---------------------------------------------------------------------------
col_X = Float64[]; col_Y = Float64[]; col_R = Int[]; col_A = Float64[]
for p in sort(chosen, by = q -> (q.ring, q.tray))
    pos = ring_positions(p.ring)[seg_cols(p.seg)]
    for (k, xyz) in enumerate(pos)
        push!(col_X, Float64(xyz[1])); push!(col_Y, Float64(xyz[2]))
        push!(col_R, p.ring); push!(col_A, mod(rad2deg(p.θ[k]), 360.0))
    end
end
# Rotate back into the scan's frame (shim_csv_frame = "scan"), exactly as export_csv.jl does, so the
# sparse CSV and a later "Export + build STL" regeneration agree. The placement plan printed above
# uses optimizer-frame tray numbers; Stage 2 re-derives the tray from the rotated (x,y).
include(joinpath(@__DIR__, "..", "utils", "viewer_frame.jl"))
if shim_csv_frame == "scan"
    sdir = read_field_direction(shell_fieldmap_path, fieldmap_path)
    sdir === nothing && (main_field_direction != "auto" ? (sdir = main_field_direction) :
        error("insert_search: can't tell the scan's B0 direction; re-run Stage 0 or set shim_csv_frame = \"optimizer\"."))
    if sdir != "+y"
        _rot = [opt_to_scan_xy(col_X[i], col_Y[i], sdir) for i in eachindex(col_X)]
        col_X = Float64.(first.(_rot)); col_Y = Float64.(last.(_rot))
        col_A = mod.(col_A .+ opt_to_scan_angle_deg(sdir), 360.0)
    end
    println("Shim CSV rotated back to the SCAN frame (B0 along ", sdir, ")")
end
shim_df = DataFrame("X (mm)" => col_X, "Y (mm)" => col_Y,
                    "RingNumber" => col_R, "Angle (deg)" => col_A)
local_csv = joinpath(OUTDIR, "insert_shim.csv")
CSV.write(local_csv, shim_df)      # column names already match the Stage-2 schema
println("Shim CSV → ", local_csv, "  (", nrow(shim_df), " magnets, ",
        length(chosen), " inserts, ", length(best_rings), " ring(s))")

# ---------------------------------------------------------------------------
# Apply: publish the shim CSV at the Stage-2 seam and record the rings used.
# NOTE: no optimizer-result jld2 is written — a sparse per-insert layout cannot be
# expressed in that format (it assumes every tray of every wished ring is filled).
# So Stage 2 must be run with "Build STL only", NOT "Export + build STL".
# ---------------------------------------------------------------------------
if insert_search_apply
    mkpath(dirname(shim_csv_path))
    CSV.write(shim_csv_path, shim_df)
    # --- standard Stage-1 result, with the sparse layout carried by final_state ---
    # final_state is the pipeline's per-magnet ON/OFF mask, so a sparse layout IS
    # expressible in the normal result format: 1 = insert placed here, 0 = empty slot.
    # Writing it means utils/eval_metrics.jl can score this run like any other, and
    # export_csv.jl reproduces exactly this sparse CSV instead of a dense one — so a
    # full pipeline run or "Export + build STL" is now safe rather than destructive.
    # Magnet order matches setup.jl: ring-major (over the rings we just wrote to
    # positions_in_tray_new_wished, in THAT ARRAY'S order — best_rings is already
    # sorted ascending, and pos_trays.jl's loop preserves array order unchanged),
    # then segment, then magnet.
    #
    # ring_slot below is a purely INTERNAL flat-array position (best_rings is sorted,
    # so slot 0 = smallest InsertPos, etc.) — it exists only to index into θ_all/st_all
    # and must never be written to the CSV or printed; RingNumber and the placement
    # plan use the real p.ring (InsertPos) everywhere else in this file.
    ring_slot = Dict(r => i - 1 for (i, r) in enumerate(best_rings))
    n_all  = num_trays * MPS * length(best_rings)
    θ_all  = fill(Float64(initial_angle_deg), n_all)   # empty slots: arbitrary (state 0 zeroes them)
    st_all = zeros(Float64, n_all)
    for p in chosen
        base = ring_slot[p.ring] * num_trays * MPS + (p.seg - 1) * MPS
        for k in 1:MPS
            θ_all[base + k]  = mod(rad2deg(p.θ[k]), 360.0)
            st_all[base + k] = 1.0
        end
    end
    let λ = 0.0, bestθ = Float32.(θ_all), final_state = Float32.(st_all), ppm = ppm_best
        mkpath(dirname(optimizer_result_path))
        @save optimizer_result_path λ bestθ final_state ppm
        @save insert_result_path    λ bestθ final_state ppm
    end
    println("Stage-1 result → ", optimizer_result_path,
            "  ($(Int(sum(st_all)))/$(n_all) magnet slots filled)")
    println("Tagged copy    → ", insert_result_path)

    lines = readlines(CONFIGTOML)
    for (i, ln) in enumerate(lines)
        if occursin(r"^\s*positions_in_tray_new_wished\s*=", ln)
            h = findfirst('#', ln); cmt = h === nothing ? "" : "   " * rstrip(ln[h:end])
            lines[i] = "positions_in_tray_new_wished = [" * join(best_rings, ", ") * "]" * cmt
        end
    end
    open(CONFIGTOML, "w") do io; for ln in lines; println(io, ln); end; end
    println("Applied: positions_in_tray_new_wished = ", best_rings)
    println("Published shim CSV → ", shim_csv_path)
    println("\nNext: Stage 2 → \"Build STL only\". \"Export + build STL\" also works now —")
    println("      export_csv.jl honours final_state, so it rebuilds this same sparse CSV.")
else
    println("\ninsert_search_apply = false → nothing published. Set it true (or copy")
    println("      $(local_csv)\n      to $(shim_csv_path)) to print this layout.")
end

# --- where each insert physically goes ---------------------------------------
println("\nPLACEMENT PLAN")
println("  Ring (folder)   tray   axial z (mm)   step   ppm after")
for p in sort(chosen, by = q -> (q.ring, q.tray))
    z = ringpos_from_tray_mm([p.ring]; tray_slot_spacing_mm = tray_slot_spacing_mm,
                             front_tray_shift_mm = front_tray_shift_mm,
                             back_tray_shift_mm = back_tray_shift_mm)[1]
    @printf("  Ring_%-11s %-6d %+9.1f    %-6d %.1f\n",
            insertpos_label(p.ring), p.tray, z, p.step, p.ppm)
end
