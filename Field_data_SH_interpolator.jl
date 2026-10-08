# Field_data_SH_interpolator.jl  —  STAGE 0 (alternative)  (measured CSV → interpolated jld2)
#
# Run on its own:        julia Field_data_SH_interpolator.jl
# Or via the pipeline:   set  field_adapter = :spherical_harmonics  in
#                         pipeline_config.jl — run_pipeline.jl will then launch
#                         THIS script instead of Field_data_file_adapter.jl.
#
# Alternative Stage 0 for measurements that are NOT a complete regular grid —
# typically a SPHERICAL-SHELL scan (e.g. Josh_NoShimMeasurement_Spherical_*.csv:
# ~7000 scattered points all sitting at one radius). Field_data_file_adapter.jl's
# plain reshape fails on this kind of input with something like:
#     AssertionError: Measured grid is incomplete: 7082 rows ≠ 885×885×61 = ...
# because a scattered/non-grid scan can't be `reshape`d — it needs true
# interpolation.
#
# Method (COMBINED_PIPELINE_PLAN.md §11). Inside the current-free bore, each
# Cartesian field component is a harmonic function, so on any source-free region
# it equals a sum of REGULAR solid spherical harmonics — the solutions of
# Laplace's equation that stay finite at the origin:
#
#     B(x,y,z)  ≈  Σ_{l=0}^{L} Σ_{m=-l}^{l}  c_lm · (r/Rn)^l · Y_lm(θ,φ)
#
# `Y_lm` are the real, fully-normalized spherical harmonics and `Rn` is a
# reference radius (here: the mean radius of the measured shell) used only to
# keep the (r/Rn)^l columns near unit scale for a well-conditioned fit — it does
# not change what's being modeled. Fitting `c_lm` by least squares to the
# scattered shell samples gives a smooth analytic model of the field that can be
# evaluated ANYWHERE inside the shell — in particular on the regular grid the
# optimizer needs. This is the same idea used for B0/shim-coil field
# characterization in MRI: a single-shell scan over (θ,φ) is enough to recover
# the c_lm via the angular orthogonality of the Y_lm; picking the REGULAR
# (not irregular, ~r^-(l+1)) solid harmonic is what licenses extrapolating
# inward from the shell to the bore center.
#
# Points the fit is evaluated at beyond the original shell radius are
# extrapolation (and can drift), but `setup.jl` masks the optimizer's cost to
# r ≤ Rmax, so as long as the output grid covers the shell out to Rmax, those
# extra corner points never influence the optimizer — see plan §11.
#
# Writes the SAME outputs as Field_data_file_adapter.jl, so every downstream
# stage (setup.jl, visualizers) is unaware of which Stage 0 adapter ran:
#     By_grid  (mT)  on axes  xg, yg, zg  (mm),  with By_grid[i,j,k] = (xg[i],yg[j],zg[k])
#
# Paths come from pipeline_config.jl:
#   in :  measured_field_path        (Measured_Field_Data_csv/<name>.csv)
#   out:  fieldmap_path              (Interpolated_Field_Data_jld2/<same name>.jld2)
#
# Measured CSV layout (same scanner-export format as the reshape adapter):
#   line 1            : "Fecha y Hora,<timestamp>"   (metadata, skipped)
#   line 2            : header  ->  X, Y, Z, Gauss, T°
#   line 3 onward     : data rows; one spurious (0,0,0) marker row to drop.
#   Unlike the parallelepiped ("medicion_*") scans, X/Y/Z here are ALREADY in mm
#   (not machine steps) — see `sh_measured_unit_mm` in pipeline_config.jl.
#
# Needs only CSV / DataFrames / JLD2 / LinearAlgebra / Statistics / Printf —
# all either already a project dependency or a stdlib. No GPU, no new deps.

import Pkg
Pkg.activate(@__DIR__)
@isdefined(ITERATION) || include(joinpath(@__DIR__, "pipeline_config.jl"))

using CSV, DataFrames, JLD2, LinearAlgebra, Statistics, Printf
include(joinpath(@__DIR__, "utils", "read_measured.jl"))   # read_measured_points, detect_field_direction
include(joinpath(@__DIR__, "utils", "sh_select.jl"))       # sh_select_columns, sh_refit, sh_summary, sh_write_csv

const GAUSS_TO_MT = 0.1          # 1 Gauss = 0.1 mT

# Sign convention (kept identical to Field_data_file_adapter.jl): the main field
# B0 points along `main_field_direction` (the LAB frame). It is stored POSITIVE
# along that axis regardless of the gaussmeter probe's polarity — valid because
# B0 never crosses zero inside the bore.
const VALID_DIRS = ("+x", "-x", "+y", "-y")
@assert main_field_direction in (VALID_DIRS..., "auto") "main_field_direction = \"$(main_field_direction)\" is invalid; use one of $(VALID_DIRS) or \"auto\"."

# Lab frame → optimizer frame. Identical mapping to Field_data_file_adapter.jl's
# §8 rotation (duplicated here rather than shared, so each Stage 0 script stays
# a standalone subprocess entry point launched directly by run_pipeline.jl — see
# that file's `run_stage` helper). BOOST always optimizes a +y field; rotating
# the whole measured field about the bore (z) axis carries the inhomogeneity
# pattern rigidly into the optimizer frame so the shim trays end up on the
# correct side. (The lab direction is still saved into the jld2 for the record.)
lab_to_optimizer_xy(x, y, dir) =
    dir == "+y" ? (x, y)  :   #   0°  : already +y
    dir == "+x" ? (-y, x) :   # +90° about z
    dir == "-x" ? (y, -x) :   # -90° about z
                  (-x, -y)    # "-y": 180° about z

# -----------------------------------------------------------------------------
# 1. Read + clean the scattered shell measurement
# -----------------------------------------------------------------------------
# Format-agnostic read (simple X,Y,Z,Gauss  OR  rich Controlled scan); averages
# repeated samples, returns |B| (Gauss) + the mean field vector when available.
mx, my, mz, bgauss, meanvec = read_measured_points(measured_field_path; unit_mm = sh_measured_unit_mm)

# Main field direction: auto-detect from the measured vector, else use config.
detected = meanvec === nothing ? nothing : detect_field_direction(meanvec)
dir = main_field_direction == "auto" ?
      (detected === nothing ?
          error("main_field_direction = \"auto\" needs a scan with field components (Controlled format).") :
          detected) :
      main_field_direction
@assert dir in VALID_DIRS "main_field_direction resolved to \"$(dir)\" — must be one of $(VALID_DIRS)."

# Rotate every (x, y) from the lab frame into the optimizer frame (field → +y).
rotxy = [lab_to_optimizer_xy(mx[i], my[i], dir) for i in eachindex(mx)]
dfx = first.(rotxy); dfy = last.(rotxy); dfz = mz

n     = length(bgauss)
radii = sqrt.(dfx .^ 2 .+ dfy .^ 2 .+ dfz .^ 2)
Rn    = mean(radii)                              # reference radius (≈ shell radius)
b     = bgauss .* GAUSS_TO_MT                     # mT, positive along main field

ncoef = (sh_degree + 1)^2
n < ncoef && error(
    "Not enough data points ($n) to fit sh_degree=$(sh_degree) ($(ncoef) basis " *
    "terms) — lower sh_degree in pipeline_config.jl.",
)

# -----------------------------------------------------------------------------
# 2. Regular (real) solid spherical harmonics basis, degree 0..sh_degree
# -----------------------------------------------------------------------------
# Associated Legendre P_l^m(cosθ), l = 0..L, m = 0..l, via the standard stable
# 3-term recurrence (Condon–Shortley phase). Stored as P[l+1, m+1] = P_l^m.
function assoc_legendre_table(L::Int, ct::Float64, st::Float64)
    P = zeros(Float64, L + 1, L + 1)
    P[1, 1] = 1.0                                          # P_0^0
    for m in 0:L-1
        P[m+2, m+2] = -(2m + 1) * st * P[m+1, m+1]          # P_{m+1}^{m+1}
    end
    for m in 0:L-1
        P[m+2, m+1] = (2m + 1) * ct * P[m+1, m+1]           # P_{m+1}^{m}
    end
    for m in 0:L, l in (m+2):L
        P[l+1, m+1] = ((2l - 1) * ct * P[l, m+1] - (l + m - 1) * P[l-1, m+1]) / (l - m)
    end
    return P
end

# (l-m)! / (l+m)!  without forming large factorials (stays accurate for l up to
# well beyond any sh_degree this project would realistically use).
fact_ratio(l::Int, m::Int) = m == 0 ? 1.0 : 1.0 / prod(Float64(k) for k in (l-m+1):(l+m))

# Real, fully-normalized spherical harmonic Y_lm(θ,φ) from the Legendre table.
function real_Ylm(l::Int, m::Int, φ::Float64, P::Matrix{Float64})
    Nlm = sqrt((2l + 1) / (4π) * fact_ratio(l, abs(m)))
    if m == 0
        return Nlm * P[l+1, 1]
    elseif m > 0
        return sqrt(2) * Nlm * P[l+1, m+1] * cos(m * φ)
    else
        am = -m
        return sqrt(2) * Nlm * P[l+1, am+1] * sin(am * φ)
    end
end

# Full basis row [R_00, R_1,-1, R_10, R_11, R_2,-2, ...] for one point (x,y,z),
# length (L+1)^2, written into `row` in place.
function solid_harmonic_row!(row::AbstractVector{Float64}, x::Float64, y::Float64, z::Float64, L::Int, Rn::Float64)
    r  = sqrt(x^2 + y^2 + z^2)
    ct = r > 0 ? clamp(z / r, -1.0, 1.0) : 1.0
    st = sqrt(max(0.0, 1 - ct^2))
    φ  = atan(y, x)
    P  = assoc_legendre_table(L, ct, st)
    idx = 1
    for l in 0:L
        rl = (r / Rn)^l
        for m in -l:l
            row[idx] = rl * real_Ylm(l, m, φ, P)
            idx += 1
        end
    end
    return row
end

# -----------------------------------------------------------------------------
# 3. Least-squares fit  A c ≈ b   (column-scaled for conditioning)
# -----------------------------------------------------------------------------
A   = Matrix{Float64}(undef, n, ncoef)
row = Vector{Float64}(undef, ncoef)
for i in 1:n
    solid_harmonic_row!(row, dfx[i], dfy[i], dfz[i], sh_degree, Rn)
    A[i, :] = row
end

colnorms = [norm(@view A[:, j]) for j in 1:ncoef]
colnorms[colnorms.==0] .= 1.0
Ascaled  = A ./ colnorms'
c_scaled = Ascaled \ b
c        = c_scaled ./ colnorms

# --- which coefficients build the field map? (config: sh_select / sh_use_degree / sh_top_k)
# The decomposition of the FULL fit is always reported; for :first_n / :top_k the kept
# columns are re-fitted to the data and everything below (grid, saved `c`) uses that.
c_full   = copy(c)
sel_cols = sh_select_columns(c_full, sh_degree, sh_select; use_degree = sh_use_degree, k = sh_top_k)
sh_select === :all || (c = sh_refit(A, b, sel_cols))

resid     = A * c .- b
rms_mT    = sqrt(mean(resid .^ 2))
maxabs_mT = maximum(abs.(resid))
condA     = cond(Ascaled)

# -----------------------------------------------------------------------------
# 4. Evaluate the fit on a regular cubic grid (this is what BOOST consumes)
# -----------------------------------------------------------------------------
R  = sh_grid_radius_mm
ng = sh_grid_n
xg = collect(range(-R, R; length = ng))
yg = collect(range(-R, R; length = ng))
zg = collect(range(-R, R; length = ng))

By_grid = Array{Float64}(undef, ng, ng, ng)
grow    = Vector{Float64}(undef, ncoef)
for (i, x) in enumerate(xg), (j, y) in enumerate(yg), (k, z) in enumerate(zg)
    solid_harmonic_row!(grow, x, y, z, sh_degree, Rn)
    By_grid[i, j, k] = abs(dot(grow, c))      # mT, positive convention (matches reshape adapter)
end

# How far the fit can be trusted: outside the measured shell radius (Rn) there
# is NO supporting data, only polynomial extrapolation of the (r/Rn)^l terms,
# which can grow quickly past Rn. Viewers (Shimming_magnets_visualizer.jl /
# field_slice_viewer.jl) read this key to hide anything beyond it -- it does
# NOT affect setup.jl / the optimizer, which already restricts its own cost to
# r ≤ Rmax via its own independent shell mask.
const field_valid_radius_mm = Rn

mkpath(dirname(fieldmap_path))
field_direction = dir                            # resolved (possibly auto-detected)
sh_select_name = String(sh_select)               # JLD2-friendly copy of the mode
@save fieldmap_path By_grid xg yg zg field_direction sh_degree c Rn field_valid_radius_mm c_full sel_cols sh_select_name

# -----------------------------------------------------------------------------
println("Stage 0 (spherical-harmonics)  wrote ", fieldmap_path)
println("  main field B0 (lab) → ", dir,
        main_field_direction == "auto" ? "  (auto-detected from field vector $(round.(meanvec, digits=1)) G)" :
        dir == "+y" ? "  (no rotation)" : "  → rotated to +y optimizer frame")
@printf("  fit: degree L=%d (%d terms), %d shell points, ref radius Rn=%.2f mm\n", sh_degree, ncoef, n, Rn)
@printf("  shell radius measured: min=%.2f max=%.2f mean=%.2f mm\n", minimum(radii), maximum(radii), Rn)
@printf("  fit residual: RMS=%.4f mT, max|resid|=%.4f mT   (cond(scaled design matrix)=%.2e)\n", rms_mT, maxabs_mT, condA)
println("  output grid ", ng, "×", ng, "×", ng, "  extent ±", R, " mm",
        "   |B0| range [", round(minimum(By_grid), digits = 3), ", ",
        round(maximum(By_grid), digits = 3), "] mT")
println("  (corners with r > shell radius are extrapolation; setup.jl's r ≤ Rmax shell mask keeps them out of the cost)")
sh_summary(stdout, c_full, c, sel_cols, sh_degree; mode = sh_select, use_degree = sh_use_degree, k = sh_top_k, A = A, b = b)
println("  decomposition → ", sh_write_csv(joinpath(sh_report_dir, "sh_decomposition.csv"), c_full, c, sel_cols, sh_degree))
