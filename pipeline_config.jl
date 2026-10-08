# =============================================================================
#  pipeline_config.jl  —  SINGLE SOURCE FOR PIPELINE (paths + derived values)
# =============================================================================
#  BOOST (field-map optimizer)  →  export_csv  →  CSV_to_STL (insert generator)
#
#  USER-EDITABLE VALUES NOW LIVE IN  config.toml  (edited by hand or the GUI).
#  This file READS config.toml and DERIVES all absolute paths + shared geometry
#  from it, so nothing can drift apart. To change a run, edit config.toml — not
#  this file. (Advanced: coupled SA sweeps are still defined here, see §4d.)
#
#  Every stage script (run_optim.jl, export_csv.jl, CSV_to_STL.jl, setup.jl) and
#  the orchestrator (run_pipeline.jl) include this file first (guarded), so all
#  paths, the iteration name and the geometry come from one place.
# =============================================================================

using TOML

const ROOT = @__DIR__
const _CFG = TOML.parsefile(joinpath(ROOT, "config.toml"))

# --- small typed accessors ---------------------------------------------------
_get(k)      = haskey(_CFG, k) ? _CFG[k] : error("config.toml is missing key: $(k)")
_getdef(k,d) = haskey(_CFG, k) ? _CFG[k] : d                     # optional key + fallback
_sym(k)      = Symbol(_get(k))                                   # "grad" → :grad
_int(k)      = Int(_get(k))
_flt(k)      = Float64(_get(k))
_intvec(k)   = Int[Int(x)         for x in _get(k)]   # typed even when empty ([])
_fltvec(k)   = Float64[Float64(x) for x in _get(k)]
function _range(k)                                               # [lo, hi] → lo:hi
    v = _get(k)
    (v isa AbstractVector && length(v) == 2) || error("config.toml: $(k) must be [lo, hi].")
    return Int(v[1]):Int(v[2])
end

# ----------------------------------------------------------------------------
# RUN MODE + OPTIMIZER + START STAGE
# ----------------------------------------------------------------------------
# pipeline_mode: "full" → Stage 0→1→1.5→2→viewers ; "lcurve" → Stage 0→SA sweep.
const pipeline_mode = _get("pipeline_mode")
# optimizer (full mode): :grad (grad_optim/, L-BFGS on operator G) or
# :sim_annealing (sim_annealing_optim/). lcurve + benchmark always use the SA.
const optimizer = _sym("optimizer")
# start_stage (a run-order index, not a stage name): -1 ask ; 0 Stage 0 ; 1 Stage 1 ; 2 Stage 1.5 ;
# 3 Stage 2 ; 4 Stage 3 (viewers) ; 5 Stage 4 (Python verifier only).
const start_stage = _int("start_stage")

# ----------------------------------------------------------------------------
# ITERATION IDENTITY
# ----------------------------------------------------------------------------
const ITERATION              = _get("iteration")
const iteration_number       = _int("iteration_number")
const measured_fieldmap_name = _get("measured_fieldmap_name")
# LAB-frame direction of B0 ("+x"/"-x"/"+y"/"-y"); Stage 0 rotates it to +y.
const main_field_direction   = _get("main_field_direction")

# ----------------------------------------------------------------------------
# 1.  ROOT DIRECTORIES
# ----------------------------------------------------------------------------
const MEASURED_FIELD_DIR     = joinpath(ROOT, "data", "inputs", "Measured_Field_Data_csv")
const INTERPOLATED_FIELD_DIR = joinpath(ROOT, "data", "cache", "Interpolated_Field_Data_jld2")
const OPTIMIZER_OUTPUT_DIR   = joinpath(ROOT, "data", "outputs", "Optimizer_Output_per_Iteration")
const MAGNET_INSERTS_DIR     = joinpath(ROOT, "assets", "Magnet_Inserts_Models_stl_step")
const STATIC_INSERTS_DIR     = joinpath(ROOT, "assets", "Static_Inserts_Models_stl_step")
const FINAL_OUTPUT_DIR       = joinpath(ROOT, "data", "outputs", "Final_3D_printing_outputs_per_Iteration")

# ----------------------------------------------------------------------------
# 2.  STAGE I/O PATHS  (artifacts handed from one stage to the next)
# ----------------------------------------------------------------------------
const measured_field_path        = joinpath(MEASURED_FIELD_DIR, measured_fieldmap_name)
const interpolated_fieldmap_name = replace(measured_fieldmap_name, r"\.csv$" => ".jld2")
const fieldmap_path              = joinpath(INTERPOLATED_FIELD_DIR, interpolated_fieldmap_name)

# Where the optimizer scores homogeneity (grad + ring search): a dense grid or a
# single spherical shell at Rmax. Shell mode uses its own Stage-0 output file.
const eval_domain      = _sym("eval_domain")         # :grid | :shell
const shell_n_points   = _int("shell_n_points")
# Shell evaluation source: :measured (raw points + |B|, no interpolation) or
# :sh_fibonacci (SH fit resampled onto shell_n_points uniform Rmax points).
const shell_source     = Symbol(_getdef("shell_source", "sh_fibonacci"))
const shell_fieldmap_path = replace(fieldmap_path, r"\.jld2$" => "_shell.jld2")

const Iteration_folder_name = ITERATION
const Iteration_file_name   = "$(ITERATION)_BOOST_result.jld2"
const optimizer_iter_dir    = joinpath(OPTIMIZER_OUTPUT_DIR, Iteration_folder_name)
const optimizer_result_path = joinpath(optimizer_iter_dir, Iteration_file_name)

# Per-optimizer result COPIES (never clobbered by the other optimizer), so an SA
# result and a gradient result coexist and can be compared (utils/eval_metrics.jl).
const sa_result_path   = joinpath(optimizer_iter_dir, "$(ITERATION)_SA_result.jld2")
const grad_result_path = joinpath(optimizer_iter_dir, "GradOpt", "grad_result.jld2")
# insert_search.jl's tagged copy. Same format; the SPARSE layout lives in
# `final_state` (1 = insert placed at that slot, 0 = empty), so eval_metrics and
# export_csv handle it with no special case.
const insert_result_path = joinpath(optimizer_iter_dir, "InsertSearch", "insert_result.jld2")

# Stage 1.5 output / Stage 2 input: the shim CSV (X, Y, RingNumber, Angle).
const shimming_magnets_csv_name = "$(ITERATION)_shim.csv"
const shim_csv_path             = joinpath(optimizer_iter_dir, shimming_magnets_csv_name)

# Companion export_csv.jl output (X, Y, Z, Angle — real physical Z instead of
# RingNumber): a convenience file for inspection/external tools. Nothing
# downstream (CSV_to_STL.jl, the viewers, the GUI) reads this one — shim_csv_path
# above remains the sole Stage-2 input / the seam.
const shimming_magnets_xyz_csv_name = "$(ITERATION)_shim_xyz.csv"
const shim_csv_xyz_path             = joinpath(optimizer_iter_dir, shimming_magnets_xyz_csv_name)

# Stage 2 insert parameters (mm)
const mag_cover_thickness  = _flt("mag_cover_thickness")
const extr_cover_thickness = _get("extr_cover_thickness")
const total_thickness      = mag_cover_thickness + extr_cover_thickness
const lc                   = _flt("lc")
const width_holes          = _flt("width_holes")
const letter_thickness     = _flt("letter_thickness")
# Overall size of every engraved digit glyph (iteration number, InsertPos tens/ones
# digits, tray digit) on the printed label — utils/helping_functions_for_JIG.jl's
# make_numb_hexadecimal scales every point coordinate by this. Defaulted to
# make_label's own default (1.3) so older configs load unchanged.
const label_scale          = Float64(_getdef("label_scale", 1.3))
const Tray_short_radii     = _flt("Tray_short_radii")
const Tray_long_radii      = _flt("Tray_long_radii")
const removing_depth       = extr_cover_thickness + 0.1

# Stage 2 output: per-tray printable geometry
const final_iteration_dir = joinpath(FINAL_OUTPUT_DIR, ITERATION)

# ----------------------------------------------------------------------------
# 3.  SHARED GEOMETRY  (must match between BOOST and the insert generator)
# ----------------------------------------------------------------------------
const num_trays             = _int("num_trays")
const mags_per_segment      = _int("mags_per_segment")
const shim_radius_mm        = _flt("shim_radius_mm")     # Float → prints "231.0"
const angle_per_segment_deg = _flt("angle_per_segment_deg")
const angular_offset_deg    = _flt("angular_offset_deg")
# Axial tray geometry, from TRUE z = 0 (see utils/pos_trays.jl `ringpos_from_tray_mm`):
#   z(+n) = +(back_tray_shift_mm  + (n−1)·tray_slot_spacing_mm)   → +Z "Back - Wall"
#   z(−n) = −(front_tray_shift_mm + (n−1)·tray_slot_spacing_mm)   → −Z "Front - Gaussmeter"
# The two ends are independent so an asymmetric bore can be described exactly.
# Supersedes the old symmetric `half_shift_mm` (h with spacing s ≡ front = back = s − h);
# defaulted to 10/10/10 = the old half_shift_mm = 0 behaviour, so old configs are unchanged.
const tray_slot_spacing_mm  = Float64(_getdef("tray_slot_spacing_mm", 10.0))
const front_tray_shift_mm   = Float64(_getdef("front_tray_shift_mm", 10.0))
const back_tray_shift_mm    = Float64(_getdef("back_tray_shift_mm",  10.0))
const positions_in_tray_occupied   = _intvec("positions_in_tray_occupied")
const positions_in_tray_new_wished = _intvec("positions_in_tray_new_wished")
const Rmin = _flt("Rmin")
const Rmax = _flt("Rmax")

# ----------------------------------------------------------------------------
# 3b. SHIM MAGNET (physical) — the ONE source for magnet strength + start angle
# ----------------------------------------------------------------------------
# Strength is entered as the magnet's REMANENCE, Br (T) — a material property from
# the manufacturer's datasheet or your own measurement setup, independent of the
# magnet's size/shape — together with its physical cube side length (mm). This
# replaced the earlier on-axis-|B|-at-1cm proxy (`magnet_B1cm_mT`) now that Br is
# measured/calculated consistently for the actual 6×6×6 mm shim magnets: Br is the
# more direct, standard quantity and needs no arbitrary calibration distance.
#
# Dipole moment of a uniformly magnetized cube (side a, volume V = a³):
#     m = Br·V/μ0 = Br·a³/μ0                          [A·m²]
# (from B = μ0·M inside the magnet, M = m/V ⇒ m = M·V = (Br/μ0)·V.)
#
# Read by setup.jl, setup_shell.jl, stages/stage1_optimize/grad_optim/ring_search.jl, stages/stage1_optimize/grad_optim/insert_search.jl
# and BOTH Stage-3 viewers, so a change here propagates everywhere. NOTE: the cached
# operator GradOpt/operator_G.jld2 (and GradOpt/ring_operator.jld2) stores the μ it
# was built with — rebuild it (Stage 1 "Grad build+solve", i.e.
# `julia stages/stage1_optimize/grad_optim/optim_grad.jl`) after changing either value; grad_core.jl warns
# if the cache and the config disagree. Defaulted so older config.toml files load.
const magnet_Br_T       = Float64(_getdef("magnet_Br_T", 1.32))     # remanence (T)
const magnet_side_mm    = Float64(_getdef("magnet_side_mm", 6.0))   # cube side length (mm)
const MU0               = 4pi * 1e-7                                 # T·m/A
const magnet_moment_Am2 = magnet_Br_T * (magnet_side_mm * 1e-3)^3 / MU0

# --- back-compat: magnet_B1cm_mT / magnet_B1cm_T / B1CM_T are now DERIVED from
# the moment above (not the source of truth). Any config that still sets
# magnet_B1cm_mT directly is IGNORED for the physics — these three constants just
# report the on-axis field at 1 cm that this Br+geometry combination implies, so
# utils/imanes.jl's B1cm_T= calling convention (used by both Stage-3 viewers)
# keeps working unchanged: m = B1cm·r³/(2·μ0/4π), r = 0.01 m ⇒ B1cm = m·2·μ0/(4π·r³).
const MAGNET_CAL_R_M    = 0.01                     # calibration distance (m)
const MU0_OVER_4PI      = MU0 / (4pi)              # = 1e-7 T·m/A
const magnet_B1cm_T     = magnet_moment_Am2 * 2 * MU0_OVER_4PI / MAGNET_CAL_R_M^3
const magnet_B1cm_mT    = magnet_B1cm_T * 1e3
const B1CM_T            = magnet_B1cm_T            # back-compat alias for the old key

# Starting angle (deg) applied to every shim magnet: the SA's initial state and
# the angle optim_grad.jl validates G at. The gradient optimizer overrides it with
# `grad_seeds` random restarts, so it barely affects that path.
const initial_angle_deg = Float64(_getdef("initial_angle_deg", 150.0))

# ----------------------------------------------------------------------------
# 4.  SIMULATED-ANNEALING SEARCH PARAMETERS
# ----------------------------------------------------------------------------
const lambda_weight   = _get("lambda_weight")
const test_ring_seq   = _intvec("test_ring_seq")
const SA_T0           = _get("SA_T0")
const SA_alpha        = _flt("SA_alpha")
const SA_iters        = _int("SA_iters")
const SA_restarts     = _int("SA_restarts")
const SA_report_every = _int("SA_report_every")
const SA_step0        = _flt("SA_step0")
const SA_step_min     = _flt("SA_step_min")

# (B1CM_T moved to §3b as a derived alias of magnet_B1cm_mT — see above.)
const w       = _flt("w")
mode          = _get("mode")             # non-const (matches historical usage)

# --- reproducibility + benchmarking -----------------------------------------
const rng_seed        = _int("rng_seed")
const benchmark_seeds = _range("benchmark_seeds")
const benchmark_dir   = joinpath(optimizer_iter_dir, "Benchmark")
const grad_seeds      = _range("grad_seeds")

# ----------------------------------------------------------------------------
# 4d. GRADIENT-OPTIMIZER OBJECTIVE  (stages/stage1_optimize/grad_optim/run_grad.jl, run_grad_lcurve.jl)
# ----------------------------------------------------------------------------
#   J(θ) = data_term(By) + grad_lambda · mean|∇B|²   (all smooth, analytic grad).
# grad_data_term :variance (mT²) or :softrange (mT, tracks ppm; β = grad_softrange_beta).
# grad_lambda is the Tikhonov weight on the field's spatial-gradient penalty.
const grad_data_term      = _sym("grad_data_term")
const grad_softrange_beta = _flt("grad_softrange_beta")
const grad_lambda         = _flt("grad_lambda")
const grad_lambda_sweep   = _fltvec("grad_lambda_sweep")

# ----------------------------------------------------------------------------
# 4c. SA SWEEP DEFINITION (pipeline_mode = "lcurve")
# ----------------------------------------------------------------------------
# Single-parameter sweep from config.toml: each point overrides sweep_xkey with a
# value from sweep_values, run over sweep_seeds → ppm mean ± std.
const sweep_seeds  = _range("sweep_seeds")
const sweep_xkey   = _sym("sweep_xkey")
const sweep_points = [NamedTuple{(sweep_xkey,)}((v,)) for v in _get("sweep_values")]
# ADVANCED — coupled multi-param sweep (overrides the line above): uncomment &
# edit, e.g. budget-constant iters×restarts with T0 scaling:
#   const sweep_xkey   = :restarts
#   const sweep_points = [(iters=50000, restarts=4,  T0=1.0),
#                         (iters=25000, restarts=8,  T0=2.0),
#                         (iters=12500, restarts=16, T0=4.0)]
const lcurve_dir = joinpath(optimizer_iter_dir, "Lcurve")

# ----------------------------------------------------------------------------
# 4a. STAGE 0 field adapter
# ----------------------------------------------------------------------------
# run_field_adapter: run measured-CSV → interpolated-jld2 at the start (false reuses).
# field_adapter: :reshape (complete regular grid) | :spherical_harmonics (shell scan).
const run_field_adapter = _get("run_field_adapter")
const field_adapter     = _sym("field_adapter")
const sh_measured_unit_mm = _flt("sh_measured_unit_mm")
const sh_degree           = _int("sh_degree")
const sh_grid_n           = _int("sh_grid_n")
const sh_grid_radius_mm   = Rmax        # keep ≥ Rmax so the shell mask is covered
# Which SH coefficients define the field map (utils/sh_select.jl): :all | :first_n | :top_k.
# Defaulted so older configs load unchanged (:all = the original behaviour).
const sh_select           = Symbol(_getdef("sh_select", "all"))
const sh_use_degree       = Int(_getdef("sh_use_degree", 4))     # :first_n → degrees 0..this
const sh_top_k            = Int(_getdef("sh_top_k", 5))          # :top_k → mean + this many largest; also the printed list length
const sh_report_dir       = joinpath(optimizer_iter_dir, "SHDecomposition")

# ----------------------------------------------------------------------------
# 4e. RING SEARCH  (stages/stage1_optimize/grad_optim/ring_search.jl) — pick the best n rings to place
# ----------------------------------------------------------------------------
const ring_search_n             = _int("ring_search_n")
const ring_search_n_max         = _int("ring_search_n_max")
const ring_search_method        = _sym("ring_search_method")    # :greedy | :exhaustive | :paired (symmetric ±a pairs; ring_search_n = total rings, even)
const ring_search_range         = _intvec("ring_search_range")  # [lo, hi] tray numbers
const ring_search_step          = _int("ring_search_step")
const ring_search_min_sep       = _flt("ring_search_min_sep")
const ring_search_min_spots_between = _int("ring_search_min_spots_between")
const ring_search_max_combos    = _int("ring_search_max_combos")
const ring_search_magnet_budget = _int("ring_search_magnet_budget")
const ring_search_apply         = _get("ring_search_apply")

# ----------------------------------------------------------------------------
# 4g. INSERT SEARCH (stages/stage1_optimize/grad_optim/insert_search.jl) — sequential per-insert placement
# ----------------------------------------------------------------------------
# Finer granularity than the ring search: places ONE insert (one tray at one ring,
# mags_per_segment magnets) at a time, freezing its angles and folding its field
# into the base before searching for the next. The number of inserts comes from the
# magnet budget: fld(magnets_available, mags_per_segment) — every insert is assumed
# FULL. Defaulted so an older config.toml still loads.
const magnets_available              = Int(_getdef("magnets_available", 336))
const insert_search_range            = Int[Int(x) for x in _getdef("insert_search_range", [-25, 25])]
const insert_search_step             = Int(_getdef("insert_search_step", 1))
const insert_search_min_spots_between = Int(_getdef("insert_search_min_spots_between", 3))
const insert_search_max_inserts      = Int(_getdef("insert_search_max_inserts", 0))
const insert_search_apply            = _getdef("insert_search_apply", true)

# ----------------------------------------------------------------------------
# 4f. OSII IMPORT (osii_to_shim.jl) — alternative Stage 1.5 from an OSII CSV
# ----------------------------------------------------------------------------
# Reads an OSII shim-config CSV from OSII_INPUT_DIR and writes the standard shim
# CSV (shim_csv_path) that Stage 2 consumes — no field map / optimizer needed.
# Keys are optional (defaulted) so older config.toml files still load everywhere.
const OSII_INPUT_DIR        = joinpath(ROOT, "data", "inputs", "OSII_shimming_outputs_toconvert")
const osii_input_name       = String(_getdef("osii_input_name", ""))
const osii_input_path       = joinpath(OSII_INPUT_DIR, osii_input_name)
const osii_invert_angle     = Bool(_getdef("osii_invert_angle", true))
const osii_angle_offset_deg = Float64(_getdef("osii_angle_offset_deg", 90))

# ----------------------------------------------------------------------------
# 4b. VERIFICATION (Stage 3) — each viewer needs a display + GPU and blocks.
# ----------------------------------------------------------------------------
const run_3d_viewer    = _get("run_3d_viewer")
const run_slice_viewer = _get("run_slice_viewer")

# GIF export from both Stage-3 viewers, so a result can be shared without asking
# someone to run the pipeline themselves. Saved under optimizer_iter_dir/Viewers/.
# viewer_gif_frames × 1/viewer_gif_fps = animation length in seconds (default: 120
# frames at 20 fps = 6 s per GIF). viewer_gif_dir lets you point exports somewhere
# else (e.g. a shared folder) without touching the rest of the iteration's outputs.
const viewer_gif_frames = Int(_getdef("viewer_gif_frames", 120))
const viewer_gif_fps    = Int(_getdef("viewer_gif_fps", 20))
const viewer_gif_dir    = joinpath(optimizer_iter_dir, _getdef("viewer_gif_subdir", "Viewers"))

# Frame the Stage-3 viewers draw in: "scan" (default; measured scan's own frame) or "optimizer"
# (B0 -> +y, the frame of the shim CSV). See utils/viewer_frame.jl.
const viewer_frame = String(_getdef("viewer_frame", "scan"))
const shim_csv_frame = String(_getdef("shim_csv_frame", "scan"))
@assert shim_csv_frame in ("scan", "optimizer") "shim_csv_frame = \"$(shim_csv_frame)\" invalid; use \"scan\" or \"optimizer\"."
@assert viewer_frame in ("scan", "optimizer") "viewer_frame = \"$(viewer_frame)\" invalid; use \"scan\" or \"optimizer\"."

# ----------------------------------------------------------------------------
# 4h. STAGE 4 — INDEPENDENT PYTHON VERIFIER (stages/stage4_verifier/Shimming_verifier/, magpylib)
# ----------------------------------------------------------------------------
# A separate, self-contained Python tool re-computes the shimmed field from the shim
# magnets with magpylib (closed-form cuboid model, no shared code with the Julia
# kernels) and reports ppm, 2D projections, a 3D scene and spherical-harmonic pyramids.
# It needs only three decisions — which measured field, which shim layout, what to
# run — which is all that is configured here. Magnet strength/size are NOT repeated:
# it uses magnet_Br_T / magnet_side_mm above, and the measurement's coordinate unit
# comes from sh_measured_unit_mm. Driven by python_verifier.jl (CLI:
# Shimming_verifier/run_verifier.py). All keys optional, so older configs still load.
const VERIFIER_DIR              = joinpath(ROOT, "stages", "stage4_verifier", "Shimming_verifier")
const run_python_verifier       = Bool(_getdef("run_python_verifier", false))     # run_pipeline: run Stage 4 at the end
const verifier_python           = String(_getdef("verifier_python", "python"))    # Python executable (needs Shimming_verifier/requirements.txt)
const verifier_measurement      = String(_getdef("verifier_measurement", ""))     # "" = measured_fieldmap_name; else file in Measured_Field_Data_csv/ or an absolute path
const verifier_use_shim         = Bool(_getdef("verifier_use_shim", true))        # false = baseline only (measured field, no shim magnets)
const verifier_shim_csv         = String(_getdef("verifier_shim_csv", ""))        # "" = this iteration's shim CSV; else a path (seam or X,Y,Z,Angle CSV)
const verifier_make_2d          = Bool(_getdef("verifier_make_2d", true))         # 2D projections, with vs without shim
const verifier_make_3d          = Bool(_getdef("verifier_make_3d", true))         # interactive 3D scene (.html)
const verifier_make_sh          = Bool(_getdef("verifier_make_sh", true))         # spherical-harmonic pyramids
const verifier_sh_max_degree    = Int(_getdef("verifier_sh_max_degree", 7))
const verifier_show_figures     = Bool(_getdef("verifier_show_figures", false))   # open matplotlib windows (blocks until closed)
const verifier_out_dir          = joinpath(optimizer_iter_dir, String(_getdef("verifier_output_subdir", "PythonVerifier")))

# ----------------------------------------------------------------------------
# 5.  Ensure per-iteration output folders exist
# ----------------------------------------------------------------------------
for d in (optimizer_iter_dir, MAGNET_INSERTS_DIR, STATIC_INSERTS_DIR, final_iteration_dir, viewer_gif_dir)
    mkpath(d)
end
