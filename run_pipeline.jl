# run_pipeline.jl  —  ORCHESTRATOR for the combined BOOST → Insert pipeline
#
#   julia run_pipeline.jl
#
# Runs the three stages, each as its OWN julia subprocess, so the heavy and
# mutually-incompatible runtimes never share a process:
#   Stage 0    Field_data_file_adapter.jl   Measured CSV → Field .jld2
#   Stage 1    optimizer (config `optimizer`):
#                :sim_annealing → stages/stage1_optimize/sim_annealing_optim/run_optim.jl  (CUDA + GLMakie)
#                :grad          → stages/stage1_optimize/grad_optim/optim_grad.jl (build G) → run_grad.jl (L-BFGS)
#   Stage 1.5  export_csv.jl   result jld2 → shim CSV  (CUDA, reuses setup.jl)
#   Stage 2    CSV_to_STL.jl   shim CSV → per-tray STL/STEP  (Gmsh)
#   Stage 3    Julia viewers   (config run_3d_viewer / run_slice_viewer)
#   Stage 4    python_verifier.jl  independent magpylib check (stages/stage4_verifier/Shimming_verifier/, Python)
#              — config run_python_verifier, or start_stage = 5 to run it alone
#
# Each stage reads everything (iteration name, paths, geometry) from
# pipeline_config.jl, so to change a run you edit THAT file, not this one.
#
# The chain stops at the first stage that fails (non-zero exit) or whose
# expected output is missing, and reports where it stopped.

include("pipeline_config.jl")   # ITERATION + all artifact paths (for summary/checks)

const JULIA = Base.julia_cmd()  # same julia executable + flags running this script

# --- Stage 0 script selection (config: field_adapter) -------------------------
# :reshape              → Field_data_file_adapter.jl   (complete regular grid)
# :spherical_harmonics  → Field_data_SH_interpolator.jl (scattered/shell scan)
S0(f) = joinpath("stages", "stage0_field", f)         # script paths are relative to ROOT
S1(parts...) = joinpath("stages", "stage1_optimize", parts...)
const STAGE0_SCRIPTS = Dict(
    :reshape             => S0("Field_data_file_adapter.jl"),
    :spherical_harmonics => S0("Field_data_SH_interpolator.jl"),
)
@assert haskey(STAGE0_SCRIPTS, field_adapter) "field_adapter = $(field_adapter) is invalid; use :reshape or :spherical_harmonics."
# eval_domain = :shell overrides the Stage-0 script (shell adapter handles either
# input) and changes the Stage-1 input file. Shell mode is gradient + ring search.
const stage0_script = eval_domain === :shell ? S0("Field_data_shell.jl") : STAGE0_SCRIPTS[field_adapter]
const stage0_title  = eval_domain === :shell ? "Stage 0   Measured CSV → shell jld2  (shell)" :
                                               "Stage 0   Measured CSV → interpolated jld2  ($(field_adapter))"
const stage1_input  = eval_domain === :shell ? shell_fieldmap_path : fieldmap_path

# --- count .stl files under a directory tree (used to confirm Stage 2 output) -
function count_stls(dir)
    isdir(dir) || return 0
    n = 0
    for (_, _, files) in walkdir(dir), f in files
        endswith(f, ".stl") && (n += 1)
    end
    return n
end

# --- run one stage as a subprocess, then verify its expected output -----------
function run_stage(title::String, script::String; expect_file=nothing)
    println("\n", "="^72)
    println("▶  ", title, "   (", script, ")")
    println("="^72)

    cmd = Cmd(`$JULIA $(joinpath(ROOT, script))`; dir = ROOT)
    t0  = time()
    try
        # stdin = devnull: stages/viewers never read stdin, and letting a child
        # inherit the console stdin we just read from (readline) can deadlock on
        # Windows. stdout/stderr stay inherited so output is still visible.
        run(pipeline(cmd; stdin = devnull))       # throws on non-zero exit
    catch
        println("\n✘  STAGE FAILED: ", title)
        println("    command: ", cmd)
        expect_file === nothing || println("    expected output: ", expect_file)
        error("Pipeline halted at: $title")
    end
    dt = round(time() - t0, digits = 1)

    if expect_file !== nothing && !ispath(expect_file)
        error("Stage '$title' exited 0 but expected output is missing:\n    $expect_file")
    end
    println("\n✓  ", title, "  done in ", dt, "s",
            expect_file === nothing ? "" : "   →  $(expect_file)")
end

# -----------------------------------------------------------------------------
println("\n", "#"^72)
println("#  COMBINED PIPELINE   —   iteration: ", ITERATION)
println("#  mode      : ", pipeline_mode)
println("#  optimizer : ", optimizer)
println("#  adapter   : ", eval_domain === :shell ? "shell" : field_adapter)
println("#  eval      : ", eval_domain)
println("#  measured  : ", measured_field_path)
println("#  field map : ", stage1_input)
println("#"^72)

@assert pipeline_mode in ("full", "lcurve") "pipeline_mode = \"$(pipeline_mode)\" is invalid; use \"full\" or \"lcurve\"."
@assert optimizer in (:grad, :sim_annealing) "optimizer = $(optimizer) is invalid; use :grad or :sim_annealing."
@assert !(eval_domain === :shell) || (pipeline_mode == "full" && optimizer === :grad) "eval_domain = :shell supports only pipeline_mode=\"full\" with optimizer=:grad (gradient + ring search)."

# Error clearly if a stage we're SKIPPING was supposed to have produced this input.
function need(path, what)
    isfile(path) || error("Can't start here — missing $what:\n    $(path)\n" *
                          "Run an earlier stage first, or choose an earlier start point.")
end

# ============================ L-CURVE MODE ============================
if pipeline_mode == "lcurve"
    if run_field_adapter
        run_stage(stage0_title, stage0_script; expect_file = fieldmap_path)
    else
        need(fieldmap_path, "field map (Stage 0 output)")
    end
    sweep_csv = joinpath(lcurve_dir, "sweep_$(sweep_xkey).csv")
    run_stage("Sweep   $(sweep_xkey) (Stage 1 only, SA)", S1("sim_annealing_optim", "run_lcurve.jl"); expect_file = sweep_csv)
    println("\n", "#"^72)
    println("#  SWEEP FINISHED  ✓   $(sweep_xkey)   iteration: ", ITERATION)
    println("#"^72)
    println("  results CSV : ", sweep_csv)
    println("  plot        : ", joinpath(lcurve_dir, "sweep_$(sweep_xkey).png"))
    println("  per-point jld2s : ", lcurve_dir)
    println("#"^72)
    exit(0)
end

# ============================ FULL MODE ============================
# Ask which stage to start from. Earlier stages are skipped and their existing
# outputs reused (so you don't re-run the whole shim process to, e.g., just
# rebuild geometry or re-open a viewer).
function ask_start_stage()
    println("\nStart from which stage?  (earlier stages are skipped; their outputs are reused)")
    println("  (option numbers are the run order; stage names are shown)")
    println("  0) Stage 0   — Measured CSV → field map     (full run)              [default]")
    println("  1) Stage 1   — BOOST optimizer              (reuse field map)")
    println("  2) Stage 1.5 — Export shim CSV              (reuse optimizer result)")
    println("  3) Stage 2   — CSV → STL/STEP geometry      (reuse shim CSV)")
    println("  4) Stage 3   — Viewers (+ Stage 4 if enabled)   (reuse shim CSV + field map)")
    println("  5) Stage 4   — Python verifier only         (reuse shim CSV + measured CSV)")
    print("Enter 0–5 [0]: ")
    flush(stdout)
    r = strip(readline())
    r == "" && return 0
    n = tryparse(Int, r)
    (n === nothing || !(0 <= n <= 5)) && (println("  (not understood — starting from Stage 0)"); return 0)
    return n
end

# Use the config override if set (0–5); otherwise ask interactively.
start = (0 <= start_stage <= 5) ? start_stage : ask_start_stage()
start == start_stage && println("\nStarting at stage $start (from pipeline_config.jl's start_stage).")

# Inputs required by the chosen start point must already exist on disk. Stage 4 needs
# neither the field map nor the optimizer result — only the measured CSV and (unless it
# runs baseline-only) a shim CSV — and python_verifier.jl checks those itself.
1 <= start <= 4 && need(stage1_input,          "field map (Stage 0 output)")
2 <= start <= 4 && need(optimizer_result_path, "optimizer result (Stage 1 output)")
3 <= start <= 4 && need(shim_csv_path,         "shim CSV (Stage 1.5 output)")

if start <= 0
    if run_field_adapter
        run_stage(stage0_title, stage0_script; expect_file = stage1_input)
    else
        println("\n(Stage 0 skipped — run_field_adapter = false.)")
        need(stage1_input, "field map (Stage 0 output)")
    end
end

if start <= 1
    if optimizer == :sim_annealing
        run_stage("Stage 1   SA optimizer", S1("sim_annealing_optim", "run_optim.jl");
                  expect_file = optimizer_result_path)
    else  # :grad — two steps: build/cache G, then L-BFGS variance minimization
        operator_G = joinpath(optimizer_iter_dir, "GradOpt", "operator_G.jld2")
        run_stage("Stage 1a  Build linear operator G", S1("grad_optim", "optim_grad.jl");
                  expect_file = operator_G)
        run_stage("Stage 1b  L-BFGS variance min",     S1("grad_optim", "run_grad.jl");
                  expect_file = optimizer_result_path)
    end
end
start <= 2 && run_stage("Stage 1.5 Export shim CSV",        joinpath("stages", "stage1_5_export", "export_csv.jl"); expect_file = shim_csv_path)
start <= 3 && run_stage("Stage 2   CSV → per-tray STL/STEP", joinpath("stages", "stage2_stl", "CSV_to_STL.jl"))

nstl = count_stls(final_iteration_dir)
start <= 3 && nstl == 0 && error("Stage 2 produced no .stl files under: $final_iteration_dir")

# -----------------------------------------------------------------------------
if start <= 4
    println("\n", "#"^72)
    println("#  PIPELINE FINISHED  ✓   iteration: ", ITERATION, "   (started at stage ", start, ")")
    println("#"^72)
    println("  field map        : ", stage1_input)
    println("  optimizer result : ", optimizer_result_path)
    println("  ppm slice images : ", joinpath(optimizer_iter_dir, "imgs"))
    println("  shim CSV (seam)  : ", shim_csv_path)
    println("  base magnet ins. : ", MAGNET_INSERTS_DIR)
    println("  base static ins. : ", STATIC_INSERTS_DIR)
    println("  printable output : ", final_iteration_dir, "   (", nstl, " .stl files)")
    println("#"^72)
end

# --- Stage 3: verification viewers (config-driven; each blocks until its window closes) ----
# Viewers are grid-based; in shell mode the shell adapter also writes the grid jld2
# (fieldmap_path) for exactly this, so the viewers work in either domain.
# (start = 5 means "Python verifier only", so the Julia viewers are skipped.)
if start <= 4
    run_3d_viewer    && run_stage("Stage 3   3D shimming-magnets viewer", joinpath("stages", "stage3_viewers", "Shimming_magnets_visualizer.jl"))
    run_slice_viewer && run_stage("Stage 3   Field slice & line viewer",  joinpath("stages", "stage3_viewers", "field_slice_viewer.jl"))
    (run_3d_viewer || run_slice_viewer) ||
        println("\n(Stage 3 viewers skipped — enable run_3d_viewer / run_slice_viewer in pipeline_config.jl.)")
end

# --- Stage 4: independent Python (magpylib) verifier — Shimming_verifier/ ---------------
# Separate runtime, needs no GPU or field map: only the measured CSV + the shim CSV.
# Runs when enabled in config (run_python_verifier) or when started at option 5 (Stage 4 only).
if run_python_verifier || start == 5
    run_stage("Stage 4   Python (magpylib) verifier", joinpath("stages", "stage4_verifier", "python_verifier.jl"))
else
    println("\n(Stage 4 Python verifier skipped — set run_python_verifier = true in config.toml, or start_stage = 5.)")
end
