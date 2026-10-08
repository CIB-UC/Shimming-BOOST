# gui/backend.jl  —  backend launcher for the shimming-pipeline GUI
#
# A thin layer the frontend calls. It holds NO optimization logic: it edits
# config.toml, imports measured CSVs, launches the existing stage scripts as
# streamed subprocesses, and opens folders. Everything the pipeline needs is
# still derived by pipeline_config.jl from config.toml.
#
# Use from the REPL / frontend:
#     include("gui/backend.jl"); using .Backend
#     Backend.read_config()                       # Dict of current settings
#     Backend.write_config(Dict("iteration"=>"run1", "optimizer"=>"grad"))
#     Backend.list_measured()                     # CSVs in the data folder
#     Backend.import_csv("C:/path/to/scan.csv")   # copy into the data folder
#     Backend.run_stage("full"; on_output=println)# run pipeline, stream the log
#     Backend.open_folder(Backend.output_dirs().final)
#
# Run directly (`julia gui/backend.jl`) for a quick self-check.

module Backend

using TOML

const REPO         = normpath(joinpath(@__DIR__, ".."))
const CONFIG       = joinpath(REPO, "config.toml")
const MEASURED_DIR = joinpath(REPO, "data", "inputs", "Measured_Field_Data_csv")
const OSII_DIR     = joinpath(REPO, "data", "inputs", "OSII_shimming_outputs_toconvert")

# Named stages the GUI can launch → script path (relative to the repo root).
const STAGES = Dict(
    "full"            => "run_pipeline.jl",                         # full run OR lcurve, per config
    "adapter_reshape" => joinpath("stages", "stage0_field", "Field_data_file_adapter.jl"),             # Stage 0 (regular grid)
    "adapter_sh"      => joinpath("stages", "stage0_field", "Field_data_SH_interpolator.jl"),          # Stage 0 (SH → grid)
    "adapter_shell"   => joinpath("stages", "stage0_field", "Field_data_shell.jl"),                    # Stage 0 (SH → shell at Rmax)
    "optim_grad"      => joinpath("stages", "stage1_optimize", "grad_optim", "optim_grad.jl"),  # build + cache operator G
    "run_grad"        => joinpath("stages", "stage1_optimize", "grad_optim", "run_grad.jl"),    # gradient L-BFGS solve
    "grad_lcurve"     => joinpath("stages", "stage1_optimize", "grad_optim", "run_grad_lcurve.jl"),
    "ring_search"     => joinpath("stages", "stage1_optimize", "grad_optim", "ring_search.jl"),     # pick best n rings
    "insert_search"   => joinpath("stages", "stage1_optimize", "grad_optim", "insert_search.jl"),   # sequential per-insert placement
    "sa"              => joinpath("stages", "stage1_optimize", "sim_annealing_optim", "run_optim.jl"),
    "sa_lcurve"       => joinpath("stages", "stage1_optimize", "sim_annealing_optim", "run_lcurve.jl"),
    "benchmark"       => joinpath("stages", "stage1_optimize", "sim_annealing_optim", "benchmark.jl"),
    "export"          => joinpath("stages", "stage1_5_export", "export_csv.jl"),
    "osii"            => joinpath("stages", "stage1_5_export", "osii_to_shim.jl"),                        # Stage 1.5 (alt): OSII CSV → shim CSV
    "stl"             => joinpath("stages", "stage2_stl", "CSV_to_STL.jl"),
    "eval"            => joinpath("core", "utils", "eval_metrics.jl"),
    "viewer_3d"       => joinpath("stages", "stage3_viewers", "Shimming_magnets_visualizer.jl"),         # Stage 3 (3D)
    "viewer_slice"    => joinpath("stages", "stage3_viewers", "field_slice_viewer.jl"),                  # Stage 3 (slice/profile)
    "verify_python"   => joinpath("stages", "stage4_verifier", "python_verifier.jl"),                     # Stage 4 (independent magpylib check)
)

stage_names() = sort(collect(keys(STAGES)))

# --- validation --------------------------------------------------------------
const ENUMS = Dict(
    "optimizer"            => ["grad", "sim_annealing"],
    "pipeline_mode"        => ["full", "lcurve"],
    "field_adapter"        => ["reshape", "spherical_harmonics"],
    "grad_data_term"       => ["variance", "softrange"],
    "main_field_direction" => ["+x", "-x", "+y", "-y", "auto"],
    "mode"                 => ["RMS", "STDIV"],
    "sweep_xkey"           => ["lambda", "T0", "alpha", "iters", "restarts", "step0", "step_min"],
    "ring_search_method"   => ["greedy", "exhaustive", "paired"],
    "eval_domain"          => ["grid", "shell"],
    "shell_source"         => ["measured", "sh_fibonacci"],
    "sh_select"            => ["all", "first_n", "top_k"],
    "viewer_frame"         => ["scan", "optimizer"],
    "shim_csv_frame"       => ["scan", "optimizer"],
)
const SEED_RANGE_KEYS = ("benchmark_seeds", "grad_seeds", "sweep_seeds")
# Keys that must be a strictly positive number. magnet_Br_T / magnet_side_mm scale
# the whole operator: 0 would make every magnet inert (and silently "optimize" to
# nothing), a negative value flips every moment (Br) or is physically meaningless
# (side length).
const POSITIVE_KEYS = ("magnet_Br_T", "magnet_side_mm", "shim_radius_mm", "tray_slot_spacing_mm")

"""Throw a descriptive error if `cfg` (a full, merged config dict) is invalid."""
function validate_config(cfg::AbstractDict)
    errs = String[]
    for (k, allowed) in ENUMS
        haskey(cfg, k) || continue
        cfg[k] in allowed || push!(errs, "$k = $(repr(cfg[k])) — must be one of $(allowed)")
    end
    if haskey(cfg, "start_stage") && !(cfg["start_stage"] in -1:5)
        push!(errs, "start_stage = $(cfg["start_stage"]) — must be -1..5")
    end
    # SH selection: the kept degree / coefficient count must fit inside the fit (sh_degree).
    if haskey(cfg, "sh_use_degree")
        v = cfg["sh_use_degree"]; L = get(cfg, "sh_degree", 8)
        (v isa Integer && v >= 1 && (!(L isa Integer) || v <= L)) ||
            push!(errs, "sh_use_degree = $(repr(v)) — must be a whole number between 1 and sh_degree ($(L))")
    end
    if haskey(cfg, "sh_top_k")
        v = cfg["sh_top_k"]; L = get(cfg, "sh_degree", 8)
        (v isa Integer && v >= 1 && (!(L isa Integer) || v <= (L + 1)^2 - 1)) ||
            push!(errs, "sh_top_k = $(repr(v)) — must be a whole number between 1 and (sh_degree+1)²−1")
    end
    if haskey(cfg, "verifier_sh_max_degree")
        v = cfg["verifier_sh_max_degree"]
        (v isa Integer && v >= 1) || push!(errs, "verifier_sh_max_degree = $(repr(v)) — must be a whole number ≥ 1")
    end
    for k in SEED_RANGE_KEYS
        haskey(cfg, k) || continue
        v = cfg[k]
        if !(v isa AbstractVector && length(v) == 2 && all(x -> x isa Integer, v) && v[1] <= v[2])
            push!(errs, "$k = $(repr(v)) — must be [lo, hi] integers with lo ≤ hi")
        end
    end
    for k in POSITIVE_KEYS
        haskey(cfg, k) || continue
        v = cfg[k]
        (v isa Real && v > 0) || push!(errs, "$k = $(repr(v)) — must be a positive number")
    end
    if haskey(cfg, "initial_angle_deg")
        v = cfg["initial_angle_deg"]
        (v isa Real && isfinite(v)) || push!(errs, "initial_angle_deg = $(repr(v)) — must be a number (deg)")
    end
    # Tray shifts are distances from z = 0 — a negative one would put tray +1 on the
    # wrong side of centre and silently interleave the two ends.
    for k in ("front_tray_shift_mm", "back_tray_shift_mm")
        haskey(cfg, k) || continue
        v = cfg[k]
        (v isa Real && isfinite(v) && v >= 0) ||
            push!(errs, "$k = $(repr(v)) — must be a non-negative distance from z = 0 (mm)")
    end
    # Insert search: the budget must afford at least one FULL insert.
    if haskey(cfg, "magnets_available")
        v = cfg["magnets_available"]; mps = get(cfg, "mags_per_segment", 7)
        if !(v isa Integer && v >= 0)
            push!(errs, "magnets_available = $(repr(v)) — must be a non-negative whole number of magnets")
        elseif v isa Integer && mps isa Integer && mps > 0 && v < mps
            push!(errs, "magnets_available = $(v) is fewer than mags_per_segment = $(mps) — " *
                        "not enough for one full insert")
        end
    end
    # Tray 0 does not exist (utils/pos_trays.jl errors on it) — catch it here, before a run.
    for k in ("positions_in_tray_new_wished", "positions_in_tray_occupied")
        haskey(cfg, k) || continue
        v = cfg[k]
        v isa AbstractVector || continue
        any(x -> x isa Integer && x == 0, v) &&
            push!(errs, "$k contains tray 0 — trays are numbered ±1, ±2, … outward from z = 0")
    end
    isempty(errs) || error("config.toml validation failed:\n  - " * join(errs, "\n  - "))
    return nothing
end

# --- config read / write (write preserves comments + key order) --------------
read_config() = TOML.parsefile(CONFIG)

_toml_val(v::AbstractString) = "\"" * v * "\""
_toml_val(v::Bool)           = v ? "true" : "false"
_toml_val(v::Integer)        = string(v)
_toml_val(v::AbstractFloat)  = string(v)                       # 231.0 → "231.0"
_toml_val(v::AbstractVector) = "[" * join(_toml_val.(v), ", ") * "]"
_toml_val(v) = error("cannot serialize $(typeof(v)) to TOML: $(repr(v))")

"""
    write_config(updates)

Apply `updates` (Dict of key => value; keys as Strings or Symbols) to config.toml,
editing values in place so comments and layout survive. Unknown keys are appended.
Validates the resulting full config before writing (throws, leaving the file
untouched, if invalid). Assumes one `key = value` per line (as in the template).
"""
function write_config(updates::AbstractDict)
    upd = Dict(string(k) => v for (k, v) in updates)
    validate_config(merge(read_config(), upd))          # validate first — fail safe

    lines = readlines(CONFIG)
    seen  = Set{String}()
    for (i, ln) in enumerate(lines)
        m = match(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=", ln)
        m === nothing && continue
        key = m.captures[2]
        haskey(upd, key) || continue
        push!(seen, key)
        h = findfirst('#', ln)                           # our values never contain '#'
        comment = h === nothing ? "" : "   " * rstrip(ln[h:end])
        lines[i] = m.captures[1] * key * " = " * _toml_val(upd[key]) * comment
    end
    for (k, v) in upd                                    # keys not already in the file
        k in seen && continue
        push!(lines, k * " = " * _toml_val(v))
    end
    open(CONFIG, "w") do io
        for ln in lines; println(io, ln); end
    end
    return CONFIG
end

# --- measured-CSV data folder ------------------------------------------------
list_measured() = sort(filter(f -> endswith(lowercase(f), ".csv"), readdir(MEASURED_DIR)))

"""Copy `src` into the measured-data folder; return the stored filename."""
function import_csv(src::AbstractString)
    isfile(src) || error("no such file: $src")
    endswith(lowercase(src), ".csv") || @warn "importing a non-.csv file" file = src
    mkpath(MEASURED_DIR)
    cp(src, joinpath(MEASURED_DIR, basename(src)); force = true)
    return basename(src)
end

# --- OSII import folder (alternative Stage 1.5 input) ------------------------
list_osii() = isdir(OSII_DIR) ?
    sort(filter(f -> endswith(lowercase(f), ".csv"), readdir(OSII_DIR))) : String[]

"""Copy `src` into the OSII import folder; return the stored filename."""
function import_osii(src::AbstractString)
    isfile(src) || error("no such file: $src")
    endswith(lowercase(src), ".csv") || @warn "importing a non-.csv file" file = src
    mkpath(OSII_DIR)
    cp(src, joinpath(OSII_DIR, basename(src)); force = true)
    return basename(src)
end

# --- what is actually in the current shim CSV --------------------------------
"""
    placement() -> (; exists, magnets, rings)

Summarise the shim CSV at the Stage-2 seam by (RingNumber, tray), so the GUI can
show which trays are really filled. A ring-level search fills all `num_trays`
trays of a ring; `stages/stage1_optimize/grad_optim/insert_search.jl` and an OSII import can fill only
some, and the ring-placement table would otherwise imply a full ring either way.

Tray is re-derived from (X, Y) exactly as `utils/helping_functions_for_JIG.jl`
`assign_tray` does, so these numbers match the printed `Ring_N/Tray_T` files.
Pure parsing — no Gmsh, no CUDA — so it is safe to call on every page load.
"""
function placement()
    cfg = read_config()
    it  = get(cfg, "iteration", "")
    csv = joinpath(REPO, "data", "outputs", "Optimizer_Output_per_Iteration", it, "$(it)_shim.csv")
    isfile(csv) || return (; exists = false, magnets = 0, rings = [])
    nt = Int(get(cfg, "num_trays", 12))
    counts = Dict{Tuple{Int,Int},Int}()          # (ring, tray) => magnets
    total = 0
    for (i, ln) in enumerate(eachline(csv))
        i == 1 && continue                        # header
        f = split(ln, ',')
        length(f) < 4 && continue
        x = tryparse(Float64, f[1]); y = tryparse(Float64, f[2]); r = tryparse(Float64, f[3])
        (x === nothing || y === nothing || r === nothing) && continue
        ang  = mod(atand(y, x), 360.0)
        tray = mod(round(Int, ang / (360.0 / nt)) + 8, nt) + 1
        k = (Int(r), tray)
        counts[k] = get(counts, k, 0) + 1
        total += 1
    end
    rings = [(; ring = rn,
               trays   = sort([t for ((r, t), _) in counts if r == rn]),
               magnets = sum(v for ((r, _), v) in counts if r == rn; init = 0))
             for rn in sort(unique(first(k) for k in keys(counts)))]
    return (; exists = true, magnets = total, rings)
end

# --- current iteration's output folders (derived from config) ----------------
function output_dirs()
    it = read_config()["iteration"]
    opt = joinpath(REPO, "data", "outputs", "Optimizer_Output_per_Iteration", it)
    return (optimizer = opt,
            final      = joinpath(REPO, "data", "outputs", "Final_3D_printing_outputs_per_Iteration", it),
            verifier   = joinpath(opt, get(read_config(), "verifier_output_subdir", "PythonVerifier")))
end

# --- run a stage as a streamed subprocess ------------------------------------
"""
    run_stage(name; on_output=println) -> exit_code

Launch stage `name` (see `stage_names()`) as its own `julia` subprocess in the
repo root, streaming merged stdout+stderr line-by-line to `on_output` (wire this
to the GUI log). stdin is /dev/null (avoids the Windows console-deadlock). Returns
the process exit code (0 = success). Blocks until the stage finishes, so the
frontend should call it on a background task to keep the UI responsive.
"""
# the currently-running stage subprocess (so stop_run() can kill it); nothing when idle
const CURRENT_PROC = Ref{Union{Base.Process,Nothing}}(nothing)

function run_stage(name::AbstractString; on_output = println)
    haskey(STAGES, name) || error("unknown stage $(repr(name)); options: $(stage_names())")
    script = joinpath(REPO, STAGES[name])
    isfile(script) || error("stage script not found: $script")
    cmd = Cmd(`$(Base.julia_cmd()) $script`; dir = REPO)
    out = Pipe()
    proc = run(pipeline(cmd; stdout = out, stderr = out, stdin = devnull); wait = false)
    CURRENT_PROC[] = proc
    close(out.in)
    try
        for line in eachline(out)
            on_output(line)
        end
        wait(proc)
    finally
        CURRENT_PROC[] = nothing
    end
    return proc.exitcode
end

"""Kill the stage subprocess currently running (if any). Returns true if one was killed."""
function stop_run()
    p = CURRENT_PROC[]
    p === nothing && return false
    try; kill(p); catch; end          # SIGTERM / TerminateProcess
    return true
end

# --- open a folder in the OS file browser ------------------------------------
function open_folder(path::AbstractString)
    p = abspath(path)
    isdir(p) || mkpath(p)
    opener = Sys.iswindows() ? `explorer $p` : Sys.isapple() ? `open $p` : `xdg-open $p`
    try; run(opener); catch; end          # explorer/xdg often return nonzero — ignore
    return p
end

end # module Backend

# quick self-check when run directly: `julia gui/backend.jl`
if abspath(PROGRAM_FILE) == @__FILE__
    using .Backend
    cfg = Backend.read_config()
    println("config.toml OK — ", length(cfg), " keys")
    println("iteration      : ", cfg["iteration"], "   optimizer: ", cfg["optimizer"])
    println("measured CSVs  : ", Backend.list_measured())
    println("stages         : ", Backend.stage_names())
    d = Backend.output_dirs()
    println("output (optim) : ", d.optimizer)
    println("output (final) : ", d.final)
    println("(no stage launched — this is a dry self-check)")
end
