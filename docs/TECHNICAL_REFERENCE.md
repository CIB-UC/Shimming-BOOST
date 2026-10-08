# Low-Field MRI Shimming Pipeline — Technical Reference

Passive-shimming toolchain for a low-field Halbach MRI magnet (Pontificia
Universidad Católica de Chile — Low-Field MRI project). It takes a measured
magnetic-field map, computes the optimal angles for a set of small permanent
"shim" magnets that flatten the field, and produces 3D-printable tray inserts —
all driven from a single config file or a browser GUI.

```
measured CSV ──▶ field map ──▶ optimal shim angles ──▶ shim CSV ──▶ STL/STEP trays
   Stage 0          Stage 1          (Stage 1)          Stage 1.5      Stage 2

 measured CSV + shim CSV ──▶ independent Python (magpylib) check of the result
                                         Stage 4
```

Everything is organized as one **iteration**: name it, set parameters, run, get
printable parts plus a verification view.

---

## Part I — Project information

**The problem.** A Halbach array gives a strong-ish B₀ in the bore, but it is not
homogeneous enough for imaging. Passive shimming corrects this by placing many
small magnets on trays around the bore; rotating each magnet changes its field
contribution. The task is to choose all the angles so the field over a target
volume is as uniform as possible.

**The measurement.** B₀ is measured on a spherical shell (or a grid) with a
gaussmeter on a robot. The pipeline reads that scan, models the field, and scores
homogeneity as **ppm** = 10⁶ · (max − min) / mean over the target region.

**The key scientific result.** The shim field is *exactly linear* in the magnet
moments, so field variance is a smooth, convex function of the angles with an
analytic gradient. A gradient method (L-BFGS) therefore finds the **global**
optimum in seconds — and shows that the remaining inhomogeneity is a **hardware
limit**, not an optimizer limit. The lever from here is physical: more / stronger /
closer shim magnets, or more rings.

**Three headline capabilities beyond "solve for the angles":**
- **Ring search** — instead of telling it where to put rings, ask it to *find* the
  best set of *n* rings from a candidate range.
- **Insert search** — a finer granularity: place ONE insert at a time (one tray at
  one ring, `mags_per_segment` magnets), driven by how many magnets you actually
  own, freezing each choice before searching the next.
- **Shell evaluation** — score homogeneity on a single spherical shell at the DSV
  radius. Because the bore field is harmonic, min/max on that surface equal min/max
  over the whole ball, so range/ppm is exact with far fewer points.

---

## Part II — How it works

### Quick start

```bash
julia install_deps.jl        # once — pins the project (CUDA, GLMakie, Gmsh, Optim, HTTP, JSON, …)

# Option A — the GUI (recommended)
julia gui/server.jl          # opens http://localhost:8010

# Option B — terminal, one config-driven run
julia run_pipeline.jl        # runs Stage 0→1→1.5→2→3 (viewers)→4 per config.toml
```

Needs an NVIDIA GPU (CUDA) for Stage 1 and the viewers. Everything is driven by
**`config.toml`**; `pipeline_config.jl` reads it and derives every path + geometry,
so nothing can drift apart.

### The stages

| Stage | Script | Does |
|-------|--------|------|
| 0 | `Field_data_*` adapters | measured CSV → field map (`.jld2`) |
| 1 | `sim_annealing_optim/` **or** `grad_optim/` | field map → optimal shim-magnet angles |
| 1.5 | `export_csv.jl` | optimizer result → shim CSV (`X, Y, RingNumber, Angle`; `RingNumber` = real InsertPos) — the seam |
| 2 | `CSV_to_STL.jl` (Gmsh) | shim CSV → per-tray STL + STEP, + the static insert |
| 3 | `Shimming_magnets_visualizer.jl`, `field_slice_viewer.jl` | 3D magnet/field view, 2D slice + profile; each has a "Save GIF" button |
| 4 | `python_verifier.jl` → `stages/stage4_verifier/Shimming_verifier/run_verifier.py` | independent **Python/magpylib** re-computation of the shimmed field: ppm, 2D projections, 3D scene, spherical-harmonic pyramids |

`run_pipeline.jl` chains them as separate subprocesses (so CUDA/GLMakie and Gmsh
never share a process), checks each exit code + output, and stops at the first
failure. `start_stage` lets you jump in and reuse earlier outputs.

**Two shortcuts that skip the optimizer:** `osii_to_shim.jl` converts an existing
**OSII** magnet layout straight into the Stage-1.5 shim CSV (then Stage 2 runs as usual —
GUI button "Build STL only"), and both **Stage-3 viewers run with no shim CSV** — after
Stage 0 alone they show just the measured field (pre-shim) for homogeneity inspection.

### Stage 4 — independent Python verifier (`Shimming_verifier/`)

A self-contained Python tool (magpylib closed-form cuboid model — **no shared code with
the Julia kernels**) that takes the measured field and the shim layout, recomputes the
shim's field at the measured points, and reports the shimmed homogeneity. Because it is
a different model of the same magnets, agreement with the optimizer's ppm is a genuine
cross-check of the operator `G` (reference run: optimizer predicted 9483 ppm, verifier
9562 ppm, unshimmed 18538 ppm). It stays independent of the pipeline: it needs only
**which field, which shim layout, what to run**, and lives in its own folder
(`main.py` = interactive/VSCode entry, `run_verifier.py` = command-line entry, deps in
`requirements.txt`).

`python_verifier.jl` is the thin adapter (Julia stdlib only — no CUDA, no field map):
- **Inputs** — `verifier_measurement` (blank = the run's own scan) and the shim CSV
  (`verifier_use_shim`; blank `verifier_shim_csv` = this iteration's `_shim.csv`, i.e.
  whatever produced the seam: optimizer export, insert search or OSII import). The
  seam's `RingNumber` (the real InsertPos) is converted back to axial z with
  `ringpos_from_tray_mm` and the current tray-geometry keys; a CSV that already has a
  `Z (mm)` column (e.g. `export_csv.jl`'s `_shim_xyz.csv`) is accepted as-is.
- **What to run** — `verifier_make_2d` / `_3d` / `_sh` (+ `verifier_sh_max_degree`,
  `verifier_show_figures`).
- **Not repeated** — magnet strength/size come from `magnet_Br_T` / `magnet_side_mm`,
  the coordinate unit from `sh_measured_unit_mm`.
- **Frame.** Stage 0 rotates the scan so B0 → +y for the optimizer, so every shim CSV is
  in that frame, while the verifier reads the *raw* scan. `run_verifier.py
  --shim-frame optimizer` rotates the shim positions + angles back into the scan's frame
  using B0's direction (`main_field_direction`, or auto-detected from the scan's mean
  field vector). Identity for the current scanner (B0 = +y); checked with synthetic
  +x / −x / −y scans, which reproduce the +y result exactly. **Now** (`shim_csv_frame = "scan"`, the
  default) the seam CSV is already in the scan frame, so `python_verifier.jl` passes `--shim-frame lab`
  (CSV as written); with `"optimizer"` it still passes `--shim-frame optimizer`.
- **Outputs** → `data/outputs/Optimizer_Output_per_Iteration/<ITER>/PythonVerifier/`
  (`comparison_2d.png`, `scene_3d.html`, `sh_pyramid.png`, plus the derived shim input).
- **Running it** — GUI Stage 4 "Run verifier"; `julia stages/stage4_verifier/python_verifier.jl`;
  `start_stage = 5` (verifier only); or `run_python_verifier = true` to append it to a
  full run. Misalignment-study knobs (remanence spread, measurement/magnet shifts) exist
  on the CLI (`stages/stage4_verifier/python Shimming_verifier/run_verifier.py --help`) but default to 0.

### Stage 0 — reading the measurement

A shared reader, `utils/read_measured.jl`, handles **two spherical-scan CSV
formats** transparently:
- **simple** — metadata line 1, header line 2 (`X,Y,Z,Gauss`) — the original Josh scan;
- **"Controlled"** — a multi-line metadata block, then columns for `radio, theta,
  phi, X,Y,Z, T_robot, muestra, …, x/y/z axis (gauss), Magnitude (gauss)`.

It finds the data header wherever it is, **averages the repeated samples per point**
(`muestra`), uses `|B|` (the `Magnitude` column), and — when the scan recorded field
components — returns the mean `(Bx,By,Bz)` vector. Temperature is read but not used
yet.

Three Stage-0 adapters consume that:
- **`Field_data_file_adapter.jl`** (`field_adapter = :reshape`) — a complete regular
  (parallelepiped) grid; just reshapes it.
- **`Field_data_SH_interpolator.jl`** (`:spherical_harmonics`) — fits regular solid
  spherical harmonics to a shell/scattered scan, evaluates on a **cube grid**.
- **`Field_data_shell.jl`** (`eval_domain = :shell`) — the optimizer's shell at Rmax.
  `shell_source = "measured"` (default) feeds the **raw measured points + measured |B|**
  directly — no interpolation, the honest choice when the scan already sits on the Rmax
  sphere; `"sh_fibonacci"` instead evaluates the SH fit on a **Fibonacci sphere** of
  `shell_n_points`. Either way it *also* writes the cube grid (`sh_grid_n` pts/axis) so
  the viewers still work.

**Spherical-harmonic decomposition + selection (SH and shell adapters).** Both SH-based
adapters already fit `B ≈ Σ a_{n,m}(r/Rn)^n Y_{n,m}` (degree `sh_degree`). Stage 0 now
always **prints the decomposition** — mean field, per-degree strength, and the `sh_top_k`
(default 5) largest `|a_{n,m}|` with `n ≥ 1` and their share of the energy — and writes every
coefficient to `data/outputs/Optimizer_Output_per_Iteration/<ITER>/SHDecomposition/sh_decomposition.csv`
(`n, m, a_full, a_used, rank, kept`). `sh_select` then chooses what the **field map is built
from** (kept columns are *re-fitted* to the data, not just truncated):

| `sh_select` | field map built from |
|---|---|
| `"all"` (default) | every degree 0..`sh_degree` — the original behaviour, bit-for-bit unchanged |
| `"first_n"` | degrees 0..`sh_use_degree` only (smooths away high-order detail) |
| `"top_k"` | the mean + the `sh_top_k` largest coefficients only (dominant harmonics) |

With `first_n` / `top_k` the optimizer and both viewers see the **filtered** field. In shell mode,
`shell_source = "measured"` then means *the measured points, with SH-filtered values* (with
`"all"` it is still the raw measured `|B|`); `sh_fibonacci` simply evaluates the filtered fit.
The saved jld2s also carry `c_full`, `sel_cols`, `sh_select_name`. Coefficients live in the
**optimizer frame** (B0 → +y), so `m` differs from the Stage-4 pyramid (raw lab frame) whenever
the scanner's B0 isn't already +y. Logic: `utils/sh_select.jl` (stdlib-only, unit-testable).

**Auto direction.** Set `main_field_direction = "auto"` and the SH/shell adapters
detect B₀'s axis from the measured field vector (the dominant in-plane component).
Otherwise use an explicit `+x/-x/+y/-y`. The direction only sets how coordinates
rotate into the optimizer frame (B₀ → +y); the stored field is `|B|`.

### Coordinate frames (scan vs optimizer) — where the rotation happens

Three frames exist; the pipeline converts between the first two only.
- **Scan (lab) frame** — the measured CSV's own `X, Y, Z`. B0 points along whatever
  `main_field_direction` says (`+x/-x/+y/-y`, or `auto` = the dominant in-plane component of the
  mean field vector the probe recorded).
- **Optimizer frame** — Stage 0 rotates the scan **about the bore (z)** so B0 → +y
  (`lab_to_optimizer_xy`: `+x → (−y, x)`, `−x → (y, −x)`, `−y → (−x, −y)`; z untouched). The field is
  stored as |B| (positive), so no vector is rotated and the sign is dropped. The operator `G`, the
  optimizer, the result jld2s and the field maps all live here. The resolved direction is saved in the
  field-map jld2 as `field_direction`.
- **Physical tray frame** — Stage 2 numbers trays from the CSV's X, Y (`assign_tray`: tray 12 → +y,
  9 → +x, 6 → −y, 3 → −x). It is **not** converted anywhere: it simply takes the CSV's frame to be the
  magnet's. Whether that is right depends on where tray 12 physically sits relative to B0 / the scan axes
  (open question).

**Rotating back (new).** Two settings in `config.toml`, both default `"scan"`:
- `shim_csv_frame` — frame the shim CSV is **written** in. With `"scan"`, `export_csv.jl` (both
  `_shim.csv` and `_shim_xyz.csv`) and `grad_optim/insert_search.jl` rotate positions **and** moment angles
  back by the scan's B0 direction right after the optimizer (`+x`: (x,y)→(y,−x), angle −90°; `−x`:
  (−y,x), +90°; `−y`: (−x,−y), +180°). Stage 2 STLs, both viewers and the Stage-4 verifier then work in the
  scan frame; Stage 2 re-derives the tray from the rotated X, Y (tray *t* → *t*+6 for a −y scan). Result
  jld2s, operators and field maps stay in the optimizer frame. `"optimizer"` = the old behaviour.
  Hand-made CSVs (OSII import, `make_halbach_ring_csv.py`) are used as written, in whichever frame this
  key declares. **CSVs written before this change are not converted — re-run Stage 1.5 export.**
- `viewer_frame` — frame the **Stage-3 viewers draw** in. `"scan"`: the field grid + axes
  (`utils/viewer_frame.jl: grid_to_scan`) and the magnets are rotated back for display, and the field is
  shown **signed** (negative for a `−x/−y` scan, label "B (mT), B0 along −y"). `"optimizer"`: |B|, B0 → +y.
  Display-only; the dipole field is still computed in the optimizer frame, then rotated.
Neither key has a GUI field yet (`config.toml` only; `gui/backend.jl` validates the values).

### Evaluation domain — grid vs shell

`eval_domain` chooses where the gradient optimizer + ring search score:
- **`"grid"`** (default) — a dense Cartesian grid; the ∇B/Tikhonov term is available,
  and the SA optimizer + viewers use this.
- **`"shell"`** — a single spherical shell at `Rmax`. Exact range/ppm with far fewer
  points; the ∇B term is disabled; gradient + ring search only. `setup_shell.jl` feeds
  the shell points as the evaluation set. `shell_source` picks those points:
  **`"measured"`** (default) scores the raw measured points + measured |B| (no
  interpolation); **`"sh_fibonacci"`** resamples the SH fit onto `shell_n_points` uniform
  points — smoother, but hides sharp extrema (measured scored ~200 ppm higher than the
  SH-smoothed shell on the reference scan).

### Stage 1 — two optimizers + ring search

The field is exactly linear in the moments:
`By = By_base + Gx·(μcosθ) + Gy·(μsinθ)` (operator `G` validated to rel. err ≈ 4e-8).

**Simulated annealing — `sim_annealing_optim/`.** The original BOOST optimizer;
minimizes `w·(max−min) + λ·√mean|∇B|²` by annealing. Kept as a baseline and for the
λ (Tikhonov) sweep. Files: `run_optim.jl`, `run_lcurve.jl`, `benchmark.jl`,
`operation.jl`.

**Gradient optimizer — `grad_optim/`.** Because the field is linear, variance /
soft-range / the ∇B penalty are smooth with analytic gradients, so **L-BFGS finds
the global optimum in seconds** (10/10 identical restarts). Two phases, split by the
cached operator `operator_G.jld2`:
- `optim_grad.jl` (GPU) — build + cache the linear operator `G`.
- `run_grad.jl` (CPU) — minimize `data_term + grad_lambda·mean|∇B|²` with L-BFGS.
- `run_grad_lcurve.jl` — sweep `grad_lambda` → range-vs-gradient L-curve (overlaid on SA).
- `grad_math.jl` — pure shared math; `grad_core.jl` — binds it to the cached operator.
- Objective knobs: `grad_data_term` (`:variance` / `:softrange`), `grad_softrange_beta`,
  `grad_lambda`.

**Shim-magnet strength.** Two numbers, `magnet_Br_T` + `magnet_side_mm` in
`config.toml`: the magnet's **remanence** Br (T, a material property from the
datasheet or your own measurement setup, independent of size) and its physical
cube side length (mm). `pipeline_config.jl` converts these to the dipole moment the
kernels consume — `μ = Br·side_m³/μ0` (`μ0 = 4π×10⁻⁷`) — and the optimizer, ring
search, insert search and **both viewers** all read it; it used to be hardcoded, in
two different units, in five separate files, then briefly lived as a single
on-axis-field-at-1cm proxy (`magnet_B1cm_mT`) before being replaced by the Br/side
pair now that Br is measured/calculated consistently for the actual 6×6×6 mm
magnets. (`magnet_B1cm_mT`/`B1CM_T` still exist internally as **derived, back-compat**
constants — reporting what on-axis field at 1 cm this Br+geometry implies, for
`utils/imanes.jl`'s calling convention — but are no longer config inputs.) Changing
either value invalidates the cached operator; `grad_core.jl` warns if the cache
disagrees with the config. `initial_angle_deg` (the SA's start point and the angle
`optim_grad` validates `G` at) sits beside it.

**Axial tray geometry.** The two ends of the bore are anchored independently, since
they need not be symmetric:
`z(+n) = +(back_tray_shift_mm + (n−1)·tray_slot_spacing_mm)` toward +Z ("Back — Wall"),
`z(−n) = −(front_tray_shift_mm + (n−1)·tray_slot_spacing_mm)` toward −Z ("Front —
Gaussmeter"). Tray 0 does not exist. This replaces the old symmetric `half_shift_mm`
(`h` at spacing `s` ≡ `front = back = s − h`); 10/10/10 reproduces `half_shift_mm = 0`
exactly. Both Stage-3 viewers now read these from config — they used to hardcode
`half_shift_mm = 0.0` and so drew magnets at the wrong z for any non-zero shift.

**Ring search — `grad_optim/ring_search.jl`.** Finds the **best set of *n* rings**
instead of you specifying them. Each candidate ring's field response is a fixed
block of `G` columns, so it builds the operator once for all candidate rings
(cached `ring_operator.jld2`), then scoring any subset is a small variance solve.
- `method = :greedy` — forward selection, `n·N` solves; scales to any *n*.
- `method = :exhaustive` — all `C(N,n)` subsets, guarded by `ring_search_max_combos`.
- `method = :paired` — only **symmetric** sets: `ring_search_n` is the *total* (even) and
  the rings come as `n/2` pairs `{−a, +a}` (same tray number either side of z = 0), e.g.
  `[-10, -5, 5, 10]`. It tries **every** choice of `n/2` pair magnitudes,
  `C(#pairs, n/2)` sets (300 for n=4 over ±25; 2 300 for n=6; 12 650 for n=8), guarded by
  `ring_search_max_combos`. Only the *positions* are paired — every magnet's angle is still
  free. Pairing is by tray number, so it is an exact z-mirror only when
  `front_tray_shift_mm == back_tray_shift_mm`. The usual `min_sep` / `min_spots_between`
  rules still apply (so with the default 3, a = 1 and adjacent pairs like ±5, ±6 are skipped).
  Needs a range symmetric about 0; `ring_search_n_max` (default 8) still caps the total.
  Writes `ring_search.csv` (top 50 sets) + a ranked plot; `ring_search_apply` works as usual.
  Enumeration lives in `grad_optim/ring_combos.jl` (pure, unit-testable).
- Rejects rings closer than `ring_search_min_spots_between` empty tray spots.
- Writes CSV + plot; `ring_search_apply` writes the winner into
  `positions_in_tray_new_wished` and saves the standard result so Stage 2 can run.

**Insert search ranking mode.** `insert_search.jl` has a local `RANK_FULL_SHELL`
toggle: `false` ranks every candidate slot cheaply on a ~250-point sub-shell and
re-solves only the winner on the full shell (the ~30× faster mode, described
below); `RANK_FULL_SHELL = true` (current default in the file) instead ranks
**every** candidate on the full shell — an accuracy/speed check on whether the
subsample ever picks a different "best slot" than scoring against everything.
Cost scales with `Ns/RANK_NS` per round (e.g. 1742 vs 250 ⇒ ~7× slower ranking),
so expect a full-shell run to take noticeably longer per insert; flip it back to
`false` for routine fast placement once the two modes have been cross-checked.

**Insert search — `grad_optim/insert_search.jl`.** Same linear algebra, finer unit.
A candidate slot is one **insert**: one tray at one ring, holding `mags_per_segment`
magnets. Because a ring's operator block already contains every tray's columns, an
insert is just an `MPS`-column slice — so this **shares `ring_operator.jld2`** with
the ring search (identical `SIG`; whichever runs first pays the one-time GPU build).

The loop is *frozen sequential placement* (matching pursuit), not the ring search's
greedy: each round scores every free legal slot against the **current** base field,
picks the best, re-solves it on the full shell, then **freezes its angles and folds
its field into the base** before the next round. Placed angles are never revisited,
so every solve is `MPS` variables no matter how full the bore gets — cost per round
is flat.

- **How many:** `magnets_available ÷ mags_per_segment`, rounded down. Every insert is
  assumed FULL; a remainder is reported and left unplaced.
- **Spacing:** two inserts in the *same tray* need at least
  `insert_search_min_spots_between` empty ring slots between them. Different trays are
  separate printed pieces, so they are unconstrained. (`spots_between` discounts the
  non-existent tray 0; `ring_search` uses the simpler `|a−b|−1`.)
- **Fixed-magnitude caveat:** a magnet can be rotated but not switched off, so ppm is
  *not* guaranteed to fall monotonically. It places the whole budget, logs ppm at each
  step, and reports the best **prefix** — publishing only that prefix.
- **Output:** a *sparse* shim CSV plus a standard result jld2 whose `final_state`
  carries the sparsity (1 = insert placed, 0 = empty slot). That mask is what lets the
  rest of the pipeline treat it normally — `eval_metrics` scores it, and `export_csv`
  regenerates the same sparse CSV rather than a dense one.
- Segment index ≠ tray number: the tray is re-derived from (x,y) exactly as
  `CSV_to_STL` does, so the spacing rule and the printed `Ring_<InsertPos>/Tray_T`
  agree.

**RingNumber = real, physical InsertPos (not a sequential index).** The shim CSV's
`RingNumber` column, `p.ring` throughout the code, and the `InsertPos` argument
`utils/helping_functions_for_JIG.jl`'s `make_label` engraves on the printed part are
all **the same number**: the actual signed tray slot (e.g. `-7`), not a 0-based
position in whatever ring set the current run happens to use. Every stage that
produces or consumes RingNumber (`export_csv.jl`, `insert_search.jl`,
`osii_to_shim.jl`, `CSV_to_STL.jl`, the GUI, both Stage-3 viewers) derives it
directly from the physical value — never by zipping a sorted/enumerated list against
`positions_in_tray_new_wished` positionally, since that array's order need not match
a CSV's ring order and a CSV's ring set can differ from the current config's (an
OSII import, a sparse insert-search result, or a re-run under a different config).
Printed parts and folder/file names spell the sign as a **letter token**, `N`/`P`,
rather than a literal `+`/`-` (fragile to engrave/type reliably) —
`insertpos_label(pos) = (pos<0 ? "N" : "P") * lpad(abs(pos), 2, '0')`, e.g. `-7 →
"N07"`, `12 → "P12"` — duplicated locally in each file per the codebase's existing
convention of small local helpers over a shared-utils import for this kind of thing.

### Metrics & verification

- `utils/eval_metrics.jl` — range / ppm / mean By / gradient-RMS for **any** saved
  result (SA or gradient), side by side with a no-shims baseline, identical
  definitions.
- `utils/verify_solution.jl` — independently re-checks a result on the GPU.
- Every gradient solve runs a finite-difference gradient self-check at startup.
- **GIF export (both Stage-3 viewers).** Each viewer has a single-view "Save GIF"
  button and a "Measured vs Shimmed" pair-comparison button: the 3D viewer records a
  360° rotating-camera GIF at its current elevation/distance, the slice viewer
  records a slice-sweep GIF. Both use `Makie.record(...)` **explicitly qualified**
  (never the bare `record`), because CUDA and GLMakie both export a conflicting
  `record` binding and the unqualified name is ambiguous once both packages are
  loaded. Config: `viewer_gif_frames` (frames/GIF), `viewer_gif_fps`, and
  `viewer_gif_subdir` (default `"Viewers"`, saved under
  `data/outputs/Optimizer_Output_per_Iteration/<ITER>/Viewers/`); length in seconds =
  `viewer_gif_frames / viewer_gif_fps`.

### Configuration (`config.toml`)

The only file you edit for a run. Grouped: run mode / optimizer / eval_domain,
iteration identity, shared geometry, Stage-2 insert params, SA params, gradient
objective, SA sweep, ring search, Stage-0 adapter, Stage-3 viewers. Conventions:
enums are strings (`optimizer = "grad"`), seed ranges are `[lo, hi]`, arrays are
TOML arrays. `pipeline_config.jl` reads each key with a typed accessor and derives
all absolute paths.

**Presets** (`gui/presets.json`): named geometry and Stage-2 insert setups —
`OSII V1.1` and `OSII V2.1 (PUC trays)` (default V2.1). A Stage-2 version pins tray
radii, inter-magnet angle, and magnet radius together.

### The GUI (`gui/`)

A local web app, no build step. `julia gui/server.jl` serves `app.html` on
`localhost:8010` and a JSON API over `backend.jl`. Header reads "BOOST — B0
Optimization Shimming Technique" / "Low Field MRI Project UC", styled with a
light-green themed palette (`--bg`/`--acc`/etc. in `app.html`'s `<style>`).

- **`backend.jl`** — reads/writes `config.toml` (comment-preserving, validated),
  imports measured CSVs, runs any stage as a streamed subprocess, stops the current
  run, opens folders, reads/writes presets. `POSITIVE_KEYS` (must be > 0) includes
  `magnet_Br_T` and `magnet_side_mm`.
- **`server.jl`** — HTTP endpoints (`/api/config`, `/api/measured`, `/api/import`,
  `/api/run` [streams the live log], `/api/stop`, `/api/presets`, `/api/open_folder`).
- **`app.html`** — a top iteration-name box, then per-stage blocks: **Stage 0**
  (measured data + geometry preset + field adapter + `eval_domain` grid/shell +
  `shell_source` measured/SH + `auto` direction + a "Shim magnets" panel for
  `magnet_Br_T` / `magnet_side_mm` / `initial_angle_deg` with a live derived-μ
  read-out), **Stage 1** (task selector: Grad build+solve / Grad L-curve / SA
  optimize / SA L-curve / Ring search / Insert search — each shows only its
  parameters; SA hides in shell mode), **Stage 1.5** (OSII import: pick an OSII file
  + angle offset/invert → shim CSV), **Stage 2** (insert preset + params — including
  `letter_thickness` and `label_scale`, both editable — with **Build STL only** for
  the OSII / no-optimizer path), **Stage 3** (viewers with **Save GIF** buttons,
  open-folder, and a **ring-placement table** showing where each printed
  `Ring_<InsertPos>` (e.g. `Ring_N07`) mounts *and which trays actually carry
  magnets*, read from the shim CSV via `/api/placement` — so a sparse layout is not
  mistaken for a full ring). A pinned run log with **Stop** streams output live;
  parameter labels are bold with one-line descriptions.

### Repository layout

```
config.toml  Project.toml  Manifest.toml  pipeline_config.jl   ← stay at the root
    config.toml        ← edit this (or use the GUI)
    pipeline_config.jl ← reads config.toml, derives paths + geometry
run_pipeline.jl        ← orchestrator (Stage 0→1→1.5→2→3→4)
install_deps.jl        ← pins the Julia project
README.md              ← plain-language guide (start here)

stages/
  stage0_field/      Field_data_file_adapter.jl    (regular grid → grid)
                     Field_data_SH_interpolator.jl (shell scan → grid, SH fit)
                     Field_data_shell.jl           (any scan → shell at Rmax, SH fit; + grid for viewers)
  stage1_optimize/   setup.jl / setup_shell.jl     (evaluation context: grid mesh / shell points + magnet positions)
                     sim_annealing_optim/  operation.jl, run_optim.jl, run_lcurve.jl, benchmark.jl
                     grad_optim/           grad_math.jl, grad_core.jl, optim_grad.jl, run_grad.jl,
                                           run_grad_lcurve.jl, ring_search.jl, ring_combos.jl, insert_search.jl
  stage1_5_export/   export_csv.jl     (optimizer result → shim CSV; honours final_state)
                     osii_to_shim.jl   (ALT: OSII layout → shim CSV; skips Stages 0–1)
  stage2_stl/        CSV_to_STL.jl     (Gmsh geometry)
  stage3_viewers/    Shimming_magnets_visualizer.jl (3D)   field_slice_viewer.jl (slice)
                     (both run field-only when no shim CSV exists)
  stage4_verifier/   python_verifier.jl  (adapter; launches stages/stage4_verifier/Shimming_verifier/run_verifier.py)
                     Shimming_verifier/  standalone Python (magpylib): main.py, run_verifier.py,
                       measurement_io, shim_magnets, field_analysis, viewer_2d/3d, sh_pyramid,
                       requirements.txt (+ its own Field_measurements/, Shimming_magnets/, output/)

core/    kernels/  f_kernel.jl (field), op_kernel.jl (SA mutation)
         utils/    read_measured, sh_select, grid_utils, pos_trays, wrap, ppm_report, imanes,
                   helping_functions_for_JIG, verify_solution, eval_metrics, viewer_frame
tools/   make_halbach_ring_csv.py   ← test tool: Halbach-dipole "bits" shim CSV (see below)
gui/     backend.jl, server.jl, app.html, presets.json
assets/  Magnet_Inserts_Models_stl_step/ , Static_Inserts_Models_stl_step/   base inserts

data/
  inputs/   data/inputs/Measured_Field_Data_csv/           input scans
            data/inputs/OSII_shimming_outputs_toconvert/   OSII layout CSVs (input to osii_to_shim.jl)
  cache/    data/cache/Interpolated_Field_Data_jld2/      Stage 0 output (grid + _shell maps)
  outputs/  Optimizer_Output_per_Iteration/<ITER>/
                <ITER>_BOOST_result.jld2        active Stage-1 result (last optimizer to run)
                <ITER>_SA_result.jld2           SA's copy      ┐ never clobber each other →
                GradOpt/grad_result.jld2        gradient's copy ┘ SA vs grad comparable
                GradOpt/operator_G.jld2         cached linear operator
                GradOpt/ring_operator.jld2      cached ring operator (SHARED: ring + insert search)
                InsertSearch/insert_result.jld2 insert search's copy (sparsity in final_state)
                Viewers/                        Stage-3 "Save GIF" output (viewer_gif_subdir)
                PythonVerifier/                 Stage-4 output (verifier_output_subdir)
                SHDecomposition/sh_decomposition.csv   Stage-0 SH coefficients (SH + shell adapters)
                Lcurve/, GradOpt/Lcurve/, RingSearch/, InsertSearch/, Benchmark/, imgs/
                <ITER>_shim.csv                 the seam (→ Stage 2); SPARSE after an insert search
                SPARSE_LAYOUT.marker            written by osii_to_shim; export_csv refuses to clobber
            Final_3D_printing_outputs_per_Iteration/<ITER>/Ring_<InsertPos>/{stl_outputs,Step_outputs}
                (folder/file names use the P/N sign token, e.g. Ring_N07, RingN07_Tray03.stl —
                 InsertPos is the real, physical, signed tray slot, not a sequential index)
```

Companion doc (in the repo root):
- `README.md` — a plain-language guide (no programming needed) to the OSII import,
  measured-shell scoring, and the ring search, with a glossary and click-by-click steps.

---

## Part III — What has been done

### Verified on the GPU box
- **Gradient optimizer** reaches the global optimum (parallelepiped scan: 7878 ppm;
  operator `G` rel. err 4e-8; 10/10 identical L-BFGS restarts).
- **Hardware limit confirmed:** the global optimum still leaves ~7878 ppm (vs 11163
  with no shims) — shims correct ~29% of the spread; the remaining lever is physical.
- **Ring search:** exhaustive n=2 on the Josh spherical scan → best pair (trays −2,
  +3), 17814.8 ppm, and an independent full grad solve on those two rings reproduces
  17814.8 **exactly**.
- Variance / soft-range gradient formulas pass finite-difference checks (~1e-7).

### Built + statically checked, not yet run on the GPU box
- **Shell mode** (`eval_domain = "shell"`): `Field_data_shell.jl`, `setup_shell.jl`,
  shell-tagged operator, `grad_core` shell branch, and the grid-for-viewers output.
  Confirm ppm ≈ a grid run (should match, since the shell captures boundary extrema),
  and that `Rmax` = the measured shell radius.
- **Controlled-CSV reader + auto direction**: `utils/read_measured.jl` verified
  against the actual files (Controlled scan → 1742 averaged points, auto-detects +x;
  Josh scan → 7082 points). Needs a live Stage-0 run to confirm.
- Greedy ring search at larger *n* (mechanism verified at n=2).

### Recent additions
- Two-optimizer split (SA vs gradient) with a config toggle + comparison tools.
- Gradient objective generalized (variance / soft-range / Tikhonov ∇B term).
- Ring search (greedy + exhaustive) with a spots-between constraint.
- The browser GUI (per-stage, live log, Stop, presets, ring-placement table).
- Shell-based evaluation; native Controlled-CSV reading with auto direction.
- **OSII import** (`osii_to_shim.jl`): convert an external OSII magnet layout into the
  Stage-1.5 shim CSV (frame remap, metres→mm, angle inverted + 90° offset), skipping
  Stages 0–1; GUI "Build STL only" feeds it to Stage 2. Iteration name/number carry
  through to the printed inserts.
- **Measured-shell scoring** (`shell_source = "measured"`, now default): score the raw
  measured points, not the SH-smoothed resample (~200 ppm more honest on the ref scan).
- **Ring search ~30× faster**: solve in Float64 (BLAS + reaches `g_tol`), and rank on a
  ~250-point sub-shell with the winner re-solved on the full shell; live "N scored" log.
- **Field-only viewers**: both Stage-3 viewers open on the measured field with no shim
  CSV (inspect homogeneity before shimming); the slice viewer also gained zoom-preserving,
  region-aware ppm + profile clipping, and a finer field grid (`sh_grid_n`, 5 mm default).
- **Insert search** (`grad_optim/insert_search.jl`): sequential per-insert placement
  driven by a magnet budget, sharing the ring operator; sparse output carried through
  the pipeline by `final_state`.
- **Magnet strength is configurable** — `magnet_Br_T` (remanence) + `magnet_side_mm`
  (cube side length) instead of hardcoded in five files in two units;
  `initial_angle_deg` likewise. (An intermediate `magnet_B1cm_mT` on-axis-field-at-1cm
  proxy was tried and then replaced by the Br/side pair, which needs no calibration
  distance; `magnet_B1cm_mT`/`B1CM_T` now survive only as derived back-compat values.)
- **GIF export from both Stage-3 viewers** — a rotating-camera GIF (3D viewer) or a
  slice-sweep GIF (slice viewer), each with a single-view button and a "Measured vs
  Shimmed" pair-comparison button, via `Makie.record` (explicitly qualified to avoid
  the CUDA/GLMakie `record` name clash). Config: `viewer_gif_frames`,
  `viewer_gif_fps`, `viewer_gif_subdir`.
- **GUI: `label_scale` + editable `letter_thickness`** — Stage 2 now exposes both the
  overall size of the engraved label digits (`label_scale`) and the engraving depth
  (`letter_thickness`) as editable fields; header/palette also refreshed to "BOOST —
  B0 Optimization Shimming Technique" / "Low Field MRI Project UC" on a light-green theme.
- **Stage 4 — independent Python verifier** (`python_verifier.jl` +
  `stages/stage4_verifier/Shimming_verifier/run_verifier.py`): inputs are just the measured field, the shim
  layout and what to run; wired into `run_pipeline.jl` (`run_python_verifier`,
  `start_stage = 5`), `config.toml` and a GUI Stage-4 block. Handles the optimizer-vs-scan
  frame rotation. Ran end-to-end on the reference iteration (9562 ppm vs the optimizer's
  9483).
- **Independent front/back tray shifts** (`front_tray_shift_mm`, `back_tray_shift_mm`,
  `tray_slot_spacing_mm`) replace the symmetric `half_shift_mm`; both viewers now read
  the config values instead of hardcoding 0.0.
- **`export_csv.jl` honours `final_state`** — it exported every magnet regardless of
  the on/off mask, which mis-exported any sparse or partially-switched-off solution.
- **RingNumber is now the real, physical InsertPos everywhere**, not a 0-based
  sequential index — `export_csv.jl`, `insert_search.jl`, `osii_to_shim.jl`,
  `CSV_to_STL.jl`, the GUI, and both Stage-3 viewers all derive/consume it directly
  from the physical tray slot; printed parts and folder/file names spell the sign as
  a letter token (`N`/`P`) instead of `+`/`-`. Fixed two real display/positional
  bugs along the way (the GUI table's lookup key, and both viewers' z/legend
  derivation).
- **Coordinate-frame handling** (see "Coordinate frames"): `shim_csv_frame` (default `scan`) rotates the shim
  CSV back to the scan frame at export; `viewer_frame` (default `scan`) draws both viewers in the scan frame
  with a signed field; `utils/viewer_frame.jl` holds the shared helpers; the verifier reads the CSV as
  written (`--shim-frame lab`).
- **Halbach test bits** — `make_halbach_ring_csv.py` (see above) for axis/angle-convention checks.
- Cleanup: removed dead code (`pipeline_config_WORKING.jl`, orphan utils) and stale
  plan docs; this reference and `README.md` are the source of truth.

### Halbach test bits — `make_halbach_ring_csv.py` (axis / angle-convention checks)

Builds a shim CSV (`X, Y, RingNumber, Angle`, plus a `_shim_xyz.csv` with the literal Z) of ideal
Halbach-dipole **bits** to check the pipeline's x/y/z and angle conventions in the viewers, the verifier
and Stage 2 — no optimizer needed. Options: `--OSII_version OSII1|OSII2` (geometry from `gui/presets.json`:
OSII2 = V2.1, r 231 mm, 20.64°/insert, field **+y**; OSII1 = V1.1, r 277 mm, 18°/insert, field **+x**),
`--tray_number` (int, `3,4` or `all`), `--z_center` *or* `--slot_center`, `--z_copies` (odd), `--flip`
(reverse the field), `--out`. Angle rule (ideal Halbach, optimizer convention θ from +x toward +y):
field +x → θ = 2φ, field +y → θ = 2φ − 90° (φ = polar angle of the magnet). `--z_center` copies sit at
`z_center + k·spacing` and each must land on a real tray slot (no slot at z = 0 unless a tray shift is 0);
`--slot_center` takes consecutive *real* slots (slot 0 is skipped). Warns if `config.toml`'s geometry
differs from the chosen preset. To build STLs from such a CSV: put it at
`data/outputs/Optimizer_Output_per_Iteration/<ITER>/<ITER>_shim.csv` under a **new** iteration name and click
**Build STL only** (never "Export + build STL" — it needs an optimizer result and would overwrite it).

### Known gaps vs. the Shimmer / OSII² preprint (arXiv 2608.22351; reviewed, NOT implemented)

A read-through of the "Shimmer" preprint against these docs (not against the code) found these gaps, most
important first:
1. **No measure → install → re-measure loop.** Shimmer iterates 3–6 times and loads the existing shim
   configuration; here a run is one-shot and nothing handles a field map measured *with* shims installed
   (subtract/freeze installed magnets, exclude occupied slots). The ~9.5k-ppm result is not an end state.
2. **Model ≠ reality, and Stage 4 only checks model vs model.** Paper: predicted vs measured total ppm
   matched (21,225 vs 21,295) but point-by-point error was up to 3,522 ppm. Their 1000-ppm tolerances for
   shims: rotation ≤ 2°, position ≤ 3 mm, Br ≤ 3 %. Here one global `magnet_Br_T`, cube side tolerance
   (moment ∝ side³), no sensitivity/Monte-Carlo study, no post-install measured-vs-predicted comparison.
3. **Temperature.** NdFeB ≈ −1200 ppm/K; a 0.9 K drift over an 8 h scan ≈ 600 ppm, and a spiral scan turns
   the drift into a spatial pattern the optimizer would try to cancel. Temperature is read but unused.
4. **Probe noise / metric.** Hall 821 ppm vs NMR 316 ppm on the same sphere; the default scores raw
   max−min of measured |B| (noise inflates ppm; the optimizer partly fits noise); |B| vs the B0 component.
5. **Frames / physical orientation.** OSII angle sign/offset still unresolved; Stage 4 cannot catch the
   physical insert orientation; scan-centre vs magnet-centre / tray z-origin offsets; no pole marker on the
   printed holders (the paper engraves one).
6. **Objective / "hardware limit".** Variance is minimised but the figure of merit is range; the unreachable
   main-array ring pattern (n ≈ 6) may be a structural floor; optimising a larger DSV worsens smaller ones;
   magnet size is not a degree of freedom here.

### Frame handling — what was tested (this session)
- **Independent frame check** (Python dipole model on the raw scan `06102026_UCScanner_NoShimming`, mean
  field (−0.4, −42.2, −3.6) mT → B0 along −y; unshimmed |B| = 19,155 ppm): the optimizer's CSV placed in the
  scan frame **with** the 180° back-rotation (positions + moments) gives **9,601 ppm**; used as if already in
  the scan frame 28,830; moments not rotated 33,861. A z-flip changed it only to 9,844 (the rings are nearly
  z-symmetric, so a **z mirror cannot be detected** this way). Conclusion: the optimizer ↔ scan frame
  conversion is self-consistent; no rotation bug found.
- `utils/viewer_frame.jl: grid_to_scan` checked against a pointwise coordinate map for all four directions
  (base Julia, synthetic arrays).
- `make_halbach_ring_csv.py`: full 12-tray ring gives a centre field of **221 µT along +y (OSII2)** and
  **128 µT along +x (OSII1, r 277 mm)** (own dipole sum, Br 1.26 T / 6 mm); z → InsertPos mapping and the
  invalid-slot error exercised; the `Halbach_Tests` iteration's verifier input shows tray 3 (slot −6) at
  z = −50…−70 mm in the scan frame.
- All edited Julia files parse (`Meta.parseall`).
- **NOT run:** `export_csv.jl` / `insert_search.jl` with the new rotation, Stage 2 on a rotated CSV, the viewers
  with `viewer_frame`/signed display (GLMakie/CUDA box), `python_verifier.jl` in `lab` mode, any GUI change.
- Environment note: on the Windows dev box Julia packages fail to load (Application Control blocks the
  unsigned compiled `.dll` caches in `~/.julia/compiled`); run the GUI/pipeline on the GPU box or WSL2, or have
  IT allowlist the depot.

### Environment
The pipeline needs the NVIDIA/CUDA box for Stage 1 and the viewers. Code edits in
recent sessions were verified statically (no GPU/Julia in the editing sandbox); runs
are launched on the GPU box.

### Extending
- More rings: raise `ring_search_n` (greedy scales linearly); watch
  `RingSearch/ring_search.png` (ppm vs #rings) for where it plateaus.
- Hardware what-ifs: change `shim_radius_mm`, `magnet_Br_T`/`magnet_side_mm` (μ),
  rings, or `Rmax`, then re-run `optim_grad.jl` + `run_grad.jl` for the new
  achievable floor.
- Not built yet: convex lower bound (Convex.jl + SCS), CMA-ES baseline, greedy ring
  search that folds already-placed rings into the base field, temperature correction.
