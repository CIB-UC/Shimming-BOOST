# Session Handoff — state, changes, and open items

> **Doc-accuracy pass note:** this file, `README.md`, `GRAD_OPTIMIZATION.md` and
> `USER_GUIDE.md` were cross-checked against the live `config.toml` /
> `pipeline_config.jl` and corrected for staleness (see "UPDATE (docs cross-checked
> against code)" below) — most notably the magnet-strength config keys, which had
> already moved on from what earlier sections of this file describe.

Context for picking this project up in a fresh chat or by a new developer. The
**reference docs** describe the system as it should be; this file describes **what was
just changed, what is and isn't verified, and what's still open.**

- `README.md` — architecture, stages (incl. Stage 4), config, repo layout (source of truth).
- `GRAD_OPTIMIZATION.md` — the gradient optimizer's code + math (incl. insert search).
- `USER_GUIDE.md` — plain-language guide (no programming needed).
- **`HANDOFF.md`** (this file) — session state + open items.

---

## -10. UPDATE (stage renumbering)

Stages are now **0, 1, 1.5, 2, 3 (viewers), 4 (Python verifier)** — formerly viewers = Stage 4 and verifier = Stage 5.
Labels only: scripts, config keys and GUI actions keep their names. `start_stage` still takes the **run-order
index** (0, 1, 2 = Stage 1.5, 3 = Stage 2, **4 = Stage 3 viewers, 5 = Stage 4 verifier only**) so existing configs and
`start_stage = 5` keep working; its comments and the start-up menu now show the stage names. GUI badges read 3 and
4. Older entries below were renamed in place (a "Stage 4" there that is not a viewer reference is unaffected).

---

## -9. UPDATE (session summary: frames, Halbach test bits, preprint review — what is tested vs not)

**Changed this session** (details in -8 / -7 below and in `README.md` "Coordinate frames"):
- `shim_csv_frame` (default `scan`): `export_csv.jl` + `grad_optim/insert_search.jl` rotate the shim CSV back
  to the scan frame; `python_verifier.jl` passes `--shim-frame lab` accordingly. `viewer_frame` (default `scan`)
  + signed field in both viewers; `utils/viewer_frame.jl` (helpers + `read_field_direction`); both keys in
  `config.toml`, `pipeline_config.jl` (asserted) and `gui/backend.jl` ENUMS (no GUI field yet).
- `make_halbach_ring_csv.py` — Halbach-dipole test bits (options in `README.md`); the generated
  `Halbach_Tests*` iterations were built from it.
- One-off data prep (not code): two scans were rotated (B0 +x → +y), plus point-inverted `_flipXYZ` copies
  (positions only; field vector unchanged), during axis debugging — keep raw scans as the source of truth.
- Reviewed the Shimmer/OSII² preprint against the docs → "Known gaps" list in `README.md` (not implemented).

**Tested:** independent Python frame check (19,155 → 9,601 ppm with the back-rotation; wrong placements
28,830 / 33,861; z-flip insensitive at 9,844 → z mirror undetectable); `grid_to_scan` all four directions;
Halbach generator centre fields (221 µT +y OSII2, 128 µT +x OSII1) and z/slot mapping; `Halbach_Tests`
verifier input (tray 3 at z −50…−70 mm in scan frame); parse check of every edited Julia file.
**NOT run:** `export_csv.jl`/`insert_search.jl`/Stage 2/both viewers/`python_verifier.jl lab` end to end with
the new rotation; GUI. **Re-run Stage 1.5 export** to regenerate any pre-existing CSV in the scan frame.

**Findings worth keeping in mind**
- The optimizer ↔ scan conversion is self-consistent (no rotation bug found). What was missing is the
  *physical tray frame*: Stage 2 takes the CSV's frame as the magnet's. Still open: where tray 12 is
  physically relative to B0 / the scan axes, and the CW/CCW sense of the printed insert (OSII open item 1).
- Auto direction reads the probe's own axes (`x/y/z axis (gauss)`), assuming they equal the robot's X/Y/Z.
  The current scan also shows ~45 G (≈5.5°) on z against ~469 G on the main axis — unusually large; probe
  tilt or real transverse field, unresolved. The latest scan (`06102026_UCScanner_NoShimming`) has 991 averaged shell points at r = 100 mm, B0 along −y (≈ −42 mT).
- `front_tray_shift_mm = back_tray_shift_mm = 0` (current config) puts InsertPos −1 and +1 at the **same
  z = 0** — a collision if both are used.
- `insert_search`'s printed placement plan uses optimizer-frame tray numbers; Stage 2/viewers number the
  trays from the rotated X, Y (t → t+6 for −y, t±3 for ±x).
- Dev-box limitation: Application Control blocks Julia's compiled `.dll` caches there (see README).

---

## -8. UPDATE (shim CSV is written in the SCAN frame: `shim_csv_frame = "scan" | "optimizer"`)

Stage 0 rotates the scan so B0 -> +y and the optimizer works there. New (default **"scan"**): right
when the optimizer result is exported, `export_csv.jl` (and `grad_optim/insert_search.jl`, which writes
its own CSV) rotate positions + moment angles BACK by the scan's B0 direction (saved as
`field_direction` in the field-map jld2): +x: (x,y)->(y,-x), angle-90; -x: (-y,x), +90; -y: (-x,-y), +180.
So Stage 2 (STLs; tray re-derived from the rotated x,y), both viewers, and the Stage-4 verifier
(now `--shim-frame lab`) all work in the scan frame. Optimizer, operators, result jld2 stay in the optimizer
frame. `shim_csv_frame = "optimizer"` restores the old behaviour. Hand-made / OSII-import /
`make_halbach_ring_csv.py` CSVs are used exactly as written, in whatever frame the key declares.
**Existing CSVs are NOT converted**: re-run Stage 1.5 export (no re-optimizing needed) to regenerate.
Caveat: insert_search's printed placement plan still uses optimizer-frame tray numbers; Stage 2's differ
by the rotation (tray t -> t+6 for -y, t+-3 for +-x). Physical tray numbering is now tied to the scan's axes.
Verified: formulas against the independent Python dipole check (9,601 ppm with this rotation); all edited
files parse. NOT run: export/insert_search/Stage 2/viewers/verifier end to end (needs the GPU/Julia box).

---

## -7. UPDATE (Stage-3 viewers: `viewer_frame = "scan" | "optimizer"`)

Stage 0 rotates the scan so B0 -> +y; the shim CSV, the field map and (before this) both viewers
were all in that OPTIMIZER frame, so a -y/±x scan looked like a +y solution. New config key
`viewer_frame` (default **"scan"**): both viewers (`Shimming_magnets_visualizer.jl`,
`field_slice_viewer.jl`) rotate the field arrays + axes (`utils/viewer_frame.jl: grid_to_scan`) and the
CSV magnets (positions + angle shift: +x -90, -x +90, -y +180) back into the scan's frame, using the
`field_direction` saved in the field-map jld2. In the scan frame the field is shown signed (negative for
-x/-y); "optimizer" = old behaviour (|B|, B0 -> +y). Display-only: optimizer, CSV, Stage 2 and
the verifier are unchanged; the dipole field is still computed in the optimizer frame, then rotated.
Verified: `grid_to_scan` against a pointwise map for all 4 directions (base Julia); files parse.
NOT run: the viewers themselves (GLMakie/CUDA box). GUI has no field for `viewer_frame` yet (config.toml only).

---

## -6. UPDATE (Stage 0 spherical-harmonic decomposition + selection: `sh_select`)

**What.** Both SH-based Stage-0 adapters (`Field_data_SH_interpolator.jl`, `Field_data_shell.jl`)
now (a) always print the SH decomposition of the measured field — mean, per-degree energy, the
`sh_top_k` largest |a(n,m)| (n ≥ 1) with % of energy — and write
`<ITER>/SHDecomposition/sh_decomposition.csv`; and (b) let the user choose what the field map is
BUILT from: `sh_select = "all" | "first_n" | "top_k"` with `sh_use_degree`, `sh_top_k`. Chosen answers
in the design conversation: report **plus** filtered field; list = k largest |a(n,m)| overall.

- **Files:** `utils/sh_select.jl` (new; stdlib only: `sh_select_columns`, `sh_refit`, `sh_summary`,
  `sh_write_csv`); both adapters (call it right after the LS fit; everything downstream —
  shell values, viewer grid, saved `c` — uses the selected coefficients); `pipeline_config.jl`
  (§4a `sh_select`/`sh_use_degree`/`sh_top_k`/`sh_report_dir`, defaulted → old configs = `all`);
  `config.toml`; `gui/backend.jl` (ENUM + range validation); `gui/app.html` (fields in the
  Stage-0 adapter panel, `sh_use_degree` only shown for first_n).
- **Semantics to remember:** kept columns are **refitted** by least squares (not truncated). With
  `:all` nothing is refitted and the outputs are bit-identical to before. In shell mode
  `shell_source = "measured"` + a selection ⇒ same measured points, `|A·c|` values (no longer raw
  data); the field the optimizer scores is then the SH-filtered one, so its ppm is NOT comparable to
  an `all` run. The jld2s gain `c_full`, `sel_cols`, `sh_select_name`. Coefficients are in the
  optimizer frame (B0 → +y) — `m` differs from Stage 4's lab-frame pyramid if B0 ≠ +y. The reshape
  adapter (`Field_data_file_adapter.jl`) does no SH fit and ignores these keys (the GUI hides them).
- **Verified (Windows box, no CSV/DataFrames/JLD2 — packages blocked there):**
  - `utils/sh_select.jl` unit tests (top-k picks the truth, refit exact, first_n block, error cases,
    summary/CSV).
  - The REAL adapter code run through a harness with only those three packages stubbed, on the
    Try2 scan: all three modes on the shell adapter (measured), top_k on the SH interpolator (grid
    41³), first_n with `sh_fibonacci`. `"all"` gives c == c_full and By_shell == raw |B| exactly.
  - **Cross-check vs the independent Python fit** (`Shimming_verifier/sh_pyramid.fit_sh`): full-fit
    coefficients agree to 8.5e-13 mT, identical top-5 set, and the Julia `first_n=3` refit equals a
    Python degree-3 fit to 6.5e-13 mT.
  - Reference numbers (Try2, L=8): measured 18,532 ppm; full fit 18,085; first_n=3 → 17,131; top_5 →
    17,008. Top-5 = a(2,+2)=−0.541, a(2,0)=+0.276, a(1,−1)=−0.260, a(1,0)=−0.170, a(4,0)=−0.099 mT
    (94% of n≥1 energy).
- **NOT run:** the adapters via `julia …` proper / the GUI (packages blocked on that machine; edits
  are parse-checked and harness-tested). GUI fields untested in a browser.
- **Open:** `sh_top_k` doubles as list length and selection size — split if you ever want a longer
  printout than the selection. No Stage-0 PNG pyramid (text table + CSV only); Stage 4 still draws
  the pyramids.

---

## -5. UPDATE (paired-ring search: `ring_search_method = "paired"`)

**What.** A third ring-search method that only considers symmetric sets. `ring_search_n`
is the TOTAL ring count (must be even, `n = 2m`); a set is `m` pair magnitudes `a`, each
contributing rings `−a` and `+a`, e.g. `[-10, -5, 5, 10]`. It scores **every** choice of
`m` magnitudes (`C(#pairs, m)`), guarded by `ring_search_max_combos`, and reuses the
existing machinery unchanged: the cached `ring_operator.jld2` (same `SIG`), the 250-point
sub-shell ranking, `set_ok` (min_sep / min_spots_between), and the full-shell multistart
re-solve of the winner.

- **Files:** `grad_optim/ring_combos.jl` (new, pure: `for_each_combination` moved here from
  `ring_search.jl`, plus `paired_magnitudes`, `for_each_paired_set`); `ring_search.jl`
  (`PAIRED`, `PAIR_MAGS`, `run_paired()`, even-n check, n-cap rounded down to even, ranked plot);
  `config.toml` / `pipeline_config.jl` comments; `gui/backend.jl` ENUM; `gui/app.html`
  (dropdown option + a live hint: pairs available, number of combinations, odd-n / range /
  max-combos warnings).
- **Semantics to remember:** only positions are paired — angles stay free. Pairing is by tray
  NUMBER (matches the requested `[-10,-5,5,10]`), so it is an exact axial mirror only when
  `front_tray_shift_mm == back_tray_shift_mm`. `ring_search_n_max` (8) still caps the total.
  Default `ring_search_min_spots_between = 3` rejects a = 1 (its two rings have 1 spot between
  by `|a−b|−1`) and adjacent pair magnitudes; lower it to allow them.
- **Verified:** `ring_combos.jl` unit test (counts C(25,2)=300, C(25,3)=2300; `[-10,-5,5,10]`
  present; every set symmetric/sorted/unique; asymmetric and stepped ranges), the GUI hint
  arithmetic in a browser with a mock config, and Julia parse of every edited file.
- **NOT run:** `ring_search.jl` itself with `"paired"` — it needs the CUDA box (operator
  build/cache). Expect first a normal cached-operator load, then `paired: 2 pair(s) of …`,
  `scoring up to 300 paired sets …`, and a `BEST 4 rings: [-a, -b, b, a]` line. Sanity check
  worth doing: the paired best ppm for n=4 must be ≥ the exhaustive n=4 best (paired ⊂
  exhaustive), and equal if the exhaustive optimum happens to be symmetric.

---

## -4. UPDATE (Stage 4 — independent Python verifier wired into the pipeline)

**What was added.** The user's standalone `Shimming_verifier/` (magpylib; recomputes
the shim field, ppm, 2D/3D views, SH pyramids) is now Stage 4. It is deliberately
still independent: it takes only *which field, which shim layout, what to run*.

- **`Shimming_verifier/run_verifier.py`** (new) — non-interactive CLI over the existing
  `main.py` (loads it, overwrites its CONFIGURATION variables from arguments, calls
  `main.main()`; `main.py`/other verifier modules are untouched). Misalignment-study
  knobs default to 0 here rather than to whatever `main.py` currently holds. Also
  `--shim-frame optimizer --b0-direction auto|±x|±y` (see Frame below). `requirements.txt` added.
- **`python_verifier.jl`** (new) — Julia-stdlib-only adapter: resolves the measurement
  and shim paths, converts the seam CSV's `RingNumber` (real InsertPos) → axial z via
  `ringpos_from_tray_mm` (or accepts a CSV that already has `Z (mm)`), pre-flights Python +
  packages, launches `run_verifier.py`, checks the requested outputs exist. Magnet
  Br/side and coord unit are read from `magnet_Br_T` / `magnet_side_mm` /
  `sh_measured_unit_mm` — not duplicated.
- **Config** (`config.toml`, all optional/defaulted in `pipeline_config.jl` §4h):
  `run_python_verifier`, `verifier_python`, `verifier_measurement`, `verifier_use_shim`,
  `verifier_shim_csv`, `verifier_make_2d/_3d/_sh`, `verifier_sh_max_degree`,
  `verifier_show_figures`, `verifier_output_subdir`. `start_stage` now accepts **5**
  (verifier only; needs neither field map nor optimizer result).
- **`run_pipeline.jl`** — Stage 4 after the Stage-3 viewers when `run_python_verifier`
  or `start_stage = 5`; start prompt has option 5; `start = 5` skips the Julia summary/viewers.
- **GUI** — new "5 Python verifier" block (measurement picker, shim options, what-to-run,
  Run + Open-output buttons); backend stage `verify_python`, `start_stage` validated −1..5,
  `/api/open_folder` accepts `verifier`. `app.html` needs a hard refresh.
- **Frame (the one non-obvious thing).** Stage 0 rotates the scan so B0 → +y
  (`Field_data_shell.jl` `lab_to_optimizer_xy`), so every shim CSV is in that frame,
  while the verifier reads the raw lab-frame scan. `run_verifier.py` rotates the shim
  positions + angles back (exact inverse; angle shifts by the same rotation). For the
  current scanner B0 is already +y ⇒ identity.

**Verified (this session, on a Windows box with Python 3.14 + magpylib 5.2.3, no GPU):**
- CLI run on the `…48inserts_14092026` iteration: unshimmed 18,537.8 ppm → shimmed
  **9,562.1 ppm** vs the optimizer's predicted **9,483** (same order/direction; the ~80 ppm
  gap is model difference — cuboid vs dipole, plus the verifier keeping a (0,0,0) marker point).
- `python_verifier.jl` end-to-end (standalone and via `run_pipeline.jl` with
  `start_stage = 5`): the derived CSV equals `export_csv.jl`'s `_shim_xyz.csv` exactly (max
  diff 0.0) and gives the same ppm.
- Frame rotation: synthetic copies of the scan rotated to B0 = +x / −x / −y, run with the
  *unrotated* shim CSV, auto-detect the direction and all reproduce 9,562.1 ppm exactly.
- **Not run:** the GUI block (edited, not opened in a browser); `verifier_show_figures =
  true`; baseline-only (`verifier_use_shim = false`) via Julia (CLI without `--shim` is
  the untouched original path); a shim CSV from an OSII import. That machine's Julia also
  couldn't load CSV/DataFrames (Application Control policy) — irrelevant to Stage 4, which
  needs only TOML.

**Open items:**
- `sh_measured_unit_mm` is used as the verifier's `--coord-unit-mm`; if scans in another
  unit are ever used the two meanings must stay the same.
- The verifier's `load_measurement` keeps the (0,0,0) marker row of a "controlled" scan
  (1743 points vs the Julia readers' 1742) — the "simple" format drops it. Harmless to ppm
  here (min radius shows 0.00) but worth aligning if exact parity with the optimizer matters.
- A shim CSV from a *different* tray geometry than the current config would get wrong z
  (RingNumber → z uses the current keys); prefer a CSV with a `Z (mm)` column in that case.
- `Shimming_verifier/output/` and `Field_measurements/`/`Shimming_magnets/` remain the
  interactive (`main.py`) workflow's folders; Stage 4 writes elsewhere.

---

## -3. UPDATE (docs cross-checked against code; magnet config, GIF export, GUI fields)

A documentation-accuracy pass (no pipeline logic changed) found and fixed several
places where `README.md` / `HANDOFF.md` / `GRAD_OPTIMIZATION.md` / `USER_GUIDE.md`
had drifted behind `config.toml` / `pipeline_config.jl`. Summary of what the code
actually does now, for the next session:

- **Magnet strength config is `magnet_Br_T` + `magnet_side_mm`, not `magnet_B1cm_mT`.**
  Sections -1/0/1/A/2/3 below (this file's older history) describe the
  `magnet_B1cm_mT` key as the live config input — that has since been replaced.
  Current source of truth (`config.toml` / `pipeline_config.jl`):
  ```
  magnet_Br_T = 1.26        # remanence (T), datasheet/measured
  magnet_side_mm = 6        # cube side length (mm)
  # mu [A*m^2] = Br[T] * side_m^3 / mu0,  mu0 = 4*pi*1e-7,  side_m = magnet_side_mm/1000
  ```
  `magnet_B1cm_mT` / `magnet_B1cm_T` / `B1CM_T` still exist in `pipeline_config.jl`
  as **derived, back-compat** constants computed *from* `magnet_moment_Am2` (so
  `utils/imanes.jl`'s `B1cm_T=` calling convention keeps working) — they are no
  longer config inputs and setting `magnet_B1cm_mT` directly in `config.toml` is
  ignored for the physics. The GUI's Stage 0 "Shim magnets" panel and
  `backend.jl`'s `POSITIVE_KEYS` were already updated to `magnet_Br_T` /
  `magnet_side_mm` before this doc pass; only the docs were behind.
- **GIF export shipped for both Stage-3 viewers.** `Shimming_magnets_visualizer.jl`
  (rotating-camera GIF) and `field_slice_viewer.jl` (slice-sweep GIF) each got a
  "Save GIF" button plus a "Measured vs Shimmed" pair-comparison button. New config
  keys `viewer_gif_frames` (120), `viewer_gif_fps` (20), `viewer_gif_subdir`
  ("Viewers", under `optimizer_iter_dir`). Both viewers call `Makie.record(...)`
  **explicitly qualified** — never the bare `record` — because CUDA and GLMakie
  both export a conflicting `record` binding once both are loaded; using the bare
  name would be ambiguous/wrong. This was **not yet in any doc** before this pass.
- **GUI Stage 2 now exposes `label_scale` and `letter_thickness` as editable
  fields** (`STAGE2_KEYS`/`STAGE2_SHOW` in `app.html`); `label_scale` (default 1.3)
  scales every engraved label digit, `letter_thickness` its engraving depth. Not
  previously mentioned in `USER_GUIDE.md`'s settings table.
- **GUI header/theme refreshed**: title bar now reads "BOOST — B0 Optimization
  Shimming Technique" / "Low Field MRI Project UC" on a light-green palette
  (`app.html`'s `<style>`, "BOOST palette v2"), replacing an earlier dark theme.
  Cosmetic only; no functional change.
- **`insert_search.jl`'s `RANK_FULL_SHELL` toggle currently defaults to `true`**
  (rank every candidate slot on the full shell, not the ~250-point sub-shell) —
  this is slower per round (~7× at the reference scan's Ns=1742) and was left set
  this way as an accuracy check against the sub-shell ranking, not necessarily the
  intended steady-state default. **Open item:** confirm whether it should be
  flipped back to `false` for routine runs (see open items list).
- Everything else in this file (InsertPos/RingNumber migration, the `fld` name
  collision fix, independent front/back tray shifts, insert search, sparse-layout
  plumbing) was checked and remains accurate as described below — only the magnet
  config key name, the GIF feature, and the two new GUI fields needed correcting.

---

## -2. UPDATE (RingNumber is now the real, physical InsertPos everywhere — 6-file migration)

**What changed.** `RingNumber` (the shim CSV's 3rd column) used to be, in most
writers, a **0-based sequential index** into whatever ring set the current run
happened to use — e.g. rings `[-14, -6, 6, 14]` got written as `0, 1, 2, 3`. That
index meant nothing physical on its own; you had to already know the run's ring set
to know which tray slot `Ring 2` actually was. The user hand-edited
`utils/helping_functions_for_JIG.jl`'s `make_label` to take a real, physical, signed
**`InsertPos`** (the actual tray slot, e.g. `-25..25`) and print it on the physical
part as `IX R±NN TX` — using letter tokens **N/P** instead of literal `+`/`-` glyphs
(engraving a minus sign reliably is fragile; `N07`/`P12` is not). Everything upstream
of that function had to be brought in line so the value flowing through the whole
pipeline as "RingNumber" is that same real InsertPos, not a sequential index — one
file at a time, confirming after each:

1. **`export_csv.jl`** — recovers each row's real InsertPos by mapping its distinct-z
   block (rows are grouped by axial position, ring-major, per `positions_from_rings_mm`'s
   build order) onto `positions_in_tray_new_wished` by **block index**, not a 0-based
   counter. Errors loudly if the block count doesn't match the wished-ring count
   (stale/mismatched geometry) instead of silently mislabeling.
2. **`grad_optim/insert_search.jl`** — writes `p.ring` (the real InsertPos) straight
   into the CSV's RingNumber column. The old `ring_index = Dict(r => i-1 for ...)`
   sequential map is gone from the CSV path entirely; the *only* remaining 0-based
   index (`ring_slot`, for addressing the shared ring-operator's column blocks) is
   now explicitly scoped local to the placement block and commented "INTERNAL ONLY,
   never exported." Added `insertpos_label(pos)` → `"N07"`/`"P12"` for the console
   placement-plan printout.
3. **`osii_to_shim.jl`** — has no pre-declared tray array (OSII data arrives with
   only measured axial positions), so it recovers each magnet's real InsertPos by
   **inverting** `pos_trays.jl`'s `ringpos_from_tray_mm` formula directly from its
   measured z, rounding to the nearest legal tray slot and `@warn`-ing if the
   residual exceeds half a slot spacing (a sign the OSII layout's axial geometry
   doesn't actually match this bore's configured tray spacing/shifts).
4. **`CSV_to_STL.jl`** — no logic change needed (it already received a real
   InsertPos post-fix-1-3 via `ring_num`); added the shared `insertpos_label` helper
   and switched folder/file naming from a bare signed int (`Ring-7`, illegal in a
   Windows path) to the P/N token (`Ring_N07`, `RingN07_Tray03.stl`), matching what
   `make_label` engraves on the part.
5. **`gui/backend.jl` + `gui/app.html`** — `backend.jl` needed no change (it already
   parsed and returned `RingNumber` verbatim). `app.html`'s ring-placement table had
   a **real bug**: `byRing` is keyed by the real InsertPos (`r.ring`), but the table
   looked it up by `byRing[i]` — the array index within
   `positions_in_tray_new_wished`, not the real ring value `t` at that index — and
   displayed the sequential `Ring_${i}` instead of the InsertPos. Post-fix-1-4 this
   would have shown "not in shim CSV" for virtually every real run. Fixed to
   `byRing[t]` and `Ring_${insertPosLabel(t)}`.
6. **`Shimming_magnets_visualizer.jl` + `field_slice_viewer.jl`** — both computed
   each ring's axial z (and, in the 3D visualizer, its legend label) by **zipping**
   `sort(unique(ring))` **positionally** against `wished_trays`/
   `positions_in_tray_new_wished`, assuming the k-th smallest RingNumber in the CSV
   is the k-th entry of that array. That assumption silently breaks the moment the
   array's order doesn't match the CSV's sorted order, or the CSV's ring set differs
   from the current config's wished set (an OSII import, a sparse insert-search
   result, or simply re-running with a different config than the CSV was built
   under) — a **wrong-z bug that would not error, just draw magnets in the wrong
   place**. Fixed by calling `ringpos_from_tray_mm` directly on each ring's own
   value (`ringpos_from_tray_mm(unique_rings; ...)`), since RingNumber *is* now the
   real InsertPos and the formula is a pure per-element mapping with no dependence
   on array order. The 3D visualizer's ring legend also changed from the old
   ambiguous "Ring N - Tray T" (two different numbers, pre-migration) to a single
   InsertPos P/N label, matching the physical part.

**Not touched, by explicit user instruction:** `utils/helping_functions_for_JIG.jl`
— the user edited this one by hand themselves; `make_label` already takes
`InsertPos::Int` and does its own N/P + 2-digit formatting internally.

**Verification status:** every file's fix was checked statically (grep for stale
references, Python-side numeric round-trip check of `osii_to_shim.jl`'s formula
inversion across asymmetric front/back shifts, and a naive paren-balance check on
the two visualizer files after editing) — **none of this has been run in Julia on
the GPU box yet.** Add to the verification checklist below: re-run `export_csv`,
`insert_search`, `osii_to_shim`, `CSV_to_STL`, the GUI, and both viewers on a real
(ideally sparse / non-contiguous / non-sorted) ring set and confirm every printed
`Ring_N##`/GUI table entry/3D legend label agrees with the actual physical tray slot.

**Terminology going forward:** "RingNumber" in a shim CSV, "ring_num"/"p.ring" in
code, and "InsertPos" in `make_label` and the labeling scheme are now **the same
number** — the real, physical, signed tray slot. There is no longer a sequential
"ring index" anywhere except the one explicitly-scoped internal `ring_slot` dict in
`insert_search.jl` used solely to address the shared operator's column blocks.

---

## -1. UPDATE (first real GPU run of insert_search — bug found & fixed)

The user ran `insert_search` (shell mode) on the GPU box for the first time. It crashed
before placing anything:

```
ERROR: LoadError: MethodError: objects of type CuArray{Float32, 3, CUDACore.DeviceMemory}
are not callable
    at grad_optim/insert_search.jl:221
```

**Root cause — a name collision, not a logic bug.** `setup_shell.jl` (and `setup.jl`) define
a top-level global `fld` holding the base-field `CuArray` (`fld = _col(By_shell)` /
`fld = Float32.(CuArray(fieldmap))`). `insert_search.jl` and `ring_search.jl` both
`include` one of those setup files and then *separately* call Julia's built-in `fld(a, b)`
(floor division) to compute the magnet budget — `fld(magnets_available, MPS)` in
`insert_search.jl`, `fld(ring_search_magnet_budget, MPR)` in `ring_search.jl`. Once the
setup file's global assignment runs, `fld` resolves to the `CuArray` instead of
`Base.fld`, so the budget-division call tries to *call* the field array → the exact error
above. `ring_search.jl` has the identical latent bug (line 185); it just hadn't been
triggered because that branch only runs when `ring_search_magnet_budget > 0`.

**Fix applied (statically, not yet re-run on the GPU box):** renamed the global
`fld` → `fld_field` everywhere it is the base-field array, restoring `fld` to Julia's
built-in floor-division function. Files touched:
- `setup.jl`, `setup_shell.jl` — the definition site.
- `grad_optim/optim_grad.jl`, `grad_optim/insert_search.jl`, `grad_optim/ring_search.jl`,
  `utils/wrap.jl` — every consumer of the global.
The two genuine `fld(...)` budget-division calls (`insert_search.jl:221`,
`ring_search.jl:185`) were left untouched — they're correct now that nothing shadows
`fld`. Nothing else in the algorithm changed.

**Next step:** re-run `insert_search` (and, since it shares the bug, exercise
`ring_search` with `ring_search_magnet_budget > 0` at least once) on the GPU box to
confirm the crash is gone and a placement actually completes. This is the very first
live Julia/CUDA run of `insert_search` — everything about the search's *own* logic
(spacing rule, sparsity mask, best-prefix selection, etc.) is still only Python-verified
per Section 1.C below, so watch the placement log and the trace CSV/plot closely on
this first successful run, not just the absence of a crash.

---

## 0. The single most important caveat

**Everything below was edited and reasoned/prototyped statically. Nothing in this
session was run in Julia, on the GPU, or in GLMakie** — the editing environment has no
CUDA/Julia/display. All runs happen on the user's **NVIDIA/CUDA box**. Where a claim is
marked "verified", it means the logic was transcribed to Python and executed against
synthetic or reconstructed data — *not* that the Julia ran. Treat the "Verify" steps as
required, not optional.

---

## 1. What changed this session (by feature)

### A. Shim-magnet strength is configurable (was hardcoded in FIVE places)
`μ = 0.06 A·m²` was hardcoded in `setup.jl`, `setup_shell.jl` and `ring_search.jl`; the
equivalent `B1cm = 0.012 T` was hardcoded again in `Shimming_magnets_visualizer.jl` and
`field_slice_viewer.jl`. `config.toml` already had a `B1CM_T = 0.012` key — read by
`pipeline_config.jl` and **used by nothing**.

- New key **`magnet_B1cm_mT = 12.0`** — the measurable quantity (on-axis |B| 1 cm from
  the magnet face). `pipeline_config.jl` derives `magnet_moment_Am2` via
  `μ = B1cm·r³/(2·μ0/4π)` (the same relation as `utils/imanes.jl`). All five sites read it.
- `B1CM_T` survives as a derived alias; both new keys use `_getdef`, so old configs load.
- Also exposed **`initial_angle_deg = 150.0`** (was hardcoded twice).
- GUI: new **Stage 0 → "Shim magnets"** panel with a live read-out of the derived μ.
- **Verified (Python):** 12 mT reproduces the legacy 0.06 A·m² exactly ⇒ this change is
  a **no-op on results** until the number is edited.
- Safety: `grad_core.jl` warns if `operator_G.jld2`'s cached μ differs from the config
  (the operator bakes μ in); `backend.jl` rejects μ ≤ 0.

### B. Independent front/back tray shifts (replaces `half_shift_mm`)
The bore is not symmetric, so each end is now anchored on its own:

```
z(+n) = +(back_tray_shift_mm  + (n−1)·tray_slot_spacing_mm)   → +Z "Back - Wall"
z(−n) = −(front_tray_shift_mm + (n−1)·tray_slot_spacing_mm)   → −Z "Front - Gaussmeter"
```

- Keys: `tray_slot_spacing_mm`, `front_tray_shift_mm`, `back_tray_shift_mm` (10/10/10).
  `half_shift_mm` **removed everywhere**, including both shipped geometry presets.
- **Tray 0 now errors** (`pos_trays.jl`), is rejected by `backend.jl`, and is flagged in
  the GUI table. It previously returned `+half_shift` silently.
- **BUG FIXED:** both Stage-3 viewers hardcoded `half_shift_mm = 0.0` while the optimizer
  used the config value — any non-zero shift drew magnets at the wrong z. They now call
  `ringpos_from_tray_mm` with the same config keys.
- `ring_search.jl`'s cache signature picked up all three keys.
- **Verified (Python):** 10/10/10 reproduces the old `half_shift_mm = 0` geometry for
  trays ±1…±25, and the general equivalence (`h` at spacing `s` ≡ `front = back = s−h`)
  holds across s ∈ {10, 8, 12.5}, h ∈ {0, 5, 2.5, −3}, trays ±30. **No-op on results.**

### C. Insert search — sequential per-insert placement (NEW)
`grad_optim/insert_search.jl`, GUI task **"Insert search (per-slot)"**. Nothing was
deleted: `run_grad.jl` is byte-identical, `ring_search.jl` only carries the tray-key rename.

- Unit = one **insert** (one tray at one ring, `mags_per_segment` magnets). Count comes
  from `magnets_available ÷ mags_per_segment`; inserts are never partially filled.
- **Frozen sequential placement (matching pursuit):** each round scores every free legal
  slot against the *current* base field, re-solves the winner on the full shell, then
  freezes its angles and folds its field into the base. Every solve is 7 variables
  regardless of how full the bore is. Deliberately *not* `ring_search`'s greedy.
- Shares `ring_operator.jld2` with the ring search (identical `SIG`) — an insert is an
  `MPS`-column slice of a ring block, so there is no second GPU build.
- Spacing rule is **same-tray only**; `spots_between` discounts the non-existent tray 0
  (`ring_search` still uses the simpler `|a−b|−1` — a deliberate, documented difference).
- Places the whole budget, logs ppm per step, publishes the best **prefix** (fixed-magnitude
  magnets mean more inserts is not always better).
- **Verified (Python, full algorithm port on a synthetic dipole problem):** monotone
  improvement, zero same-tray violations, no duplicate slots, exact magnet accounting.
- **Verified (Python):** the segment→tray map is **not** identity (segment 1 → tray 3),
  so assuming it would have mis-assigned every insert; all 7 magnets of a segment agree
  on their tray and all 12 trays are covered exactly once.

### D. Sparse layouts now travel through the whole pipeline
An insert-search layout is sparse. Audited every downstream consumer:

- **`export_csv.jl` ignored `final_state`** — a latent bug that also mis-exported any SA
  solution with magnets switched off. It now filters by the mask, so `insert_search`
  writes a standard result jld2 (sparsity carried in `final_state`: 1 = placed, 0 = empty)
  and "Export + build STL" **regenerates the same sparse CSV** instead of a dense one.
- `utils/eval_metrics.jl` gained `insert_result_path` in its defaults (so "Compare
  metrics" shows the insert run, with `#on` = magnets actually placed) and an
  operator/result size-mismatch guard with a clear rebuild instruction.
- `osii_to_shim.jl` writes a `SPARSE_LAYOUT.marker`; `export_csv.jl` refuses to clobber a
  marked CSV. (The OSII path has no result jld2 at all, so it needs the marker — the
  insert path does not, because its result round-trips.) The marker records the file it
  describes, so it self-clears once anything else rewrites the CSV.
- GUI: new `/api/placement` endpoint + a **"Trays filled" / "Magnets"** column in the
  ring-placement table, read from the shim CSV on disk, so a sparse layout is no longer
  displayed as a full 84-magnet ring.
- **Verified (Python):** a realistic sparse CSV (6 inserts / 4 rings, two rings partial)
  passes both viewers' `@assert`, `CSV_to_STL`'s radius + angle-step checks, and builds
  exactly 6 inserts each with 7 pockets; the marker guard behaves correctly in all four
  states (fresh / just-written / rewritten / deleted).

### E. Docs
All four .md files brought in line with the code (this file rewritten).

---

## 2. Verification checklist (run these on the GPU box)

- [ ] **No-op check (do this first):** with the current `magnet_Br_T`/`magnet_side_mm`,
      `10/10/10` tray keys and the old ring set, re-run Stage 0 + "Grad build+solve" and
      confirm the ppm matches your last known-good run. Both refactors are supposed to
      change nothing.
- [ ] **Magnet strength:** change `magnet_Br_T` or `magnet_side_mm`, re-run **only**
      `run_grad`, and confirm `grad_core` prints the stale-operator warning. Then
      rebuild and confirm ppm moves.
- [ ] **Tray shifts:** set `front_tray_shift_mm ≠ back_tray_shift_mm`, check the GUI ring
      table, then open the 3-D viewer and confirm the magnets sit at those z values (this
      is the path that was silently wrong before).
- [ ] **Insert search:** start with `insert_search_max_inserts = 5` to time one round
      (≈600 slots scored per round at the default range) before running the full budget.
      Confirm the ppm trace, the best-prefix report, and the placement plan.
- [ ] **Insert search → Stage 2:** "Build STL only", then confirm the printed
      `Ring_<InsertPos>/Tray_T` files match the placement plan and the GUI "Trays filled" column.
- [ ] **Round-trip:** after an insert search, click "Export + build STL" and confirm the
      shim CSV is unchanged (i.e. `final_state` filtering works) — this is the path that
      used to silently produce a dense layout.
- [ ] **Compare metrics** after rebuilding the operator for the new ring set; check `#on`.
- [ ] **OSII import:** run it, then deliberately run `julia export_csv.jl` and confirm it
      REFUSES with the marker message.
- [ ] **Ring search:** unchanged behaviour, but its operator cache is now shared with the
      insert search — confirm it still loads/rebuilds correctly.
- [ ] **InsertPos migration (all 6 files, see Section -2):** run a ring set that is
      sparse and/or NOT already ascending-sorted (e.g. `[6, -14, 14, -6]` in
      `positions_in_tray_new_wished`, or an insert-search result that only fills some
      rings), then confirm every one of these agrees on the same physical tray slot for
      each magnet: the shim CSV's `RingNumber` column, the console placement-plan
      printout (`insertpos_label`), the GUI ring-placement table, the printed
      `Ring_N##/Tray##.stl` folder/file names, and both viewers' legend/labels and drawn
      z-positions. This is the scenario the old positional-zip bugs would have gotten
      wrong silently (no crash, just the wrong tray).

- [ ] **Frames (new):** re-run Stage 1.5 export on the −y iteration; confirm the log says "rotated back to the
      SCAN frame", `_shim.csv` rows are the optimizer rows turned 180° (angle +180°), Stage 2 "Build STL only"
      runs, and the verifier (`lab`) still reports ≈ 9.6k ppm on the 06102026 scan.
- [ ] **Viewers:** with `viewer_frame = "scan"` the field cloud and magnets agree in place/orientation (use the
      Halbach test bits); flip to `"optimizer"` and confirm the old picture.
- [ ] **Physical tray frame:** decide where tray 12 sits relative to B0 / the scan axes before printing; if tied to
      B0, set `shim_csv_frame = "optimizer"`. Do a Halbach test print to settle the insert's CW/CCW sense.

---

## 3. Current config state (`config.toml` at handoff)

**Superseded — this snapshot is from an earlier session.** The live `config.toml`
now reads (treat this list, not the paragraph above, as current):

- `iteration = "InsertSearch_Test_UcNoShim_48inserts_10092026"`, `iteration_number = 1`
- `measured_fieldmap_name = "Try2_UCScanner_NoShim_Sphere200mmDiam_20DegC_08082026.csv"`
- `eval_domain = "shell"`, `shell_source = "measured"`, `shell_n_points = 2000`
- `shim_radius_mm = 231`, `num_trays = 12`, `mags_per_segment = 7`
- `Rmax = 100`, `angle_per_segment_deg = 20.64`
- `tray_slot_spacing_mm = 10`, `front_tray_shift_mm = 10`, `back_tray_shift_mm = 10`
- `magnet_Br_T = 1.26`, `magnet_side_mm = 6` (⇒ μ = Br·side_m³/μ0 — see
  `pipeline_config.jl` §3b), `initial_angle_deg = 150`
- `positions_in_tray_new_wished = [-25, -23, -22, -12, -11, ..., 24, 25]` (33 slots)
- `label_scale = 1.3`, `letter_thickness = 1`
- `ring_search_method = "greedy"`, `ring_search_n = 4`, `ring_search_range = [-25, 25]`
- `magnets_available = 336` (⇒ 48 full inserts), `insert_search_range = [-20, 20]`,
  `insert_search_min_spots_between = 1`, `insert_search_max_inserts = 0`
- `run_3d_viewer = true`, `run_slice_viewer = false`
- `viewer_gif_frames = 120`, `viewer_gif_fps = 20`, `viewer_gif_subdir = "Viewers"`
- OSII: `osii_input_name = "Tryout_angles_90deg.csv"`, `osii_invert_angle = true`,
  `osii_angle_offset_deg = 90`

Always trust the actual `config.toml` in the repo over either list above — this
section is a point-in-time snapshot, not a live mirror, and it will drift again the
next time someone edits `config.toml` for a real run.

---

## 4. Open items / decisions not yet made

1. **OSII angle sign/offset** — still needs a printed tray to settle CW vs CCW. Knobs:
   `osii_invert_angle`, `osii_angle_offset_deg`.
2. **OSII ring → physical tray slot** — deferred by the user; `RingNumber` still only
   names the output folder.
3. **±200 mm viewer labels vs small bores** — offered, not done. Less pressing now that
   `Rmax = 100`, but still fixed rather than adaptive.
4. **Profile clip scope** — one-axis vs box clipping; awaiting preference.
5. **3-D viewer region tools** — the zoom/region-ppm work is slice-viewer only.
6. **`spots_between` divergence** — `insert_search` counts real slots (tray 0 excluded),
   `ring_search` uses `|a−b|−1`. Deliberate; unify if it ever confuses.
7. **Insert search has no combinatorial guard** — ~600 slots × budget rounds. Fine at the
   current scale, but there is no `max_combos` equivalent to stop a huge range × budget.
8. **Insert search is grid-mode capable but untested there** — it follows `eval_domain`
   like `ring_search`; only shell mode was reasoned through.
9. ~~Dead config keys `DISC_5` / `BATCH_M`~~ — removed from `config.toml` and `pipeline_config.jl`.
10. **`magnets_available` is global, not per-grade** — μ is a per-magnet vector internally,
    so mixed magnet grades could be supported later without touching the math.
11. **Applying a shipped geometry preset overwrites the tray shifts** with 10/10/10, since
    they are now preset keys. Save your measured values as your own preset.
12. **`insert_search.jl`'s `RANK_FULL_SHELL` is currently `true`** (full-shell ranking
    every round, not the faster ~250-point sub-shell). Decide whether this should be the
    steady-state default or flipped back to `false` once its ppm has been cross-checked
    against a sub-shell run on a real dataset — right now it's left on for that
    accuracy check, at a real speed cost per insert-search round.
13. **`viewer_gif_frames`/`viewer_gif_fps`/`viewer_gif_subdir` have no GUI fields yet**
    — set via `config.toml` only. Add to `app.html`'s Stage 3 block if the GIF export
    feature should be reachable without hand-editing the config.

14. **Physical tray frame is unspecified** — Stage 2 numbers/rotates trays from the CSV's frame
    (`assign_tray`). A single "tray frame" setting (scan vs B0-relative) was proposed, not built.
15. **z orientation / handedness** of the scan vs the magnet is unchecked (rings are nearly z-symmetric, so the
    frame check can't see a z mirror).
16. **`viewer_frame` / `shim_csv_frame` have no GUI fields** (config.toml only).
17. **Preprint gaps** (iteration loop, tolerance/sensitivity study, temperature correction, probe noise, pole
    marker, range-based objective) — see "Known gaps vs. the Shimmer preprint" in `README.md`.
18. **Zero tray shifts** (`front/back_tray_shift_mm = 0`) make InsertPos ±1 coincide at z = 0.

---

## 5. Environment & gotchas

- Runs need the **NVIDIA/CUDA + display** box (Stage 1 optimizer, viewers, Stage-0 GPU
  dipole in viewers). Gmsh (Stage 2) and CUDA/GLMakie are kept in separate subprocesses
  by `run_pipeline.jl`.
- The GUI is `julia gui/server.jl` → `http://localhost:8010`. After editing `app.html`,
  **hard-refresh the browser** (the in-app "Reload" only re-reads config).
- Field-only viewers auto-detect via `has_shim = isfile(shim_csv_path)`; no flag needed.
- **Changing `magnet_Br_T`/`magnet_side_mm` or the tray geometry invalidates cached operators.**
  `ring_operator.jld2` self-invalidates via `SIG`; `operator_G.jld2` does not — it only
  warns, so rebuild it (Stage 1 "Grad build+solve") after such a change.
- The ring operator is **shared** between `ring_search.jl` and `insert_search.jl` through
  an identical `SIG` string. If you edit one file's `SIG`, edit the other's.
