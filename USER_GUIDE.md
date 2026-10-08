# A Plain-Language Guide to the Shimming Tool

**Who this is for:** anyone using this tool — you do **not** need to know programming.
This guide explains, in everyday language, what the tool does and the newest
things it can do. Every technical word is explained in the **Glossary** at the end.

For the deep technical details, see `README.md`. This
document is the friendly version.

---

## 1. What the tool does, in 30 seconds

An MRI magnet needs a very even magnetic field in the space where the sample or
patient goes. Straight out of the box, the field is a bit lumpy. We smooth it out by
placing lots of small magnets ("shims") in printed trays around the tube, and rotating
each one to just the right angle so their combined nudges cancel the lumps.

The tool does this in steps:

- **Stage 0 — Field map:** read a measurement of the current (lumpy) field.
- **Stage 1 — Optimize:** calculate the best angle for every shim magnet.
- **Stage 1.5 — Export:** write those results into a simple table (a list of magnets, positions, and angles).
- **Stage 2 — Inserts / STL:** turn that table into 3D files you can print.
- **Stage 3 — Verify:** look at the result in 3D.
- **Stage 4 — Python check:** an independent second opinion — a separate program
  recomputes the shimmed field and reports the score, pictures and spherical harmonics.

You normally drive all of this from a **web page** on your own computer: you start it
(a colleague can show you the one command), and it opens in your browser at
`http://localhost:8010`. Everything below is done by clicking in that page.

At the very top of the page there are two boxes: **Iteration name** and **Label #**.
Whatever you type there names your run — it becomes the output folder's name and the
number engraved on the printed trays. Set these first, every time.

---

## 2. The new features (overview)

1. **Insert search** — tell it how many magnets you own; it decides, one insert at a
   time, where each should go and how each magnet in it should be turned.
2. **OSII import** — you already have a magnet layout from the OSII program? Skip our
   calculation entirely and go straight to printing.
3. **Score on real measurements** — let the calculation judge the field using your
   actual measured points instead of a smoothed computer model. More honest.
4. **Faster ring search** — the "find the best magnet positions for me" tool now runs
   in minutes (or seconds) instead of the better part of an hour.
5. **Look at the field before shimming** — open the viewers on the raw measured field,
   with no magnets needed, to inspect how uneven it is up front.
6. **Magnet strength is now a setting** — type in your magnet's remanence and size
   instead of it being fixed in the code.
7. **Save a GIF of the result** — both viewers can now export an animated GIF of what
   you're looking at, so you can share it without asking someone to run the tool.

8. **Independent Python check (Stage 4)** — a separate program double-checks the result
   with its own physics, so you don't have to trust the optimizer's own arithmetic.

9. **Spherical-harmonic breakdown of the measured field (Stage 0)** — see which "field
   shapes" make up the unevenness, and optionally rebuild the field from only the first few
   degrees or only the biggest few shapes.

10. **Frames made explicit** — the magnet list is now turned back into your scanner's own directions,
    and both viewers draw (and sign) the field the same way. Plus **Halbach test bits** to check your axes.

Each is explained in full below.

---

## 3. Feature 1 — Insert search (place one insert at a time)

### What it's for

The ring search fills a whole **ring** at once — all 12 trays, 84 magnets. That is a
lot of magnets, and you may not have that many. Insert search works in the unit you
actually buy and print: **one insert** — one tray at one ring position, holding 7
magnets.

You tell it how many magnets you have. It works out how many full inserts that is
(magnets ÷ 7, rounded down — an insert is never planned half-empty), and then places
them one at a time.

### How it decides

1. It tries every empty, legal slot — every tray at every candidate ring — and for
   each one works out the best rotations for its 7 magnets.
2. It keeps the single best slot and fixes it in place.
3. It then treats those 7 magnets as **part of the magnet**, not as something still
   being decided, and goes looking for the next insert against that new situation.

So each insert is chosen knowing exactly what the previous ones did. Earlier inserts
are never re-turned, which is what makes this different from the ring search.

### The spacing rule

Two inserts **in the same tray** must have at least a few empty ring slots between
them (setting: `insert_search_min_spots_between`), because they'd otherwise collide.
Inserts in *different* trays are separate printed pieces, so they can sit at the same
ring position with no problem.

### Something important about using all your magnets

A shim magnet can be **rotated, but not switched off**. So adding one more insert is
not automatically an improvement — sometimes the extra magnets do more harm than good.

The tool therefore places your whole budget, records the score after every single
insert, and then tells you which point was actually best. If inserts 10 to 12 made
things worse, it says so, and the layout it hands you contains only the first 9. You
use fewer magnets *and* get a better field.

### How to use it — step by step

1. Open the web page and set **Iteration name** and **Label #** at the top.
2. Run **Stage 0 "Prepare field map"** as usual.
3. In the **"1 — Optimize"** box, pick the **"Insert search (per-slot)"** task.
4. Type your magnet count into **`magnets_available`**. The line underneath tells you
   how many full inserts that is, and how many magnets would be left over.
5. Click **"Run optimization."** Watch the log: one line per insert, showing where it
   went and the score afterwards.
6. When it finishes, read the **BEST PREFIX** line — that is what it kept.
7. Go to **"2 — Inserts / STL"** and click **"Build STL only."**

### Where to see what it decided

- The **"3 — Verify & output"** box lists each printed ring *and which trays actually
  carry magnets* — so you can see at a glance that, say, Ring_N07 holds only trays 2 and 7.
- `InsertSearch/insert_search.png` plots the score against the number of inserts, with
  the best point marked.
- `InsertSearch/insert_search.csv` lists every step, with a `kept` column.

---

## 4. Feature 2 — Importing an OSII layout

### What it's for

Sometimes another program (called **OSII**) has *already* decided where the magnets go
and how each should be rotated. In that case you don't need our tool to figure it out —
you only need to turn OSII's answer into printable trays. This feature does exactly
that, letting you jump in near the end of the process.

### What it actually does

It reads OSII's file and rewrites it in the format our tray-builder understands. Think
of it as a translator between two languages that describe the same thing.

### Why a translation is needed (the honest details, in plain terms)

OSII and our tool both locate each magnet with three numbers (like describing a spot by
its left–right, up–down, and front–back distances), and both give each magnet a
rotation angle. But they don't label the directions the same way, and they use
different units. So the tool automatically:

- **Converts the units** — OSII uses metres, we use millimetres (every distance is
  multiplied by 1000).
- **Matches up the directions** — one axis is the same in both; the other two are
  swapped, and one is flipped. The practical result: the ring of magnets OSII describes
  lands in the correct plane in our system, at its real, physical tray-slot number —
  positive going toward the wall end, negative going toward the gaussmeter end,
  counted outward from the centre of the tube.
- **Adjusts each magnet's rotation** — the two programs measure a magnet's turn from a
  different starting line **and** in opposite directions, so the tool flips the turn and
  adds a fixed quarter-turn (90°).

You don't do any of this by hand. But two settings let you correct the rotation if a
test print ever comes out turned the wrong way:

- **`osii_invert_angle`** (on by default) — flips the *direction* of the turn. If the
  magnets look **mirrored**, toggle this.
- **`osii_angle_offset_deg`** (90 by default) — the fixed extra turn. If every magnet
  looks rotated by the **same amount**, change this number.

Both are adjustable right in the web page, so a fix is a click away — no code.

### How to use it — step by step

1. Open the web page.
2. At the top, type your **Iteration name** and **Label #**. These will appear on the
   output folder and be engraved on the trays.
3. Find the box labelled **"1.5 — OSII import."**
4. In **"Use OSII file,"** pick your OSII file. (Not listed? Paste its full path into
   **"Import CSV (full path)"** and click **"Import into OSII folder"** — then pick it.)
5. Leave the **Transform** settings at their defaults the first time (invert on, offset 90).
6. Click **"Convert OSII → shim CSV."**
7. Go to the box labelled **"2 — Inserts / STL"** and click **"Build STL only."**
8. Your printable trays appear in the output folder, named with your iteration.

### Important: which Stage 2 button to press

Stage 2 has **two** buttons:

- **"Build STL only"** — use this after an OSII import (the magnet list already exists).
- **"Export + build STL"** — use this only after you've run our optimizer.

If you press "Export + build STL" after an OSII import, it will fail with a
"file not found" error — because that button first tries to export an optimizer result
that doesn't exist (you skipped the optimizer on purpose). **After OSII import, always
use "Build STL only."**

---

## 5. Feature 3 — Scoring on your real measurements

### The idea

Before the tool can improve the field, it has to measure how uneven the field is. It
checks the field at many points on the surface of an imaginary sphere (called the
**shell**) the size of your imaging region. There are now two ways to get those check
points:

- **`measured`** (new, and now the default) — use the *actual* points your scanner
  measured, with their *actual* readings. Nothing is invented or smoothed.
- **`sh_fibonacci`** (the older way) — build a smooth mathematical model that best fits
  your measurements, then read that model at a set of evenly spaced, invented points.

### Why "measured" is usually better

The smooth model quietly rounds off the sharpest bumps in the real field. On your last
scan, the smoothed version reported the field spread as about **18,050 ppm**, but your
real measured points showed about **18,255 ppm** — the smoothing was hiding roughly
**200 ppm** of genuine unevenness. It's better to improve the field against the honest
number. And because your scan already sits neatly on the sphere, using the real points
is both possible and more faithful.

**When would you keep `sh_fibonacci`?** If your measurement is scattered or doesn't sit
on a clean sphere, or if you specifically want a set of perfectly evenly spaced points.

### How to use it

In the **"0 — Field map"** box, under **"Field adapter,"** set **`shell_source`** to
`measured` (it's already set for you). Click **"Prepare field map."** You'll see a line
in the log like `shell: 1742 MEASURED points (no interpolation)`. Then run the optimizer
as usual. If instead you see `... SH-fit points ... (interpolated)`, you're on the old
mode — switch the dropdown.

---

## 6. Feature 4 — A faster "ring search"

### What ring search is

Instead of *you* choosing which tray positions ("rings") to fill with magnets, ring
search tries combinations and reports the best set. It has two styles:

- **Exhaustive** — literally tries *every* possible combination. Thorough, but slow.
- **Paired** — only tries *symmetric* layouts, where every ring on one side of the
  centre has a twin at the same number on the other side (for example rings −10, −5, +5
  and +10). You say how many rings you want in total (it must be an even number: 4 rings
  = 2 pairs), and it tries **every** possible combination of pairs. Because it only
  considers symmetric layouts there are far fewer to try (300 for 4 rings across ±25),
  so it is quick and thorough at once. The individual magnets are still turned freely —
  only the *ring positions* are paired. Two twins are exactly mirror images in
  distance only if the front and back tray distances are equal.
- **Greedy** — builds the set up one ring at a time, always adding the most helpful
  next ring. Almost always lands on the same (or nearly the same) answer, much faster.

### What was slow, and what we fixed

Trying every group of 4 rings out of 50 candidates is about **103,000 trial fits**, and
each was taking around 26 milliseconds — so the whole thing would run about **45
minutes**. We fixed two things:

- The trial fits were running in a slower numeric mode; we switched to the fast one.
- The big win: each trial was checking the field at all 1,742 points, but for merely
  *ranking* which group is best, a few hundred points give the same ordering. So ranking
  now uses about 250 points, and only the single winning group is re-checked on the full
  set. That's roughly **10–35× faster** per trial, with no meaningful change to the
  answer.

Exhaustive now takes about **3–5 minutes** instead of 45.

### The practical advice

For a search this size, **use "greedy."** It does about 200 trials instead of 103,000 —
so it finishes in **seconds** — and almost always finds the same answer. Choose it in
the **"1 — Optimize"** box: pick the **"Ring search (best n)"** task and set the method
to **greedy**. Only use exhaustive if you specifically need the guaranteed-best group.

### Watching it run

While it works, the log now prints `1000 scored`, `2000 scored`, and so on, so you can
see progress. (Previously a display quirk hid these updates until the very end — that's
fixed.)

---

## 7. Feature 5 — Looking at the field — before or after shimming

### What it's for

You don't have to shim before you can *see* the field. The two viewers now open on the
**raw measured field** on their own, so you can check how uneven it is right after
measuring — before deciding anything. (Once you've shimmed, the same viewers also show
the shimmed result, as before.)

### The two viewers

- **Slice viewer** — a flat "map" of the field through a chosen plane, plus a line
  profile across it and a homogeneity number (ppm). Best for judging uniformity.
- **3D viewer** — a 3-D cloud of the field (and, once shimmed, the magnets too).

### How to open the field on its own

1. Run **Stage 0 "Prepare field map."**
2. In the **"3 — Verify & output"** box, click **"Open slice viewer"** (or "Open 3D
   viewer"). With no shim yet, they come up showing only the measured field — the title
   says "pre-shim." No optimization, no magnets required.

### Handy tools inside the slice viewer

- **Zoom into a region** (drag/scroll) to focus on the client's region of interest. The
  zoom now **stays put** when you move the sliders, the **ppm updates to that region only**
  (it says "visible region"), and the line profile **clips to the region** too — so you're
  always looking at just the part that matters.
- **Sphere mask:** a button crops the view (and the ppm) to a sphere of a chosen radius,
  hiding the corners that are usually the least trustworthy.
- The field is now sampled every **5 mm** for smoother pictures.

---

## 8. Feature 6 — Saving a GIF of the result (new)

### What it's for

Once you've got a view you like in the 3D viewer or the slice viewer — a good angle,
a good slice, the shimmed result — you can save it as an animated GIF instead of
taking a screenshot or asking a colleague to open the tool themselves.

### The two kinds

- **3D viewer** — "Save GIF" spins the camera a full 360° around the bore at
  whatever elevation/distance you've currently zoomed/rotated to, and saves that
  as a GIF.
- **Slice viewer** — "Save GIF" instead sweeps the slice position across the field
  (a "slice sweep"), so you see the field change plane by plane.

Each viewer also has a second button, **"Save GIF pair: Measured vs Shimmed,"**
which saves two matching GIFs side by side — one of the raw measured field, one of
the shimmed result — for a clean before/after.

### Where the files go

GIFs are saved under your iteration's output folder, in a `Viewers` subfolder
(`data/outputs/Optimizer_Output_per_Iteration/<your iteration>/Viewers/`). Two settings control
length: how many frames, and how many frames per second — by default a GIF is
120 frames at 20 frames/second, i.e. 6 seconds long.

---

## 9. Feature 7 — Independent Python check (Stage 4)

### What it's for

The optimizer predicts a score (ppm) for your magnet layout, but it uses its own
calculation to do so. Stage 4 hands the same layout to a **separate program** (written
in Python, in the `Shimming_verifier` folder) that calculates the magnets' field a
different way. If both programs agree, you can trust the number. On the reference run
the optimizer predicted about 9,480 ppm and the Python check found about 9,560 ppm
(the unshimmed field was about 18,540 ppm).

### What you choose

Only three things — everything else (magnet strength, size, units) is taken from the
settings you already made:

1. **Which field** — the measured scan to compare against. Leave it on
   *"same as Stage 0 field map"* unless you want a different scan.
2. **Which shim layout** — normally this run's magnet list (from the optimizer, the
   insert search or an OSII import). Untick *verifier_use_shim* to look at the measured
   field alone, without any shim magnets.
3. **What to run** — any of: 2D pictures of the field with vs without shim, an
   interactive 3D scene (opens in any browser), and the spherical-harmonic pyramids
   (how much of each "field shape" the shim removes).

### How to use it

1. Finish your run (you need a shim CSV, so at least Stage 1.5 or an insert search).
2. In the **"4 — Python verifier"** box, check the three choices above.
3. Click **"Run verifier."** It takes a few seconds; the log ends with the
   with/without-shim scores and where the files went.
4. Click **"Open verifier output"** to see the pictures and open `scene_3d.html`.

You can also make it run automatically at the end of a full run (*run_python_verifier*),
or run only this stage from the command line by setting `start_stage = 5`.

### First-time setup

The Python program needs a few Python packages. Once, in a terminal:
`python -m pip install -r stages/stage4_verifier/Shimming_verifier/requirements.txt`. If Python isn't on
your PATH, put its full path in *verifier_python*.

---

## 9a. Feature 9 — Spherical-harmonic breakdown at Stage 0

### What it's for

Any uneven field can be split into standard "shapes" called spherical harmonics, each with
a *degree* n (1 = a simple slope, 2 = a saddle/bow shape, then finer and finer detail) and a
strength in mT. Stage 0 now shows this breakdown for your measured field every time it
builds the field map, so you can see at a glance which shapes dominate — and therefore what
the shims actually have to fight.

### What you see

In the run log, after Stage 0: the average field, a short table of how strong each degree
is, and **a list of the 5 biggest shapes** (say how many with *sh_top_k*), each with its
share of the total unevenness. A full table is saved next to your results
(`SHDecomposition/sh_decomposition.csv`).

### What you can choose (*sh_select*, Stage 0 → Field adapter)

- **all** (default) — use every shape the fit found. Nothing changes from before.
- **first_n** — keep only degrees 1…n (set *sh_use_degree*). Throws away the fine detail and
  keeps the broad shape.
- **top_k** — keep only the average plus the *sh_top_k* biggest shapes (for example 5).

With the last two, the optimizer and the viewers work on this **simplified** field, not the
raw measurement — so the ppm numbers you see afterwards describe the simplified field. The
log's last line tells you how far the simplified field is from your real data (RMS and ppm).
For the honest number, keep *all* and use the shim on the real field; use *first_n* / *top_k*
when you want to study or shim only the dominant shapes.

In shell mode with *shell_source = measured*, the same measured points are used, but their
values come from the simplified field when first_n / top_k is on.

---

## 9b. Feature 10 — Which way is the field pointing? (frames) and Halbach test bits

### The problem in plain words

Your scanner's axes (X, Y, Z) and the magnet's field direction don't have to line up. The tool
calculates in an internal "tidy" view where the field always points up (+y) — it quietly **turns** your
measurement to get there. Your scan says the field points along −y, so the tidy view is turned half a
turn (180°) compared with the scanner. That is fine for the maths, but the results (magnet positions and
angles) have to be turned **back** before they mean something in the scanner's own directions.

### What happens now (by default)

- Right after the optimizer, the magnet list is turned back into the **scanner's view** (`shim_csv_frame
  = "scan"`). The printed trays, both viewers and the Python check then all use the scanner's view.
- Both viewers draw in the scanner's view too (`viewer_frame = "scan"`) and show the field **with its sign**
  (negative numbers if the field points −x or −y). Set either key to `"optimizer"` (in `config.toml`) to get
  the old behaviour.
- A magnet list made **before** this change is not converted — just click **Export** again (no need to
  re-optimize).
- Tray numbers are worked out from where each magnet sits, so after the turn a −y scan's tray 3 becomes tray 9
  and so on (every tray shifts by 6). Check that the physical tray numbering on your magnet matches the
  scanner's directions — this is the one thing the tool cannot know for you.

### How well was it checked?

An independent calculation on your real −y scan showed the turned-back layout cuts the unevenness from
about 19,200 ppm to about 9,600 ppm, while using the un-turned list made it worse (about 28,800 ppm). It
cannot tell front from back (the rings are nearly symmetric), so a mirrored z axis would not show up. The
viewers and the export step themselves have not been run since this change.

### Halbach test bits (checking your axes without the optimizer)

`make_halbach_ring_csv.py` writes a small magnet list that imitates a Halbach ring, so you can see in the
viewers and the Python check whether directions come out as expected. You choose the OSII version (OSII2: field
+y, OSII1: field +x), which tray(s), where along the bore (a z in mm, or a slot number) and how many
positions (an odd number). To make STLs from it: copy it into a **new** iteration folder as `<iteration>_shim.csv`
and use **Build STL only**.

## 10. Glossary

- **Shim (shim magnet)** — a small magnet added to nudge the main field toward being even.
- **Ring** — one position along the tube where a full circle of shim magnets sits. In this setup a full ring holds 84 magnets.
- **Insert position (InsertPos)** — the real, physical slot number for a ring, counted
  outward from the centre of the tube: positive toward the wall end, negative toward
  the gaussmeter end (e.g. `-7` or `12`). This is what "Ring number" actually means
  everywhere in the tool now — it is never just a count of "which ring is this in
  the current run" — and it's what gets engraved on the printed part.
- **N / P label** — how a negative or positive insert position is written on a
  printed part and in folder/file names, since a printed minus sign is easy to
  misread or lose: `N07` means insert position `-7`, `P12` means `+12`.
- **Tray** — the 3D-printed holder that carries the magnets for one part of a ring.
- **Insert** — one tray's worth of magnets at one ring position: 7 magnets. The unit the insert search places, and the unit you print.
- **Front / Back** — Front is the gaussmeter end of the tube, Back is the wall end. Their distances from the centre are set separately, because the magnet isn't symmetric.
- **Bore / imaging region / DSV** — the tube (and the space inside it) where the sample or patient goes; the region we want to be even.
- **Field uniformity / ppm** — a score for how even the field is. "ppm" means *parts per million*. It's the gap between the strongest and weakest field, divided by the average, times a million. **Lower is better.**
- **Shell** — the surface of an imaginary sphere where we check the field. Thanks to physics, if the field is even on this surface, it's even everywhere inside — so checking the surface is enough.
- **Optimizer** — the part of the tool that calculates the best magnet angles.
- **Field map** — the tool's internal picture of the current field, built from your measurement.
- **CSV** — a plain table file, like a spreadsheet saved as simple text.
- **STL / STEP** — 3D-model file formats: STL for printing, STEP for CAD editing.
- **Iteration** — one named run. Its name labels the output folder and is engraved on the printed trays.
- **Stage 0 / 1 / 1.5 / 2 / 3 / 4** — the steps of the process: read the measurement (0), calculate angles (1), write the magnet list (1.5), make printable trays (2), view/verify (3), independent Python double-check (4).
- **magpylib** — the Python physics library the Stage 4 check uses to compute each magnet's field. Separate from the optimizer's own code, which is the point.
- **Spherical harmonics (SH) pyramid** — a chart splitting the field's unevenness into standard "shapes" (degree 1, 2, 3 …). Shows *which* shapes the shim removes and which remain.

- **Scan frame (scanner view)** — the X, Y, Z directions of your measurement file.
- **Optimizer frame (tidy view)** — the internal view where the field always points along +y; the optimizer works here.
- **Halbach ring** — magnets turned in a regular pattern (angle = twice the position angle) so their fields add up inside the ring.

---

## 11. Quick reference — where each new setting lives

| Setting | Where in the web page | Plain meaning |
|---|---|---|
| `shell_source` | Stage 0 → Field adapter | Use real measured points (`measured`) or a smoothed model (`sh_fibonacci`). |
| OSII file | Stage 1.5 → OSII import | Which OSII layout file to convert. |
| `osii_invert_angle` | Stage 1.5 → Transform | Flip magnet turn direction if a print looks mirrored. |
| `osii_angle_offset_deg` | Stage 1.5 → Transform | Fixed extra turn (90°); change if a print is rotated by a constant amount. |
| Ring search method | Stage 1 → Ring search task | `greedy` (fast, recommended), `exhaustive` (thorough, slow) or `paired` (only symmetric ±pairs, every combination). For `paired`, the ring count is the total and must be even. |
| `magnets_available` | Stage 1 → Insert search task | How many shim magnets you own. Divided by 7 to get the number of full inserts. |
| `insert_search_min_spots_between` | Stage 1 → Insert search task | How far apart two inserts must be *within the same tray*. |
| `magnet_Br_T` | Stage 0 → Shim magnets | How strong one magnet's material is: its remanence, in tesla (from the datasheet). |
| `magnet_side_mm` | Stage 0 → Shim magnets | The physical cube side length of one magnet, in mm. |
| `letter_thickness` | Stage 2 → Insert params | How deeply the label digits are engraved into the printed part. |
| `label_scale` | Stage 2 → Insert params | Overall size of the engraved label digits (iteration, insert position, tray). |
| `viewer_gif_frames` / `viewer_gif_fps` | (config.toml; not yet a GUI field) | How many frames a Save-GIF export has, and how fast it plays — together set the length in seconds. |
| `sh_select` | Stage 0 → Field adapter | Build the field map from `all` shapes, only the `first_n` degrees, or only the `top_k` biggest. |
| `sh_use_degree` | Stage 0 → Field adapter | first_n: keep degrees up to this number. |
| `sh_top_k` | Stage 0 → Field adapter | top_k: how many biggest shapes to keep — and how long the "biggest shapes" list in the log is. |
| `verifier_measurement` | Stage 4 → Which field | Scan the Python check compares against (blank = the same as Stage 0). |
| `verifier_use_shim` / `verifier_shim_csv` | Stage 4 → Which shim layout | Include the shim magnets; blank path = this run's magnet list. |
| `verifier_make_2d` / `_3d` / `_sh` | Stage 4 → What to run | Which outputs to produce: 2D pictures, 3D scene, spherical-harmonic pyramids. |
| `run_python_verifier` | Stage 4 → What to run | Also run Stage 4 automatically at the end of a full run. |
| `front_tray_shift_mm` | Stage 0 → Geometry | Distance from the centre to the first tray on the gaussmeter side. |
| `back_tray_shift_mm` | Stage 0 → Geometry | Distance from the centre to the first tray on the wall side. |
| `tray_slot_spacing_mm` | Stage 0 → Geometry | Gap between neighbouring tray positions (10 mm). |
| `main_field_direction` | Stage 0 → Field adapter | Which way the field points in your scan (`auto` reads it from the measurement). |
| `shim_csv_frame` | (config.toml only) | `scan` (default): the magnet list is written in your scanner's view; `optimizer`: old internal view. |
| `viewer_frame` | (config.toml only) | `scan` (default): viewers draw in your scanner's view with a signed field; `optimizer`: old view. |

All of these can also be set by editing the `config.toml` file directly, but the web
page is the easy way and is recommended.
