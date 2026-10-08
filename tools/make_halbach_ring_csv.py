"""Build a shim CSV (X (mm), Y (mm), RingNumber, Angle (deg)) of Halbach-dipole "bits":
one or more trays (7-magnet inserts) repeated over several axial positions, to test the
axis / angle conventions of the pipeline.

    python tools/make_halbach_ring_csv.py --OSII_version OSII2 --tray_number 3 \
           --z_center 0 --z_copies 3  [--out name_shim.csv] [--flip]
    python tools/make_halbach_ring_csv.py --OSII_version OSII1 --tray_number 3,4 \
           --slot_center 5 --z_copies 5

--OSII_version   OSII2 -> geometry of preset "OSII V2.1 (PUC trays)" (r = 231 mm, 20.64 deg
                          per insert), field pointing +y in the CSV frame (= OSII +z, since
                          osii_to_shim maps BOOST Y = OSII z).
                 OSII1 -> geometry of preset "OSII V1.1" (r = 277 mm, 18 deg per insert),
                          field pointing +x in the CSV frame (= OSII +x).
--tray_number    tray(s) to fill: an int 1..num_trays, a comma list "3,4", or "all".
                 Tray centre angle = (tray-9)*360/num_trays (tray 9 -> +x, 12 -> +y, 3 -> -x).
--z_center       axial centre (mm); copies sit at z_center + k*tray_slot_spacing_mm,
                 k = -(n-1)/2 .. (n-1)/2.  Every copy must land on a real tray slot
                 (z(+n)=+(back+(n-1)*s), z(-n)=-(front+(n-1)*s); no slot at z=0 unless a
                 shift is 0).
--slot_center    alternative: centre on this InsertPos; copies are the n consecutive REAL
                 slots around it (slot 0 does not exist, so ... -2, -1, +1, +2 ...).
--z_copies       how many axial positions to fill (odd).
Geometry (radius, insert arc, tray shifts) comes from gui/presets.json for the chosen
version; a warning is printed if config.toml disagrees (Stage 2 reads config.toml).

Angle convention (optimizer's): moment = (cos t, sin t), t from +X toward +Y. Ideal Halbach
dipole with magnet at polar angle phi:   field +x: t = 2*phi    field +y: t = 2*phi - 90 deg
(--flip reverses the field).  Magnets of a tray are ordered by decreasing phi, like
export_csv / the optimizer output. A companion *_xyz.csv with literal Z (mm) is also written.
"""
import argparse, csv, json, math, os, tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root (this script lives in tools/)
PRESET = {"OSII1": "OSII V1.1", "OSII2": "OSII V2.1 (PUC trays)"}
FIELD_DEG = {"OSII1": 0.0, "OSII2": 90.0}   # direction of the interior field in the CSV frame

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("--OSII_version", required=True, choices=["OSII1", "OSII2"])
ap.add_argument("--tray_number", required=True)
ap.add_argument("--z_center", type=float)
ap.add_argument("--slot_center", type=int)
ap.add_argument("--z_copies", type=int, default=1)
ap.add_argument("--out", default=None)
ap.add_argument("--flip", action="store_true", help="field reversed")
a = ap.parse_args()

if (a.z_center is None) == (a.slot_center is None):
    raise SystemExit("give exactly one of --z_center / --slot_center")
if a.z_copies < 1 or a.z_copies % 2 == 0:
    raise SystemExit("--z_copies must be a positive odd number")

with open(os.path.join(ROOT, "gui", "presets.json"), encoding="utf-8") as f:
    g = json.load(f)["geometry"][PRESET[a.OSII_version]]
R, arc, nmag, ntray = g["shim_radius_mm"], g["angle_per_segment_deg"], g["mags_per_segment"], g["num_trays"]
sp, fr, bk = g["tray_slot_spacing_mm"], g["front_tray_shift_mm"], g["back_tray_shift_mm"]

with open(os.path.join(ROOT, "config.toml"), "rb") as f:
    cfg = tomllib.load(f)
for key, val in [("shim_radius_mm", R), ("angle_per_segment_deg", arc), ("mags_per_segment", nmag),
                 ("num_trays", ntray), ("tray_slot_spacing_mm", sp),
                 ("front_tray_shift_mm", fr), ("back_tray_shift_mm", bk)]:
    if key in cfg and abs(float(cfg[key]) - float(val)) > 1e-9:
        print(f"WARNING: config.toml {key} = {cfg[key]} but {PRESET[a.OSII_version]} uses {val}; "
              f"set config.toml to the preset before Stage 2.")

def slot_z(n):   # InsertPos -> axial z (mm)
    return bk + (n - 1) * sp if n > 0 else -(fr + (-n - 1) * sp)

def z_to_slot(z, tol=1e-6):
    for n in range(1, 200):
        if abs(slot_z(n) - z) < tol: return n
        if abs(slot_z(-n) - z) < tol: return -n
    raise SystemExit(f"z = {z} mm is not a tray slot (front={fr}, back={bk}, spacing={sp} mm)")

half = (a.z_copies - 1) // 2
if a.z_center is not None:
    slots = [z_to_slot(a.z_center + k * sp) for k in range(-half, half + 1)]
else:
    if a.slot_center == 0: raise SystemExit("slot 0 does not exist")
    real = [n for n in range(-200, 201) if n != 0]
    i = real.index(a.slot_center)
    if i - half < 0 or i + half >= len(real): raise SystemExit("slot range out of bounds")
    slots = real[i - half: i + half + 1]

trays = list(range(1, ntray + 1)) if a.tray_number == "all" else [int(t) for t in a.tray_number.split(",")]
for t in trays:
    if not 1 <= t <= ntray: raise SystemExit(f"tray {t} not in 1..{ntray}")

step = arc / (nmag - 1)
rows = []
for s in slots:
    for t in trays:
        centre = ((t - 9) * 360.0 / ntray) % 360.0
        for j in range(nmag):
            phi = centre + arc / 2 - j * step
            x, y = R * math.cos(math.radians(phi)), R * math.sin(math.radians(phi))
            th = (2 * phi - FIELD_DEG[a.OSII_version] + (180.0 if a.flip else 0.0)) % 360.0
            rows.append((x, y, float(s), th, slot_z(s)))

out = a.out or os.path.join(ROOT, f"Halbach_{a.OSII_version}_T{a.tray_number.replace(',', '-')}_"
                                  f"{'z' + format(a.z_center, 'g') if a.z_center is not None else 'slot' + str(a.slot_center)}"
                                  f"_x{a.z_copies}_shim.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["X (mm)", "Y (mm)", "RingNumber", "Angle (deg)"])
    w.writerows([r[:4] for r in rows])
xyz = out[:-len("_shim.csv")] + "_shim_xyz.csv" if out.endswith("_shim.csv") else out + "_xyz.csv"
with open(xyz, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["X (mm)", "Y (mm)", "Z (mm)", "Angle (deg)"])
    w.writerows([(r[0], r[1], r[4], r[3]) for r in rows])
print(f"{a.OSII_version}: r={R} mm, {arc} deg/insert, field {'+' if not a.flip else '-'}"
      f"{'x' if a.OSII_version == 'OSII1' else 'y'}")
print(f"trays {trays} x InsertPos {slots} (z = {[slot_z(s) for s in slots]} mm) -> {len(rows)} magnets")
print(f"wrote {out}\n      {xyz}")
