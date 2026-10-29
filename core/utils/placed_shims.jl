# utils/placed_shims.jl — shim inserts that are ALREADY physically placed in the magnet.
#
# The user can list zero, one or several shim CSVs (GUI: Stage 0 → Measured data →
# "Already placed shims"; config key `placed_shim_csvs`). The measured field map is assumed to
# ALREADY INCLUDE those shims (they were installed before the scan), so the optimizer never
# touches the field — it only has to know WHICH SLOTS ARE TAKEN, and keep new inserts away:
#   • insert search : those exact (ring, tray) slots are not candidates, and the same-tray
#                     spacing rule (insert_search_min_spots_between) also applies against them;
#   • ring search   : a ring that holds any placed insert is not a candidate (a ring is a whole
#                     12-tray block), nor is any ring too close to one (ring_search_min_spots_between).
#
# Accepted CSV formats (the same ones the pipeline itself writes; auto-detected by column names):
#   seam   X (mm), Y (mm), RingNumber, Angle (deg)      RingNumber = real InsertPos
#   xyz    X (mm), Y (mm), Z (mm),     Angle (deg)      InsertPos is recovered from Z
#
# FRAME. The CSVs are written in the frame declared by `shim_csv_frame` ("scan" = the scanner's own
# axes, the default; "optimizer" = B0 → +y). The optimizer and its slot geometry live in the optimizer
# frame, so for "scan" the positions are rotated scan → optimizer with the scan's B0 direction before
# the tray is derived (the same rotation export_csv.jl undoes when it writes the file).
#
# TRAY. Derived from (x, y) in the optimizer frame exactly as insert_search.jl / CSV_to_STL do
# (assign_tray), so a placed insert and a candidate slot share one numbering.
#
# Pure base-Julia + CSV/DataFrames — no GPU, no config globals: every geometry value is an argument,
# so it is unit-testable.

using CSV, DataFrames

_pl_scan_to_opt_xy(x, y, dir) =
    dir == "+y" ? (x, y) : dir == "+x" ? (-y, x) : dir == "-x" ? (y, -x) : (-x, -y)

_pl_assign_tray(x, y, num_trays) =
    mod(round(Int, mod(atan(y, x) * 180 / pi, 360.0) / (360.0 / num_trays)) + 8, num_trays) + 1

# InsertPos (signed tray slot) → axial z (mm) and back; same formulas as pos_trays.jl / osii_to_shim.jl
_pl_z_of_ring(t, spacing, front, back) =
    t > 0 ? back + (t - 1) * spacing : -(front + (-t - 1) * spacing)
function _pl_ring_of_z(z, spacing, front, back)
    if z >= 0
        n = round(Int, (z - back) / spacing) + 1
        resid = z - (back + (n - 1) * spacing)
    else
        n = -(round(Int, (-z - front) / spacing) + 1)
        resid = z - (-(front + (-n - 1) * spacing))
    end
    return n, resid
end

"""
    read_placed_magnets(paths; tray_slot_spacing_mm, front_tray_shift_mm, back_tray_shift_mm)
        -> (; x, y, ring, ang, file)

Every magnet of every placed-shim CSV, positions and angles AS WRITTEN (i.e. in the CSV's own frame,
`shim_csv_frame`), with the real InsertPos (`RingNumber`, or recovered from `Z (mm)`). Used both to derive the
occupied slots (`load_placed_inserts`) and to DRAW the placed shims (viewers, layout figure). Empty `paths` →
empty vectors. Warns when axial positions do not line up with the configured tray geometry.
"""
function read_placed_magnets(paths::AbstractVector{<:AbstractString};
                             tray_slot_spacing_mm::Real, front_tray_shift_mm::Real, back_tray_shift_mm::Real)
    x = Float64[]; y = Float64[]; ring = Int[]; ang = Float64[]; file = String[]
    for p in paths
        isfile(p) || error("placed shims: file not found:
    $(p)")
        df   = CSV.read(p, DataFrame)
        cols = names(df)
        ringcol = "RingNumber" in cols
        zcol    = "Z (mm)" in cols
        for c in ("X (mm)", "Y (mm)", "Angle (deg)")
            c in cols || error("placed shims: $(basename(p)) has no column '$(c)' — columns found: $(cols)
" *
                               "    expected  X (mm), Y (mm), RingNumber|Z (mm), Angle (deg)")
        end
        (ringcol || zcol) || error("placed shims: $(basename(p)) needs a 'RingNumber' or a 'Z (mm)' column — found $(cols)")
        worst = 0.0
        for r in eachrow(df)
            ringv = if ringcol
                round(Int, Float64(r["RingNumber"]))
            else
                n, resid = _pl_ring_of_z(round(Float64(r["Z (mm)"]); digits = 3), tray_slot_spacing_mm,
                                         front_tray_shift_mm, back_tray_shift_mm)
                worst = max(worst, abs(resid)); n
            end
            ringv == 0 && error("placed shims: $(basename(p)) has a row at ring 0 (no such slot).")
            push!(x, Float64(r["X (mm)"])); push!(y, Float64(r["Y (mm)"]))
            push!(ring, ringv); push!(ang, Float64(r["Angle (deg)"])); push!(file, basename(p))
        end
        worst > tray_slot_spacing_mm / 2 &&
            @warn "placed shims: $(basename(p)) — axial positions do not line up with the configured tray geometry" worst_residual_mm = round(worst, digits = 2)
    end
    return (; x, y, ring, ang, file)
end

"""
    load_placed_inserts(paths; num_trays, mags_per_segment, tray_slot_spacing_mm,
                        front_tray_shift_mm, back_tray_shift_mm, csv_frame = "scan", scan_dir = nothing)
        -> Vector{NamedTuple{(:ring, :tray, :nmags, :file)}}

One entry per occupied INSERT SLOT (a ring slot × a tray), sorted by (ring, tray). `paths` may be empty
(→ empty result). Throws a descriptive error for an unreadable file, unknown columns, or the same slot
claimed twice (within one file or across files). Warns for partially filled inserts.
"""
function load_placed_inserts(paths::AbstractVector{<:AbstractString};
                             num_trays::Integer, mags_per_segment::Integer,
                             tray_slot_spacing_mm::Real, front_tray_shift_mm::Real, back_tray_shift_mm::Real,
                             csv_frame::AbstractString = "scan", scan_dir = nothing)
    out = NamedTuple{(:ring, :tray, :nmags, :file), Tuple{Int,Int,Int,String}}[]
    csv_frame == "scan" && scan_dir === nothing && !isempty(paths) &&
        error("placed shims: the shim CSVs are in the scan frame (shim_csv_frame = \"scan\") but the scan's " *
              "B0 direction is unknown. Re-run Stage 0 (it saves field_direction), or set main_field_direction.")
    claimed = Dict{Tuple{Int,Int},String}()
    for p in paths                                   # one file at a time, so a clash can name both files
        M = read_placed_magnets([p]; tray_slot_spacing_mm = tray_slot_spacing_mm,
                                front_tray_shift_mm = front_tray_shift_mm, back_tray_shift_mm = back_tray_shift_mm)
        counts = Dict{Tuple{Int,Int},Int}()
        for i in eachindex(M.x)
            x, y = csv_frame == "scan" ? _pl_scan_to_opt_xy(M.x[i], M.y[i], scan_dir) : (M.x[i], M.y[i])
            key = (M.ring[i], _pl_assign_tray(x, y, num_trays))
            counts[key] = get(counts, key, 0) + 1
        end
        for (key, n) in sort!(collect(counts))
            haskey(claimed, key) &&
                error("placed shims: ring $(key[1]), tray $(key[2]) is occupied by BOTH $(claimed[key]) and $(basename(p)).")
            claimed[key] = basename(p)
            n == mags_per_segment ||
                @warn "placed shims: $(basename(p)) — ring $(key[1]), tray $(key[2]) holds $(n) magnets, not $(mags_per_segment); the whole insert slot is treated as occupied"
            push!(out, (ring = key[1], tray = key[2], nmags = n, file = basename(p)))
        end
    end
    sort!(out; by = s -> (s.ring, s.tray))
    return out
end

"""Distinct rings (InsertPos) that hold at least one placed insert."""
placed_rings(placed) = sort!(unique([s.ring for s in placed]))

"""
    insert_slot_blocked(ring, tray, placed, min_spots_between; spots_between)

True if a NEW insert at (ring, tray) is not allowed because of the placed inserts: the slot itself is taken,
or a placed insert sits in the SAME tray with fewer than `min_spots_between` empty ring slots between them.
`spots_between(a, b)` is the caller's counting rule (it knows about the missing tray 0).
"""
function insert_slot_blocked(ring::Integer, tray::Integer, placed, min_spots_between::Integer; spots_between)
    for s in placed
        s.tray == tray || continue
        spots_between(ring, s.ring) >= min_spots_between || return true     # same ring ⇒ n = -1 ⇒ always blocked
    end
    return false
end

"""
    ring_blocked(ring, placed, min_spots_between)

True if a NEW whole ring at `ring` is not allowed: some placed insert is on that ring, or closer than
`min_spots_between` empty slots (|a−b|−1, the ring search's counting rule) — a whole ring touches every tray.
"""
function ring_blocked(ring::Integer, placed, min_spots_between::Integer)
    for r in placed_rings(placed)
        (abs(ring - r) - 1) >= min_spots_between || return true
    end
    return false
end
