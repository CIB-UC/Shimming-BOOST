# utils/viewer_frame.jl — optimizer frame <-> scan (lab) frame, for the Stage-3 viewers.
# Stage 0 rotates the scan about the bore (z) so that B0 -> +y (the optimizer frame):
#     lab_to_optimizer_xy: "+y" (x,y)   "+x" (-y,x)   "-x" (y,-x)   "-y" (-x,-y)
# These helpers apply the INVERSE (optimizer -> scan) for display. Base Julia only.

opt_to_scan_xy(x, y, dir) =
    dir == "+y" ? (x, y) : dir == "+x" ? (y, -x) : dir == "-x" ? (-y, x) : (-x, -y)

# a moment angle theta (deg, CCW from +x) in the optimizer frame, seen from the scan frame
opt_to_scan_angle_deg(dir) = dir == "+y" ? 0.0 : dir == "+x" ? -90.0 : dir == "-x" ? 90.0 : 180.0

"""
    grid_to_scan(A, xg, yg, dir) -> (A', xg', yg')

Re-express a field array A[i,j,k] on the optimizer-frame axes (xg, yg) in the scan frame:
the point (x,y) is moved to opt_to_scan_xy(x,y,dir) and the axes are re-sorted ascending.
(z, the bore axis, is unchanged.)
"""
function grid_to_scan(A, xg, yg, dir)
    dir == "+y" && return A, xg, yg
    dir == "-y" && return A[end:-1:1, end:-1:1, :], -reverse(xg), -reverse(yg)
    B = permutedims(A, (2, 1, 3))
    dir == "+x" && return B[:, end:-1:1, :], yg, -reverse(xg)
    return B[end:-1:1, :, :], -reverse(yg), xg          # "-x"
end

# --- scan -> optimizer (inverse of opt_to_scan_*), and the shim-CSV frame helpers ------------
scan_to_opt_xy(x, y, dir) =
    dir == "+y" ? (x, y) : dir == "+x" ? (-y, x) : dir == "-x" ? (y, -x) : (-x, -y)
scan_to_opt_angle_deg(dir) = -opt_to_scan_angle_deg(dir)

"""
    read_field_direction(paths...) -> String or nothing

Resolved scan direction ("+x"/"-x"/"+y"/"-y") that Stage 0 saved in the first field-map jld2
that has it (`field_direction`, or `main_field_direction` for the reshape adapter).
Needs JLD2's `jldopen` in scope in the including script.
"""
function read_field_direction(paths...)
    for p in paths
        isfile(p) || continue
        d = nothing
        jldopen(p, "r") do f
            haskey(f, "field_direction") ? (d = f["field_direction"]) :
            haskey(f, "main_field_direction") && (d = f["main_field_direction"])
        end
        d === nothing || return String(d)
    end
    return nothing
end
