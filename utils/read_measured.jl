# read_measured.jl  —  format-agnostic reader for SPHERICAL-scan measurement CSVs.
#
# Handles BOTH layouts the SH / shell Stage-0 adapters accept:
#   • simple : metadata line 1, header line 2 = X,Y,Z,Gauss[,T°]           (Josh scan)
#   • rich   : multi-line metadata block, then a header like
#       Timestamp,radio,theta_deg,phi_deg,X,Y,Z,T_robot (C),muestra,Time (s),
#       x axis (gauss),y axis (gauss),z axis (gauss),Magnitude (gauss)      (Controlled_V2)
#
# `read_measured_points(path; unit_mm)` returns
#     (x, y, z  in mm ; b = |B| in Gauss (positive) ; meanvec | nothing)
# averaging repeated samples per point (rich format: `muestra`) and dropping the
# spurious (0,0,0) marker. `meanvec` is the mean (Bx,By,Bz) Gauss vector when the
# scan recorded field COMPONENTS (rich format) — used to auto-detect B0's direction.
#
# `detect_field_direction(meanvec)` → "+x"/"-x"/"+y"/"-y" (dominant in-plane axis).
# Temperature (T_robot) is read but NOT used yet.

using CSV, DataFrames, Statistics

_exactcol(df, name) = (for n in names(df); lowercase(n) == lowercase(name) && return n; end; nothing)
function _findcol(df, kws...)
    for n in names(df)
        ln = lowercase(n)
        all(k -> occursin(lowercase(k), ln), kws) && return n
    end
    return nothing
end

function read_measured_points(path::AbstractString; unit_mm::Real = 1.0)
    lines = readlines(path)
    isxyz(l) = occursin(r"(^|,)\s*x\s*(,|$)"i, l) && occursin(r"(^|,)\s*y\s*(,|$)"i, l) &&
               occursin(r"(^|,)\s*z\s*(,|$)"i, l)
    hdr = findfirst(isxyz, lines)
    hdr === nothing && error("read_measured_points: no X,Y,Z header found in\n    $path")
    df = CSV.read(IOBuffer(join(lines[hdr:end], "\n")), DataFrame)

    xcol = _exactcol(df, "X"); ycol = _exactcol(df, "Y"); zcol = _exactcol(df, "Z")
    (xcol === nothing || ycol === nothing || zcol === nothing) &&
        error("read_measured_points: missing X/Y/Z columns in $path")

    magcol = _findcol(df, "magnitude")                 # rich format
    magcol === nothing && (magcol = _exactcol(df, "Gauss"))   # simple format
    magcol === nothing && error("read_measured_points: no 'Magnitude'/'Gauss' field column in $path")

    xax = _findcol(df, "x axis"); yax = _findcol(df, "y axis"); zax = _findcol(df, "z axis")
    has_vec = xax !== nothing && yax !== nothing && zax !== nothing

    # drop the (0,0,0) marker row(s)
    df = filter(r -> !(r[xcol] == 0 && r[ycol] == 0 && r[zcol] == 0), df)

    # average repeated samples per point (rich format has a `muestra` index)
    if _findcol(df, "muestra") !== nothing
        aggs = Any[magcol => mean => magcol]
        has_vec && append!(aggs, Any[xax => mean => xax, yax => mean => yax, zax => mean => zax])
        df = combine(groupby(df, [xcol, ycol, zcol]), aggs...)
    end

    x = Float64.(df[!, xcol]) .* unit_mm
    y = Float64.(df[!, ycol]) .* unit_mm
    z = Float64.(df[!, zcol]) .* unit_mm
    b = abs.(Float64.(df[!, magcol]))                  # |B| in Gauss, positive
    meanvec = has_vec ? (mean(Float64.(df[!, xax])), mean(Float64.(df[!, yax])), mean(Float64.(df[!, zax]))) : nothing
    return x, y, z, b, meanvec
end

# dominant IN-PLANE axis of the mean field vector → lab-frame B0 direction string.
# (Halbach field lies in the x–y plane; z is ignored for the direction label.)
function detect_field_direction(meanvec)
    bx, by, _ = meanvec
    return abs(bx) >= abs(by) ? (bx >= 0 ? "+x" : "-x") : (by >= 0 ? "+y" : "-y")
end
