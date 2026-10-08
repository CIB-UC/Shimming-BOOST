using Statistics


############ Rings/Trays helpers (mm coherentes con tu pipeline) ############

# TrayNr -> axial position (mm), measured from TRUE z = 0 (magnet centre plane).
#
# The two innermost trays are anchored INDEPENDENTLY, because the two ends of the
# bore are not necessarily symmetric:
#
#   back_tray_shift_mm   |z| of tray +1  → +Z, the "Back - Wall" end
#   front_tray_shift_mm  |z| of tray −1  → −Z, the "Front - Gaussmeter" end
#
# Every further tray steps out by a constant tray_slot_spacing_mm:
#
#   z(+n) = +( back_tray_shift_mm  + (n−1)·tray_slot_spacing_mm )     n ≥ 1
#   z(−n) = −( front_tray_shift_mm + (n−1)·tray_slot_spacing_mm )     n ≥ 1
#
# e.g. back=10, front=10, spacing=10 → +1:+10, +2:+20, −1:−10, −2:−20
#      back=5,  front=8,  spacing=10 → +1: +5, +2:+15, −1: −8, −2:−18
#
# Tray 0 does not exist: numbering starts at ±1 either side of z = 0.
# (Replaces the old single `half_shift_mm`, which forced a symmetric bore:
#  half_shift_mm = h with spacing s is exactly front = back = s − h.)
function ringpos_from_tray_mm(trays::AbstractVector{<:Integer};
                              tray_slot_spacing_mm::Real = 10.0,
                              front_tray_shift_mm::Real  = 10.0,
                              back_tray_shift_mm::Real   = 10.0) :: Vector{Float64}
    any(==(0), trays) && error(
        "tray 0 is not a valid slot — trays are numbered ±1, ±2, … outward from z = 0 " *
        "(got $(collect(trays))). Fix positions_in_tray_new_wished / the ring-search range.")
    return Float64[ t > 0 ?  ( back_tray_shift_mm  + (t  - 1) * tray_slot_spacing_mm) :
                             -(front_tray_shift_mm + (-t - 1) * tray_slot_spacing_mm)
                    for t in trays ]
end

# Filtra posiciones deseadas sacando las ya ocupadas
function filter_free_trays(wished::AbstractVector{<:Integer},
                           occupied::AbstractVector{<:Integer}=Int[])::Vector{Int}
    free = Int[]
    occ = Set(occupied)
    for p in wished
        if p in occ
            @info "Tray $p ocupado: se omite"
        else
            push!(free, p)
        end
    end
    return free
end

"""
positions_from_rings_mm(wished_trays; occupied_trays=[], shim_radius_mm=235.0,
                        mags_per_segment=7, num_segments=12, angle_per_segment_deg=2*(180-169.68),
                        angular_offset_deg=0.0)

Devuelve un Vector{NTuple{3,Float64}} con posiciones (mm) de TODOS los imanes
según anillos/trays y tu geometría. Aplica la transformación de ejes:
X → -X ; Z → Y ; Y → Z
"""
function positions_from_rings_mm(wished_trays::AbstractVector{<:Integer};
                                 occupied_trays::AbstractVector{<:Integer}=Int[],
                                 shim_radius_mm::Real = 275.0,
                                 mags_per_segment::Integer = 6,
                                 num_segments::Integer = 12,
                                 angle_per_segment_deg::Real = 2*(180 - 169.68),
                                 angular_offset_deg::Real = 0.0,
                                 tray_slot_spacing_mm::Real = 10.0,
                                 front_tray_shift_mm::Real  = 10.0,
                                 back_tray_shift_mm::Real   = 10.0)

    # 1) trays válidos y posiciones axiales (mm)
    free_trays = wished_trays #filter_free_trays(wished_trays, occupied_trays)
    ring_z_mm  = ringpos_from_tray_mm(free_trays;
                                      tray_slot_spacing_mm = tray_slot_spacing_mm,
                                      front_tray_shift_mm  = front_tray_shift_mm,
                                      back_tray_shift_mm   = back_tray_shift_mm)  # mm

    # 2) ángulos por segmento y por imán (grados)
    segment_angles_deg = collect(range(0, stop=360, length=num_segments+1))[1:end-1]
    mag_angles_deg     = collect(range(-angle_per_segment_deg/2,
                                       stop=+angle_per_segment_deg/2,
                                       length=mags_per_segment))
    offset_deg = angular_offset_deg

    # 3) construye posiciones en mm, aplica tu transformación de ejes
    pos = []
    pos_size = length(ring_z_mm) * num_segments * mags_per_segment
    sizehint!(pos, pos_size)

    for z_mm in ring_z_mm
        for seg_deg in segment_angles_deg
            θ_seg = seg_deg + offset_deg
            for mag_deg in mag_angles_deg
                θ = deg2rad(θ_seg + mag_deg)
                x_mm = shim_radius_mm * cos(θ)
                y_mm = shim_radius_mm * sin(θ)                        

                pos_trans = [-x_mm, y_mm, z_mm]
                push!(pos, pos_trans)
            end
        end
    end

    @info "Generadas $(length(ring_z_mm)) rings, $(length(pos)) posiciones."
    return pos
end
