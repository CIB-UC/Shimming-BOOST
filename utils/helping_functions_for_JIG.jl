####### MAGNET INSERT STL  - JULIA + GMSH ########
# Created by: Low Field Project - Pontificia Universidad Católica de Chile
# Date: December 2025 - July 2026



import Gmsh: gmsh
using Dates

# Base-insert output dirs come from pipeline_config.jl. Guarded so this file
# stays usable standalone and won't double-load when CSV_to_STL already did.
@isdefined(MAGNET_INSERTS_DIR) || include(joinpath(@__DIR__, "..", "pipeline_config.jl"))


############ Function to print ############
function printingFunction(texts::Vector{String}, printTime = false; width=60, char='*')
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    ts_line = "[$ts]"

    # Grow the banner so any over-long line (e.g. a long file path) still fits.
    longest = maximum(length, texts)
    printTime && (longest = max(longest, length(ts_line)))
    w = max(width, longest + 4)

    println(char^w)
    for text in texts
        pad = w - length(text) - 2
        left = pad ÷ 2
        right = pad - left
        println(char * " "^left * text * " "^right * char)
    end
    if printTime
        pad = w - length(ts_line) - 2
        left = pad ÷ 2
        right = pad - left
        println(char * " "^left * ts_line * " "^right * char)
    end


    println(char^w)
    println()
end

############ Function to create magnet_insert ############
function create_magnet_insert_stl(lc::Float64 = 0.5, shim_radius_mm = 231, theta_deg = 3.44, mags_per_segment= 7, mag_cover_thickness = 0.5 , extr_cover_thickness = 6, Test::Bool = false)
    # lc : mesh length characteristic
    # shim_radius_mm: radii of the circumference where magnets are placed
    # theta_deg: degrees between magnets inside a tray
    # mag_cover_thickness: Thickness of the magnetic covers
    # extr_cover_thickness: Thickness where magnets will be placed

    
    sm_circDiam = 2 #Small circles Diam: 2mm, just a way to see through the insert
    Big_cirDiam  =12.6 #Big circles Diam: 12.6mm, which are extruded and cover the magnets.
    
    small_radii = shim_radius_mm-6.5 # external radii of insert 
    big_radii = shim_radius_mm+6.5 # internal radii of insert
    printingFunction(["Creating Insert STL with radii: ", string(small_radii), string(big_radii)], true)
    theta = theta_deg * (π / 180) # radians
    half_n  = (mags_per_segment - 1) / 2   # = 3.0 for 7, 2.5 for 6, 2.0 for 5
    
    # Initializing GMSH
    gmsh.initialize()
    gmsh.model.add("Base Mesh")
    factory = gmsh.model.occ

    
    small_circle = factory.addCircle(0, 0, 0, small_radii, -1, π/2 - half_n*theta, π/2 + half_n*theta)
    big_circle   = factory.addCircle(0, 0, 0, big_radii,   -1, π/2 - half_n*theta, π/2 + half_n*theta)
    # small_circle = factory.addCircle(0, 0, 0, small_radii, -1, π/2 - 3*theta, π/2 + 3*theta)
    # big_circle = factory.addCircle(0, 0, 0, big_radii, -1, π/2 - 3*theta, π/2 + 3*theta)


    ########### Left curveline ############
   leftmost_circleCenter   = factory.addPoint(shim_radius_mm*cos(π/2 + half_n*theta), shim_radius_mm*sin(π/2 + half_n*theta), 0, lc)
    leftmost_circleExtreme1 = factory.addPoint(small_radii*cos(π/2 + half_n*theta),    small_radii*sin(π/2 + half_n*theta),    0, lc)
    leftmost_circleExtreme2 = factory.addPoint(big_radii*cos(π/2 + half_n*theta),      big_radii*sin(π/2 + half_n*theta),      0, lc)
    alpha_left = π/2 + half_n*theta
    
    # left_curveline = factory.addCircleArc(leftmost_circleExtreme1, leftmost_circleCenter, leftmost_circleExtreme2)
    # this last bit is a 180° circle arc, very unstable as to where does the arc go

    left_mid = factory.addPoint(
        shim_radius_mm * cos(alpha_left) + 6.5 * (-sin(alpha_left)),
        shim_radius_mm * sin(alpha_left) + 6.5 * (cos(alpha_left)),
        0, lc
    )
    left_arc1 = factory.addCircleArc(leftmost_circleExtreme1, leftmost_circleCenter, left_mid)
    left_arc2 = factory.addCircleArc(left_mid, leftmost_circleCenter, leftmost_circleExtreme2)


    ########### Right curveline ############
    rightmost_circleCenter   = factory.addPoint(shim_radius_mm*cos(π/2 - half_n*theta), shim_radius_mm*sin(π/2 - half_n*theta), 0, lc)
    rightmost_circleExtreme1 = factory.addPoint(small_radii*cos(π/2 - half_n*theta),    small_radii*sin(π/2 - half_n*theta),    0, lc)
    rightmost_circleExtreme2 = factory.addPoint(big_radii*cos(π/2 - half_n*theta),      big_radii*sin(π/2 - half_n*theta),      0, lc)
    # right_curveline = factory.addCircleArc(rightmost_circleExtreme2, rightmost_circleCenter, rightmost_circleExtreme1)
    # this last bit is a 180° circle arc, very unstable as to where does the arc go

    alpha_right = π/2 - half_n*theta    

    right_mid = factory.addPoint(
        shim_radius_mm * cos(alpha_right) + 6.5 * (sin(alpha_right)),
        shim_radius_mm * sin(alpha_right) + 6.5 * (-cos(alpha_right)),
        0, lc
    )
    right_arc1 = factory.addCircleArc(rightmost_circleExtreme2, rightmost_circleCenter, right_mid)
    right_arc2 = factory.addCircleArc(right_mid, rightmost_circleCenter, rightmost_circleExtreme1)

    ########### General Loop ############
    General_Loop = factory.addCurveLoop([small_circle, -left_arc1, -left_arc2, big_circle, right_arc1, right_arc2])
    General_Disk = factory.addPlaneSurface([General_Loop])


    ############ Extruding disks ############
    bigCircles = Int[]
    for i in 1:mags_per_segment
        phi = π/2 + (i - (mags_per_segment + 1)/2) * theta
        tag = factory.addCircle(
            shim_radius_mm*cos(phi),
            shim_radius_mm*sin(phi),
            0.0,
            Big_cirDiam/2
        )
        push!(bigCircles, tag)
    end

    bigDisks = Int[]
    for c in bigCircles
        loop = factory.addCurveLoop([c])
        s = factory.addPlaneSurface([loop])
        push!(bigDisks, s)
    end



    # ############ Extruding Time!! ############
    Mag_covers = factory.extrude([(2, General_Disk)], 0.0, 0.0, mag_cover_thickness)
    factory.extrude([(2, s) for s in bigDisks], 0.0, 0.0, extr_cover_thickness + mag_cover_thickness)
    factory.synchronize()

    vols = gmsh.model.getEntities(3)
    fuse_out = factory.fuse([vols[1]], vols[2:end])
    factory.synchronize()
    vols = gmsh.model.getEntities(3)
    factory.synchronize()

    fused_vols = fuse_out[1]          # list of (3, tag)
    # @show fused_vols



    ########### Removing disks ############
    # # rm_circ1 = factory.addCircle(49.749, 238.96, 0, 6.355) # previously hardcoded
    # # rm_circ2 = factory.addCircle(-49.749, 238.96, 0, 6.355) # previously hardcoded

    # corner_angle = π/2 - 3*theta
    # rm_x = shim_radius_mm * cos(corner_angle)   
    # rm_y = shim_radius_mm * sin(corner_angle)   
    # rm_circ1 = factory.addCircle( rm_x + 1.2*Big_cirDiam/2, rm_y + 1.2*Big_cirDiam/2, 0, Big_cirDiam/2)
    # rm_circ2 = factory.addCircle(-rm_x - 1.2*Big_cirDiam/2, rm_y + 1.2*Big_cirDiam/2, 0, Big_cirDiam/2)


    # removingLoop1 = factory.addCurveLoop([rm_circ1])
    # removingDisk1 = factory.addPlaneSurface([removingLoop1])

    # removingLoop2 = factory.addCurveLoop([rm_circ2])
    # removingDisk2 = factory.addPlaneSurface([removingLoop2])


    # # ########### Intruding small circles ############

    # smallCircles = Int[]
    # for i in -2:4
    #     phi = π/2 + (i-1)*theta
    #     tag = factory.addCircle(
    #         shim_radius_mm*cos(phi),
    #         shim_radius_mm*sin(phi),
    #         0.0,
    #         sm_circDiam/2
    #     )
    #     push!(smallCircles, tag)
    # end

    # smallDisks = Int[]
    # for c in smallCircles
    #     loop = factory.addCurveLoop([c])
    #     s = factory.addPlaneSurface([loop])
    #     push!(smallDisks, s)
    # end
    # factory.synchronize()

    ############ Removing small cilinders ############
    # out_small = factory.extrude(
    #     [(2, s) for s in smallDisks],
    #     0.0, 0.0,
    #     extr_cover_thickness + mag_cover_thickness
    # )
    # smallCutVols = [(d,t) for (d,t) in out_small if d == 3]
    # cut_out = factory.cut(
    #     fused_vols,
    #     smallCutVols,
    #     -1, true, true
    # )
    # factory.synchronize()

    # newMainVol = cut_out[1]    # (3, tag)

    # ########### Removing corner disks (Helps to include in static insert) ############
    # removingCil1 = factory.extrude([(2, removingDisk1)], 0.0, 0.0, extr_cover_thickness + mag_cover_thickness)
    # removingCil2 = factory.extrude([(2, removingDisk2)], 0.0, 0.0, extr_cover_thickness + mag_cover_thickness)
    # rmCutVols = [(d,t) for (d,t) in vcat(removingCil1, removingCil2) if d == 3]
    # Newcut_out = factory.cut(
    #     newMainVol,
    #     rmCutVols,
    #     -1, true, true
    # )
    # factory.synchronize()
    # finalMainVol = Newcut_out[1]  # (3, tag)
    # finalVolTag = finalMainVol[1][2]

    finalMainVol = fuse_out[1]  # (3, tag)
    finalVolTag = finalMainVol[1][2]


    ############ Cleaning up ############
    factory.removeAllDuplicates()
    factory.synchronize()
    for (d,t) in gmsh.model.getEntities(3)
        if t != finalVolTag
            factory.remove([(3,t)], true)
        end
    end
    factory.synchronize()

    
    
    ############ To remove anything past 240 - 1.84############
    clip_z = mag_cover_thickness + extr_cover_thickness
    clip_box = factory.addBox(-50.0, 240.0 - 1.84, -0.1, 100.0, 15.0, clip_z + 0.2)
    factory.synchronize()
    clip_result = factory.cut(
        finalMainVol,
        [(3, clip_box)],
        -1, true, true
    )
    factory.synchronize()
    finalMainVol = clip_result[1]
    finalVolTag = finalMainVol[1][2]
    factory.synchronize()


    ############ To remove anything below 223.5 ############
    clip_z = mag_cover_thickness + extr_cover_thickness
    clip_box_bottom = factory.addBox(-50.0, 193.5- 1.85, -0.1, 100.0, 30.0, clip_z + 0.2)
    factory.synchronize()
    clip_result_bottom = factory.cut(
        finalMainVol,
        [(3, clip_box_bottom)],
        -1, true, true
    )
    factory.synchronize()
    finalMainVol = clip_result_bottom[1]
    finalVolTag = finalMainVol[1][2]


    
    ############ Cleaning up ############
    factory.removeAllDuplicates()
    factory.synchronize()
    for (d,t) in gmsh.model.getEntities(3)
        if t != finalVolTag
            factory.remove([(3,t)], true)
        end
    end
    factory.synchronize()
    
    if Test
        if !("-nopopup" in ARGS)
                gmsh.fltk.run()
        end
    else
        ############ Export CAD (STEP) ############
        outdir = MAGNET_INSERTS_DIR
        mkpath(outdir)  # creates the folder if it doesn't exist
        gmsh.write(joinpath(outdir, "MagnetInsert_radius_$(shim_radius_mm)radius_$(mags_per_segment)mags.step"))


        ############ Creating and Exporting Mesh as STL ############
        finalSurfaces = gmsh.model.getBoundary([(3, finalVolTag)], true, false, false)
        gmsh.model.mesh.clear()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)   # linear triangles (best for STL)

        ############ Generate mesh ############
        gmsh.model.mesh.generate(2)

        ############ Sanity Check ############
        types, elemTags, nodeTags = gmsh.model.mesh.getElements(2)
        println("2D element types: ", types)
        println("Total 2D elements: ", sum(length.(elemTags)))

        gmsh.write(joinpath(outdir,"MagnetInsert_radius_$(shim_radius_mm)radius_$(mags_per_segment)mags.stl"))
    end

    gmsh.finalize()
    printingFunction(["Magnet Insert created!!"], false)

end

############ Function to create static insert - OSII v2.1 ############
function create_static_insert_stl(lc::Float64 = 0.5, shim_radius_mm = 231, theta_deg = 3.44, removing_depth = 6.1, mags_per_segment = 7, Test::Bool = false)
    # This function creates the static insert for the Trays used in Low Field MRI UC Project, for the OSII 2.1 Halbach Array Magnet.
    # Its created based off the shimming parameters used

    theta = theta_deg * (π / 180) # radians
    half_n = (mags_per_segment - 1) / 2
    Removing_disks_diam =12.8 # which are the ones that allow the magnet insert to fit.
    static_insert_width = 7.1 # in mm

    scanner_contact_width = 96 # in mm 
    main_body_width = 100 # in mm
    tray_contact_width= 92 # in mm

    main_to_scanner_change_height = 2.3 # in mm
    main_body_height = 10.5 # in mm
    tray_to_main_change_height = 3.8 #in mm

    Tray_short_radii =221.7094 
    # Tray_long_radii = 238.2094


    # Initializing GMSH
    gmsh.initialize()
    gmsh.model.add("Base Mesh")
    factory = gmsh.model.occ


    ############ Main Polygon ############
    p1 = factory.addPoint(scanner_contact_width/2, Tray_short_radii, 0.0, lc)
    p2 = factory.addPoint(main_body_width/2, Tray_short_radii + main_to_scanner_change_height, 0.0, lc)
    p3 = factory.addPoint(main_body_width/2, Tray_short_radii + main_to_scanner_change_height + main_body_height, 0.0, lc)
    p4 = factory.addPoint(tray_contact_width/2, Tray_short_radii + main_to_scanner_change_height + main_body_height + tray_to_main_change_height, 0.0, lc)
    p5 = factory.addPoint(-tray_contact_width/2, Tray_short_radii + main_to_scanner_change_height + main_body_height + tray_to_main_change_height, 0.0, lc)
    p6 = factory.addPoint(-main_body_width/2, Tray_short_radii + main_to_scanner_change_height + main_body_height, 0.0, lc)
    p7 = factory.addPoint(-main_body_width/2, Tray_short_radii + main_to_scanner_change_height, 0.0, lc)
    p8 = factory.addPoint(-scanner_contact_width/2, Tray_short_radii, 0.0, lc)


    l1 = factory.addLine(p1, p2)
    l2 = factory.addLine(p2, p3)
    l3 = factory.addLine(p3, p4)
    l4 = factory.addLine(p4, p5)
    l5 = factory.addLine(p5, p6)
    l6 = factory.addLine(p6, p7)
    l7 = factory.addLine(p7, p8)
    l8 = factory.addLine(p8, p1)

    loop    = factory.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8])
    surface = factory.addPlaneSurface([loop])
    factory.synchronize()

    
    main_vol = factory.extrude([(2, surface)], 0.0, 0.0, static_insert_width)
    factory.synchronize()


    ############ Removing Disks ############
    Removing_Circles = Int[]
    for i in 1:mags_per_segment
        phi = π/2 + (i - (mags_per_segment + 1)/2) * theta
        tag = factory.addCircle(
            shim_radius_mm*cos(phi),
            shim_radius_mm*sin(phi),
            0.0,
            Removing_disks_diam/2
        )
        push!(Removing_Circles, tag)
    end

    Removing_Disks = Int[]
    for c in Removing_Circles
        loop = factory.addCurveLoop([c])
        s = factory.addPlaneSurface([loop])
        push!(Removing_Disks, s)
    end

    Removing_Disks_extruded = factory.extrude([(2, s) for s in Removing_Disks], 0.0, 0.0, removing_depth)
    factory.synchronize()


    ########### Removing Disks ############
    
    Static_Cutting = [(d,t) for (d,t) in Removing_Disks_extruded if d == 3]
    main_vol_vols = [(d,t) for (d,t) in main_vol if d == 3]
    cut_out = factory.cut(
        main_vol_vols,
        Static_Cutting,
        -1, true, true
    )
    factory.synchronize()

    newMainVol = cut_out[1]    # (3, tag)
    finalVolTag = newMainVol[1][2]



    ############ Cleaning up ############
    factory.removeAllDuplicates()
    factory.synchronize()
    for (d,t) in gmsh.model.getEntities(3)
        if t != finalVolTag
            factory.remove([(3,t)], true)
        end
    end
    factory.synchronize()


    if Test
        if !("-nopopup" in ARGS)
                gmsh.fltk.run()
        end
    else
        ############ Exporting! ############
        outdir = STATIC_INSERTS_DIR
        mkpath(outdir)  # creates the folder if it doesn't exist
        
        ############ Export CAD (STEP) ############
        gmsh.write(joinpath(outdir, "StaticInsert_radius_$(shim_radius_mm)radius_$(mags_per_segment)mags.step"))


        ############ Creating and Exporting Mesh as STL ############
        finalSurfaces = gmsh.model.getBoundary([(3, finalVolTag)], true, false, false)
        gmsh.model.mesh.clear()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)   # linear triangles (best for STL)

        ############ Generate mesh ############
        gmsh.model.mesh.generate(2)

        ############ Sanity Check ############
        types, elemTags, nodeTags = gmsh.model.mesh.getElements(2)
        println("2D element types: ", types)
        println("Total 2D elements: ", sum(length.(elemTags)))

        gmsh.write(joinpath(outdir, "StaticInsert_radius_$(shim_radius_mm)radius_$(mags_per_segment)mags.stl"))
    end

    gmsh.finalize()
    printingFunction(["Static Insert created!!"], false)
end

############ Function to create static insert - OSII v1.2 ############
function create_static_insert_stl_forJosh(lc::Float64 = 0.5, shim_radius_mm = 275, theta_deg = 3.44, removing_depth = 6.1, mags_per_segment = 7, Test::Bool = false)
    # This function creates the static insert for the Trays used in the OSII 1 Halbach Array Magnet.
    # Its created based off the shimming parameters used

    theta = theta_deg * (π / 180) # radians
    half_n = (mags_per_segment - 1) / 2
    Removing_disks_diam = 12.9 # which are the ones that allow the magnet insert to fit.
    static_insert_width = 7.1 # in mm

    mainbody_arc = 21.4 * (π / 180) # radians
    extrusion_arc_out = 23 * (π / 180) # radians
    extrusion_arc_in = 25 * (π / 180) # radians

    Tray_short_radii = 269.0
    Tray_long_radii = 285.6
    extrusion_radii_1 = 269.0 + 9.0
    extrusion_radii_2 = 269.0 + 9.7
    extrusion_radii_3 = 269.0 + 4

    main_bod_depth = 7
    extrusion_depth = 4.6
    extrusion_height = 1.0

    # Initializing GMSH
    gmsh.initialize()
    gmsh.model.add("Base Mesh")
    factory = gmsh.model.occ

    
    ############ Main Bod ############ 
    center_z0 = factory.addPoint(0.0, 0.0, 0.0, lc)

    p1 = factory.addPoint(-Tray_short_radii * sin(mainbody_arc/2), Tray_short_radii * cos(mainbody_arc/2), 0.0, lc)
    p2 = factory.addPoint( Tray_short_radii * sin(mainbody_arc/2), Tray_short_radii * cos(mainbody_arc/2), 0.0, lc)
    p3 = factory.addPoint(-Tray_long_radii  * sin(mainbody_arc/2), Tray_long_radii  * cos(mainbody_arc/2), 0.0, lc)
    p4 = factory.addPoint( Tray_long_radii  * sin(mainbody_arc/2), Tray_long_radii  * cos(mainbody_arc/2), 0.0, lc)


    arc_mainbod_short = factory.addCircleArc(p1, center_z0, p2)
    line_mainbod_right = factory.addLine(p2, p4)
    arc_mainbod_long  = factory.addCircleArc(p3, center_z0, p4)
    line_mainbod_left = factory.addLine(p3, p1)

    mainbod_loop = factory.addCurveLoop([arc_mainbod_short, line_mainbod_right, -arc_mainbod_long, line_mainbod_left])
    mainbod_surface_tag = factory.addPlaneSurface([mainbod_loop])
    mainbod_vol_tag = factory.extrude([(2,mainbod_surface_tag)], 0, 0, main_bod_depth)


    ############ Extrusion Bod ############ 
    center_z1 = factory.addPoint(0.0, 0.0, extrusion_height, lc)

    e1 = factory.addPoint(-Tray_short_radii * sin(extrusion_arc_in/2),  Tray_short_radii * cos(extrusion_arc_in/2),  extrusion_height, lc)
    e2 = factory.addPoint( Tray_short_radii * sin(extrusion_arc_in/2),  Tray_short_radii * cos(extrusion_arc_in/2),  extrusion_height, lc)
    e3 = factory.addPoint(-extrusion_radii_1 * sin(extrusion_arc_in/2), extrusion_radii_1 * cos(extrusion_arc_in/2), extrusion_height, lc)
    e4 = factory.addPoint(-extrusion_radii_1 * sin(extrusion_arc_out/2),extrusion_radii_1 * cos(extrusion_arc_out/2),extrusion_height, lc)
    e5 = factory.addPoint(-Tray_long_radii  * sin(extrusion_arc_out/2), Tray_long_radii  * cos(extrusion_arc_out/2), extrusion_height, lc)
    e6 = factory.addPoint( Tray_long_radii  * sin(extrusion_arc_out/2), Tray_long_radii  * cos(extrusion_arc_out/2), extrusion_height, lc)
    e7 = factory.addPoint( extrusion_radii_2 * sin(extrusion_arc_out/2),extrusion_radii_2 * cos(extrusion_arc_out/2), extrusion_height, lc)
    e8 = factory.addPoint( extrusion_radii_3 * sin(extrusion_arc_in/2), extrusion_radii_3 * cos(extrusion_arc_in/2), extrusion_height, lc)

    line_extrbod_1 = factory.addCircleArc(e1, center_z1, e2)   # inner arc
    line_extrbod_2 = factory.addLine(e1, e3)
    line_extrbod_3 = factory.addCircleArc(e3, center_z1, e4)   # left intermediate arc (~1°)
    line_extrbod_4 = factory.addLine(e4, e5)
    line_extrbod_5 = factory.addCircleArc(e5, center_z1, e6)   # outer arc
    line_extrbod_6 = factory.addLine(e6, e7)
    line_extrbod_7 = factory.addLine(e7, e8)
    line_extrbod_8 = factory.addLine(e8, e2)
    
    extrbod_loop = factory.addCurveLoop([-line_extrbod_1, line_extrbod_2, line_extrbod_3, line_extrbod_4, line_extrbod_5, line_extrbod_6, line_extrbod_7, line_extrbod_8])
    extrbod_surface_tag = factory.addPlaneSurface([extrbod_loop])
    extrbod_vol_tag = factory.extrude([(2,extrbod_surface_tag)], 0, 0, extrusion_depth)
    factory.synchronize()

    vols = gmsh.model.getEntities(3)
    main_vol = factory.fuse([vols[1]], [vols[2]])
    factory.synchronize()
    main_vol = main_vol[1]          # list of (3, tag)
    
    factory.synchronize()


    ############ Removing Disks ############
    Removing_Circles = Int[]
    for i in 1:mags_per_segment
        phi = π/2 + (i - (mags_per_segment + 1)/2) * theta
        tag = factory.addCircle(
            shim_radius_mm*cos(phi),
            shim_radius_mm*sin(phi),
            0.0,
            Removing_disks_diam/2
        )
        push!(Removing_Circles, tag)
    end

    Removing_Disks = Int[]
    for c in Removing_Circles
        loop = factory.addCurveLoop([c])
        s = factory.addPlaneSurface([loop])
        push!(Removing_Disks, s)
    end

    Removing_Disks_extruded = factory.extrude([(2, s) for s in Removing_Disks], 0.0, 0.0, removing_depth)
    factory.synchronize()


    ########### Removing Disks ############
    
    Static_Cutting = [(d,t) for (d,t) in Removing_Disks_extruded if d == 3]
    main_vol_vols = [(d,t) for (d,t) in main_vol if d == 3]
    cut_out = factory.cut(
        main_vol_vols,
        Static_Cutting,
        -1, true, true
    )
    factory.synchronize()

    newMainVol = cut_out[1]    # (3, tag)
    finalVolTag = newMainVol[1][2]



    ############ Cleaning up ############
    factory.removeAllDuplicates()
    factory.synchronize()
    for (d,t) in gmsh.model.getEntities(3)
        if t != finalVolTag
            factory.remove([(3,t)], true)
        end
    end
    factory.synchronize()


    if Test
        if !("-nopopup" in ARGS)
                gmsh.fltk.run()
        end
    else
        ############ Exporting! ############
        outdir = STATIC_INSERTS_DIR
        mkpath(outdir)  # creates the folder if it doesn't exist
        
        ############ Export CAD (STEP) ############
        gmsh.write(joinpath(outdir, "StaticInsert_radius_$(shim_radius_mm)radius_$(mags_per_segment)mags.step"))


        ############ Creating and Exporting Mesh as STL ############
        finalSurfaces = gmsh.model.getBoundary([(3, finalVolTag)], true, false, false)
        gmsh.model.mesh.clear()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)   # linear triangles (best for STL)

        ############ Generate mesh ############
        gmsh.model.mesh.generate(2)

        ############ Sanity Check ############
        types, elemTags, nodeTags = gmsh.model.mesh.getElements(2)
        println("2D element types: ", types)
        println("Total 2D elements: ", sum(length.(elemTags)))

        gmsh.write(joinpath(outdir, "StaticInsert_radius_$(shim_radius_mm)radius_$(mags_per_segment)mags.stl"))
    end

    gmsh.finalize()
    printingFunction(["Josh's Static Insert created!!"],false) 
end

############ Function to create square ############
function make_square(factory, center::NTuple{2,Float64}, L::Real, θ::Real, lc::Float64, total_thickness = 6.5)
    # total_thickness : height of total insert
    cx, cy = center
    L = float(L)
    h = L/2

    # Create axis-aligned square centered at origin, then move+rotate
    rect_tag = factory.addRectangle(-h, -h, total_thickness, L, L) 
    ent_rect = [(2, rect_tag)]
    factory.translate(ent_rect, cx, cy, 0.0)
    factory.rotate(ent_rect, cx, cy, 0, 0.0, 0.0, 1.0, float(θ))
    factory.extrude(ent_rect, 0,0,-(total_thickness-0.3)) # 6.2 in order to not fully cut the insert

    circle= factory.addCircle(cx + (0.8 + h)* cos(θ), cy + (0.8 + h) * sin(θ), total_thickness, 0.8)
    circle_loop = factory.addCurveLoop([circle])
    circle_tag = factory.addPlaneSurface([circle_loop])
    ent_circle = [(2, circle_tag)]
    factory.extrude(ent_circle, 0,0,-(total_thickness-0.3))
    factory.synchronize()  

end

############ Function to create hexadecimal number ############
function make_numb_hexadecimal(factory, number, center::NTuple{2,Float64}, lc::Float64, scale::Float64 = 1.3)
    # Auxiliary function, in able to create the hexadecimal labels
    if number == 0
        ce_zero_1 = factory.addPoint(center[1]+1*scale, center[2], 0.0, lc)
        zero_1 = factory.addPoint(center[1], center[2]-3.0*scale, 0.0, lc)
        zero_2 = factory.addPoint(center[1]+2.3*scale, center[2], 0.0, lc)
        zero_3 = factory.addPoint(center[1], center[2]+3.0*scale, 0.0, lc)
        zero_4 = factory.addPoint(center[1]-2.3*scale, center[2], 0.0, lc)
        zero_corner1 = factory.addPoint(center[1]+1.65*scale, center[2]-2.3*scale, 0.0, lc)
        zero_corner2 = factory.addPoint(center[1]+1.65*scale, center[2]+2.3*scale, 0.0, lc)
        zero_corner3 = factory.addPoint(center[1]-1.65*scale, center[2]+2.3*scale, 0.0, lc)
        zero_corner4 = factory.addPoint(center[1]-1.65*scale, center[2]-2.3*scale, 0.0, lc)
        arczero_1 = factory.addSpline([zero_1, zero_corner1, zero_2])
        arczero_2 = factory.addSpline([zero_2, zero_corner2, zero_3])
        arczero_3 = factory.addSpline([zero_3, zero_corner3, zero_4])
        arczero_4 = factory.addSpline([zero_4, zero_corner4, zero_1])
        bigloop_0 = factory.addCurveLoop([arczero_1, arczero_2, arczero_3, arczero_4])

        zero_5 = factory.addPoint(center[1], center[2]-2.1*scale, 0.0, lc)
        zero_6 = factory.addPoint(center[1]+1*scale, center[2], 0.0, lc)
        zero_7 = factory.addPoint(center[1], center[2]+2.1*scale, 0.0, lc)
        zero_8 = factory.addPoint(center[1]-1*scale, center[2], 0.0, lc)
        zero_corner5 = factory.addPoint(center[1]+0.6*scale, center[2]-1.6*scale, 0.0, lc)
        zero_corner6 = factory.addPoint(center[1]+0.6*scale, center[2]+1.6*scale, 0.0, lc)
        zero_corner7 = factory.addPoint(center[1]-0.6*scale, center[2]+1.6*scale, 0.0, lc)
        zero_corner8 = factory.addPoint(center[1]-0.6*scale, center[2]-1.6*scale, 0.0, lc)
        arczero_5 = factory.addSpline([zero_5, zero_corner5, zero_6])
        arczero_6 = factory.addSpline([zero_6, zero_corner6, zero_7])
        arczero_7 = factory.addSpline([zero_7, zero_corner7, zero_8])
        arczero_8 = factory.addSpline([zero_8, zero_corner8, zero_5])
        smallloop_0 = factory.addCurveLoop([arczero_5, arczero_6, arczero_7, arczero_8])
        
        return factory.addPlaneSurface([bigloop_0, smallloop_0])

    elseif number == 1
        one_1 = factory.addPoint(center[1]+0.55 *scale, center[2]-3.0*scale, 0.0, lc)
        one_2 = factory.addPoint(center[1]+0.55 *scale, center[2]+1.3 *scale, 0.0, lc)
        one_3 = factory.addPoint(center[1]+2 *scale, center[2]+0.4 *scale, 0.0, lc)
        one_4 = factory.addPoint(center[1]+2 *scale, center[2]+1.4 *scale, 0.0, lc)
        one_5 = factory.addPoint(center[1]+0.55 *scale, center[2]+3 *scale, 0.0, lc)
        one_6 = factory.addPoint(center[1]-0.55 *scale, center[2]+3 *scale, 0.0, lc)
        one_7 = factory.addPoint(center[1]-0.55 *scale, center[2]-3 *scale, 0.0, lc)
        line1_1 = factory.addLine(one_1, one_2)
        line1_2 = factory.addLine(one_2, one_3)
        line1_3 = factory.addLine(one_3, one_4)
        line1_4 = factory.addLine(one_4, one_5)
        line1_5 = factory.addLine(one_5, one_6)
        line1_6 = factory.addLine(one_6, one_7)
        line1_7 = factory.addLine(one_7, one_1)
        loop_1 = factory.addCurveLoop([line1_1, line1_2, line1_3, line1_4, line1_5, line1_6, line1_7])
        return factory.addPlaneSurface([loop_1])
    
    elseif number == 2
        two_1 = factory.addPoint(center[1]+2.0*scale, center[2]-3.0*scale, 0.0, lc)
        two_2 = factory.addPoint(center[1]+1.65*scale, center[2]-2.0*scale, 0.0, lc)
        two_23 = factory.addPoint(center[1], center[2]-0.1*scale, 0.0, lc)
        two_3 = factory.addPoint(center[1]-0.77*scale, center[2]+0.75*scale, 0.0, lc)
        ce_two_1 = factory.addPoint(center[1]-0.67*scale, center[2]+1.86*scale, 0.0, lc)
        ce_two_2 = factory.addPoint(center[1]-0.26*scale, center[2]+2.05*scale, 0.0, lc)
        two_4 = factory.addPoint(center[1]+0.77*scale, center[2]+1*scale, 0.0, lc)
        two_5 = factory.addPoint(center[1]+1.9*scale, center[2]+1.1*scale, 0.0, lc)
        ce_two_3 = factory.addPoint(center[1], center[2]+3.0*scale, 0.0, lc)
        ce_two_4 = factory.addPoint(center[1]-1.52*scale, center[2]+2.54*scale, 0.0, lc)
        two_6 = factory.addPoint(center[1]-1.9*scale, center[2]+0.81*scale, 0.0, lc)
        two_7 = factory.addPoint(center[1]-0.97*scale, center[2]-0.73*scale, 0.0, lc)
        two_8 = factory.addPoint(center[1]+0.34*scale, center[2]-2.0*scale, 0.0, lc)
        two_9 = factory.addPoint(center[1]-2*scale, center[2]-2.0*scale, 0.0, lc)
        two_10 = factory.addPoint(center[1]-2*scale, center[2]-3.0*scale, 0.0, lc)
        line2_1 = factory.addLine(two_1, two_2)
        line2_2 = factory.addSpline([two_2, two_23, two_3])
        line2_3 = factory.addSpline([two_3, ce_two_1, ce_two_2, two_4])
        line2_4 = factory.addLine(two_4, two_5)
        line2_5 = factory.addSpline([two_5, ce_two_3, ce_two_4, two_6])
        line2_6 = factory.addSpline([two_6, two_7,two_8 ])
        line2_7 = factory.addLine(two_8, two_9)
        line2_8 = factory.addLine(two_9, two_10)
        line2_9 = factory.addLine(two_10, two_1)
        loop_2 = factory.addCurveLoop([line2_1, line2_2, line2_3, line2_4, line2_5, line2_6, line2_7, line2_8, line2_9])
        return factory.addPlaneSurface([loop_2])
    
    elseif number == 3
        ce_three_1 = factory.addPoint(center[1], center[2]+1.5*scale, 0.0, lc)
        ce_three_2 = factory.addPoint(center[1], center[2]-1.5*scale, 0.0, lc)
        three_1 = factory.addPoint(center[1]+0.3*scale, center[2]+0.3*scale, 0.0, lc)
        three_2 = factory.addPoint(center[1]-0.3*scale, center[2]+0.7*scale, 0.0, lc)
        three_3 = factory.addPoint(center[1]-0.8*scale, center[2]+1.5*scale, 0.0, lc)
        three_4 = factory.addPoint(center[1], center[2]+2.3*scale, 0.0, lc)
        three_5 = factory.addPoint(center[1]+0.8*scale, center[2]+1.5*scale, 0.0, lc)
        three_6 = factory.addPoint(center[1]+1.5*scale, center[2]+1.5*scale, 0.0, lc)
        three_7 = factory.addPoint(center[1], center[2]+3*scale, 0.0, lc)
        three_8 = factory.addPoint(center[1]-0.9*scale, center[2]+0.3*scale, 0.0, lc)
        three_89 = factory.addPoint(center[1]-0.6*scale, center[2], 0.0, lc)
        three_9 = factory.addPoint(center[1]-0.9*scale, center[2]-0.3*scale, 0.0, lc)
        three_10 = factory.addPoint(center[1], center[2]-3*scale, 0.0, lc)
        three_11 = factory.addPoint(center[1]+1.5*scale, center[2]-1.5*scale, 0.0, lc)
        three_12 = factory.addPoint(center[1]+0.8*scale, center[2]-1.5*scale, 0.0, lc)
        three_13 = factory.addPoint(center[1], center[2]-2.3*scale, 0.0, lc)
        three_14 = factory.addPoint(center[1]-0.8*scale, center[2]-1.5*scale, 0.0, lc)
        three_15 = factory.addPoint(center[1]-0.3*scale, center[2]-0.7*scale, 0.0, lc)
        three_16 = factory.addPoint(center[1]+0.3*scale, center[2]-0.3*scale, 0.0, lc)

        line3_1 = factory.addSpline([three_1, three_2, three_3])
        line3_3 = factory.addCircleArc(three_3, ce_three_1, three_4) 
        line3_4 = factory.addCircleArc(three_4, ce_three_1, three_5)
        line3_5 = factory.addLine(three_5, three_6)
        line3_6 = factory.addCircleArc(three_6, ce_three_1, three_7)
        line3_7 = factory.addCircleArc(three_7, ce_three_1, three_8)
        line3_889 = factory.addLine(three_8, three_89)
        line3_899 = factory.addLine(three_89, three_9)
        line3_9 = factory.addCircleArc(three_9, ce_three_2, three_10)
        line3_10 = factory.addCircleArc(three_10, ce_three_2, three_11)
        line3_11 = factory.addLine(three_11, three_12)
        line3_12 = factory.addCircleArc(three_12, ce_three_2, three_13)
        line3_13 = factory.addCircleArc(three_13, ce_three_2, three_14)
        line3_14 = factory.addSpline([three_14,three_15, three_16])
        line3_15 = factory.addLine(three_16, three_1)
        loop_3 = factory.addCurveLoop([line3_1, line3_3, line3_4, line3_5, line3_6, line3_7, line3_889, line3_899, line3_9, line3_10, line3_11, line3_12, line3_13, line3_14, line3_15])
        return factory.addPlaneSurface([loop_3])

    elseif number ==4
        four_1 =  factory.addPoint(center[1]-0.3*scale, center[2]-3*scale, 0.0, lc)
        four_2 =  factory.addPoint(center[1]-0.3*scale, center[2]-1.8*scale, 0.0, lc)
        four_3 =  factory.addPoint(center[1]+2.15*scale, center[2]-1.8*scale, 0.0, lc)
        four_4 =  factory.addPoint(center[1]+2.15*scale, center[2]-0.67*scale, 0.0, lc)
        four_5 =  factory.addPoint(center[1]-0.3*scale, center[2]+3*scale, 0.0, lc)
        four_7 =  factory.addPoint(center[1]-1*scale, center[2]+3*scale, 0.0, lc)
        four_8 =  factory.addPoint(center[1]-1*scale, center[2]-0.67*scale, 0.0, lc)
        four_9 =  factory.addPoint(center[1]-2.15*scale, center[2]-0.67*scale, 0.0, lc)
        four_10 =  factory.addPoint(center[1]-2.15*scale, center[2]-1.8*scale, 0.0, lc)
        four_11 =  factory.addPoint(center[1]-1*scale, center[2]-1.8*scale, 0.0, lc)
        four_12 =  factory.addPoint(center[1]-1*scale, center[2]-3*scale, 0.0, lc)

        line4_1 = factory.addLine(four_1, four_2)
        line4_2 = factory.addLine(four_2, four_3)
        line4_3 = factory.addLine(four_3, four_4)
        line4_4 = factory.addLine(four_4, four_5)
        line4_5 = factory.addLine(four_5, four_7)
        line4_7 = factory.addLine(four_7, four_8)
        line4_8 = factory.addLine(four_8, four_9)
        line4_9 = factory.addLine(four_9, four_10)
        line4_10 = factory.addLine(four_10, four_11)
        line4_11 = factory.addLine(four_11, four_12)
        line4_12 = factory.addLine(four_12, four_1)
        bigloop_4 = factory.addCurveLoop([line4_1, line4_2, line4_3, line4_4, line4_5, line4_7, line4_8, line4_9, line4_10, line4_11, line4_12])

        four13 = factory.addPoint(center[1]-0.3*scale, center[2]-0.67*scale, 0.0, lc)
        four14 = factory.addPoint(center[1]+1.5*scale, center[2]-0.67*scale, 0.0, lc)
        four15 = factory.addPoint(center[1]-0.3*scale, center[2]+2.02*scale, 0.0, lc)
        line13 = factory.addLine(four13, four14)
        line14 = factory.addLine(four14, four15)
        line15 = factory.addLine(four15, four13)
        smallloop_4 = factory.addCurveLoop([line13, line14, line15])
        return factory.addPlaneSurface([bigloop_4, smallloop_4])
    
    elseif number ==5
        five_1 = factory.addPoint(center[1]+0.8*scale, center[2]-1.3*scale, 0.0, lc)
        five_2 = factory.addPoint(center[1], center[2]-2.1*scale, 0.0, lc)
        five_3 = factory.addPoint(center[1]-0.6*scale, center[2]-1.7*scale, 0.0, lc)
        five_4 = factory.addPoint(center[1]-0.6*scale, center[2]-0.4*scale, 0.0, lc)
        five_5 = factory.addPoint(center[1], center[2]+0.3*scale, 0.0, lc)
        five_6 = factory.addPoint(center[1]+0.8*scale, center[2], 0.0, lc)
        five_7 = factory.addPoint(center[1]+1.5*scale, center[2], 0.0, lc)
        five_8 = factory.addPoint(center[1]+1.3*scale, center[2]+3*scale, 0.0, lc)
        five_9 = factory.addPoint(center[1]-1.5*scale, center[2]+3*scale, 0.0, lc)
        five_10 = factory.addPoint(center[1]-1.5*scale, center[2]+2*scale, 0.0, lc)
        five_11 = factory.addPoint(center[1]+0.3*scale, center[2]+2*scale, 0.0, lc)
        five_12 = factory.addPoint(center[1]+0.45*scale, center[2]+1*scale, 0.0, lc)
        five_123 = factory.addPoint(center[1]-0.85*scale, center[2]+0.6*scale, 0.0, lc)
        five_13 = factory.addPoint(center[1]-1.5*scale, center[2]-1.05*scale, 0.0, lc)
        five_134 = factory.addPoint(center[1]-1.2*scale, center[2]-2.25*scale, 0.0, lc)
        five_14 = factory.addPoint(center[1], center[2]-3*scale, 0.0, lc)
        five_15 = factory.addPoint(center[1]+1.5*scale, center[2]-1.3*scale, 0.0, lc)

        line5_1 = factory.addSpline([five_1, five_2, five_3, five_4, five_5, five_6])
        line5_2 = factory.addLine(five_6, five_7)
        line5_3 = factory.addLine(five_7, five_8)
        line5_4 = factory.addLine(five_8, five_9)
        line5_5 = factory.addLine(five_9, five_10)
        line5_6 = factory.addLine(five_10, five_11)
        line5_7 = factory.addLine(five_11, five_12)
        line5_8 = factory.addSpline([five_12, five_123,five_13, five_134, five_14, five_15])
        line5_9 = factory.addLine(five_15, five_1)
        loop_5 = factory.addCurveLoop([line5_1, line5_2, line5_3, line5_4, line5_5, line5_6, line5_7, line5_8, line5_9])
        return factory.addPlaneSurface([loop_5])

    elseif number ==6
        six_1 = factory.addPoint(center[1], center[2]-2*scale, 0.0, lc)
        six_2 = factory.addPoint(center[1]-0.73*scale, center[2]-1.15*scale, 0.0, lc)
        six_3 = factory.addPoint(center[1]-0.7*scale, center[2]-0.49*scale, 0.0, lc)
        six_4 = factory.addPoint(center[1], center[2]+0.2*scale, 0.0, lc)
        six_5 = factory.addPoint(center[1]+0.7*scale, center[2]-0.49*scale, 0.0, lc)
        six_6 = factory.addPoint(center[1]+0.73*scale, center[2]-1.15*scale, 0.0, lc)
        six_7 = factory.addPoint(center[1]-1.82*scale, center[2]+1.63*scale, 0.0, lc)
        six_8 = factory.addPoint(center[1]-0.1*scale, center[2]+3*scale, 0.0, lc)
        six_9 = factory.addPoint(center[1]+1.47*scale, center[2]+2.4*scale, 0.0, lc)
        six_10 = factory.addPoint(center[1]+2.08*scale, center[2], 0.0, lc)
        six_11 = factory.addPoint(center[1]+1.76*scale, center[2]-1.76*scale, 0.0, lc)
        six_12 = factory.addPoint(center[1], center[2]-3*scale, 0.0, lc)
        six_13 = factory.addPoint(center[1]-1.82*scale, center[2]-1.72*scale, 0.0, lc)
        six_14 = factory.addPoint(center[1]-1.89*scale, center[2]-0.57*scale, 0.0, lc)
        six_15 = factory.addPoint(center[1]-0.96*scale, center[2]+0.82*scale, 0.0, lc)
        six_16 = factory.addPoint(center[1]+0.12*scale, center[2]+0.96*scale, 0.0, lc)
        six_17 = factory.addPoint(center[1]+0.89*scale, center[2]+0.49*scale, 0.0, lc)
        six_18 = factory.addPoint(center[1]+0.81*scale, center[2]+1.2*scale, 0.0, lc)
        six_19 = factory.addPoint(center[1]+0.3*scale, center[2]+2.09*scale, 0.0, lc)
        six_20 = factory.addPoint(center[1]-0.49*scale, center[2]+2.01*scale, 0.0, lc)
        six_21 = factory.addPoint(center[1]-0.7*scale, center[2]+1.5*scale, 0.0, lc)

        line6_1 = factory.addSpline([six_1, six_2, six_3, six_4, six_5, six_6, six_1])
        line6_2 = factory.addSpline([six_7, six_8, six_9, six_10, six_11, six_12, six_13, six_14, six_15, six_16, six_17])
        line6_3 = factory.addSpline([six_17, six_18, six_19, six_20, six_21])
        line6_4 = factory.addLine(six_21, six_7)
        bigloop_6 = factory.addCurveLoop([line6_2, line6_3, line6_4])
        smallloop_6 = factory.addCurveLoop([line6_1])
        return factory.addPlaneSurface([bigloop_6, smallloop_6])

    elseif number ==7
        seven_1 = factory.addPoint(center[1]+2*scale, center[2]+3*scale, 0.0, lc)
        seven_2 = factory.addPoint(center[1]-2*scale, center[2]+3*scale, 0.0, lc)
        seven_3 = factory.addPoint(center[1]-2*scale, center[2]+2*scale, 0.0, lc)
        seven_4 = factory.addPoint(center[1]-0.01*scale, center[2]-3*scale, 0.0, lc)
        seven_5 = factory.addPoint(center[1]+1.2*scale, center[2]-3*scale, 0.0, lc)
        seven_6 = factory.addPoint(center[1]-0.7*scale, center[2]+2*scale, 0.0, lc)
        seven_7 = factory.addPoint(center[1]+2*scale, center[2]+2*scale, 0.0, lc)

        line7_1 = factory.addLine(seven_1, seven_2)
        line7_2 = factory.addLine(seven_2, seven_3)
        line7_3 = factory.addLine(seven_3, seven_4)
        line7_4 = factory.addLine(seven_4, seven_5)
        line7_5 = factory.addLine(seven_5, seven_6)
        line7_6 = factory.addLine(seven_6, seven_7)
        line7_7 = factory.addLine(seven_7, seven_1)
        loop_7 = factory.addCurveLoop([line7_1, line7_2, line7_3, line7_4, line7_5, line7_6, line7_7])
        return factory.addPlaneSurface([loop_7])

    elseif number ==8
        ceeight_1 = factory.addPoint(center[1], center[2]-1.5*scale, 0.0, lc)
        eight_1 = factory.addPoint(center[1] +2*scale, center[2]-1.5*scale, 0.0, lc)
        eight_2 = factory.addPoint(center[1] -2*scale, center[2]-1.5*scale, 0.0, lc)
        eight_13 = factory.addPoint(center[1] +1.5*scale, center[2]-0.5*scale, 0.0, lc)
        eight_3 = factory.addPoint(center[1] +0.9*scale, center[2]+0.1*scale, 0.0, lc)
        eight_24 = factory.addPoint(center[1] -1.5*scale, center[2]-0.5*scale, 0.0, lc)
        eight_4 = factory.addPoint(center[1] -0.9*scale, center[2]+0.1*scale, 0.0, lc)
        eight_5 = factory.addPoint(center[1]-1.8*scale, center[2]+1.5*scale, 0.0, lc)
        eight_56 = factory.addPoint(center[1]-1*scale, center[2]+2.7*scale, 0.0, lc)
        eight_6 = factory.addPoint(center[1], center[2]+3*scale, 0.0, lc)
        eight_67 = factory.addPoint(center[1]+1*scale, center[2]+2.7*scale, 0.0, lc)
        eight_7 = factory.addPoint(center[1]+1.8*scale, center[2]+1.5*scale, 0.0, lc)

        line8_1 = factory.addCircleArc(eight_1, ceeight_1, eight_2)
        line8_13 = factory.addSpline([eight_3, eight_13, eight_1])
        line8_24 = factory.addSpline([eight_2, eight_24, eight_4])
        line8_2 = factory.addSpline([eight_4, eight_5, eight_56, eight_6, eight_67, eight_7, eight_3])
        loop_8 = factory.addCurveLoop([line8_1, line8_24, line8_2, line8_13])
        return factory.addPlaneSurface([loop_8])
    
    elseif number ==9
        nine_1 = factory.addPoint(center[1], center[2]-3.0*scale, 0.0, lc)
        nine_2 = factory.addPoint(center[1]+1.7*scale, center[2], 0.0, lc)
        nine_21 = factory.addPoint(center[1]+1.3*scale, center[2]-0.5*scale, 0.0, lc)
        nine_22 = factory.addPoint(center[1], center[2]-0.9*scale, 0.0, lc)
        nine_23 = factory.addPoint(center[1] - 0.5*scale, center[2]-0.4*scale, 0.0, lc)
        nine_3 = factory.addPoint(center[1], center[2]+3.0*scale, 0.0, lc)
        nine_4 = factory.addPoint(center[1]-2.05*scale, center[2], 0.0, lc)
        nine_corner1 = factory.addPoint(center[1]+1.6*scale, center[2]-2.2*scale, 0.0, lc)
        nine_corner2 = factory.addPoint(center[1]+1.4*scale, center[2]+2.3*scale, 0.0, lc)
        nine_corner3 = factory.addPoint(center[1]-1.4*scale, center[2]+2.3*scale, 0.0, lc)
        nine_corner_34 = factory.addPoint(center[1]-1.7*scale, center[2]-1.8*scale, 0.0, lc) 
        nine_corner4 = factory.addPoint(center[1]-1.4*scale, center[2]-2.3*scale, 0.0, lc)
        nine_corner12 = factory.addPoint(center[1]+0.8*scale, center[2]-1.7*scale, 0.0, lc)
        nine_5 = factory.addPoint(center[1], center[2]-1.9*scale, 0.0, lc)
        line9_1 = factory.addLine(nine_corner1, nine_corner12)
        arcnine_1 = factory.addSpline([nine_corner12, nine_5, nine_23])
        arcnine_2 = factory.addSpline([nine_23, nine_22, nine_21, nine_2, nine_corner2, nine_3])
        arcnine_3 = factory.addSpline([nine_3, nine_corner3, nine_4])
        arcnine_4 = factory.addSpline([nine_4, nine_corner_34, nine_corner4, nine_1, nine_corner1])
        bigloop_9 = factory.addCurveLoop([line9_1, arcnine_1, arcnine_2, arcnine_3, arcnine_4])

        nine_6 = factory.addPoint(center[1] - 0.5*scale, center[2]+0.25*scale, 0.0, lc)
        nine_7 = factory.addPoint(center[1] + 0.6*scale, center[2]+0.3*scale, 0.0, lc)
        nine_8 = factory.addPoint(center[1] + 0.6*scale, center[2]+1.6*scale, 0.0, lc)
        nine_9 = factory.addPoint(center[1] - 0.5*scale, center[2]+1.9*scale, 0.0, lc)
        line9_2 = factory.addLine(nine_9, nine_6)
        arcnine_5 = factory.addSpline([nine_6, nine_7, nine_8, nine_9])
        smallloop_9 = factory.addCurveLoop([line9_2, arcnine_5])
        return factory.addPlaneSurface([bigloop_9, smallloop_9])
    
    elseif number ==10 # A
        ten_1 = factory.addPoint(center[1]+3*scale, center[2]-3*scale, 0.0, lc)
        ten_2 = factory.addPoint(center[1]+0.66*scale, center[2]+3*scale, 0.0, lc)
        ten_3 = factory.addPoint(center[1]-0.66*scale, center[2]+3*scale, 0.0, lc)
        ten_4 = factory.addPoint(center[1]-3*scale, center[2]-3*scale, 0.0, lc)
        ten_5 = factory.addPoint(center[1]-1.71*scale, center[2]-3*scale, 0.0, lc)
        ten_6 = factory.addPoint(center[1]-1.22*scale, center[2]-1.64*scale, 0.0, lc)
        ten_7 = factory.addPoint(center[1]+1.22*scale, center[2]-1.64*scale, 0.0, lc)
        ten_8 = factory.addPoint(center[1]+1.71*scale, center[2]-3*scale, 0.0, lc)

        line10_1 = factory.addLine(ten_1, ten_2)
        line10_2 = factory.addLine(ten_2, ten_3)
        line10_3 = factory.addLine(ten_3, ten_4)
        line10_4 = factory.addLine(ten_4, ten_5)
        line10_5 = factory.addLine(ten_5, ten_6)
        line10_6 = factory.addLine(ten_6, ten_7)
        line10_7 = factory.addLine(ten_7, ten_8)
        line10_8 = factory.addLine(ten_8, ten_1)
        loop_10 = factory.addCurveLoop([line10_1, line10_2, line10_3, line10_4, line10_5, line10_6, line10_7, line10_8])
        
        ten_9 = factory.addPoint(center[1]+0.85*scale, center[2]-0.63*scale, 0.0, lc)
        ten_10 = factory.addPoint(center[1], center[2]+1.6*scale, 0.0, lc)
        ten_11 = factory.addPoint(center[1]-0.85*scale, center[2]-0.63*scale, 0.0, lc)

        line10_9 = factory.addLine(ten_9, ten_10)
        line10_10 = factory.addLine(ten_10, ten_11)
        line10_11 = factory.addLine(ten_11, ten_9)
        smallloop_10 = factory.addCurveLoop([line10_9, line10_10, line10_11])

        return factory.addPlaneSurface([loop_10, smallloop_10])
    
    elseif number ==11 # B
        eleven_1 = factory.addPoint(center[1]+2.5*scale, center[2]-3*scale, 0.0, lc)
        eleven_2 = factory.addPoint(center[1]+2.5*scale, center[2]+3*scale, 0.0, lc)
        eleven_3 = factory.addPoint(center[1]-1*scale, center[2]+3*scale, 0.0, lc)
        ceeleven_34 = factory.addPoint(center[1]-1*scale, center[2]+1.7*scale, 0.0, lc)
        eleven_4 = factory.addPoint(center[1]-1*scale, center[2]+0.4*scale, 0.0, lc)
        ceeleven_45 = factory.addPoint(center[1]-1*scale, center[2]-1.3*scale, 0.0, lc)
        eleven_5 = factory.addPoint(center[1]-1*scale, center[2]-3*scale, 0.0, lc)
        line11_1 = factory.addLine(eleven_2, eleven_1)
        line11_2 = factory.addLine(eleven_3, eleven_2)
        arc11_3 = factory.addCircleArc(eleven_4, ceeleven_34, eleven_3)
        arc11_4 = factory.addCircleArc(eleven_5, ceeleven_45, eleven_4)
        line11_5 = factory.addLine(eleven_1, eleven_5)
        bigloop_11 = factory.addCurveLoop([line11_1, line11_5, arc11_4, arc11_3, line11_2])

        eleven_6 = factory.addPoint(center[1]+1.5*scale, center[2]-2*scale, 0.0, lc)
        eleven_7 = factory.addPoint(center[1]-1*scale, center[2]-2*scale, 0.0, lc)
        eleven_8 = factory.addPoint(center[1]-1*scale, center[2]-0.6*scale, 0.0, lc)
        eleven_9 = factory.addPoint(center[1]+1.5*scale, center[2]-0.6*scale, 0.0, lc)
        line11_6 = factory.addLine(eleven_6, eleven_7)
        arc_11_7 = factory.addCircleArc(eleven_7, ceeleven_45, eleven_8)
        line11_8 = factory.addLine(eleven_8, eleven_9)
        line11_9 = factory.addLine(eleven_9, eleven_6)
        smallloop_11_1 = factory.addCurveLoop([line11_6, arc_11_7, line11_8, line11_9])

        eleven_10 = factory.addPoint(center[1]+1.5*scale, center[2]+1.1*scale, 0.0, lc)
        eleven_11 = factory.addPoint(center[1]-0.4*scale, center[2]+1.1*scale, 0.0, lc)
        ceeleven_11_12 = factory.addPoint(center[1]-0.4*scale, center[2]+1.7*scale, 0.0, lc)
        eleven_12 = factory.addPoint(center[1]-0.4*scale, center[2]+2.3*scale, 0.0, lc)
        eleven_13 = factory.addPoint(center[1]+1.5*scale, center[2]+2.3*scale, 0.0, lc)
        line11_10 = factory.addLine(eleven_10, eleven_11)
        arc_11_11 = factory.addCircleArc(eleven_11, ceeleven_11_12, eleven_12)
        line11_12 = factory.addLine(eleven_12, eleven_13)
        line11_13 = factory.addLine(eleven_13, eleven_10)
        smallloop_11_2 = factory.addCurveLoop([line11_10, arc_11_11, line11_12, line11_13])

        return factory.addPlaneSurface([bigloop_11, smallloop_11_1, smallloop_11_2])

    elseif number==12 # C
        twelve_1 = factory.addPoint(center[1]-0.18*scale, center[2]-3*scale, 0.0, lc)
        twelve_12 = factory.addPoint(center[1]-1.68*scale, center[2]-2.52*scale, 0.0, lc)
        twelve_2 = factory.addPoint(center[1]-2.67*scale, center[2]-0.7*scale, 0.0, lc)
        twelve_3 = factory.addPoint(center[1]-1.5*scale, center[2]-1.06*scale, 0.0, lc)
        twelve_4 = factory.addPoint(center[1]-0.17*scale, center[2]-1.96*scale, 0.0, lc)
        twelve_5 = factory.addPoint(center[1]+1.3*scale, center[2]+0.1*scale, 0.0, lc)
        twelve_6 = factory.addPoint(center[1]-0.19*scale, center[2]+2.17*scale, 0.0, lc)
        twelve_7 = factory.addPoint(center[1]-1.45*scale, center[2]+1.16*scale, 0.0, lc)
        twelve_8 = factory.addPoint(center[1]-2.65*scale, center[2]+1.45*scale, 0.0, lc)
        twelve_9 = factory.addPoint(center[1]-0.25*scale, center[2]+3.2*scale, 0.0, lc)
        twelve_910 = factory.addPoint(center[1]+1.78*scale, center[2]+2.38*scale, 0.0, lc)
        twelve_10 = factory.addPoint(center[1]+2.55*scale, center[2], 0.0, lc)
        twelve_101 = factory.addPoint(center[1]+2.13*scale, center[2]-1.68*scale, 0.0, lc)

        line12_23 = factory.addLine(twelve_2, twelve_3)
        line12_37 = factory.addSpline([twelve_3, twelve_4, twelve_5, twelve_6, twelve_7])
        line12_78 = factory.addLine(twelve_7, twelve_8)
        line12_82 = factory.addSpline([twelve_8, twelve_9,twelve_910, twelve_10, twelve_101, twelve_1, twelve_12, twelve_2]) 
        loop_12 =  factory.addCurveLoop([line12_23,line12_37,  line12_78, line12_82])
        return factory.addPlaneSurface([loop_12])
    else
        error("make_numb_hexadecimal: number $number is out of range (expected 1-12)")
    end
end

############ Function to create labels ############
function make_label(factory, iteration::Int, InsertPos::Int, tray::Int, lc::Float64, shim_radius_mm = 235, letter_thickness = 0.5, scale = 1.3)
    ### Function to create a label given Ring "X", Tray "Y" and Iteration "Z"
    insert = (3, 1)
    ring = abs(InsertPos)
    back_or_front = sign(InsertPos)
    tens_digit_insert = abs(InsertPos) ÷ 10
    ones_digit_insert = abs(InsertPos) % 10 

    factory.synchronize()
    ##### Creating letter I, center = 37.5 ###### 
    # I1 = factory.addPoint(37.5, shim_radius_mm+1.0, 0.0, lc)
    # I2 = factory.addPoint(37.5, shim_radius_mm-6.0, 0.0, lc)
    # I3 = factory.addPoint(35.5, shim_radius_mm-6.0, 0.0, lc)
    # I4 = factory.addPoint(35.5, shim_radius_mm+1.0, 0.0, lc)
    # I_line1 = factory.addLine(I1, I2)
    # I_line2 = factory.addLine(I2, I3)
    # I_line3 = factory.addLine(I3, I4)
    # I_line4 = factory.addLine(I4, I1)
    # loop_I = factory.addCurveLoop([I_line1, I_line2, I_line3, I_line4])
    # surface_I = factory.addPlaneSurface([loop_I])
    # volume_I = factory.extrude([(2, surface_I)], 0.0, 0.0, letter_thickness)
    I1 = factory.addPoint(42.01, shim_radius_mm-8.34, 0.0, lc)
    I2 = factory.addPoint(37.4, shim_radius_mm-7.53, 0.0, lc)
    I3 = factory.addPoint(37.6, shim_radius_mm-6.13, 0.0, lc)
    I4 = factory.addPoint(38.93, shim_radius_mm-6.36, 0.0, lc)
    I5 = factory.addPoint(39.82, shim_radius_mm-1.27, 0.0, lc)
    I6 = factory.addPoint(38.53, shim_radius_mm-1.05, 0.0, lc)
    I7 = factory.addPoint(38.77, shim_radius_mm+0.345, 0.0, lc)
    I8 = factory.addPoint(43.39, shim_radius_mm-0.46, 0.0, lc)
    I9 = factory.addPoint(43.14, shim_radius_mm-1.85, 0.0, lc)
    I10 = factory.addPoint(41.85, shim_radius_mm-1.63, 0.0, lc)
    I11 = factory.addPoint(40.96, shim_radius_mm-6.71, 0.0, lc)
    I12 = factory.addPoint(42.25, shim_radius_mm-6.94, 0.0, lc)
    I_line1 = factory.addLine(I1, I2)
    I_line2 = factory.addLine(I2, I3)
    I_line3 = factory.addLine(I3, I4)
    I_line4 = factory.addLine(I4, I5)
    I_line5 = factory.addLine(I5, I6)
    I_line6 = factory.addLine(I6, I7)
    I_line7 = factory.addLine(I7, I8)
    I_line8 = factory.addLine(I8, I9)
    I_line9 = factory.addLine(I9, I10)
    I_line10 = factory.addLine(I10, I11)
    I_line11 = factory.addLine(I11, I12)
    I_line12 = factory.addLine(I12, I1)
    loop_I = factory.addCurveLoop([I_line1, I_line2, I_line3, I_line4,I_line5, I_line6, I_line7, I_line8, I_line9, I_line10, I_line11, I_line12])
    surface_I = factory.addPlaneSurface([loop_I])
    volume_I = factory.extrude([(2, surface_I)], 0.0, 0.0, letter_thickness)
    
    ###### Creating Iteration Number Z, center = 22.5, 234 ######
    number_z = make_numb_hexadecimal(factory, iteration, (32.22, shim_radius_mm-2.61), lc, scale)
    volume_z = factory.extrude([(2, number_z)], 0.0, 0.0, letter_thickness)

    ###### Creating letter R, center = 7.5 ######
    # R1 = factory.addPoint(9, shim_radius_mm-3.5, 0.0, lc)
    # R2 = factory.addPoint(9, shim_radius_mm+3.5, 0.0, lc)
    # R3 = factory.addPoint(5.7, shim_radius_mm+3.5, 0.0, lc)
    # CR34 = factory.addPoint(5.7, shim_radius_mm+1.6, 0.0, lc)
    # R4 = factory.addPoint(5.7, shim_radius_mm-0.3, 0.0, lc)
    # R5 = factory.addPoint(3.2, shim_radius_mm-3.5, 0.0, lc)
    # R6 = factory.addPoint(4.8, shim_radius_mm-3.5, 0.0, lc)
    # R7 = factory.addPoint(7, shim_radius_mm-0.3, 0.0, lc)
    # R8 = factory.addPoint(7.5, shim_radius_mm-0.3, 0.0, lc)
    # R9 = factory.addPoint(7.5, shim_radius_mm-3.5, 0.0, lc)
    
    R1 = factory.addPoint(21.48, shim_radius_mm-5.40, 0.0, lc)
    R2 = factory.addPoint(21.48, shim_radius_mm+2.55, 0.0, lc)
    R3 = factory.addPoint(16.20, shim_radius_mm+2.55, 0.0, lc)
    R34 = factory.addPoint(15.50, shim_radius_mm+2.00, 0.0, lc)
    R4 = factory.addPoint(16.09, shim_radius_mm-1.65, 0.0, lc)
    R5 = factory.addPoint(14.00, shim_radius_mm-5.40, 0.0, lc)
    R6 = factory.addPoint(15.91, shim_radius_mm-5.40, 0.0, lc)
    R7 = factory.addPoint(17.82, shim_radius_mm-2.33, 0.0, lc)
    R8 = factory.addPoint(18.90, shim_radius_mm-2.33, 0.0, lc)
    R9 = factory.addPoint(18.90, shim_radius_mm-5.40, 0.0, lc)
    lineR1 = factory.addLine(R1, R2)
    lineR2 = factory.addLine(R2, R3)
    arcR3 = factory.addSpline([R3, R34, R4])
    lineR4 = factory.addLine(R4, R5)
    lineR5 = factory.addLine(R5, R6)
    lineR6 = factory.addLine(R6, R7)
    lineR7 = factory.addLine(R7, R8)
    lineR8 = factory.addLine(R8, R9)
    lineR9 = factory.addLine(R9, R1)    
    bigloop_R = factory.addCurveLoop([arcR3, lineR4, lineR5, lineR6, lineR7, lineR8, lineR9, lineR1, lineR2])
    surface_R = factory.addPlaneSurface([bigloop_R])
    volume_R = factory.extrude([(2, surface_R)], 0.0, 0.0, letter_thickness)

    if back_or_front == -1
        N1 = factory.addPoint(11.23, shim_radius_mm-4.74, 0.0, lc)
        N2 = factory.addPoint(9.35, shim_radius_mm-4.68, 0.0, lc)
        N3 = factory.addPoint(9.30, shim_radius_mm+0.81, 0.0, lc)
        N4 = factory.addPoint(5.95, shim_radius_mm-4.57, 0.0, lc)
        N5 = factory.addPoint(3.97, shim_radius_mm-4.50, 0.0, lc)
        N6 = factory.addPoint(4.23, shim_radius_mm+3.49, 0.0, lc)
        N7 = factory.addPoint(6.12, shim_radius_mm+3.43, 0.0, lc)
        N8 = factory.addPoint(5.97, shim_radius_mm-1.15, 0.0, lc)
        N9 = factory.addPoint(9.04, shim_radius_mm+3.33, 0.0, lc)
        N10 = factory.addPoint(11.50, shim_radius_mm+3.25, 0.0, lc)

        lineN1 = factory.addLine(N1, N2)
        lineN2 = factory.addLine(N2, N3)
        lineN3 = factory.addLine(N3, N4)
        lineN4 = factory.addLine(N4, N5)
        lineN5 = factory.addLine(N5, N6)
        lineN6 = factory.addLine(N6, N7)
        lineN7 = factory.addLine(N7, N8)
        lineN8 = factory.addLine(N8, N9)
        lineN9 = factory.addLine(N9, N10)
        lineN10 = factory.addLine(N10, N1)

        loop_N = factory.addCurveLoop([lineN1, lineN2, lineN3, lineN4, lineN5, lineN6, lineN7, lineN8, lineN9, lineN10])
        surface_N = factory.addPlaneSurface([loop_N])
        volume_N = factory.extrude([(2, surface_N)], 0.0, 0.0, letter_thickness)
    else
        P1 = factory.addPoint(10.80, shim_radius_mm-4.65, 0.0, lc)
        P2 = factory.addPoint(9.18, shim_radius_mm-4.65, 0.0, lc)
        P3 = factory.addPoint(9.18, shim_radius_mm-1.63, 0.0, lc)
        ArcP4 = factory.addPoint(6.45, shim_radius_mm-1.52, 0.0, lc)
        ArcP5 = factory.addPoint(4.67, shim_radius_mm + 0.89, 0.0, lc)
        ArcP6 = factory.addPoint(6.45, shim_radius_mm + 3.23, 0.0, lc)
        P7 = factory.addPoint(9.18, shim_radius_mm+3.35, 0.0, lc)
        P8 = factory.addPoint(10.80, shim_radius_mm+3.35, 0.0, lc)

        lineP1 = factory.addLine(P1, P2)
        lineP2 = factory.addLine(P2, P3)
        lineP3 = factory.addLine(P3, ArcP4)
        arcP4 = factory.addSpline([ArcP4, ArcP5, ArcP6])
        lineP6 = factory.addLine(ArcP6, P7)
        lineP7 = factory.addLine(P7, P8)
        lineP8 = factory.addLine(P8, P1)
        loop_P = factory.addCurveLoop([lineP1, lineP2, lineP3, arcP4, lineP6, lineP7, lineP8])
        surface_P = factory.addPlaneSurface([loop_P])
        volume_P = factory.extrude([(2, surface_P)], 0.0, 0.0, letter_thickness)
    end

    ###### Creating first digit of InsertPos, center = -7.01, shim_radius_mm -0.54 ######
    number_x1 = make_numb_hexadecimal(factory, tens_digit_insert, (-7.01, shim_radius_mm-0.54), lc, scale)
    volume_x1 = factory.extrude([(2, number_x1)], 0.0, 0.0, letter_thickness)

    ###### Creating first second of InsertPos, center = -16.50, shim_radius_mm -1.01 ######
    number_x2 = make_numb_hexadecimal(factory, ones_digit_insert, (-16.50, shim_radius_mm-1.01), lc, scale)
    volume_x2 = factory.extrude([(2, number_x2)], 0.0, 0.0, letter_thickness)

    ###### Creating letter T, center = 22.5 ######
    T1 = factory.addPoint(-28.71, shim_radius_mm-6.30, 0.0, lc)
    T2 = factory.addPoint(-30.76, shim_radius_mm-6.55, 0.0, lc)
    T3 = factory.addPoint(-31.59, shim_radius_mm-0.15, 0.0, lc)
    T4 = factory.addPoint(-34.07, shim_radius_mm-0.48, 0.0, lc)
    T5 = factory.addPoint(-34.26, shim_radius_mm+1.06, 0.0, lc)
    T6 = factory.addPoint(-27.26, shim_radius_mm+1.96, 0.0, lc)
    T7 = factory.addPoint(-27.07, shim_radius_mm+0.43, 0.0, lc)
    T8 = factory.addPoint(-29.54, shim_radius_mm+0.11, 0.0, lc)
    
    lineT1 = factory.addLine(T1, T2)
    lineT2 = factory.addLine(T2, T3)
    lineT3 = factory.addLine(T3, T4)
    lineT4 = factory.addLine(T4, T5)
    lineT5 = factory.addLine(T5, T6)
    lineT6 = factory.addLine(T6, T7)
    lineT7 = factory.addLine(T7, T8)
    lineT8 = factory.addLine(T8, T1)
    loop_T = factory.addCurveLoop([lineT1, lineT2, lineT3, lineT4, lineT5, lineT6, lineT7, lineT8])
    surface_T = factory.addPlaneSurface([loop_T])
    volume_T = factory.extrude([(2, surface_T)], 0.0, 0.0, letter_thickness)

    ###### Creating Tray Number Y, center = 39.21, shim_radius_mm  -3.72 ######
    number_y = make_numb_hexadecimal(factory, tray, (-39.21, shim_radius_mm-3.72), lc, scale)
    volume_y = factory.extrude([(2, number_y)], 0.0, 0.0, letter_thickness)
    factory.synchronize()
    
    ### Removing the labels from the main volume ####
    allVols = factory.getEntities(3)
    removingVols = [v for v in allVols if v != insert]
    CutInsert = factory.cut([insert], removingVols)
    factory.synchronize()
end

############ Function to check if package is installed ############
function ensure_pkg(pkg::String)
    try
        @eval import $(Symbol(pkg))
    catch
        @info "Installing missing package: $pkg"
        Pkg.add(pkg)
        @eval import $(Symbol(pkg))
    end
end

############ Function to assign tray number ############
function assign_tray(x::Real, y::Real; num_trays::Int=12)
    angle_deg = mod(atan(y, x) * 180/pi, 360.0)
    # Tray 12 → 90°, tray 9 → 0°, tray 6 → 270°, tray 3 → 180°
    tray_idx = mod(round(Int, angle_deg / (360.0 / num_trays)) + 8, num_trays)
    return tray_idx + 1
end