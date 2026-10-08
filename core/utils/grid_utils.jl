# Grilla 3D en GPU (coordenadas en **metros**, Float32)
# Cada X, Y, Z es un arreglo con 3 canales que representa un espacio escalar. Cada uno de estos espacios escalares contiene
# el componente resprectivo del vector que esta represntando la grilla. Puede ser tanto posiciones como vectores
struct GridGPU
    X::CuArray{Float32,3}
    Y::CuArray{Float32,3}
    Z::CuArray{Float32,3}
    nx::Int32
    ny::Int32
    nz::Int32
end


# Toma los vectores con los puntos de evaluacion en cada eje (-w:Δ:w) y genera 3 grillas escalares con las que se genera un CustomConstruct
# Se usa broadcasting para expandir las dimensiones automaticamente al tamaño apropiado (Se multiplica por 0 las contribuciones de los otros ejes)
function make_grid_gpu_from_axes_mm(xg_mm::AbstractVector, yg_mm::AbstractVector, zg_mm::AbstractVector)
    x = Float32.(xg_mm .* 1f-3)  # mm → m
    y = Float32.(yg_mm .* 1f-3)
    z = Float32.(zg_mm .* 1f-3)
    nx, ny, nz = length(x), length(y), length(z)

    # Creamos mesh 3D en device (evitando construir arrays gigantes en CPU)
    dX = CuArray(reshape(x, nx, 1, 1)) .+ zero(Float32) .* CuArray(reshape(y, 1, ny, 1)) .+ zero(Float32) .* CuArray(reshape(z, 1, 1, nz))
    dY = zero(Float32) .* CuArray(reshape(x, nx, 1, 1)) .+ CuArray(reshape(y, 1, ny, 1)) .+ zero(Float32) .* CuArray(reshape(z, 1, 1, nz))
    dZ = zero(Float32) .* CuArray(reshape(x, nx, 1, 1)) .+ zero(Float32) .* CuArray(reshape(y, 1, ny, 1)) .+ CuArray(reshape(z, 1, 1, nz))

    return GridGPU(dX, dY, dZ, Int32(nx), Int32(ny), Int32(nz))
end

# Construye los ejes dadas las restricciones en la entrada
function build_axes_mm(dims::NTuple{3,Int}, resmm::NTuple{3,<:Real};
                       center_grid::Bool = true,
                       center_mm::NTuple{3,<:Real} = (0.0, 0.0, 0.0),
                       origin_mm::NTuple{3,<:Real} = (0.0, 0.0, 0.0))
    nx,ny,nz = dims
    dx,dy,dz = resmm

    if center_grid
        cx,cy,cz = center_mm
        x0 = cx - (nx-1)*dx/2
        y0 = cy - (ny-1)*dy/2
        z0 = cz - (nz-1)*dz/2
        xg = collect(x0:dx:(x0 + (nx-1)*dx))
        yg = collect(y0:dy:(y0 + (ny-1)*dy))
        zg = collect(z0:dz:(z0 + (nz-1)*dz))
    else
        ox,oy,oz = origin_mm
        xg = collect(ox:dx:(ox + (nx-1)*dx))
        yg = collect(oy:dy:(oy + (ny-1)*dy))
        zg = collect(oz:dz:(oz + (nz-1)*dz))
    end
    return xg, yg, zg
end
