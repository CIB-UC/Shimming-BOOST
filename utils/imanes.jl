#############################
# imanes.jl  —  versión GPU
#############################

using CUDA

# =======================
# Constantes / utilidades
# =======================
const μ0_4π_f32 = 1f-7     # μ0/(4π) en SI (T·m/A), Float32
const EPS_R2_f32 = 1f-12   # regularización para evitar división por cero

# Convierte un valor de campo axial B a 1 cm (en Tesla) al momento dipolar m (A·m²)
# Fórmula: B_axis = μ0/(4π) * (2 m) / r^3  =>  m = B * 4π * r^3 / (2 μ0)
# Aquí usamos μ0/(4π) = 1e-7, así: m = B * r^3 / (2 * 1e-7)
@inline function m_from_B1cm_T_f32(B1cm_T::Float32; r_m::Float32 = 1f-2)
    return B1cm_T * (r_m^3) / (2f0 * μ0_4π_f32)
end

# =======================
# Construcción de ejes/grilla
# =======================
"""
    build_axes_mm(dims, resmm; center_grid=true, center_mm=(0,0,0), origin_mm=(0,0,0))

Devuelve `(xg_mm, yg_mm, zg_mm)` como vectores en **mm**.
- `dims = (nx,ny,nz)`
- `resmm = (dx,dy,dz)` resolución en **mm**
- Si `center_grid=true`, centra la malla en `center_mm`.
- Si `center_grid=false`, usa `origin_mm` como esquina mínima (x_min,y_min,z_min).
"""
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

# Grilla 3D en GPU (coordenadas en **metros**, Float32)
struct GridGPU
    X::CuArray{Float32,3}
    Y::CuArray{Float32,3}
    Z::CuArray{Float32,3}
    nx::Int32
    ny::Int32
    nz::Int32
end

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

# =======================
# Kernel: By de M dipolos
# B(r) = μ0/(4π) * (1/ρ^3) * [3(m·r̂) r̂ - m]
# Salida: componente Y (By)
# =======================
@inline function _idx(i,j,k, nx,ny,nz)
    return i + (j-1)*nx + (k-1)*nx*ny
end

function _by_dipoles_kernel!(
    By::CuDeviceArray{Float32,3},
    X::CuDeviceArray{Float32,3}, Y::CuDeviceArray{Float32,3}, Z::CuDeviceArray{Float32,3},
    px::CuDeviceVector{Float32}, py::CuDeviceVector{Float32}, pz::CuDeviceVector{Float32},
    mx::CuDeviceVector{Float32}, my::CuDeviceVector{Float32}, mz::CuDeviceVector{Float32},
    μ0_4π::Float32, ϵ::Float32, nx::Int32, ny::Int32, nz::Int32
)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    N = nx*ny*nz
    if idx > N; return; end

    tmp  = idx - 1
    k    = (tmp ÷ (nx*ny)) + 1
    tmp2 = tmp % (nx*ny)
    j    = (tmp2 ÷ nx) + 1
    i    = (tmp2 % nx) + 1

    x = X[i,j,k]; y = Y[i,j,k]; z = Z[i,j,k]
    acc = 0.0f0
    M = length(px)

    @inbounds for m in 1:M
        rx = x - px[m]; ry = y - py[m]; rz = z - pz[m]
        ρ2 = rx*rx + ry*ry + rz*rz + ϵ
        invρ = inv(sqrt(ρ2))
        invρ3 = invρ * invρ * invρ
        mdotr_hat = (mx[m]*rx + my[m]*ry + mz[m]*rz) * invρ
        term_y = 3f0 * mdotr_hat * (ry * invρ) - my[m]
        acc += μ0_4π * invρ3 * term_y
    end

    By[idx] = acc
    return
end

# =======================
# Construcción de momentos m a partir de θdeg
# =======================
"""
    build_dipoles_device(posiciones_mm, θdeg; axis=:y, m_mag=1f0, B1cm_T=nothing)

Devuelve (px,py,pz,mx,my,mz) como **CuArray{Float32}**.
- `posiciones_mm`: Vector{NTuple{3,Real}} en mm.
- `θdeg`: vector de grados por imán.
- `axis`: eje base del imán (`:x`, `:y`, `:z`) desde el que se rota.
- `m_mag`: magnitud del momento (A·m²) si ya está calibrado.
- `B1cm_T`: si lo entregas (en Tesla), ignora `m_mag` y calibra con B a 1 cm.
  (usa la fórmula dipolo axial).
"""
# ACEPTA Vector{Tuple{…}} o Vector{SVector{3,…}} etc.
# Reemplaza tu build_dipoles_device por esta versión "CPU→GPU"
function build_dipoles_device(posiciones_mm::AbstractVector{<:NTuple{3,<:Real}},
                              θdeg::AbstractVector{<:Real};
                              axis::Symbol = :y,
                              m_mag::Float32 = 1.0f0,
                              B1cm_T::Union{Nothing,Real} = nothing)

    @assert length(posiciones_mm) == length(θdeg)
    M = length(posiciones_mm)

    # --- Construimos en CPU (Float32) ---
    px_h = Vector{Float32}(undef, M)
    py_h = Vector{Float32}(undef, M)
    pz_h = Vector{Float32}(undef, M)
    mx_h = Vector{Float32}(undef, M)
    my_h = Vector{Float32}(undef, M)
    mz_h = Vector{Float32}(undef, M)

    # Calibración por B@1cm (si la diste)
    m_use = if B1cm_T === nothing
        m_mag
    else
        m_from_B1cm_T_f32(Float32(B1cm_T))
    end

    @inbounds for k in 1:M
        x_mm, y_mm, z_mm = posiciones_mm[k]
        px_h[k] = Float32(x_mm * 1e-3)  # mm → m
        py_h[k] = Float32(y_mm * 1e-3)
        pz_h[k] = Float32(z_mm * 1e-3)

        θ = Float32(θdeg[k] * (π/180))
        if axis === :y
            # base +y, rotación en plano y–z
            mx_h[k] = 0f0
            my_h[k] =  m_use * cos(θ)
            mz_h[k] =  m_use * sin(θ)
        elseif axis === :z
            # base +z, rotación en plano x–z
            mx_h[k] =  m_use * sin(θ)
            my_h[k] =  0f0
            mz_h[k] =  m_use * cos(θ)
        else
            # base +x, rotación en plano x–y
            mx_h[k] =  m_use * cos(θ)
            my_h[k] =  m_use * sin(θ)
            mz_h[k] =  0f0
        end
    end

    # --- Subimos a device al final (sin scalar indexing en GPU) ---
    px = CuArray(px_h);  py = CuArray(py_h);  pz = CuArray(pz_h)
    mx = CuArray(mx_h);  my = CuArray(my_h);  mz = CuArray(mz_h)
    return px,py,pz,mx,my,mz
end

# =======================
# Campo de imanes (GPU) — By solamente
# =======================
"""
    campo_imanes_gpu(posiciones, θdeg, dims, resmm;
                     center_grid=true, center_mm=(0,0,0), origin_mm=(0,0,0),
                     axis=:y, m_mag=1f0, B1cm_T=nothing, to_mT=true)

Devuelve `CuArray{Float32,3}` con **By** en la grilla especificada.
- `dims` = (nx,ny,nz)
- `resmm` = (dx,dy,dz) en **mm**
- Malla centrada en `center_mm` (mm) o desde `origin_mm` si `center_grid=false`.
- `axis`, `m_mag` o `B1cm_T` para la orientación/magnitud del momento magnético.
- `to_mT=true` convierte T → mT al final (útil si tu pipeline usa mT).
"""
# dims y resmm aceptan cualquier Tuple de 3 elementos con Int/Real
function campo_imanes_gpu(posiciones::AbstractVector{<:NTuple{3,<:Real}},
                          θdeg::AbstractVector{<:Real},
                          dims::Tuple{<:Integer,<:Integer,<:Integer},
                          resmm::Tuple{<:Real,<:Real,<:Real};
                          center_grid::Bool = true,
                          center_mm::Tuple{<:Real,<:Real,<:Real} = (0.0,0.0,0.0),
                          origin_mm::Tuple{<:Real,<:Real,<:Real} = (0.0,0.0,0.0),
                          axis::Symbol = :y,
                          m_mag::Float32 = 1.0f0,
                          B1cm_T::Union{Nothing,Real} = nothing,
                          to_mT::Bool = true)

    # 1) Ejes y grilla en GPU
    xg_mm, yg_mm, zg_mm = build_axes_mm(dims, resmm; center_grid=center_grid, center_mm=center_mm, origin_mm=origin_mm)
    grid = make_grid_gpu_from_axes_mm(xg_mm, yg_mm, zg_mm)

    # 2) Posiciones/momentos en GPU
    px,py,pz,mx,my,mz = build_dipoles_device(posiciones, θdeg; axis=axis, m_mag=m_mag, B1cm_T=B1cm_T)

    # 3) Kernel de acumulación
    dBy = CUDA.zeros(Float32, Int(grid.nx), Int(grid.ny), Int(grid.nz))
    N = Int(grid.nx)*Int(grid.ny)*Int(grid.nz)
    @cuda threads=256 blocks=cld(N,256) _by_dipoles_kernel!(
        dBy, grid.X, grid.Y, grid.Z,
        px,py,pz, mx,my,mz,
        μ0_4π_f32, EPS_R2_f32, grid.nx, grid.ny, grid.nz
    )
    CUDA.synchronize()

    # 4) Unidades: T → mT si se solicita
    if to_mT
        dBy .*= 1_000f0
    end
    return dBy  # CuArray{Float32,3}
end

# =======================
# Wrapper compatible con tu pipeline
# =======================
"""
    campo_imanes(posiciones, θdeg, dims, resmm; kwargs...) -> (By, nothing, nothing, nothing)

Versión compatible con tu código existente. Calcula By en **GPU** y lo devuelve como **Array** (CPU).
"""
# Wrapper compatible
function campo_imanes(posiciones::AbstractVector{<:NTuple{3,<:Real}},
                      θdeg::AbstractVector{<:Real},
                      dims::Tuple{<:Integer,<:Integer,<:Integer},
                      resmm::Tuple{<:Real,<:Real,<:Real};
                      center_grid::Bool = true,
                      center_mm::Tuple{<:Real,<:Real,<:Real} = (0.0,0.0,0.0),
                      origin_mm::Tuple{<:Real,<:Real,<:Real} = (0.0,0.0,0.0),
                      axis::Symbol = :y,
                      m_mag::Float32 = 1.0f0,
                      B1cm_T::Union{Nothing,Real} = nothing,
                      to_mT::Bool = true)

    dBy = campo_imanes_gpu(posiciones, θdeg, dims, resmm;
                           center_grid=center_grid, center_mm=center_mm, origin_mm=origin_mm,
                           axis=axis, m_mag=m_mag, B1cm_T=B1cm_T, to_mT=to_mT)

    return Array(dBy), nothing, nothing, nothing
end
