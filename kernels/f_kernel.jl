using CUDA
using LinearAlgebra

const to_rad_f32 = 1.74532925f-2 # π/180 precalculado como Float32
const scale = 1.0f-7 # μ0 / 4π

function _M!(θ, μ, current, m, M)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx > m; return; end
    @inbounds begin
        θin = θ[idx] * to_rad_f32
        μin = μ[idx]
        state = current[idx]
        M[1, idx] = cos(θin) * μin * state
        M[2, idx] = sin(θin) * μin * state
        M[3, idx] = 0.0
    end
    return
end

function _Btotmasked!(fieldmap, B,                           # Valores base y alocaciones
                X, Y, Z,                                        # Grid de evaluación
                P, M, m,                                                    # Pos y θ de vec momento dipolo
                mask, N)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
                                   

    _By_from_shim(X, Y, Z, P, M, B, m, N, idx)

    if idx <= N
        B[idx] += fieldmap[idx] # Bytot
        B[idx] *= mask[idx]    # Aplicamos mascara
    end

    return # Es un kernel mutable, no debemos retornar nada
end



function _Btot!(fieldmap, B,                                                   # Valores base y alocaciones
                X, Y, Z,                                                       # Grid de evaluación
                P, M, m, N)                                                 # Pos y θ de vec momento dipolo

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x                                   

    _By_from_shim(X, Y, Z, P, M, B, m, N, idx)
    
    if idx <= N
        @inbounds B[idx] += fieldmap[idx] # Bytot
    end
    return 
end

@inline function _By_from_shim(X, Y, Z,  # X, Y y Z Son grillas de tamaño nx x ny x nz, las que contienen valores escalares. 
                                  # En vez de entregar los vectores posicion entregamos 3 campos escalares en grilla con x y z cada uno
                            P, M, B, m,
                            N, idx)
    
    # Inicializamos acumulador de By
    By = 0.0f0
    
    # Cargamos punto de eval
    Rx, Ry, Rz = 0.0f0, 0.0f0, 0.0f0
    @inbounds if idx <= N
        Rx = X[idx]
        Ry = Y[idx]
        Rz = Z[idx]
    end

    @inbounds if idx <= N
        # 1. Main Unrolled Loop
        # We step by 4. We stop at m - 3 to avoid going out of bounds.
        k = 1
        while k <= m - 3
            # --- Dipole 1 ---
            μx1, μy1, μz1 = M[1, k],   M[2, k],   M[3, k]
            px1, py1, pz1 = P[1, k],   P[2, k],   P[3, k]
            dx1, dy1, dz1 = Rx - px1, Ry - py1, Rz - pz1
            r2_1 = dx1*dx1 + dy1*dy1 + dz1*dz1
            
            # --- Dipole 2 ---
            μx2, μy2, μz2 = M[1, k+1], M[2, k+1], M[3, k+1]
            px2, py2, pz2 = P[1, k+1], P[2, k+1], P[3, k+1]
            dx2, dy2, dz2 = Rx - px2, Ry - py2, Rz - pz2
            r2_2 = dx2*dx2 + dy2*dy2 + dz2*dz2

            # --- Dipole 3 ---
            μx3, μy3, μz3 = M[1, k+2], M[2, k+2], M[3, k+2]
            px3, py3, pz3 = P[1, k+2], P[2, k+2], P[3, k+2]
            dx3, dy3, dz3 = Rx - px3, Ry - py3, Rz - pz3
            r2_3 = dx3*dx3 + dy3*dy3 + dz3*dz3

            # --- Dipole 4 ---
            μx4, μy4, μz4 = M[1, k+3], M[2, k+3], M[3, k+3]
            px4, py4, pz4 = P[1, k+3], P[2, k+3], P[3, k+3]
            dx4, dy4, dz4 = Rx - px4, Ry - py4, Rz - pz4
            r2_4 = dx4*dx4 + dy4*dy4 + dz4*dz4

            # --- Math Block 1 ---
            if r2_1 > 1.0f-18
                inv_r_1 = CUDA.rsqrt(r2_1)
                inv_r2_1 = inv_r_1 * inv_r_1
                inv_r3_1 = inv_r2_1 * inv_r_1
                inv_r5_1 = inv_r3_1 * inv_r2_1
                dot_mr_1 = dx1*μx1 + dy1*μy1 + dz1*μz1
                By += 3.0f0 * dot_mr_1 * dy1 * inv_r5_1 - μy1 * inv_r3_1
            end

            # --- Math Block 2 ---
            if r2_2 > 1.0f-18
                inv_r_2 = CUDA.rsqrt(r2_2)
                inv_r2_2 = inv_r_2 * inv_r_2
                inv_r3_2 = inv_r2_2 * inv_r_2
                inv_r5_2 = inv_r3_2 * inv_r2_2
                dot_mr_2 = dx2*μx2 + dy2*μy2 + dz2*μz2
                By += 3.0f0 * dot_mr_2 * dy2 * inv_r5_2 - μy2 * inv_r3_2
            end

            # --- Math Block 3 ---
            if r2_3 > 1.0f-18
                inv_r_3 = CUDA.rsqrt(r2_3)
                inv_r2_3 = inv_r_3 * inv_r_3
                inv_r3_3 = inv_r2_3 * inv_r_3
                inv_r5_3 = inv_r3_3 * inv_r2_3
                dot_mr_3 = dx3*μx3 + dy3*μy3 + dz3*μz3
                By += 3.0f0 * dot_mr_3 * dy3 * inv_r5_3 - μy3 * inv_r3_3
            end

            # --- Math Block 4 ---
            if r2_4 > 1.0f-18
                inv_r_4 = CUDA.rsqrt(r2_4)
                inv_r2_4 = inv_r_4 * inv_r_4
                inv_r3_4 = inv_r2_4 * inv_r_4
                inv_r5_4 = inv_r3_4 * inv_r2_4
                dot_mr_4 = dx4*μx4 + dy4*μy4 + dz4*μz4
                By += 3.0f0 * dot_mr_4 * dy4 * inv_r5_4 - μy4 * inv_r3_4
            end

            k += 4
        end

        # 2. Remainder Loop (Cleanup)
        # Handles the last 1, 2, or 3 items if m is not divisible by 4
        while k <= m
            μx, μy, μz = M[1, k], M[2, k], M[3, k]
            px, py, pz = P[1, k], P[2, k], P[3, k]
            dx, dy, dz = Rx - px, Ry - py, Rz - pz
            r2 = dx*dx + dy*dy + dz*dz

            if r2 > 1.0f-18
                inv_r = CUDA.rsqrt(r2)
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r
                inv_r5 = inv_r3 * inv_r2
                dot_mr = dx*μx + dy*μy + dz*μz
                By += 3.0f0 * dot_mr * dy * inv_r5 - μy * inv_r3
            end
            k += 1
        end
    end
        
    # Write final result to Global Memory
    if idx <= N
        @inbounds B[idx] = By * scale * 1000 # mT
    end
    return
end

# Determina el indice 1D de acceso al array
@inline function _idx(i,j,k, nx,ny,nz)
    return i + (j-1)*nx + (k-1)*nx*ny
end

function _grad!(B, Gx, Gy, Gz, d, nx, ny, nz, N)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    tmp  = idx - 1
    k    = (tmp ÷ (nx*ny)) + 1
    tmp2 = tmp % (nx*ny)
    j    = (tmp2 ÷ nx) + 1
    i   = (tmp2 % nx) + 1

    if idx > N; return; end
    @inbounds begin
        #grad x
        if 1 < i < nx
            Gx[idx] = (B[_idx(i+1,j,k,nx,ny,nz)] - B[_idx(i-1,j,k,nx,ny,nz)]) / (2f0*d)
        elseif i == 1
            Gx[idx] = (B[_idx(2,j,k,nx,ny,nz)] - B[_idx(1,j,k,nx,ny,nz)]) / d
        else
            Gx[idx] = (B[_idx(nx,j,k,nx,ny,nz)] - B[_idx(nx-1,j,k,nx,ny,nz)]) / d
        end

        #grad y
        if 1 < j < ny
            Gy[idx] = (B[_idx(i,j+1,k,nx,ny,nz)] - B[_idx(i,j-1,k,nx,ny,nz)]) / (2f0*d)
        elseif j == 1
            Gy[idx] = (B[_idx(i,2,k,nx,ny,nz)]   - B[_idx(i,1,k,nx,ny,nz)])   / d
        else
            Gy[idx] = (B[_idx(i,ny,k,nx,ny,nz)]  - B[_idx(i,ny-1,k,nx,ny,nz)]) / d
        end

        #grad z
        if 1 < k < nz
            Gz[idx] = (B[_idx(i,j,k+1,nx,ny,nz)] - B[_idx(i,j,k-1,nx,ny,nz)]) / (2f0*d)
        elseif k == 1
            Gz[idx] = (B[_idx(i,j,2,nx,ny,nz)] - B[_idx(i,j,1,nx,ny,nz)]) / d
        else
            Gz[idx] = (B[_idx(i,j,nz,nx,ny,nz)] - B[_idx(i,j,nz-1,nx,ny,nz)]) / d
        end
    end
    sync_threads()

    return 
end


function _metrics!(B, by_min, by_max, grad_rms, Gx, Gy, Gz, mask, N, Nmask)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid = threadIdx().x

    shared_min = CuDynamicSharedArray(eltype(B), blockDim().x)                                        
    shared_max = CuDynamicSharedArray(eltype(B), blockDim().x, blockDim().x * sizeof(eltype(B)))      
    shared_sum = CuDynamicSharedArray(eltype(B), blockDim().x, 2 * blockDim().x * sizeof(eltype(B)))  

    # 1. Standard Initialization
    shared_min[tid] = typemax(eltype(B))
    shared_max[tid] = typemin(eltype(B))
    shared_sum[tid] = zero(eltype(B))

    if idx == 1
        by_min[idx] = typemax(eltype(B))
        by_max[idx] = typemin(eltype(B))
        grad_rms[idx] = 0.0f0
    end

    # Apply mask mathematically to the global arrays
    if idx <= N
        B[idx]  *= mask[idx]
        Gx[idx] *= mask[idx]
        Gy[idx] *= mask[idx]
        Gz[idx] *= mask[idx]
    end
    sync_threads()
    
    # 2. Smart Loading into Shared Memory
    if idx <= N
        val = B[idx]
        is_valid = mask[idx] > 0.0f0 # Check if point is inside the mask
        
        # If masked out, assign typemax/typemin so they naturally fail min/max comparisons
        shared_min[tid] = is_valid ? val : typemax(eltype(B))
        shared_max[tid] = is_valid ? val : typemin(eltype(B))
        shared_sum[tid] = is_valid ? (Gx[idx]^2 + Gy[idx]^2 + Gz[idx]^2)/Nmask : 0.0f0
    end
    sync_threads()

    # 3. Clean Tree Reduction (No 0.0 checks needed!)
    s = blockDim().x ÷ 2
    while s > 0
        if tid <= s
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s])
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s])
            shared_sum[tid] += shared_sum[tid + s]
        end
        sync_threads()
        s ÷= 2
    end

    # 4. Updates to Global Memory
    if tid == 1
        CUDA.@atomic by_min[] = min(by_min[], shared_min[1])
        CUDA.@atomic by_max[] = max(by_max[], shared_max[1])
        CUDA.@atomic grad_rms[] += shared_sum[1]
    end
    return
end



function _mean!(B, by_mean, N, Nmask)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid = threadIdx().x

    shared_sum = CuDynamicSharedArray(eltype(B), blockDim().x)

    # Inicializamos shmem con un valor
    shared_sum[tid] = zero(eltype(B))

    # Cargamos un punto y adicionalmente guardamos val²
    if idx <= N
        val = B[idx]
        shared_sum[tid] = val / Nmask
    end
   
    # Aseguramos que toda la memoria termino de ser cargada
    sync_threads()

    # Tree Reduction
    s = blockDim().x ÷ 2
    while s > 0
        if tid <= s
            shared_sum[tid] += shared_sum[tid + s]
        end
        sync_threads()
        s ÷= 2
    end

    # Updates to Global Memory
    if tid == 1
        CUDA.@atomic by_mean[] += shared_sum[1]
    end
    
    return
end

function _std!(B, by_mean,stdiv, N, Nmask)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    tid = threadIdx().x

    shared_sum = CuDynamicSharedArray(eltype(B), blockDim().x)

    # Inicializamos shmem con un valor
    shared_sum[tid] = zero(eltype(B))

    # Cargamos un punto y adicionalmente guardamos val²
    if idx <= N
        val = B[idx]
        if val == 0.0
            shared_sum[tid] = 0.0
        else
            μ = by_mean[1]
            shared_sum[tid] = ((val - μ)^2)/Nmask
        end
    end
   
    # Aseguramos que toda la memoria termino de ser cargada
    sync_threads()

    # Tree Reduction
    s = blockDim().x ÷ 2
    while s > 0
        if tid <= s
            shared_sum[tid] += shared_sum[tid + s]
        end
        sync_threads()
        s ÷= 2
    end

    # Updates to Global Memory
    if tid == 1
        CUDA.@atomic stdiv[] += shared_sum[1]
    end

    return
end

function f_val!(by_min, by_max, grad_rms, coef, λ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx == 1
        coef[idx] = w * (by_max[idx] - by_min[idx]) + λ * sqrt(grad_rms[idx])
    end
    return
end