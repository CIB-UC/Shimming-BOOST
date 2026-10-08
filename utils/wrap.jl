include("../kernels/f_kernel.jl")
include("../kernels/op_kernel.jl")

function RMS_operation( θop, λ, current_state )

    @cuda threads=threads blocks=blocks                   _M!(θop, mu, current_state, m, M)

    @cuda threads=threads blocks=blocks                   _Btot!(fld_field, B,                                
                                                            grid.X, grid.Y, grid.Z,              
                                                            P, M, m, N)                          
    
    @cuda threads=threads blocks=blocks                   _grad!(B, Gx, Gy, Gz, dy_m, grid.nx, grid.ny, grid.nz, N)
    
    @cuda threads=threads blocks=blocks shmem=shmem_RMS   _metrics!(B, by_min, by_max, grad_rms, Gx, Gy, Gz, msk, N, Nmsk)        
    @cuda threads=threads blocks=blocks                    f_val!(by_min, by_max, grad_rms, coef, λ)
    
    return
end


function STDIV_operation( θop, current_state )

    @cuda threads=threads blocks=blocks                   _M!(θop, mu, current_state, m, M)

    @cuda threads=threads blocks=blocks                   _Btotmasked!(fld_field, B,                                             
                                                                grid.X, grid.Y, grid.Z,     
                                                                P, M, m,                                              
                                                                msk, N)                                                 

    @cuda threads=threads blocks=blocks shmem=shmem_sum   _mean!(B, by_mean, N, Nmsk)

    @cuda threads=threads blocks=blocks shmem=shmem_sum   _std!(B, by_mean, stdiv, N, Nmsk)

    return
end