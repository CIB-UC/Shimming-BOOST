
using GPUArrays: @allowscalar
using Printf

function _next_ring_activation!(on_off, ring_id, Nmags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx > Nmags; return; end

    @inbounds if ((ring_id - 1)*84 + 1 <= idx  &&  idx <= ring_id * 84)
        on_off[idx] = 1.0
    end
    return
end

function _mutate!(θ, ring_id, step_deg, lo, hi, Nmags, tresh)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx > Nmags; return; end
    if !((ring_id - 1)*84 + 1 <= idx  &&  idx <= ring_id * 84) ; return; end

    r_val = rand()

    @inbounds if r_val < tresh
        θ[idx] += step_deg * randn()
        θ[idx] = min(max(θ[idx], lo[idx]), hi[idx])

    end
    return

end

function _reset!(θ, on_off, ring_id, lo, hi, Nmags)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx > Nmags; return; end
    if !((ring_id - 1)*84 + 1 <= idx  &&  idx <= ring_id * 84) ; return; end

    @inbounds θ[idx] = lo[idx] + rand() * (hi[idx] - lo[idx])
    return
end

@inline function cool(t0, alpha, k)
    return t0 * (alpha^k)
end


function overwrite_cuarray!(x, xop)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x  

    N = length(x)
    if idx > N; return; end

    @inbounds xop[idx] = x[idx]
    return
end

function naive_SA_RMS!(f, λ, ring_sequence; lower=lower, upper=upper,
                     iters=10, restarts=10,
                     T0=10.0, alpha=0.85, step0=10.0, step_min=0.5,
                     report_every=10)
    
    lo = CuArray(lower)
    hi = CuArray(upper)

    f(θ_init, λ, on_off)
    bestθ_global = copy(θ_init)
    bestf_global = CuArray([0.0f0])
    #best_state_global = copy(on_off)

    @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(coef, bestf_global)

    Θ = copy(bestθ_global)

    nrings = length(ring_sequence)
    for (ring_pos, ring_id) in enumerate(ring_sequence)
        @cuda threads=512 blocks=cld(Nmagshim, 512) _next_ring_activation!(on_off, ring_id, Nmagshim)
        active = Int(sum(on_off))
        @printf("\n[ring %d/%d]  id=%d  active magnets=%d\n", ring_pos, nrings, ring_id, active)
        for r in 1:restarts
            if r == 1
            else
                @cuda threads=512 blocks=cld(Nmagshim, 512) _reset!(Θ, on_off, ring_id, lo, hi, Nmagshim)
            end

            f(Θ, λ, on_off)

            step = step0
            T = T0
            step_decay = (step0 > step_min) ? (step_min/step0)^(1/iters) : 1.0

            for k in 1:iters

                @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(Θ, Θmew)          # copia el valor de θ en Θmew
                #@cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(on_off, state_new)
                @cuda threads=512 blocks=cld(Nmagshim, 512) _mutate!(Θmew, ring_id, step, lo, hi, Nmagshim, 0.3)

                @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(coef, f_prev)

                f(Θmew, λ, on_off)

                @allowscalar if (coef[1] < f_prev[1]) || (rand() < exp(-(coef[1] - f_prev[1])/max(T,1e-9)))
                    @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(Θmew, Θ)
                    #@cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(state_new, on_off)
                    @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(coef, f_prev)
                    if coef[1] < bestf_global[1]
                        @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(coef, bestf_global)
                        @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(Θ, bestθ_global)
                        #@cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(on_off, best_state_global)
                    end
                end

                T = cool(T0, alpha, k)
                step = max(step*step_decay, step_min)
                @allowscalar begin
                    if (k % report_every == 0) || (k == iters)
                        @printf("\r  restart %d/%d  iter %5d/%-5d  f=%.5f  best=%.5f  T=%7.3f  step=%.2f   ",
                                r, restarts, k, iters, coef[1], bestf_global[1], T, step)
                        flush(stdout)
                    end
                end
            end
        end
        println()  # newline so each ring's final progress line is preserved
        @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(bestθ_global, Θ) # Fijamos la mejor configuracion antes de pasar al siguiente anillo
    end
    best_state_global = copy(on_off)
    return bestθ_global, best_state_global
end

function naive_SA_STDIV!(f, ring_sequence; lower=lower, upper=upper,
                     iters=2_000, restarts=5,
                     T0=0.1, alpha=0.995, step0=10.0, step_min=0.5,
                     report_every=50)
    
    lo = CuArray(lower)
    hi = CuArray(upper)

    bestf_global = copy(stdiv)
    bestθ_global = copy(θ_init)
    Θ = copy(bestθ_global)
    
    f(Θ, on_off)

    @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(stdiv, bestf_global)

    for ring_id in ring_sequence
        @cuda threads=512 blocks=cld(Nmagshim, 512) _next_ring_activation!(on_off, ring_id, Nmagshim)
        @info "Starting SA for ring=$ring_id"
        for r in 1:restarts
            if r == 1
            else
                @cuda threads=512 blocks=cld(Nmagshim, 512) _reset!(Θ, on_off, ring_id, lo, hi, Nmagshim)
            end

            f(Θ, on_off)

            step = step0
            T = T0
            step_decay = (step0 > step_min) ? (step_min/step0)^(1/iters) : 1.0

            for k in 1:iters
        
                @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(Θ, Θmew)          # copia el valor de θ en Θmew
                @cuda threads=512 blocks=cld(Nmagshim, 512) _mutate!(Θmew, ring_id, step, lo, hi, Nmagshim, 0.3)
                
                @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(stdiv, f_prev)

                f(Θmew, on_off)

                @allowscalar if (stdiv[1] < f_prev[1]) || (rand() < exp(-(stdiv[1] - f_prev[1])/max(T,1e-9)))
                    @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(Θmew, Θ)
                    @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(stdiv, f_prev)
                    if stdiv[1] < bestf_global[1]
                        @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(stdiv, bestf_global)
                        @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(Θ, bestθ_global)
                    end
                end

                T = cool(T0, alpha, k)
                step = max(step*step_decay, step_min)
                @allowscalar begin
                    if (k % report_every == 0)
                        @info "SA restart=$r iter=$k  f=$stdiv best=$bestf_global  T=$(round(T,digits=4))  step=$(round(step,digits=2))"
                    end
                    @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(CuArray([0.0f0]), stdiv)
                    @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(CuArray([0.0f0]), by_mean)
                end

                

            end
        end
        @cuda threads=512 blocks=cld(Nmagshim, 512) overwrite_cuarray!(bestθ_global, Θ) # Fijamos la mejor configuracion antes de pasar al siguiente anillo
    end
    best_state_global = copy(on_off)
    return bestθ_global, best_state_global
end