# The Gradient Optimizer — code & math

How the gradient-based shim optimizer works: the core idea, every file and
function, how they connect, the math, and how the optimization is actually solved.

---

## 1. The one idea everything rests on

The shim field is **exactly linear in the magnet moments**. Each magnet $i$ has a
fixed magnitude $\mu$ and a free angle $\theta_i$, so its in-plane moment is
$(\mu\cos\theta_i,\ \mu\sin\theta_i)$. The field component along $B_0$ (the "$B_y$"
we score) at any evaluation point is

$$
B_y(\theta)=b+G_x\,u+G_y\,v,\qquad u_i=\mu\cos\theta_i,\quad v_i=\mu\sin\theta_i,
$$

where

- $b\in\mathbb{R}^{N_s}$ — the measured base field (no shims) at the $N_s$ evaluation points,
- $G_x,G_y\in\mathbb{R}^{N_s\times N_m}$ — response matrices; column $i$ of $G_x$ is
  the $B_y$ response of magnet $i$ to a **unit x-moment** ($G_y$: unit y-moment).
  They depend only on **geometry**, not on the angles.

Because the field is linear, the homogeneity metrics become **smooth functions of
$\theta$ with analytic gradients**, so L-BFGS finds the global optimum in seconds.
The system splits into two phases: **build $G$ once (GPU)**, then **solve on $G$
repeatedly (CPU)**.

---

## 2. File map (dependency graph)

```
pipeline_config.jl ─┐  consts: paths, geometry, grad_* knobs, eval_domain
setup.jl / setup_shell.jl ─┤  GPU globals: grid, fld, msk, P, mu, M, B, N, threads, positions
kernels/f_kernel.jl ─┘  _M!, _Btot!  (dipole-field CUDA kernels)
        │
        ▼   PHASE 1 (GPU)
optim_grad.jl ──writes──▶ GradOpt/operator_G.jld2      ← the seam
        │
        ▼   PHASE 2 (CPU)
grad_math.jl   (pure math, no globals, no file loads)
   └── included by ──▶ grad_core.jl   (binds the math to the cached operator)
                            └── used by ──▶ run_grad.jl         (the driver)
                                         ──▶ run_grad_lcurve.jl (λ sweep)
                                         ──▶ utils/eval_metrics.jl (scoring)
grad_math.jl also used directly by ──▶ ring_search.jl    (per-ring operator, cached)
                                   ──▶ insert_search.jl  (SAME cache, sliced per tray)
```

The **seam is `operator_G.jld2`**: Phase 1 writes it, Phase 2 reads it. This
decouples the expensive one-time GPU build from the cheap, repeatable CPU solve.

---

## 3. Supporting inputs (grad depends on these)

**`pipeline_config.jl`** — every const the grad path reads: `optimizer_iter_dir`,
`grad_data_term`, `grad_softrange_beta`, `grad_lambda`, `grad_seeds`,
`grad_lambda_sweep`, `grad_result_path`, `optimizer_result_path`, `eval_domain`,
geometry.

**`setup.jl` (grid) / `setup_shell.jl` (shell)** — build the GPU evaluation context:

- `grid` — a `GridGPU` holding evaluation-point coords $(X,Y,Z)$ in metres. Grid
  mode: a Cartesian mesh; shell mode: the shell points reshaped to $(N_s,1,1)$.
  `_Btot!` indexes them linearly, so the shape doesn't matter.
- `fld` — base field $b$ at those points (mT).
- `msk` — shell mask (1 inside the scored region) → `shell_idx`.
- `P` (3×$N_m$ magnet positions, metres), `mu` (per-magnet $\mu$), `M`,`B` (kernel
  scratch), `m`, `N`, `threads`, `blocks`, `Nmagshim`, `θ0`.

**`kernels/f_kernel.jl`** — the two CUDA kernels the build uses:

- `_M!(θ, μ, current, m, M)` — moment matrix: $M[1,i]=\cos\theta_i\,\mu_i\,\text{state}_i$,
  $M[2,i]=\sin(\cdot)$, $M[3,i]=0$.
- `_Btot!(fieldmap, B, X, Y, Z, P, M, m, N)` — for every eval point, sums the base
  `fieldmap` plus each magnet's dipole $B_y$ contribution → total $B_y$ into `B`.

---

> **Frame note.** Everything on this page — the field map $b$, the shell points, the magnet positions,
> $G$ and the solved $\theta$ — lives in the **optimizer frame**: Stage 0 has rotated the scan about the bore so
> that $B_0\to+y$, and the field is stored as $|B|>0$ (so "$B_y$" is the component along $B_0$). The
> optimizer therefore never sees the scan frame. The conversion back is *outside* this code: `export_csv.jl`
> and `insert_search.jl` rotate the exported positions and angles by the scan's $B_0$ direction
> (`shim_csv_frame`, default `"scan"`; $-y$: $(x,y)\to(-x,-y)$, $\theta\to\theta+180^\circ$). The result jld2s and
> cached operators stay in the optimizer frame. Checked independently (Python dipole model on a real
> $-y$ scan): the back-rotated layout gives 9,601 ppm vs 19,155 unshimmed; see `README.md`, "Coordinate frames".

## 4. PHASE 1 — `optim_grad.jl` (build & cache $G$, GPU)

Picks `setup.jl` or `setup_shell.jl` from `eval_domain`, then `f_kernel.jl`.

- **`build_G()`** — extracts $G$ column by column. For each magnet $i$: set a unit
  x-moment on **only** that magnet (`Munit[1,i]=1`), run `_Btot!` with a **zero base
  field**, read the resulting $B_y$ — that *is* column $i$ of $G_x$. Repeat with a
  unit y-moment for $G_y$. Because it drives the *same kernel* the SA uses, $G$
  matches it by construction. Stores both the shell-restricted $G_x,G_y$ (rows =
  shell points, for the variance term) and the full-grid $G_x^{\text{full}},
  G_y^{\text{full}}$ (all points, needed by the $\nabla B$ term which requires
  neighbours). Returns all four.
- **`validate_G(Gx, Gy; θdeg, state)`** — sanity check: compute the field two ways at
  the initial angles — (1) directly via the kernel (`_M!` then `_Btot!`), and (2) via
  the linear model $b+G_x u+G_y v$ in Float64 — and report max/relative difference
  (the "rel err ≈ 4e-8"). A large value means the moment convention or indexing is
  wrong.
- The script then computes `By_base_shell/full`, `mask_full`, dims, spacing
  `d_grad_m` (= `dy_m`), sets `is_shell`, and **`@save`s `operator_G.jld2`** with keys:
  `shell_idx, Ns, Nmag, Gx, Gy, By_base_shell, mu_cpu, Gx_full, Gy_full,
  By_base_full, mask_full, Nmask, nx, ny, nz, d_grad_m, is_shell`.

---

## 5. `grad_math.jl` — pure, shared math (no globals, no file loads)

Side-effect-free, so it can be included without any operator present.

- **`_data_variance(By)`** → $(\text{value},\ \partial/\partial B_y)$:
  $\mathrm{Var}=\tfrac1n\sum_k(B_{y,k}-\bar B_y)^2$, gradient $\tfrac2n(B_y-\bar B_y)$.
- **`_data_softrange(By, β)`** → smooth $(\max-\min)$ via log-sum-exp; gradient
  $p-q$ (softmax minus softmin weights).
- **`make_data_cost(GX, GY, b, mu; data_term, beta)`** — operator-**agnostic** cost
  builder: for any shell operator, returns an Optim closure `fg!(F,G,θ)`. This is
  what `ring_search` uses on its per-ring column subsets.
- **`shell_metrics(GX, GY, b, mu, θ)`** → $(\text{range},\ \text{mean},\ \mathrm{ppm})$.
- **`fd_gradcheck(fg!, θ)`** — finite-difference check of any analytic gradient
  (the "self-check" printed at startup).

---

## 6. `grad_core.jl` — binds the math to the cached operator

At include time: loads `operator_G.jld2`, exposes the arrays as Float64 globals
(`GXF,GYF,BF` full-grid; `GXS,GYS` shell; `MU,SHELL,MASKV,NS,NM`, dims), reads
`IS_SHELL`, and includes `grad_math.jl`.

- **`_fd_axis(nx,ny,nz,d,axis)`** — sparse finite-difference matrix for one axis
  (central $/(2d)$ interior, one-sided $/d$ at faces), matching `kernels/_grad!`.
  `DX,DY,DZ` are these three — or `nothing` in shell mode (no mesh).
- **`byfield(θ, state)`** — the full-grid field $BF+GXF\,u+GYF\,v$.
- **`field_metrics(θ, state)`** → $(\text{range},\text{mean},\mathrm{ppm},\text{grad\_rms})$;
  range/ppm over the shell slice, $\text{grad\_rms}=\sqrt{\text{mean}|\nabla B|^2}$
  (or `NaN` in shell mode). Defined identically to the SA's, so the two optimizers
  are comparable.
- **`_var_term(By)` / `_softrange_term(By, β)`** — wrap the pure `_data_*` functions
  on `By[SHELL]`, scattering the gradient back to a full-grid vector.
- **`_gradpen_term(By)`** — the $\nabla B$ (Tikhonov) penalty + adjoint.
- **`build_cost(; data_term, grad_lambda, beta)`** — assembles the objective closure
  `fg!(F,G,θ)`: compute $B_y$, the data term, optionally add $\lambda\cdot\nabla B$,
  then chain to $\theta$ (below). In shell mode it forces $\lambda=0$ and warns.

The **Optim `fg!(F,G,θ)` convention**: one closure serves value *and* gradient. If
`G` isn't `nothing`, fill it in place; if `F` isn't `nothing`, return the value —
so $B_y$ is computed once, not twice.

---

## 7. PHASE 2 — `run_grad.jl` (the driver)

- Includes config + `grad_core.jl`; builds `fg! = build_cost(...)` from config knobs.
- Runs `fd_gradcheck` once (prints ~1e-6).
- Splits `fg!` into Optim callbacks: `f_cost(θ)=fg!(true,nothing,θ)`,
  `g_cost!(G,θ)=(fg!(nothing,G,θ);G)`.
- **`run_lbfgs(θ0)`** — one L-BFGS solve (`g_tol=1e-12`, ≤5000 iters).
- **`run_multistart(seeds)`** — solves from `grad_seeds` random starts, scores each
  with `field_metrics`, keeps the best ppm (10/10 identical ⇒ global optimum).
- Saves the winner (`λ, bestθ` in degrees, `final_state=ones`, `ppm`) to **both**
  `grad_result_path` (tagged copy) and `optimizer_result_path` (the active Stage-1
  result that `export_csv` → `CSV_to_STL` consume).

**Insert search — `insert_search.jl`.** Same operator, finer unit and a different
search. A ring block of the cached `ring_operator.jld2` is `num_trays × mags_per_segment`
columns, so one **insert** (one tray at one ring) is the `MPS`-column slice
`((s−1)·MPS+1 : s·MPS)` — no new GPU build, and the cache is shared with the ring
search via an identical `SIG`.

The `s` above indexes into `best_rings` (the candidate rings this search is
choosing among) purely to address that column slice — it is an internal,
never-exported array position (`ring_slot` in the code, explicitly scoped local to
the placement block). What actually gets **written to the CSV's RingNumber column**
and printed in the placement-plan log is `p.ring`, the real physical InsertPos —
the two are related only by a lookup dict, never conflated.

Because `make_data_cost(GX, GY, b, mu; …)` takes the base field `b` as an argument, a
placed insert can be *folded into `b`* instead of staying in the variable set:

$$
b^{(k+1)} \;=\; b^{(k)} \;+\; G_x^{(k)}\,\mu\cos\theta^{(k)} \;+\; G_y^{(k)}\,\mu\sin\theta^{(k)},
\qquad
\theta^{(k)} \;=\; \arg\min_\theta\; \mathrm{data}\big(b^{(k)} + G_x^{(k)}u + G_y^{(k)}v\big)
$$

with the slot $(k)$ chosen at each round as the feasible slice minimising that same
objective. This is **matching pursuit**: each round is an $M_{PS}$-variable solve
regardless of how many inserts are already placed, so cost per round is flat, and the
frozen angles are never revisited. That is the deliberate difference from
`ring_search`'s greedy, which re-solves all placed rings jointly at every round
(and whose step $k$ therefore costs $M_{PR}\!\cdot\!k$ variables).

The trade is the usual one for greedy pursuit: it decides *where* very cheaply but
cannot undo an early angle choice, so its ppm is an upper bound on what a joint
re-solve of the same slots would reach. Note also that $\lVert m_i\rVert = \mu$ is
fixed — a magnet rotates but never switches off — so the sequence of ppm values is
**not** guaranteed monotone, and the script reports the best prefix rather than
assuming the last step is best.

**Ranking mode — `RANK_FULL_SHELL` (`insert_search.jl`).** A local `const` toggle
controls how each round's candidate slots are ranked:
- `false` — rank every free legal slot cheaply on a **~250-point sub-shell**
  (`RANK_NS = min(Ns, 250)`, rows picked evenly across the shell), then re-solve
  only the winner on the full `Ns`-point shell for the reported ppm. This is the
  fast mode (≈30× faster ranking than full-shell).
- `true` (current default in the file) — rank **every** candidate on the **full**
  shell (`RANK_NS = Ns`, all rows). An accuracy/speed check on whether the
  subsample would ever have picked a different "best slot" than scoring against
  every point; costs `Ns/RANK_NS` more per round (e.g. 1742 vs 250 ⇒ ~7× slower).

Both modes share the same downstream logic (frozen sequential placement, best-prefix
reporting); only which rows feed the per-round ranking solve differs. Toggle it back
to `false` for routine fast placement once cross-checked against a full-shell run.

**Related:** `run_grad_lcurve.jl` sweeps `grad_lambda` → range-vs-gradient L-curve;
`ring_search.jl` builds its own per-ring operator, cached and then converted to
**Float64** so the tens of thousands of subset solves hit BLAS (and actually reach
`g_tol` instead of stalling at the Float32 noise floor), and **ranks** each candidate
subset with `grad_math` on a small **~250-point sub-shell** — only the winning subset is
re-solved on the full shell for the reported ppm (≈30× faster than full-shell ranking);
`utils/eval_metrics.jl` scores any saved result with `field_metrics`.

---

## 8. The math

**Field model** ($r=B_y-\bar B_y\mathbf 1$):

$$
B_y(\theta)=b+G_x u+G_y v,\qquad u_i=\mu\cos\theta_i,\ v_i=\mu\sin\theta_i.
$$

**Variance:**

$$
\mathrm{Var}(\theta)=\frac{\lVert r\rVert^2}{N_s},\qquad
\frac{\partial\,\mathrm{Var}}{\partial B_y}=\frac{2}{N_s}r.
$$

**Soft-range** (sharpness $\beta$):

$$
S=\frac1\beta\log\!\sum_k e^{\beta B_{y,k}}+\frac1\beta\log\!\sum_k e^{-\beta B_{y,k}},\qquad
\frac{\partial S}{\partial B_y}=p-q,\quad
p_k=\frac{e^{\beta B_{y,k}}}{\sum_j e^{\beta B_{y,j}}},\ \
q_k=\frac{e^{-\beta B_{y,k}}}{\sum_j e^{-\beta B_{y,j}}}.
$$

**Gradient penalty** (FD operators $D_a$, mask $m$):

$$
\mathrm{GradPen}=\frac{1}{N_{\text{mask}}}\sum_{a\in\{x,y,z\}}\lVert m^{1/2}\!\odot D_aB_y\rVert^2,\qquad
\frac{\partial\,\mathrm{GradPen}}{\partial B_y}=\frac{2}{N_{\text{mask}}}\sum_a D_a^{\!\top}(m\odot D_aB_y).
$$

**Full objective:**

$$
J(\theta)=\mathrm{data}(\theta)+\lambda\,\mathrm{GradPen}(\theta).
$$

**Gradient chain to the angles.** With the field-space adjoint
$\mathrm{adj}=\partial J/\partial B_y$ and $B_y=b+G_xu+G_yv$:

$$
\frac{\partial J}{\partial u}=G_x^{\!\top}\mathrm{adj},\quad
\frac{\partial J}{\partial v}=G_y^{\!\top}\mathrm{adj},\qquad
\boxed{\ \frac{\partial J}{\partial\theta_i}
=-\mu\sin\theta_i\,(G_x^{\!\top}\mathrm{adj})_i
+\mu\cos\theta_i\,(G_y^{\!\top}\mathrm{adj})_i\ }
$$

(from $\partial u_i/\partial\theta_i=-\mu\sin\theta_i$, $\partial v_i/\partial\theta_i=\mu\cos\theta_i$).
The fixed magnitude $\lVert m_i\rVert=\mu$ is built in for free by using $\theta$.

**Metrics:**

$$
\text{range}=\max_kB_{y,k}-\min_kB_{y,k},\quad
\mathrm{ppm}=10^6\frac{\text{range}}{\bar B_y},\quad
\text{grad\_rms}=\sqrt{\tfrac{1}{N_{\text{mask}}}\textstyle\sum_a\lVert m^{1/2}\!\odot D_aB_y\rVert^2}.
$$

---

## 9. How the optimization is solved

With $J(\theta)$ and $\nabla J(\theta)$ available analytically, this is **smooth
unconstrained minimization** over $\theta\in\mathbb{R}^{N_m}$:

$$
\theta^\star=\arg\min_\theta J(\theta).
$$

**Tool:** the `Optim.jl` package. **Function:** `Optim.optimize` with the **L-BFGS**
algorithm. The literal call (`run_grad.jl`):

```julia
Optim.minimizer(
    optimize(f_cost, g_cost!, θ0, LBFGS(),
             Optim.Options(g_tol = 1e-12, iterations = 5000))
)
```

- `f_cost(θ)` returns $J$; `g_cost!(G,θ)` writes $\nabla J$ into `G` in place.
- `LBFGS()` selects the algorithm; `g_tol=1e-12` stops when
  $\lVert\nabla J\rVert_\infty\le10^{-12}$; `Optim.minimizer` extracts $\theta^\star$.

**What L-BFGS is.** A quasi-Newton method. Newton would step
$\theta\leftarrow\theta-[\nabla^2J]^{-1}\nabla J$, but the Hessian
$\nabla^2J$ ($N_m\times N_m$) is expensive to form/invert. BFGS approximates the
inverse Hessian $H_k\approx[\nabla^2J]^{-1}$ from the history of gradient changes
$y_k=\nabla J_{k+1}-\nabla J_k$ and steps $s_k=\theta_{k+1}-\theta_k$; **L-BFGS**
(limited-memory) keeps only the last few $\{s_k,y_k\}$ pairs, so each iteration is
$O(N_m)$. Per iteration: evaluate $g=\nabla J$, form $d=-H_kg$, line-search along
$d$, update $\theta$ and the stored pairs. It needs **only values and gradients** —
exactly what we supply — no Hessian.

**Why it reaches the global optimum.** $\mathrm{Var}$ is convex in the moments
$(u,v)$; the $\theta$-parametrization makes it formally non-convex, but the
landscape is empirically unimodal. The safety net is multi-start:

$$
\theta^{(s)}=\text{L-BFGS}(\theta_0^{(s)}),\quad
\theta_0^{(s)}\sim\mathrm{Unif}[0,2\pi)^{N_m},\quad
\theta^\star=\arg\min_s\mathrm{ppm}(\theta^{(s)}).
$$

When all starts converge to the same $\theta^\star$ and identical ppm, that is the
global minimum.

**End-to-end:** `build_cost` → the `fg!` closure ($J$, $\nabla J$) →
`f_cost`/`g_cost!` → `Optim.optimize(…, LBFGS())` → `Optim.minimizer` →
$\theta^\star$, repeated over `grad_seeds`, best kept.
