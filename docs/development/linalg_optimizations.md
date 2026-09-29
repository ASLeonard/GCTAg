# Linear Algebra Optimisations in GCTAg

This document catalogues the performance-relevant design of `GCTAg`, a fork of GCTA built to scale genomic mixed-model analyses (GRM construction, REML, MLMA, PCA) to cohorts of hundreds of thousands of individuals. It is split into two parts:

- **Part I — Core Performance** covers the algorithms, numerical kernels, and memory-scaling strategies. Mathematical context is given for each change.
- **Part II — Supporting Changes** covers build/BLAS portability, statistical libraries, I/O, and the command-line flags that activate the Part I features.

Three ideas recur throughout Part I:

1. **Touch one triangle only.** Symmetric matrices are stored, read, and updated as a single triangle, halving memory traffic and, when untouched pages are never faulted in, resident memory.
2. **Never require the whole GRM in memory.** The GRM can be streamed from disk in tiles within an explicit memory budget.
3. **Replace $V^{-1}$ by structure.** A low-rank-plus-isotropic (Woodbury) surrogate for the GRM, and stochastic (Hutch++) trace estimates, remove the $O(n^2)$ matrices and $O(n^3)$ operations from REML and MLMA.

For the list of user-facing flags see [GCTAg.md](../changes/GCTAg.md); §10 maps each flag to the section that describes it.

---

# Part I: Core Performance

---

## 1. GRM Construction (`src/GRM.cpp`)

The genomic relationship matrix is defined as:

$$G = \frac{1}{m} Z Z^T, \qquad Z_{ij} = \frac{x_{ij} - 2p_j}{\sqrt{2p_j(1-p_j)}}$$

where $x_{ij}$ is the raw allele count for individual $i$ at marker $j$ and $p_j$ is the allele frequency. The computation proceeds in blocks of $k$ markers at a time, applying rank-$k$ updates $G \mathrel{+}= \frac{1}{m} Z_k Z_k^T$ per block.

### 1.1 Raw BLAS → Eigen `selfadjointView` / `rankUpdate`

```cpp
// Full-sample partition (symmetric rank-k update = DSYRK)
Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>> A(stdGeno, grm_n, k, Eigen::OuterStride<>(stdGenoLD));
Eigen::Map<Eigen::MatrixXd>(grm, grm_n, grm_n)
    .selfadjointView<Eigen::Lower>().rankUpdate(A, alpha);

// Cross-partition (rectangular block = DGEMM + DSYRK)
Eigen::Map<...> A_top(stdGeno, grm_s_n, k, stride);
Eigen::Map<...> A_bot(stdGeno + part_keep_indices.first, grm_m, k, stride);
Eigen::Map<Eigen::MatrixXd>(grm, grm_m, grm_s_n).noalias() += alpha * (A_bot * A_top.transpose());
Eigen::Map<Eigen::MatrixXd>(grm_start, grm_m, grm_m)
    .selfadjointView<Eigen::Lower>().rankUpdate(A_bot, alpha);
```

`selfadjointView<Lower>::rankUpdate` maps to BLAS `DSYRK`, writing only the lower triangle. This removes platform-specific `#if` guards around raw Fortran/C BLAS ABIs and makes the intent explicit to both the compiler and the reader.

### 1.2 Aligned column stride (`stdGenoLD`)

AVX/AVX-512 BLAS kernels prefer 64-byte aligned leading dimensions. The stride is rounded up:

$$\text{stdGenoLD} = \left\lceil \frac{n}{8} \right\rceil \times 8$$

```cpp
stdGenoLD = (grm_n + 7) / 8 * 8;
```

Every column of the standardised genotype buffer is then 64-byte aligned, avoiding the scalar fallback path for unaligned loads.

### 1.3 Cached derived constants

Values derived from the partition are computed once in the constructor rather than per call:

```cpp
grm_n = static_cast<int>(part_keep_indices.second) + 1;
grm_m = static_cast<int>(part_keep_indices.second) - static_cast<int>(part_keep_indices.first) + 1;
grm_s_n = grm_n - grm_m;
grm_bytes_std_geno = sizeof(double) * grm_n;
stdGenoLD = (grm_n + 7) / 8 * 8;
```

### 1.4 Loop hoisting: fewer OpenMP fork/join barriers

The per-block missingness bookkeeping is restructured so all blocks are filled first (serial, cheap) and a **single** parallel region then iterates over GRM pairs:

```
for i in 0..numNblock:
    fill sampleMissBuf[i]

#pragma omp parallel for       ← single fork/join
for pair in index_grm_pairs:
    for i in 0..numNblock:
        N_thread(sampleMissBuf[i])
```

This reduces OpenMP overhead from $2 \times n_{\text{block}}$ barriers to one, and keeps each thread's `N[]` output range hot in L1/L2 across the inner block loop.

### 1.5 Pre-allocated scratch buffers

```cpp
validIndexBuf.reserve(nMarkerBlock);
sampleMissBuf.resize(numNSampleBlock * markerPerN * numNblockMax);
```

Scratch buffers are allocated once in the constructor and reused, eliminating per-block heap allocation in the hot marker loop.

### 1.6 `std::popcount`

```cpp
sub_miss[k] += std::popcount(block_buf[k]);  // C++20, inlines to POPCNT
```

### 1.7 Block and tile budgets

- **`--nMarkers N`** sets the number of SNPs processed per rank-$k$ block. The default is 1024 (previously a hardcoded 128), giving 8× fewer rank-$k$ dispatches. Larger values use more memory and eventually become cache-bound, so larger is not always faster.
- **`--GRM-tile-budget G`** processes at most `G` gigabytes of GRM tiling at a time within a single call — the same idea as `--make-grm-part`, but automatic. The budget is a soft constraint because other buffers coexist with the tiles.
- **`--merge-grms [G=0.5]`** merges the GRMs listed in the `--mgrm` file as an $N$-weighted average, entirely streamed (see §2.2): no input or output GRM is ever held as a dense matrix.

---

## 2. Triangular Storage and Streaming GRM Access

At $n = 150{,}000$ a dense double-precision GRM is 180 GB. This section describes how GCTAg avoids holding it, or half of it, in memory.

### 2.1 Upper-triangle-only storage

The GRM loader has an `upper_only` mode that skips the upper→lower mirror pass. Only the upper triangle ($\text{row} \le \text{col}$) holds valid data. Two effects:

- **Half the write bandwidth** on load, since there is no mirror pass.
- **Half the resident memory** when the destination matrix is freshly allocated: the untouched lower-triangle pages are never faulted in by the kernel, so they never count towards RSS. (This depends on the allocator returning lazily-faulted pages; with `USE_JEMALLOC`, transparent huge pages are disabled for exactly this reason, see §7.)

Every consumer then reads the matrix as upper-triangle-only:

- dense products use `selfadjointView<Eigen::Upper>()`;
- column reads use `col(j).head(j+1)` (rows on or above the diagonal), which is contiguous;
- the exact symmetric eigensolver uses LAPACK `dsyevr` (see §3), which references only the upper triangle and needs $O(k)$ workspace, versus `dsyevd`, which overwrites its whole input with eigenvectors.

The same principle is applied to $V$, $V^{-1}$, and $P$ in REML (§4.1).

### 2.2 Sequential `read()` streaming

The GRM binary (`.grm.bin`, float32, packed lower triangle by row) is read with a single blocking `read()` stream in row-aligned chunks (default 1 GiB), not `mmap`. With `mmap`, several threads faulting different regions concurrently present the kernel and the parallel filesystem with interleaved access offsets, defeating readahead and fragmenting reads across storage targets. Now only the float→double cast and scatter into the destination are multithreaded, once a chunk is already in memory. Chunk boundaries come from a closed-form triangular row partition, so no chunk ever splits a packed row.

The same pattern is used for `--merge-grms`, which reads each input block sequentially and parallelises only the in-memory mixing arithmetic.

### 2.3 Out-of-core symmetric matrix-vector products

`chunked_symmetric_matvec` computes $Y = KX$ for a symmetric $K$ that is never held densely. Rows and columns are split into blocks $B_0, \dots, B_{m-1}$ of size $b$, and each lower-triangular tile $K[B_i, B_j]$ ($j \le i$) is read **exactly once** per product:

- **Off-diagonal tile** ($j < i$): used directly for $Y[B_i] \mathrel{+}= K_{ij} X[B_j]$ and, via its transpose, for $Y[B_j] \mathrel{+}= K_{ij}^T X[B_i]$. The upper triangle is never read; it is supplied algebraically.
- **Diagonal tile** ($j = i$): only its own lower triangle is valid, so it is used through a lazy `selfadjointView<Lower>`, which dispatches to a symmetric BLAS product without materialising a mirrored copy.

Companion routines need only the diagonal tiles' diagonals ($\text{tr}\,K$), or one pass over all tiles ($\text{tr}\,K^2 = \sum_{ij} K_{ij}^2$).

`ChunkedGrmReader` supplies the tiles. It serves the GRM in *analysis* sample order (after `--keep`/`--remove`), keeps only a bounded row-band cache resident, and reads with `pread`. It requires the analysis-to-file index map to be order-preserving, since the band cache only pays off when each band is read once and reused by every tile touching those rows. Tiles are stored as float32 and widened to double once, per tile.

### 2.4 Memory budget

All streaming features share `--grm-chunked-budget <G>`. The block size is solved from the budget:

$$b = \Big\lfloor \sqrt{\frac{\text{budget} - \text{reserved} - 8\,n\,k_{\text{ext}}}{2 \cdot 8}} \Big\rfloor$$

- $8 n k_{\text{ext}}$ bytes is the output accumulator $Y$ ($n \times k_{\text{ext}}$ doubles), allocated in full up front. $k_{\text{ext}}$ is the worst-case number of columns the chunk size will be multiplied against, since $b$ is solved once and held fixed.
- `reserved` is the reader's row-band cache, about 10% of the budget capped at 512 MB. It is computed in one shared function so the reader and the block-size solver cannot drift apart.
- The denominator accounts for two $b \times b$ double tile buffers, independent of $n$ and $k_{\text{ext}}$.
- If $b \ge n$, chunking buys nothing and the dense path is used. If the budget cannot fit even one row, the run stops with an error.

---

## 3. Randomised Eigendecomposition and PCA

`symmetric_eigendecomp.hpp` is shared by PCA and by the Woodbury basis construction (§4.4). Every entry point takes a **matvec functor** $X \mapsto AX$ rather than a matrix, so identical code runs on a dense upper-triangle GRM (`selfadjointView<Upper>`) or on the out-of-core product of §2.3.

| Method | Selected by | Notes |
|---|---|---|
| Randomised range finder + power iteration + Rayleigh–Ritz | `--pca-approx rSVD`, `--svd-method power` (default) | $(q+1)$ block products of width $k + \text{oversample}$ dominate the cost; $q$ set by `--svd-power-iter` |
| Nyström (single pass) | `--svd-method nystrom` | One matvec pass instead of $q+1$. Uses a *signed* pseudo-inverse so the small negative eigenvalues of a GRM with missing genotypes are retained rather than discarded. Can fail on poorly conditioned matrices |
| Lanczos (Spectra) | `--pca-approx Lanczos` | Implicitly restarted; suited to well-separated top spectra. Wrapped around the same block matvec |
| Exact | `--pca` alone | `dsyevr`/`dsyevd` on the upper triangle |

For $k \ll n$, Lanczos and randomised methods cost $O(k n^2)$ against $O(n^3)$ for the full decomposition, a ratio of about $k/n$.

Implementation choices:

- **CholeskyQR2 orthogonalisation.** Householder QR (`dgeqrf`/`dorgqr`) is replaced by DSYRK + DPOTRF + DTRSM, applied twice. This is pure Level-3 BLAS, so it keeps scaling with thread count at large $k$, where the Householder panel factorisation stops parallelising. If a Gram matrix is not positive definite it falls back to Householder QR.
- **Rayleigh–Ritz on the upper triangle** (`dsyevr`). Peak memory for the small $k \times k$ problem is about $k^2/2 + k^2$, versus about $3k^2$ for `dsyevd`. Every Rayleigh–Ritz call site reads the same triangle, which matters because the Woodbury rank criteria read tail eigenvalues straight from this step.
- **Locked expansion.** To grow a converged rank-$k_0$ basis $(V_0, \theta_0)$ to $k_{\text{ext}}$, only $m = k_{\text{ext}} - k_0$ *new* directions are sketched and power-iterated in the orthogonal complement of $V_0$ (iterating $(I-P_0)A(I-P_0)$, $P_0 = V_0V_0^T$), followed by **one** exact Rayleigh–Ritz on the orthonormal union $W = [V_0, Q_1]$:

  $$W^T A W = \begin{bmatrix} \operatorname{diag}(\theta_0) & V_0^T A Q_1 \\ Q_1^T A V_0 & Q_1^T A Q_1 \end{bmatrix}$$

  $AV_0$ is never needed, and across a doubling schedule each column is touched once. By Cauchy interlacing the captured trace at any fixed rank cannot decrease between rounds. Two runtime checks guard the construction (an operator probe carried on the first product, and an orthogonality check after orthonormalisation); any failure falls back to a cold sketch.
- **Warm start.** A previous basis or a PCA eigenvector file can seed the random sketch.
- **Parallel Gaussian sketch** with one generator per thread, each seeded from the shared generator. The sketch therefore depends on thread count and is not bit-reproducible across thread counts.

**Streaming PCA.** `--pca [N]` with no `N` returns all eigenvalues. `--grm-chunked-budget` streams the GRM through §2.3, which requires an approximate method (an exact decomposition needs the dense matrix). When the solved block size reaches $n$, the dense path is used, because a single block would pay the diagonal-tile cost across the whole matrix.

---

## 4. REML

REML maximises the restricted log-likelihood:

$$\ell(\boldsymbol{\sigma}^2) = -\frac{1}{2}\left(\log|V| + \log|X^T V^{-1} X| + y^T P y\right)$$

where $V = \sum_{i} \sigma_i^2 A_i + \sigma_e^2 I$ and $P = V^{-1} - V^{-1}X(X^T V^{-1}X)^{-1}X^T V^{-1}$. Each Newton step needs the score $\partial\ell/\partial\sigma_i^2 = -\frac{1}{2}\bigl(\text{tr}(PA_i) - y^T PA_i Py\bigr)$ and the average-information (AI) matrix $H_{ij} = \frac{1}{2}\,y^T PA_i PA_j Py$.

`RemlEngine.cpp` implements single-GRM univariate REML with three interchangeable representations of $V^{-1}$: a dense Cholesky factor, a Woodbury low-rank form, and (optionally) an explicit dense $V^{-1}$. Sections 4.1–4.3 describe the dense kernels, 4.4 the Woodbury representation, 4.5 Hutch++, and 4.6 the iteration itself.

### 4.1 Dense-$V$ kernels

**Upper-triangle assembly.** $V = \sum_i \sigma_i^2 A_i$ is symmetric positive definite and LAPACK `dpotrf('U')` reads only its upper triangle, so only that triangle is assembled. With upper-triangle-only $A_i$ (§2.1), both the read and the write are column-contiguous (`col(j).head(j+1)`), directly vectorisable and parallel over columns. When the GRM is streamed, $V$'s upper triangle is filled block by block from the tiles, so the GRM itself is never resident; $V$ still is (a single $n \times n$ upper triangle).

**Factorise-only.** When only $\log|V|$ and solves against $V$ are needed (Hutch++ and MLMA modes), `dpotri` is skipped. The Cholesky factor $V = U^TU$ is kept and

$$\log|V| = 2 \sum_{i=1}^n \log U_{ii}$$

is read from its diagonal. This avoids the $O(n^3)$ inversion and the second $n \times n$ matrix.

**$P$ by symmetric rank update.** When $P$ is materialised, let $L = \text{chol}(X^T V^{-1}X)$ (a $c \times c$ factor, $c$ = number of covariates) and $Z = V^{-1}X L$ (an $n \times c$ matrix). Then

$$P = V^{-1} - Z Z^T,$$

a symmetric rank-$c$ update evaluated with `selfadjointView<Upper>().rankUpdate(Z, -1)` (DSYRK), writing only one triangle. The upper triangle is what `dpotri('U')` produced, so no mirror copy is made; a full mirror is built on demand only in the rare case that $X^TV^{-1}X$ is not positive definite.

### 4.2 Implicit $P$ products

Rather than materialising $P$, `applyP_vec` / `applyP_mat` compute

$$Pv = V^{-1} v - V^{-1}X \underbrace{(X^T V^{-1} X)^{-1}}_{c \times c} X^T V^{-1} v .$$

The $V^{-1}v$ term dispatches on the active representation: the Woodbury form (§4.4), two triangular solves against the Cholesky factor, or a symmetric product with an explicit $V^{-1}$. The correction term involves only $c$-dimensional quantities. Neither AI-REML nor Hutch++ therefore ever loads $P$.

### 4.3 Exact traces and the AI matrix

With $P$ available, the trace and information entries reduce to Hadamard sums via $\text{tr}(BC) = \sum_{kl} B_{kl}C_{lk}$:

```cpp
Hi(i, j) = PA[i].cwiseProduct(PA[j].transpose()).sum();    // tr(PA_i PA_j)
```

- $PA_i$ is formed with a symmetric product against the stored triangle of $A_i$ (`selfadjointView<Upper>() * P`). $P$ itself is symmetrised first and used as a plain dense operand, because on AOCL/Zen4 `dsymm` is slower than `dgemm` for the rectangular products against $U_k$.
- The tiny $c \times c$ traces in the score use the same `cwiseProduct(...).sum()` identity instead of a $c \times c$ GEMM, whose call overhead would exceed the arithmetic.
- With a streamed GRM, $PA_0$ is produced by the chunked matvec of §2.3.

### 4.4 Woodbury low-rank REML

The GRM is replaced by a rank-$k$ plus isotropic surrogate built from its top eigenpairs $(U_k, d_k)$ and a flat tail eigenvalue $\lambda_{\text{tail}}$:

$$K \approx \lambda_{\text{tail}} I + U_k\,(D_k - \lambda_{\text{tail}} I)\,U_k^T, \qquad V = \sigma_g^2 K + \sigma_e^2 I .$$

With $\sigma^2_{\text{eff}} = \sigma_g^2 \lambda_{\text{tail}} + \sigma_e^2$, $\delta_j = \max(0, d_j - \lambda_{\text{tail}})$ and $c_j = \sigma_g^2\delta_j / (\sigma^2_{\text{eff}} + \sigma_g^2\delta_j)$:

$$V^{-1}v = \frac{v - U_k\,\operatorname{diag}(c)\,U_k^T v}{\sigma^2_{\text{eff}}}, \qquad Kv = \lambda_{\text{tail}} v + U_k (d - \lambda_{\text{tail}}) \odot U_k^T v ,$$

$$\log|V| = (n-k)\log\sigma^2_{\text{eff}} + \sum_{j=1}^k \log(\sigma^2_{\text{eff}} + \sigma_g^2\delta_j) - \tfrac12 r^2 (n-k)\,\operatorname{Var}_{\text{tail}}, \quad r = \sigma_g^2/\sigma^2_{\text{eff}} .$$

The last term is a second-order (delta-method) correction for the spread of the unretained eigenvalues, with $\operatorname{Var}_{\text{tail}}$ the empirical variance of the tail. Traces and AI terms treat the tail as exactly flat at $\lambda_{\text{tail}}$. Ritz values are projected onto a PSD surrogate before the tail moments are computed, since GRM spectra can have negative tails.

Every product $Kv$, $V^{-1}v$, $Pv$ costs $O(nk)$ and needs no $n \times n$ matrix; the Woodbury path never materialises $V^{-1}$.

**Choosing the rank.** `--reml-woodbury-basis [MP|EIG|VAR|k]` selects how $k$ is set:

| Mode | Rank rule |
|---|---|
| `k` | user-supplied rank |
| `MP` | eigenvalues above the upper Marchenko–Pastur bulk edge $\lambda_+$, extended by `--reml-woodbury-basis-MP-margin` and confirmed by `--reml-woodbury-basis-MP-confirm` further sub-threshold eigenvalues |
| `EIG` | smallest rank capturing the fraction `--reml-woodbury-basis-EIG-mass` of the total eigenvalue mass |
| `VAR` | smallest rank leaving relative Frobenius tail error at most `--reml-woodbury-basis-VAR-tail` |

For the adaptive modes the basis grows in rounds using locked expansion (§3), starting from `--reml-woodbury-basis-range <initial=2000> <maximum=25000>`. The next rank follows a doubling schedule, tightened by a mass-sufficiency bound once the target is near. All Ritz pairs of a round are locked (not just the leading $k$), so the oversample vectors are refined by the next round rather than recomputed. Convergence of the rank criterion is monitored per mode: captured mass for `EIG` (non-decreasing across rounds by interlacing), tail error for `VAR`, and the bulk-edge confirmation run for `MP`.

**Reuse across phenotypes.** The phenotype-independent basis $(U_k, d_k, \lambda_{\text{tail}}, \operatorname{Var}_{\text{tail}})$ is stored in the `.reml` file. `--reml-woodbury-reuse <.reml>` re-estimates variance components and fixed effects for a new phenotype without reading a GRM, provided the analysed samples are identical (checked by sample count and a hash of the ordered IDs before anything else is read).

### 4.5 Hutch++ stochastic trace estimation

Computing $\text{tr}(PA_i)$ exactly needs either $P$ ($O(n^2)$ storage) or an $O(n^3)$ product. The Girard–Hutchinson estimator $\text{tr}(M) \approx \frac1s\sum_j r_j^T M r_j$ has variance $O(\|M\|_F^2/s)$. **Hutch++** (Meyer et al., 2021) captures the dominant subspace of $M$ exactly and estimates only the remainder stochastically. With $Q$ an orthonormal basis of the range of $MS$ ($S$ a Rademacher sketch) and $G$ independent Rademacher probes:

$$\text{tr}(M) \approx \text{tr}(Q^T M Q) + \frac{1}{k}\sum_{j=1}^k g_j^T(I - QQ^T)\,M\,(I - QQ^T)\,g_j ,$$

with variance decaying as $O(\|M\|_F^2/k^2)$. Here $M = PA_i$, applied as $Z \mapsto P(A_i Z)$ using the products of §4.2 and, for the GRM component, the streamed (§2.3) or Woodbury (§4.4) product for $A_iZ$.

`--reml-trace-hutchpp [N=100]` uses $N$ operator applications per component per iteration: $k = N/3$ for the sketch, and $Q$ and $G$ ($2k$ columns) are pushed through the operator in a single fused $n \times 2k$ product, which amortises BLAS call overhead and the triangular-solve setup. Eliminates the $8n^2$-byte $P$ matrix from peak memory.

**Probe policy.**

- *Default:* fresh probes at every iteration. The estimator's sampling variance is a free by-product and feeds a stochastic stopping rule (§4.6). To save work far from convergence, the probe count drops to $\max(N/3, 30)$ while $|\Delta\log L| \ge 1$.
- `--reml-trace-hutchpp-fixed-probes`: probes are drawn once and reused. The estimate is then a deterministic (but biased) function of the variance components, which is more stable on ill-conditioned problems.

### 4.6 AI-REML iteration

- **HE-regression start** (default; `--reml-no-HE-start` restores an equal split of $V_g$ and $V_e$). Starting values solve the two Haseman–Elston moment equations $E[y^TQKQy]$ and $E[y^TQy]$ with $Q = I - X(X^TX)^{-1}X^T$. The required $\text{tr}K$ and $\text{tr}K^2$ come from the dense GRM, the streamed tiles (§2.3), or, for Woodbury, from $\sum d_j + (n-k)\lambda_{\text{tail}}$ and $\|d\|^2 + (n-k)(\operatorname{Var}_{\text{tail}} + \lambda_{\text{tail}}^2)$.
- **Fraction-to-boundary safeguard** ($\tau = 0.995$, always active): a Newton step that would drive a variance component to or through zero is shortened so the iterate stays strictly inside the feasible region.
- **`--reml-ai-robust`.** By default, convergence uses a fixed threshold on the change in log-likelihood ($|\Delta\log L| < 10^{-4}$ or a relative form of it), and the step is shrunk by the fixed factor $10^{-1/2}$ whenever the *previous* iteration's $|\Delta\log L|$ exceeded 1 — a backward-looking gate unrelated to whether the current step's local quadratic model can be trusted. `--reml-ai-robust` replaces both with quantities derived from the Newton step itself:

  - *Stopping.* AI-REML already forms $H^{-1}$ (the inverse AI matrix) and the score $U$ to take its step $\delta = H^{-1}U$, so the Newton decrement $\lambda^2 = U^T\delta$ is a free by-product, and by the standard quadratic-approximation identity $\log L(\theta^\*) - \log L(\theta) \approx \tfrac12\lambda^2$: it is (twice) the estimated log-likelihood gap remaining to the optimum, in the same units as the tolerance `--reml-ai-robust-tol`. The run stops once $\lambda^2 \le \max(2\cdot\text{tol},\ q_{1-\alpha})$, where $q_{1-\alpha}$ is the $(1-\alpha)$ quantile of $\lambda^2$'s own null distribution under Hutch++ probe noise (`--reml-ai-robust-risk` sets $\alpha$ per iteration): with $U \sim \mathcal N(0,\operatorname{diag}(\text{var}_U))$ from the Hutch++ trace variance, $\lambda^2$ is a weighted sum of independent $\chi^2_1$ variables, Satterthwaite/Welch-matched to a scaled $\chi^2_h$ from the $2\times2$-scale eigenproblem of $\text{diag}(\sqrt{\text{var}_U})\,H^{-1}\,\text{diag}(\sqrt{\text{var}_U})$. This makes each stopping check a single-iteration hypothesis test with a chosen false-stop probability, rather than a threshold that needs repeated small steps to trust. For exact or Woodbury REML, $\text{var}_U = 0$ and the quantile collapses to 0, so the test reduces to $\lambda^2 \le 2\cdot\text{tol}$.
  - *Step size.* The step is scaled by $\min\!\bigl(1,\ \sqrt{0.5/\lambda^2}\bigr)$, converging to a full Newton step as $\lambda^2 \to 0$, **unless** the previous step already proved trustworthy: after each step, the predicted gain $\tfrac12\lambda^2\cdot s(1 - s/2)$ (for the step scale $s$ taken) is compared to the realised $\Delta\log L$, and if their ratio exceeds a threshold the next step is taken at full Newton strength regardless of the new $\lambda^2$. This is one bit of memory carried between iterations, not a persistent trust region — no ratchet, no expanding or shrinking radius.
  - Both quantities are then subject to the same fraction-to-boundary safeguard as the default path (below), applied identically either way.
- **`--reml-force-dense-V`.** By default REML stores the Cholesky factor of $V$ rather than an explicit $V^{-1}$. This selects a different BLAS operation in MLMA (§5). In principle the explicit-inverse MLMA step can be faster, at greater cost in REML; not recommended.
- **`--save-reml` / `--load-reml <file>`** run the REML stage and the MLMA stage separately; the REML state is written to `${out}.reml`.

---

## 5. Streaming MLMA (`--mlma-stream`)

For each SNP $j$ the mixed-model association test uses the GLS estimator and Wald statistic

$$\hat{\beta}_j = \frac{x_j^T V^{-1} y}{x_j^T V^{-1} x_j}, \qquad \operatorname{se}(\hat\beta_j) = \big(x_j^T V^{-1} x_j\big)^{-1/2}, \qquad T_j^2 = \frac{(x_j^T V^{-1} y)^2}{x_j^T V^{-1} x_j} \sim \chi^2_1 .$$

Two facts make the scan cheap. First, $V^{-1}y$ is the same for every SNP and is computed **once**. Second, only the *diagonal* of $X^TV^{-1}X$ is needed, so $V^{-1}X$ is never formed: each quadratic form reduces to a squared norm of a triangularly transformed column.

### 5.1 Streaming the genotypes

SNPs are streamed from the genotype backend in blocks rather than pre-loaded, keeping peak RSS independent of the SNP count. Each block is decoded directly into a single float design matrix `X_block` ($n \times B$): for BED with a SIMD lookup table, for PGEN/BGEN via the double path followed by a cast. Decoding is parallel over SNPs with dynamic scheduling. The single-precision buffer is the only $O(nB)$ allocation, so the block width is sized from `--mlma-stream <G>` as

$$B = \operatorname{clamp}\!\left(\left\lfloor \frac{G \cdot 10^9}{4n} \right\rfloor,\ 256,\ 65536\right),$$

defaulting to 10,000 markers when no budget is given. Results are formatted into a 4 MB buffer and written in large flushes.

### 5.2 Three representations of $V^{-1}$

The REML state determines which of three forms the scan uses. In each case the per-SNP numerator is $x^T(V^{-1}y)$ from one SGEMV against the precomputed vector, and the denominator comes from a block operation:

| Representation | $V^{-1}y$ (once) | $\operatorname{diag}(X^TV^{-1}X)$ per block |
|---|---|---|
| **Woodbury** (`--reml-woodbury-basis`) | $(y - U_k^T(c \odot U_ky))/\sigma^2_{\text{eff}}$ | one SGEMM $U_kX$ ($k \times n$ by $n \times B$), then $\big(\lVert x\rVert^2 - \lVert\sqrt{c}\odot U_kx\rVert^2\big)/\sigma^2_{\text{eff}}$, clamped at 0 |
| **Cholesky of $V$**, $V = U^TU$ (default) | two `STRSV` solves | one `STRSM` $U^{-T}X$, then column squared norms: $\lVert U^{-T}x\rVert^2 = x^TV^{-1}x$ |
| **Cholesky of $V^{-1}$**, $V^{-1} = U_i^TU_i$ (`--reml-force-dense-V`) | two triangular products | one `STRMM` $U_iX$, then column squared norms; no linear solve |

The Woodbury eigenvector matrix is stored transposed ($k \times n$) and in single precision so the $k \times n$ by $n \times B$ product is a plain column-major GEMM. In all three forms the work per block is one Level-3 triangular or general product plus $O(nB)$ reductions, with no per-SNP $O(n^2)$ or $O(p^3)$ work.

### 5.3 Per-SNP statistics

$\hat\beta = \text{num}/\text{den}$, $\operatorname{se} = 1/\sqrt{\text{den}}$, and $T^2 = \text{num}^2/\text{den}$, with SNPs whose denominator is at most $10^{-30}$ (monomorphic or degenerate) reported as `NA`. Fixed covariates are handled once, before the SNP loop, rather than per SNP: the phenotype is pre-adjusted to $y_{\text{adj}} = y - Xb$ using the fixed-effect estimate $b$ from the REML fit (from `--load-reml`'s saved state or from the inline REML run), and the streamed scan then runs entirely against $y_{\text{adj}}$, with no covariate-dependent term inside the per-SNP statistic. Pre-adjustment is the only supported mode; `--mlma-no-preadj-covar` is accepted as a flag but not yet implemented; passing it stops the run with an explicit error.

`--log-pval` evaluates the $\chi^2_1$ tail on the log scale (§8), avoiding underflow of extreme p-values.

`--model [additive|nonadditive]` recodes genotypes on the fly as they are decoded.

---

## 6. Other Kernels

### 6.1 LD pruning: block GEMM (`LinAlg.cpp`)

Pairwise squared correlations within a block of $b$ SNPs are the Hadamard square of $\frac1n Z^TZ$, where $z_i$ are standardised genotype vectors:

$$R^2_{ij} = \left(\frac{z_i^T z_j}{n}\right)^2 = \left[\frac{1}{n} Z^T Z\right]_{ij}^2 .$$

```cpp
eigenMatrix R = (X_sub.transpose() * X_sub) * (1.0 / n);  // DGEMM: b×b
R = R.array().square();                                     // r²_ij = R_ij²
```

One Level-3 BLAS call replaces $O(b^2)$ Level-1 dot products.

### 6.2 Median: $O(n\log n)$ → $O(n)$ (`CommFunc.cpp`)

`std::stable_sort` is replaced by `std::nth_element` (linear average time), used when reporting the median $r^2$ across thousands of genomic windows.

### 6.3 Log-determinant from the factor diagonal (`Matrix.hpp`, `LinAlg.cpp`)

The determinant is read from the diagonal of the factor in a single pass, avoiding both overflow of the raw determinant and a second decomposition:

$$\log|V| = 2\sum_{i=1}^n \log L_{ii} \quad (\text{LLT}), \qquad \log|\det A| = \sum_i \log|U_{ii}| \quad (\text{LU}).$$

---

## Performance Summary Table

| Area | Change | Technique | Impact |
|------|--------|-----------|--------|
| GRM build | DSYRK via `rankUpdate` | Symmetry exploit | Writes $n(n+1)/2$ vs $n^2$ elements |
| GRM build | `stdGenoLD` 64-byte alignment | Memory layout | Aligned AVX-512 BLAS kernels |
| GRM build | `--nMarkers` 128 → 1024 | Tuning | 8× fewer rank-$k$ dispatches |
| GRM build | OpenMP hoist: $2n_b$ → 1 barrier | Loop hoisting | Removes repeated fork/join overhead |
| GRM build | Pre-allocated buffers, `std::popcount` | Memory / instruction | No per-block allocation; single `POPCNT` |
| GRM I/O | Upper-triangle-only load | Symmetry exploit | ~$n^2/2 \times 8$ bytes less RSS; no mirror pass |
| GRM I/O | Sequential `read()` replaces `mmap` | I/O pattern | Restores readahead on parallel filesystems |
| GRM I/O | Tiled out-of-core $KX$ | Streaming | No dense $K$; memory set by `--grm-chunked-budget` |
| Eigensolvers | CholeskyQR2 | Level-3 BLAS | Scales with threads at large $k$ |
| Eigensolvers | Locked basis expansion | Algorithm | Each column sketched once across rank rounds |
| Eigensolvers | Nyström single pass | Algorithm | 1 vs $q+1$ matvec passes |
| REML | $V$ assembled in upper triangle only | Symmetry exploit | ~$2\times$ fewer writes; contiguous columns |
| REML | Factorise-only (no `dpotri`) | Algorithm | Saves $O(n^3)$ flops and one $n \times n$ matrix |
| REML | $P$ via `rankUpdate` (DSYRK) | Symmetry exploit | ~$2\times$ flops and bandwidth |
| REML | Implicit $Pv$ | Algorithm | $O(cn)$ correction; no $P$ load |
| REML | AI matrix as Hadamard sum | Vectorisation | No loop, no heap allocation |
| REML | Woodbury $V^{-1}$, $\log|V|$ | Low-rank structure | $O(nk)$ per product; no $n \times n$ matrix |
| REML | Hutch++ | Stochastic trace | Removes the $8n^2$-byte $P$; $N$ products per component |
| REML | Fused $[Q\,G]$ product | Batching | Half the operator calls per trace |
| REML | HE-regression start, Newton-decrement stop | Solver | Fewer iterations; noise-aware convergence |
| MLMA | $V^{-1}y$ hoisted out of SNP loop | Loop hoisting | Saves $m \times 2n^2$ flops |
| MLMA | Diagonal-only $x^TV^{-1}x$ via triangular transform | Avoid work | No $V^{-1}X$; one STRSM/STRMM/SGEMM per block |
| MLMA | Streaming float decode, budget-sized blocks | Memory | No resident genotype matrix; one $4nB$-byte buffer |
| LD | Block GEMM for $r^2$ | Level-3 BLAS | $O(b^2)$ dot products → one DGEMM |
| Utilities | `nth_element` median, logdet from factor diagonal | Algorithm / numerical | $O(n)$ median; one decomposition, no overflow |

---

# Part II: Supporting Changes

These changes improve portability, correctness, and I/O. They are not individually performance changes, but several remove constraints (x86-only build, MKL requirement, OpenMP spin-polling) that would otherwise prevent the Part I improvements from delivering their full benefit.

---

## 7. Build and BLAS Portability

The Eigen-based paths of Part I serve every platform (x86, Apple Silicon, ARM Linux) and every supported BLAS backend; vendor-specific code paths and Fortran/C ABI `#if` guards were removed.

- **Explicit backend.** `GCTA_BLAS_BACKEND` is required: `OpenBLAS`, `MKL`, `AOCL` (Linux) or `Accelerate` (macOS). Library and include paths are given explicitly, avoiding fragile auto-detection. Backend and version are reported at configure time and compiled into the binary.
- **Native optimisation.** Release builds use `-O3` with `-march=native` (`-mcpu=native` on Apple Silicon) and link-time optimisation (full LTO for AOCC, ThinLTO for Clang, `-flto=auto` for GCC). This makes the aligned strides and Eigen expression templates effective on each machine, and means the binary is tied to the CPU generation it was built on.
- **Library pinning.** On Linux the BLAS, `libgomp`/`libomp` and `libstdc++` directories are written as `DT_RPATH`, which takes precedence over `LD_LIBRARY_PATH`, so HPC modules loaded at run time cannot shadow the libraries the binary was built against.
- **Allocator.** `-DUSE_JEMALLOC=/path/to/libjemalloc.so` links jemalloc with `thp:never`, preventing transparent-huge-page over-allocation on very large triangular allocations such as the GRM, and preserving the untouched-page savings of §2.1.
- **OpenMP spin-wait elimination.**

  ```cpp
  setenv("KMP_BLOCKTIME", "0", 0);   // LLVM/Intel OpenMP: sleep after parallel region
  setenv("GOMP_SPINCOUNT", "0", 0);  // GNU OpenMP: yield after parallel region
  ```

  Without these, OpenMP threads spin-poll after each parallel region, consuming memory bandwidth and starving the BLAS call that follows.

---

## 8. Statistical Libraries: Boost.Math

About 100 lines of hand-rolled numerical series (`betai`, `betacf`, `gammln`, `gammp`, `gser`, `gcf`) were replaced by Boost.Math:

```cpp
boost::math::cdf(boost::math::complement(boost::math::chi_squared(df), x))
```

Boost.Math uses Lanczos-approximated special functions with guaranteed relative error bounds.

`--log-pval` adds a log-space path for extreme $\chi^2$ values, using the asymptotic expansion $\ln Q(a,x) \approx -x + (a-1)\ln x - \ln\Gamma(a)$ for $x \gg a$, which avoids underflow to $p = 0$ for GWAS hits below $10^{-300}$.
<!-- TODO: confirm the reported quantity/base of --log-pval (ln p vs -log10 p); GCTAg.md says -log10(p). -->

`rankContrast` (`StatLib.cpp`) replaced 40 lines of `dgeqrf`/`dormqr` LAPACK calls with 8 lines of `Eigen::HouseholderQR`.

---

## 9. I/O and Memory Management

| Change | Location | Before | After |
|--------|----------|--------|-------|
| gzip read/write | `grm.cpp` | Custom `gzifstream`/`gzofstream` (~200-line C wrapper) | `boost::iostreams::filtering_stream` |
| FastFAM covariate conditioning | `FastFAM.cpp` | 2 CBLAS `dgemv` + 10-line `#if` block | `y -= covar * (H * y)` |
| GRM `sub_miss` | `GRM.h` / `GRM.cpp` | raw `uint32_t*`, manual `delete[]` | `std::vector<uint32_t>` |
| FastFAM scratch arrays | `FastFAM.cpp` | `new double[n]` × 4, manual `delete[]` | `std::vector<double>` × 4 |
| `quantile` | `CommFunc.cpp` | Mapped `vector` into `Eigen::VectorXd` | Direct `quantile(data, n, prob)`, no temporary |
| `rand_seed()` | `CommFunc.cpp` | Reversed time string, `abs(atoi(...))` | `std::random_device{}() & 0x7FFFFFFF` |

---

## 10. Flags and the Features They Activate

| Flag | Feature | Section |
|------|---------|---------|
| `--nMarkers`, `--GRM-tile-budget`, `--merge-grms` | GRM block/tile budgets and streamed merge | 1.7, 2.2 |
| `--grm-chunked-budget <G>` | Out-of-core GRM for PCA, REML, MLMA | 2.3, 2.4 |
| `--pca [N]`, `--pca-approx [Lanczos\|rSVD]`, `--svd-method [power\|nystrom]` | Exact and randomised PCA | 3 |
| `--reml-woodbury-basis [MP\|EIG\|VAR\|k]` and its `-MP-margin`, `-MP-confirm`, `-EIG-mass`, `-VAR-tail`, `-range` options | Low-rank REML | 4.4 |
| `--reml-woodbury-reuse <.reml>` | Basis reuse across phenotypes | 4.4 |
| `--reml-trace-hutchpp [N]`, `--reml-trace-hutchpp-fixed-probes` | Stochastic traces | 4.5 |
| `--reml-ai-robust`, `--reml-no-HE-start`, `--reml-force-dense-V` | AI-REML iteration | 4.6 |
| `--save-reml`, `--load-reml <file>` | Split REML / MLMA stages | 4.6, 5 |
| `--mlma-stream [G]`, `--model [additive\|nonadditive]` | Streaming MLMA | 5 |
| `--log-pval` | Log-scale p-values | 5.3, 8 |

---

## References

- Meyer, R. A., Musco, C., Musco, C., & Woodruff, D. P. (2021). *Hutch++: Optimal Stochastic Trace Estimation*. SIAM Symposium on Simplicity in Algorithms (SOSA).
- Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). *Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions*. SIAM Review.
- Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Mathematics of the USSR-Sbornik* 1, 457–483.
- Nocedal, J., & Wright, S. J. *Numerical Optimization* (Ch. 19); Wächter, A., & Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. *Mathematical Programming* 106, 25–57.
- Jiang, J. (2026). *Genomic Dimensionality Bounds Mixed-Model Association Power, Fine-Mapping Resolution, and Genomic Prediction Reliability*. bioRxiv.
- Eigen documentation: `selfadjointView`, `rankUpdate`, `triangularView`, `LLT`.
- Boost.Math documentation: *Statistical distributions and special functions*.
- Spectra documentation: `SymEigsSolver` (implicitly restarted Lanczos).