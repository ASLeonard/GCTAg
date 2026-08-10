# GCTAg

Notes on `GCTAg`, a performance-oriented fork of `GCTA`.

## Notable changes

### General

- Reworked chromosome name handling: all chromosomes are assumed to be included unless otherwise stated (you should no longer need to set `--autosome-num`).
- Homogametic/heterogametic sex chromosomes can now be specified manually (`--chr-homogametic`). Note: this has not been tested extensively.

### GRM Building

- `--GRM-tile-budget <G>` — Process at most `G` gigabytes of GRM tiling at a time, similar to `--make-grm-part N n` but within a single call and more intuitive.
- `--nMarkers <N=1024>` - Process N SNPs per block in GRM building. Raised the hardcoded value from 128 to now default at 1024. Higher values use more memory and eventually become cache bottlenecks, so higher is not always faster.

### PCA

Note: `--pca` now dispatches to the V2 paths; use `--pca-v1` for the legacy PC solvers.

- `--pca-approx [Lanczos|rSVD]` — Use an approximate solver for eigenvalues. If the number of requested PCs is much smaller than the GRM dimension, this is highly accurate.
- `--pca <N>` — If `N` is not given, defaults to all eigenvalues rather than the top 20.
- `--svd-method [power|nystrom]` — By default, use power iterations during rSVD, or swap to a Nystrom approach (can fail on poorly conditioned matrices).
- `--svd-chunked-budget <G>` — Use a chunked approach to loading the GRM when doing rSVD, using `G` gigabytes of the GRM at a time.

### REML

- `--reml-trace-hutchpp [N=200]` — Use the Hutch++ stochastic approximator for matrix traces. Lowers memory (and consequently can improve runtime). Optional `N` sets the number of probes.
- `--reml-trace-hutchpp-fixed-probes` — By default, fresh probes are used at every REML iteration with a stochastic stopping condition. This flag reuses fixed probes instead, converging to a biased estimator but deterministically (better for an unstable matrix).
- `--reml-woodbury-basis [MP|EIG|VAR|k]` — Use a Woodbury basis to exploit the low effective rank of the GRM. MP uses Marchenko–Pastur theory to detect the effective rank from the GRM; `EIG` uses the eigenmass approach from Jiang 2026; VAR limits the variance remaining in the truncated tail;`k` manually sets the effective rank to that value.
    - `--reml-woodbury-basis-MP-margin <M>` - Calculate the Woodbury basis using MP theory and extend beyond with margin `M`.
    - `--reml-woodbury-basis-MP-confirm <C>` - Confirm the dropoff after the MP-k is robust using `C` additional eigenvalues.
    - `--reml-woodbury-basis-EIG-mass <M>` — Calculate the Woodbury basis based on the number of eigenvalues needed to explain the fraction `M` of the total eigenvalue mass.
    - `--reml-woodbury-basis-VAR-tail <V>` - Calculate the Woodbury basis that leaves at most `V` variance left in the truncated tail.
- `--reml-ai-robust` — Derive REML convergence from the remaining curvature in the model (forward looking), as opposed to a fixed threshold (backward looking). The maximum step between changes is also based on the curvature, rather than a fixed `0.316` ($`10^{-1/2}`$) proportion when $`\Delta \log(L)`$ is large.
- `--save-reml` and `--load-reml <file>` — Run only the REML or MLMA stage respectively. The REML file is saved to `${out}.reml`, based on the `--out` prefix.

The `--svd-*` flags from the PCA section above can also be used here alongside `--reml-woodbury-basis` rSVD paths.

### MLMA

Reworked linear algebra to maintain multi-threading performance.

- `--mlma-stream [G]` (instead of `--mlma`) — Uses more of the "V2" GCTA path; only streams the BED file on demand rather than loading it up-front, lowering peak RSS and improving CPU utilization. Optionally takes a memory budget `G` in gigabytes for how much genotype matrix to stream per block.
- `--model [additive|nonadditive]` — Recodes the BED genotypes as additive or nonadditive on the fly. Both use more memory compared to the hardcall approach.
- `--log-pval` — Calculate an asymptotic version of the chi-squared distribution to return -log10(p) values beyond typical machine precision.
