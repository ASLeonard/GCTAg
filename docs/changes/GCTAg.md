# GCTAg

Notes on `GCTAg`, a performance-oriented fork of `GCTA`.

## Notable changes

### General

- Subcommand CLI via `CLI11` (`GCTAg --help`, `GCTAg <subcommand> --help`, `GCTAg --version`): `grm` (`make`, `merge`, `prune`, `sparse`, `subtract`, `unify`), `pca`, `reml`, `mlma`, and `legacy` (passthrough to unchanged GCTA v1 analyses).
- Reworked chromosome name handling: all chromosomes are assumed to be included unless otherwise stated (you should no longer need to set `--autosome-num`).
- Homogametic/heterogametic sex chromosomes can now be specified manually (`--chr-homogametic`). Note: this has not been tested extensively.
- Extensively use `triangularView<Eigen::Upper/Lower>` where possible. If the allocator doesn't fault in untouched memory, this can lead to large memory savings.

### GRM Building (`GCTAg grm make`, `GCTAg grm merge`)

- `--tile-budget <G>` (alias `--GRM-tile-budget`) — Process at most `G` gigabytes of GRM tiling at a time, similar to `--part N n` (`--make-grm-part`) but within a single call and more intuitive. The budget is not strict due to other matrices and buffers co-existing, but is a soft constraint.
- `--n-markers <N=1024>` (alias `--nMarkers`) — Process N SNPs per block in GRM building. Raised the hardcoded value from 128 to now default at 1024. Higher values use more memory and eventually become cache bottlenecks, so higher is not always faster.
- `GCTAg grm merge --mgrm <file> [--budget <G>]` — Merge GRMs using a streaming chunk approach. Uses ~0.5 GB by default (`--budget 0`), or takes an integer budget in GB.

### PCA (`GCTAg pca`)

Note: `GCTAg pca` dispatches to the V2 paths; use `GCTAg legacy --pca-v1 ...` for the legacy PC solvers.

- `--approx [Lanczos|rSVD]` (alias `--pca-approx`) — Use an approximate solver for eigenvalues. If the number of requested PCs is much smaller than the GRM dimension, this is highly accurate.
- `--pcs <N>` — Number of top PCs (`0`, the default, computes all eigenvalues rather than the top 20).
- `--svd-method [power|nystrom]` — By default, use power iterations during rSVD, or swap to a Nystrom approach (can fail on poorly conditioned matrices).
- `--grm-chunked-budget <G>` — Use a chunked approach to loading the GRM. This loads a small buffer (about 10% of the budget) and the remaining memory goes to decoding the packed GRM for as many rows as the budget can fit.

### REML (`GCTAg reml` or inline in `GCTAg mlma`)

- `--trace-hutchpp [N=100]` (alias `--reml-trace-hutchpp`) — Use the Hutch++ stochastic approximator for matrix traces. Lowers memory (and consequently can improve runtime). Optional `N` sets the number of probes.
- `--trace-hutchpp-fixed-probes` (alias `--reml-trace-hutchpp-fixed-probes`) — By default, fresh probes are used at every REML iteration with a stochastic stopping condition. This flag reuses fixed probes instead, converging to a biased estimator but deterministically (better for an unstable matrix).
- `--woodbury-basis [MP|EIG|VAR|k]` (alias `--reml-woodbury-basis`) — Use a Woodbury basis to exploit the low effective rank of the GRM. `MP` uses Marchenko–Pastur theory to detect the effective rank from the GRM; `EIG` uses the eigenmass approach from Jiang 2026; `VAR` limits the variance remaining in the truncated tail; `k` manually sets the effective rank to that value.
    - `--woodbury-mp-margin <M>` (alias `--reml-woodbury-basis-MP-margin`) — Calculate the Woodbury basis using MP theory and extend beyond with margin `M`.
    - `--woodbury-mp-confirm <C>` (alias `--reml-woodbury-basis-MP-confirm`) — Confirm the dropoff after the MP-k is robust using `C` additional eigenvalues.
    - `--woodbury-eig-mass <M>` (alias `--reml-woodbury-basis-EIG-mass`) — Calculate the Woodbury basis based on the number of eigenvalues needed to explain the fraction `M` of the total eigenvalue mass.
    - `--woodbury-var-tail <V>` (alias `--reml-woodbury-basis-VAR-tail`) — Calculate the Woodbury basis that leaves at most `V` variance left in the truncated tail.
- `--woodbury-range <initial=2000> <maximum=25000>` (alias `--reml-woodbury-basis-range`) — Change the initial guess and maximum value for the Woodbury basis k.
- `--ai-robust` (alias `--reml-ai-robust`) — Derive REML convergence from the remaining curvature in the model (forward looking), as opposed to a fixed threshold (backward looking). The maximum step between changes is also based on the curvature, rather than a fixed `0.316` ($`10^{-1/2}`$) proportion when $`\Delta \log(L)`$ is large.
- `GCTAg reml` (or `GCTAg mlma --save-reml`) and `GCTAg mlma --load-reml <file>` — Run only the REML or MLMA stage respectively. The REML file is saved to `${out}.reml`, based on the `--out` prefix.
- `--no-he-start` (alias `--reml-no-HE-start`) — By default, REML now uses a (cheap) HE estimator rather than assuming equal partition of V(g) and V(e). This restores the previous default behaviour.
- `--force-dense-V` (alias `--reml-force-dense-V`) — By default, REML can store the solve of V rather than V itself. This affects MLMA using a different BLAS operation. In theory, the dense-V MLMA step can be faster, but at a greater cost during REML. Not recommended.

The `--svd-method` and `--grm-chunked-budget` flags from the PCA section above can also be used here.

If the non-missing samples are identical across different phenotypes, we can use `--woodbury-reuse <.reml>` (alias `--reml-woodbury-reuse`) to reuse the same Woodbury basis as calculated from a previous REML run and estimate variance components for a new phenotype.

### MLMA (`GCTAg mlma`)

Reworked linear algebra to maintain multi-threading performance. The legacy `--mlma` path (`GCTAg legacy --mlma ...`) should be considered deprecated in favour of `GCTAg mlma`.

- `GCTAg mlma [--stream-budget <G>]` — Uses the streaming V2 path; streams genotypes on demand rather than loading them up-front, lowering peak RSS and improving CPU utilization. Optionally takes a memory budget `G` in gigabytes (`--stream-budget` / `--mlma-stream`) for how much genotype matrix to stream per block. Defaults to 10,000 markers.
- `--model [additive|nonadditive|dominance]` — Recodes the BED genotypes on the fly. `additive`/`nonadditive` use more memory compared to the hardcall approach.
- `--log-pval` — Calculate an asymptotic version of the chi-squared distribution to return -log10(p) values beyond typical machine precision.
