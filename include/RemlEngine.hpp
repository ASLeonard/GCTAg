/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * RemlEngine — free-function REML engine for the v2 MLMALoco path.
 * Scope: single-GRM, univariate, non-bivariate, non-within-family.
 *
 * The public API consists of:
 *   reml::compute()                — run REML to convergence
 *   reml::compute_woodbury_basis() — pre-compute low-rank basis (called
 *                                    automatically from compute() if needed)
 *   reml::build_reml_state()       — pack RemlCtx output into a RemlState
 *                                    for the downstream MLMA streaming step
 */
#pragma once

#include "RemlCtx.hpp"
#include "RemlState.hpp"
#include <vector>

namespace reml {

// ── Top-level entry point ─────────────────────────────────────────────────────
//
// Runs REML to convergence on the GRM + phenotype stored in ctx.
// On return, ctx.varcmp and ctx.b are filled in.
// If ctx.woodbury_rank != 0, the Woodbury basis is computed first.
// If ctx.reml_trace_approx, Hutch++ probes are computed on first iteration.
//
void compute(RemlCtx& ctx,
             const std::vector<double>& priors,
             const std::vector<double>& priors_var,
             bool no_constrain);

// Computes the low-rank Woodbury basis (Uk, dk, lambda_tail) for the GRM
// stored in ctx.A[ctx.r_indx[0]].  Called automatically by compute() when
// ctx.woodbury_rank != 0.  Safe to call explicitly before compute().
void compute_woodbury_basis(RemlCtx& ctx);

// Packs the post-convergence RemlCtx into a RemlState ready for MLMA.
// Converts double-precision Vi/Uk/b to float for memory efficiency and then
// releases the heavy double workspace immediately so the RSS stays low.
RemlState build_reml_state(RemlCtx& ctx);

} // namespace reml
