/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * RemlCtx — flat workspace struct that carries all state for a single-GRM
 * univariate REML computation.  This mirrors the REML-relevant private
 * members of the legacy `gcta` class but is usable from any v2 translation
 * unit without instantiating a `gcta` object.
 *
 * Uses Eigen types directly (double precision throughout) so it can be
 * included without pulling in gcta.h.
 *
 * Scope: single-GRM, univariate, non-bivariate, non-within-family.
 * Bivariate / within-family REML remain in gcta and are out of scope here.
 */
#pragma once

#include "mlma_woodbury.hpp"
#include "chunked_grm_matvec.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>

// REML is always double-precision irrespective of SINGLE_PRECISION build flag.
using RemlMat = Eigen::MatrixXd;
using RemlVec = Eigen::VectorXd;

enum class WoodburyMode {
    Off      = 0,
    Fixed    = 1,
    MP   = -1, // Marchenko-Pastur bulk edge
    EIG  = -2, // Eigenvalue spectral mass fraction
    VAR = -3  // Relative tail variance dispersion ratio
};

// ──────────────────────────────────────────────────────────────────────────────
// RemlCtx: input + config + workspace + output
// Caller fills the "Input" and "Config" sections, then calls reml::compute().
// After compute() returns, "Output" holds variance components and fixed effects;
// "Workspace" holds the V^{-1} state needed for the subsequent association test.
// ──────────────────────────────────────────────────────────────────────────────
struct RemlCtx {
    WoodburyMode woodbury_mode() const {
        if (woodbury_basis_rank == -1) return WoodburyMode::MP;
        if (woodbury_basis_rank == -2) return WoodburyMode::EIG;
        if (woodbury_basis_rank == -3) return WoodburyMode::VAR;
        if (woodbury_basis_rank > 0)   return WoodburyMode::Fixed;
        return WoodburyMode::Off;
    }
    // ── Input (set by caller before reml::compute()) ──────────────────────────
    int  n     = 0;   // sample size
    int  X_c   = 0;   // columns in design matrix (1 + n_covariates)
    RemlMat  X;       // n × X_c (intercept in column 0)
    RemlVec  y;       // phenotype vector (n)
    double y_Ssq = 0.0; // sample variance of y (computed by caller before call)

    // GRM components: A[r_indx[i]] is the i-th variance component matrix.
    // For standard single-GRM REML: A = {GRM, empty} and r_indx = {0, 1}.
    // An empty matrix (size() == 0) represents the identity component (residual).
    std::vector<RemlMat> A;
    std::vector<int>     r_indx;  // typically {0, 1}
    RemlMat grm_N;   // from _grm_N — used only for Woodbury MP-k M estimation

    // ── Chunked K@X (avoids a dense n x n K resident during basis construction) ──
    // Only affects compute_woodbury_basis_basis's matvecs. When enabled, ctx.A[...]
    // is expected to stay EMPTY (never densely loaded) and grm_tile_reader
    // must be set by the caller before reml::compute() — see
    // chunked_grm_matvec.hpp for the exact callback contract (reads a single
    // lower-triangular tile; float32-on-disk should be widened to double at
    // the tile level, not for the whole file up front).
    double                      svd_chunked_budget = 0.0;  // GB budget for streaming chunk rows, off by default
    gcta_chunked::TileReader grm_tile_reader;               // caller-populated when chunked

    // Hard cap on rSVD sketch memory (Omega/Y/qr_scratch/Q, each ~n*k_ext*8
    // bytes, several live simultaneously during power iteration) as k_ext
    // escalates. Chunking K (svd_chunked) removes K's own O(n^2)
    // footprint but does nothing to bound this — it scales with k_ext
    // identically whether K is chunked or dense, and unbounded escalation on
    // a pathological (near-full-rank) GRM can reach hundreds of GB before
    // the "may not offer a computational advantage" warning below even
    // fires. 0 = no explicit cap (only bounded by n-1, as before).
    double svd_mem_budget_gb = 0.0;

    // ── Config (algorithm control) ────────────────────────────────────────────
    int    reml_mtd                  = 0;     // 0=AI-REML, 1=Fisher, 2=EM-REML
    int    reml_max_iter             = 100;
    int    reml_inv_mtd              = 0;     // 0=LLT, 1=LU (for V inversion fallback)
    double reml_diag_mul             = 0.01;  // diagonal jitter when V is near-singular
    int    reml_diagV_adj            = 0;     // 0=none, 1=diag jitter, 2=bending
    int    woodbury_basis_rank             = 0;     // >0 fixed k; -1 MP-k; -2 EIG-k; -3 VAR-k; 0 off
    // MP-k: eigenvalues fall off a cliff at the Marchenko-Pastur bulk edge,
    // but the exact crossing point is noisy over a band of eigenvalues near
    // the edge (rSVD tail estimation error + finite-M lambda_plus estimate +
    // edge fluctuations). That band's *size* doesn't grow with k_signal —
    // it's a property of the local spectral density at the edge, not of how
    // far the edge sits from zero — so it's confirmed by scanning past
    // k_signal for a run of eigenvalues consistently below the margin,
    // rather than by multiplying k_signal by a fixed factor.
    double woodbury_basis_edge_margin      = 0.15;  // relative margin below lambda_plus to confirm past the edge
    int    woodbury_basis_edge_confirm     = 20;    // consecutive sub-margin eigenvalues required to confirm
    // Maximum relative Frobenius error of replacing the residual spectrum by
    // its mean: sqrt((n-k) * tail_d_var / tr(K^2)). Unlike a coefficient of
    // variation, this remains meaningful when an indefinite tail has mean
    // eigenvalue near zero. 0.001 limits its non-isotropic energy to 0.001% of
    // the GRM's total spectral energy.
    double woodbury_basis_var_thresh       = 0.001;
    // EIG-k: trace(K) ~ n (GRM diagonals are ~1), and most of that mass
    // sits in the Marchenko-Pastur bulk — many eigenvalues of similar size,
    // not a few outliers — so capturing "the last 0.5%" of trace mass can
    // require a lot more eigenvalues than capturing the first 99%, not a
    // proportional bit more. Buffering by raising the mass target (99% ->
    // 99.5%) inherits that cost. Buffer by a small fixed eigenvalue *count*
    // past the raw crossing instead: those extra eigenvalues are usually
    // already sitting in the existing k_svd budget's headroom (oversample /
    // starting-budget slack), so this is normally free — no extra rSVD pass.
    int    woodbury_basis_EIG_k_buffer   = 0;    // extra eigenvalues past the raw reml_eigen_mass crossing
    int    woodbury_basis_k_init            = 2000;     // starting rank for Woodbury basis formation
    int    woodbury_basis_k_max            = 25000;     // rank cap for Woodbury basis
    bool   woodbury_basis_posthoc_correction = false; // true → apply c_j corrections to V^{-1} after basis is formed
    bool   reml_trace_hutchpp         = false; // Hutch++ trace (skips n x n P)
    int    reml_trace_hutchpp_nprobes = 200;
    int    reml_trace_power_iter     = 0;     // power-iter for Hutch++ range sketch
    bool   reml_force_inv            = false;
    bool   reml_force_dense_vi       = false; // true → force dense V^{-1}
    bool   reml_force_converge       = false;
    bool   reml_no_converge          = false;
    bool   reml_fixed_var            = false;
    bool   reml_allow_constrain_run  = false;
    bool   reml_no_HE_start          = false; // Active by default

    bool   svd_nystrom          = false; // true → single-pass Nystrom basis

    // Output naming (for .hsq file and LOGGER lines)
    std::string out;                       // output file prefix
    std::vector<std::string> var_name;     // e.g. {"V(G)", "V(e)"}
    std::vector<std::string> hsq_name;     // e.g. {"V(G)/Vp"}

    // ── Woodbury basis (set by reml::compute_woodbury_basis_basis()) ───────────────
    bool    Vi_use_woodbury_basis = false;
    int     woodbury_basis_rank_  = 0;     // actual rank used (<= woodbury_basis_rank or MP)
    float woodbury_basis_eigen_mass = 0.99f;    // fraction of eigenvalue mass captured by Uk
    bool woodbury_basis_eigen_adaptive = true; // true → adaptively raise k until mass target is reached rather than doubling
    RemlMat Uk;                      // n x k leading eigenvectors of K
    RemlVec dk;                      // k eigenvalues (clamped >= 0)
    double  lambda_tail     = 0.0;   // average bulk eigenvalue
    double  tail_d_var      = 0.0;   // var of tail eigenvalues (2nd-order logdet corr)
    double  sigma2_eff      = 0.0;   // sigma2_g * lambda_tail + sigma2_e (updated per AI iter)
    double  sg2             = 0.0;   // sigma2_g (cached for calcu_tr_PA_woodbury)
    RemlVec ck;                      // Woodbury corrections c_j

    // ── Iteration workspace (managed by reml::compute()) ─────────────────────
    bool    Vi_use_llt = false; // true when Vi_L is valid (factorize_only path)
    RemlMat Vi;                 // full V^{-1} (or empty when Woodbury)
    RemlMat Vi_L;               // Cholesky L from in-place dpotrf
    RemlMat Vi_X;               // V^{-1} X  (n x X_c)
    RemlMat Xt_Vi_X_i;          // (X' V^{-1} X)^{-1}  (X_c x X_c)
    RemlMat Uk_Vi_X;            // U_k^T V^{-1} X  (k x X_c, Woodbury)
    RemlMat UkTX;               // U_k^T X  (k x X_c, constant, cached)
    RemlVec UkTy;               // U_k^T y  (k, constant, cached)
    RemlMat hutchpp_S;          // Hutch++ Rademacher probes  (n x k)
    RemlMat hutchpp_G;          // Hutch++ Rademacher probes  (n x k)
    bool reml_hutchpp_fixed_probes = false; // true → don't redraw every call/iteration (default false)
    // Persistent working-set scratch for calcu_tr_PA_hutchpp, sized once per
    // (n,k) and reused across every REML iteration and every component (ci)
    // -- this call sits in the hottest loop in the engine (once per ci per
    // AI-REML iteration). QG/MQG are n x 2k: Q and G are pushed through the
    // PA operator in one fused call rather than two (see calcu_tr_PA_hutchpp).
    // All cleared (resized to 0) in build_reml_state once REML exits.
    RemlMat hutchpp_qr_scratch; // n x k
    RemlMat hutchpp_K;          // n x k
    RemlMat hutchpp_Q;          // n x k
    RemlMat hutchpp_QG;         // n x 2k
    RemlMat hutchpp_MQG;        // n x 2k
    RemlMat hutchpp_QtG;        // k x k
    RemlMat hutchpp_R;          // n x k
    RemlMat hutchpp_MR;         // n x k
    RemlVec reml_tmp_n;         // n — shared scratch for reml_equation / em_reml
    RemlMat P;                  // projection matrix (freed after convergence)
    bool   reml_ai_robust      = false;   // opt-in: legacy dlogL gate is default until validated
    double reml_ai_robust_tol  = 1e-4;    // logL-unit tolerance; lambda_sq_floor = 2*this
    double reml_ai_robust_risk = 0.01;  // per-iteration false-stop probability under H0

    // ── Status flags ─────────────────────────────────────────────────────────
    bool reml_AI_not_invertible = false;
    bool reml_have_bend_A       = false;

    // ── Output (populated by reml::compute()) ────────────────────────────────
    std::vector<double> varcmp; // variance components in r_indx order
    RemlVec b;                  // fixed-effect estimates (X_c elements)

    // Minimal REML summary for MLMA-style .hsq output.
    // Mirrors the non-bivariate mlmassoc summary fields in main/est_hsq.cpp.
    std::vector<double> varcmp_se; // SE for each variance component (sqrt(diag(Hi)))
    double Vp = 0.0;               // total phenotypic variance
    double Vp_se = 0.0;            // SE(Vp)
    std::vector<double> hsq;       // V(G_i)/Vp for non-residual components
    std::vector<double> hsq_se;    // SE for hsq entries
    double logL = 0.0;             // converged REML log-likelihood
    bool has_logL = false;
    int reml_iterations = 0;       // total REML loop rounds executed
};
