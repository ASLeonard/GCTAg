/*
 * GCTA: a tool for Genome-wide Complex Trait Analysis
 *
 * RemlState — shared post-REML descriptor used by both MLMA_stream and
 * MLMA_loco.  Holds either a dense V^{-1} (GOBY path) or the Woodbury
 * low-rank factors (TUNA path), plus the fixed-effect vector b.
 *
 * This header intentionally has no heavy includes so it can be included
 * by both legacy (main/) and v2 (src/) translation units.
 */
#pragma once

#include "mlma_woodbury.hpp"
#include <Eigen/Dense>
#include <cstdint>

struct Header {
    char    magic[4];     // 4-byte magic signature ("TUNA" or "GOBY")
    int32_t n;            // Matrix size (e.g., number of samples)
    int32_t x_c;          // Number of covariates
    int32_t num_varcmp;   // Number of variance components
    int32_t num_r_indx;   // Index count metadata
};

struct RemlState {
    RemlState() = default;
    // Vi_L_f (and wb.Uk_f, for the Woodbury path) can be tens of GB at large
    // n -- deleting copy and defaulting move turns any future accidental
    // copy (e.g. a by-value parameter, or a return path that defeats NRVO)
    // into a compile error instead of a silent multi-GB duplication. Move
    // stays exactly as cheap as before: every member here is an Eigen type
    // or scalar, so the generated move constructor is pointer swaps only.
    RemlState(const RemlState&)            = delete;
    RemlState& operator=(const RemlState&) = delete;
    RemlState(RemlState&&)                 = default;
    RemlState& operator=(RemlState&&)      = default;

    int32_t n           = 0;
    int32_t x_c         = 0;
    bool    is_woodbury = false;
    // Non-Woodbury path: Vi_L_f always holds a lower-triangular Cholesky
    // factor, contiguous n×n. is_llt selects which one:
    //   is_llt=true  -> Vi_L_f = L,  where V      = L  L^T. Apply V^{-1}
    //                   via two triangular SOLVES (TRSV/TRSM): L^{-T}(L^{-1}·v).
    //   is_llt=false -> Vi_L_f = Li, where V^{-1} = Li Li^T. Apply V^{-1}
    //                   via two triangular PRODUCTS (TRMV/TRMM): Li(Li^T·v).
    // V^{-1} itself is never stored explicitly. Whichever factor REML
    // produced cheaply (the common is_llt=true case costs one dpotrf, no
    // inversion) — or, in the reml_diagV_adj fallback where only the
    // explicit inverse is available, whichever factor is derived from it —
    // is computed exactly once (at REML exit / save time) and reused as-is
    // by every downstream consumer. No consumer re-factorizes.
    bool    is_llt      = false;
    Eigen::MatrixXf Vi_L_f;  // Cholesky factor per is_llt above
    WoodburyMLMACache wb;    // TUNA: Woodbury low-rank factors (is_woodbury=true)
    Eigen::VectorXf dk_f;    // TUNA: top-k eigenvalues of K (needed for save-reml)
    float lambda_tail_f = 0.0f; // TUNA: tail mean eigenvalue of K
    Eigen::VectorXf b;       // fixed-effect coefficients (x_c elements)
    Eigen::VectorXf varcmp;  // variance components from REML state file
};
