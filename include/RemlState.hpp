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

struct RemlState {
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
