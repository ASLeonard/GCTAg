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
#include <optional>

struct Header {
    char    magic[4];     // 4-byte magic signature ("TUN3", "TREF", or "GOBY")
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
    // Non-Woodbury path: Vi_L_f always holds an upper-triangular Cholesky
    // factor (from dpotrf('U')), contiguous n×n with only the upper triangle
    // populated (strict lower triangle stays unfaulted). is_llt selects which one:
    //   is_llt=true  -> Vi_L_f = U,  where V      = U^T U. Apply V^{-1}
    //                   via two upper-triangular SOLVES (TRSV/TRSM): U^{-1}(U^{-T}·v).
    //   is_llt=false -> Vi_L_f = Ui, where V^{-1} = Ui^T Ui. Apply V^{-1}
    //                   via two upper-triangular PRODUCTS (TRMV/TRMM): Ui^T(Ui·v).
    // V^{-1} itself is never stored explicitly. Whichever factor REML
    // produced cheaply (the common is_llt=true case costs one dpotrf('U'), no
    // inversion) — or, in the reml_diagV_adj fallback where only the
    // explicit inverse is available, whichever factor is derived from it —
    // is computed exactly once (at REML exit / save time) and reused as-is
    // by every downstream consumer. No consumer re-factorizes.
    bool    is_llt      = false;
    Eigen::MatrixXf Vi_L_f;  // Upper-triangular Cholesky factor per is_llt above
    WoodburyMLMACache wb;    // TUN3/TREF: Woodbury low-rank factors (is_woodbury=true)
    std::optional<QuantizedWoodburyUk> prebuilt_q_uk; // 1-pass streaming --fast-mlma cache (wb.Uk_f stays empty)
    Eigen::VectorXf cached_Vi_y; // Precomputed exact V^{-1} y_adj from TUN3/TREF header or streaming pass
    bool            has_cached_Vi_y = false;
    Eigen::VectorXf dk_f;    // TUN3/TREF: top-k eigenvalues of K
    float lambda_tail_f = 0.0f; // TUN3/TREF: tail mean eigenvalue of K
    Eigen::VectorXf b;       // fixed-effect coefficients (x_c elements)
    Eigen::VectorXf varcmp;  // variance components from REML state file
};
