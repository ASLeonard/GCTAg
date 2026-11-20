#pragma once

#include <cstdint>

enum class RemlAlg : int {
    AI     = 0,
    Fisher = 1,
    EM     = 2
};

constexpr const char* reml_alg_name(RemlAlg alg) {
    switch (alg) {
        case RemlAlg::AI:     return "AI-REML";
        case RemlAlg::Fisher: return "Fisher-scoring REML";
        case RemlAlg::EM:     return "EM-REML";
    }
    return "AI-REML";
}

enum class DiagVAdj : int {
    None    = 0,
    Jitter  = 1,
    Bending = 2
};

struct WoodburyBasisSpec {
    enum class Kind : uint8_t { MP, EIG, VAR, Fixed };
    Kind kind = Kind::MP;
    int fixed_rank = 0;
};

constexpr const char* woodbury_mode_name(WoodburyBasisSpec::Kind kind) {
    switch (kind) {
        case WoodburyBasisSpec::Kind::MP:    return "MP-k";
        case WoodburyBasisSpec::Kind::EIG:   return "EIG-k";
        case WoodburyBasisSpec::Kind::VAR:   return "VAR-k";
        case WoodburyBasisSpec::Kind::Fixed: return "fixed-k";
    }
    return "off";
}

enum class FastMlmaMode : int {
    Off    = 0,
    Slice1 = 1,
    Slice2 = 2,
    Hybrid = 3
};

constexpr const char* fast_mlma_mode_name(FastMlmaMode mode) {
    switch (mode) {
        case FastMlmaMode::Slice1: return "1slice";
        case FastMlmaMode::Slice2: return "2slice";
        case FastMlmaMode::Hybrid: return "hybrid";
        case FastMlmaMode::Off:
        default:                   return "off";
    }
}
