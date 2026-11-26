#pragma once

#include "Cli.hpp"
#include "Marker.h"
#include "Pheno.h"
#include "pgen_hardcall.hpp"

#include <string>
#include <vector>

namespace gcta_grm {

// Clean-room BED / native hardcall PGEN autosomal additive VanRaden INT8 GRM engine.
// Decoupled from legacy GRM / Geno / GenoBufItem / BedBackend / PgenBackend / loopDouble.
void run_bed_int8_grm(Pheno& pheno,
                      Marker& marker,
                      const gcta_cli::GrmMakeOptions& opts,
                      const std::string& out_prefix,
                      std::vector<pgen_hardcall::PgenFileIndex> preloaded_pgen_indices = {});

} // namespace gcta_grm
