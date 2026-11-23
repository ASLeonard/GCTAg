#pragma once

#include "Cli.hpp"
#include "Marker.h"
#include "Pheno.h"

#include <string>

namespace gcta_grm {

// Clean-room BED autosomal additive VanRaden INT8 GRM engine.
// Decoupled from legacy GRM / Geno / GenoBufItem / BedBackend / loopDouble.
void run_bed_int8_grm(Pheno& pheno,
                      Marker& marker,
                      const gcta_cli::GrmMakeOptions& opts,
                      const std::string& out_prefix);

} // namespace gcta_grm
