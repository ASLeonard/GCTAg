#pragma once

#include "Cli.hpp"
#include <string>

class MLMA {
public:
    static void run(const gcta_cli::MlmaOptions& opts, const std::string& out_prefix);
};
