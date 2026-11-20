#pragma once

#include "Cli.hpp"
#include <string>

class PCAStream {
public:
    static void run(const gcta_cli::PcaOptions& opts, const std::string& out_prefix);
};
