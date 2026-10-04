// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "llrt/rtoptions.hpp"
#include <cstdio>
#include <exception>
#include <string_view>

int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "--api") {
        tt::llrt::RunTimeOptions::set_root_dir(argv[2]);
    } else if (argc != 1) {
        return 2;
    }
    try {
        const tt::llrt::RunTimeOptions options;
        std::printf("SDK_ROOT=%s\n", options.get_root_dir().c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
