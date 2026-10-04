// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "jit_build/types.hpp"
#include <nlohmann/json.hpp>
namespace tt::tt_metal::native_firmware_bundle {
inline nlohmann::json describe_recipe(const jit_build::TargetRecipe& recipe) {
    return {{"target_name", recipe.target_name}, {"cflags", recipe.cflags}, {"defines", recipe.defines},
            {"includes", recipe.includes}, {"pch_umbrella", recipe.pch_umbrella},
            {"compiler_opt_level", recipe.compiler_opt_level}, {"srcs", recipe.srcs}, {"objs", recipe.objs},
            {"lflags", recipe.lflags}, {"extra_link_objs", recipe.extra_link_objs},
            {"linker_script", recipe.linker_script}, {"linker_opt_level", recipe.linker_opt_level},
            {"weakened_firmware_name", recipe.weakened_firmware_name},
            {"firmware_is_kernel_object", recipe.firmware_is_kernel_object}};
}

}
