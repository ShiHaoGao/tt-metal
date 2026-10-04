// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "configuration.hpp"
#include <reflect>
#include <stdexcept>
#include <type_traits>

namespace tt::tt_metal::experimental::deployment_detail {
// Versioned structural encoding. No pointers, padding, hash fallback, compiler
// version, source string or filesystem state enters the compatibility record.
class RecordWriter {
public:
    std::vector<std::byte> bytes;
    template<class T> void append(const T& value) {
        if constexpr (std::is_same_v<T, bool>) {
            bytes.push_back(std::byte(value));
        } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
            auto number = static_cast<uint64_t>(value);
            for (unsigned i = 0; i != 8; ++i) bytes.push_back(std::byte(number >> (8 * i)));
        } else if constexpr (std::is_same_v<T, CoreCoord>) {
            append(value.x); append(value.y);
        } else if constexpr (std::is_same_v<T, tt_cxy_pair>) {
            append(value.chip); append(value.x); append(value.y);
        } else if constexpr (requires { value.has_value(); }) {
            append(value.has_value());
            if (value) append(*value);
        } else if constexpr (requires { std::variant_size<T>::value; }) {
            append(value.index());
            std::visit([&](const auto& member) { append(member); }, value);
        } else if constexpr (std::is_same_v<T, std::monostate>) {
            throw std::invalid_argument("missing dispatch kernel configuration");
        } else if constexpr (requires { value.size(); value.begin(); }) {
            append(value.size());
            for (const auto& member : value) append(member);
        } else {
            static_assert(std::is_aggregate_v<T>, "publication requires a structural value owner");
            reflect::for_each([&](auto i) { append(reflect::get<i>(value)); }, value);
        }
    }
};
} // namespace tt::tt_metal::experimental::deployment_detail
