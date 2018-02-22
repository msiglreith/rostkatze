#pragma once

#include <cstdint>

namespace stdx {

/// 8-bit signed integer.
using i8 = int8_t;
/// 16-bit signed integer.
using i16 = int16_t;
/// 32-bit signed integer.
using i32 = int32_t;
/// 64-bit signed integer.
using i64 = int64_t;

/// 8-bit unsigned integer.
using u8 = uint8_t;
/// 16-bit unsigned integer.
using u16 = uint16_t;
/// 32-bit unsigned integer.
using u32 = uint32_t;
/// 64-bit unsigned integer.
using u64 = uint64_t;

/// Unsigned integer which can hold an array index.
using usize = size_t;

constexpr auto operator "" _uz (unsigned long long int x) -> usize {
    return x;
}

} // namespace stdx
