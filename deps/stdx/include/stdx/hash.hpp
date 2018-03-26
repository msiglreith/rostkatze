#pragma once

#include "types.hpp"

namespace stdx {

static auto hash_combine(usize& seed) -> void { }

template<typename T0, typename... Ts>
static auto hash_combine(usize& hash, T0 const& t0, Ts... ts) -> void {
    auto h0 { std::hash<T0>{}(t0) };
    hash = hash * 37u + h0;
    hash_combine(hash, ts...);
}

}