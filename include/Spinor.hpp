#pragma once
#include "GLOBAL.hpp"

namespace klft {
template <size_t Nc>
struct Spinor {
  Kokkos::Array<Kokkos::Array<complex_t, Nc>, 4> s;

  Spinor() = default;
};

}  // namespace klft
