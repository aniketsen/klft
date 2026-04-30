#pragma once
#include "GLOBAL.hpp"

namespace klft {
namespace Reducer {
template <class ScalarType, size_t N>
struct ArrayType {
  Kokkos::Array<ScalarType, N> array;
  KOKKOS_FORCEINLINE_FUNCTION ArrayType() = default;
  KOKKOS_FORCEINLINE_FUNCTION
  ArrayType(const ArrayType& rhs) {
    for (int i = 0; i < N; i++) {
      array[i] = rhs.array[i];
    }
  }
  KOKKOS_INLINE_FUNCTION  // add operator
      ArrayType&
      operator+=(const ArrayType& src) {
    for (int i = 0; i < N; i++) {
      array[i] += src.array[i];
    }
    return *this;
  }
};
}  // namespace Reducer
}  // namespace klft

namespace Kokkos {  // reduction identity must be defined in Kokkos namespace
                    // Usage with Kokkos::Sum<klft::Reducer::ArrayType<T, N>>(.)
template <class T, size_t N>
struct reduction_identity<klft::Reducer::ArrayType<T, N> > {
  KOKKOS_FORCEINLINE_FUNCTION static klft::Reducer::ArrayType<T, N> sum() {
    return klft::Reducer::ArrayType<T, N>();
  }
};
}  // namespace Kokkos
