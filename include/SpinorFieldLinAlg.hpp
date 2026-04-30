//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/
#pragma once

#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GammaMatrix.hpp"
#include "Spinor.hpp"

//  For  now in an external file, should be in SpinorField.hpp
namespace klft {
template <typename DSpinorFieldType>
struct resetSpinorFieldFunctor {
  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType a;

  static constexpr index_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static constexpr index_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;
  resetSpinorFieldFunctor(
      SpinorFieldType& a,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : a(a), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
#pragma unroll
    for (index_t c1 = 0; c1 < RepDim; ++c1) {
#pragma unroll
      for (index_t c2 = 0; c2 < Nc; ++c2) {
        a(Idcs...)[c1][c2] = 0;
      }
    }
  }
};
template <typename DSpinorFieldType>
KOKKOS_FORCEINLINE_FUNCTION void resetSpinorField(
    typename DSpinorFieldType::type& a) {
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};

  resetSpinorFieldFunctor<DSpinorFieldType> SDP(a, a.dimensions);

  KTune::parallel_for(
      "resetSpinorField" + a.getPrecision(),
      Policy<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>(
          start, a.dimensions),
      SDP);
  Kokkos::fence();
}
template <typename DSpinorFieldType>
struct SpinorDotProduct {
  using SpinorFieldType = typename DSpinorFieldType::type;
  const SpinorFieldType a;
  const SpinorFieldType b;
  using FieldType = typename DeviceFieldType<
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>::type;
  FieldType dot_product_per_site;

  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;
  SpinorDotProduct(
      const SpinorFieldType& a,
      const SpinorFieldType& b,
      FieldType& dot_product_per_site,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : a(a),
        b(b),
        dot_product_per_site(dot_product_per_site),
        dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    complex_t temp_result = spinor_inner_prod(a(Idcs...), b(Idcs...));
    dot_product_per_site(Idcs...) = temp_result;
  }
};

template <typename DSpinorFieldType>
KOKKOS_FORCEINLINE_FUNCTION complex_t
spinor_dot_product(const typename DSpinorFieldType::type& a,
                   const typename DSpinorFieldType::type& b,
                   typename DeviceFieldType<DeviceFermionFieldTypeTraits<
                       DSpinorFieldType>::Rank>::type& dot_product_per_site) {
  assert(a.dimensions == b.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(a.field)::execution_space,
          typename decltype(b.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction
  complex_t result = 0.0;
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};

  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  // using FieldType = typename
  // DeviceFieldType<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>::type;
  // FieldType dot_product_per_site(end, complex_t(0.0, 0.0));
  SpinorDotProduct<DSpinorFieldType> SDP(a, b, dot_product_per_site,
                                         a.dimensions);

  KTune::parallel_for(
      "SpinorField_dot_product" + a.getPrecision(),
      Policy<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>(
          start, a.dimensions),
      SDP);
  Kokkos::fence();
  result = dot_product_per_site.sum();
  Kokkos::fence();
  return result;
}
template <typename DSpinorFieldType>
KOKKOS_FORCEINLINE_FUNCTION complex_t
spinor_dot_product(const typename DSpinorFieldType::type& a,
                   const typename DSpinorFieldType::type& b) {
  assert(a.dimensions == b.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(a.field)::execution_space,
          typename decltype(b.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction

  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceFieldType<
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>::type;
  FieldType dot_product_per_site(a.dimensions, complex_t(0.0, 0.0));

  Kokkos::fence();
  return spinor_dot_product<DSpinorFieldType>(a, b, dot_product_per_site);
}
template <typename DSpinorFieldType>
struct SpinorNorm {
  using SpinorFieldType = typename DSpinorFieldType::type;
  const SpinorFieldType a;
  using FieldType = typename DeviceScalarFieldType<
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>::type;
  FieldType norm_per_site;
  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;

  SpinorNorm(
      const SpinorFieldType& a,
      FieldType& norm_per_site,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : a(a), norm_per_site(norm_per_site), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    real_t temp_result = sqnorm(a(Idcs...));
    norm_per_site(Idcs...) = temp_result;
  }
};

template <typename DSpinorFieldType>
KOKKOS_FORCEINLINE_FUNCTION real_t
spinor_norm_sq(const typename DSpinorFieldType::type& a,
               typename DeviceScalarFieldType<
                   DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>::type&
                   norm_per_site) {
  real_t result = 0.0;
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};

  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  // using FieldType = typename
  // DeviceScalarFieldType<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>::type;
  // real_t init = 0;
  // FieldType norm_per_site(end, init);
  SpinorNorm<DSpinorFieldType> norm(a, norm_per_site, a.dimensions);
  KTune::parallel_for(
      "SpinorField_norm" + a.getPrecision(),
      Policy<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>(
          start, a.dimensions),
      norm);
  Kokkos::fence();
  result = norm_per_site.sum();
  Kokkos::fence();
  return result;
}
template <typename DSpinorFieldType>
KOKKOS_FORCEINLINE_FUNCTION real_t
spinor_norm_sq(const typename DSpinorFieldType::type& a) {
  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceScalarFieldType<
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>::type;
  real_t init = 0;
  FieldType norm_per_site(a.dimensions, init);
  Kokkos::fence();

  return spinor_norm_sq<DSpinorFieldType>(a, norm_per_site);
}
template <typename DSpinorFieldType>
KOKKOS_FORCEINLINE_FUNCTION real_t
spinor_norm(const typename DSpinorFieldType::type& a) {
  return Kokkos::sqrt(spinor_norm_sq<DSpinorFieldType>(a));
}
template <typename DSpinorFieldType>
KOKKOS_FORCEINLINE_FUNCTION real_t
spinor_norm(const typename DSpinorFieldType::type& a,
            typename DeviceScalarFieldType<
                DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>::type&
                norm_per_site) {
  return Kokkos::sqrt(spinor_norm_sq<DSpinorFieldType>(a, norm_per_site));
}
template <typename DSpinorFieldType>
struct axpyFunctor {
  using SpinorFieldType = typename DSpinorFieldType::type;
  const SpinorFieldType x;
  const SpinorFieldType y;
  const complex_t alpha;
  SpinorFieldType c;
  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;
  axpyFunctor(
      const SpinorFieldType::value_type& alpha,
      const SpinorFieldType& x,
      const SpinorFieldType& y,
      SpinorFieldType& c,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : x(x), y(y), c(c), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = (y(Idcs...) + (alpha * x(Idcs...)));
  }
};

template <typename DSpinorFieldType>
struct axpyG5Functor {
  using SpinorFieldType = typename DSpinorFieldType::type;
  const SpinorFieldType x;
  const SpinorFieldType y;
  const complex_t alpha;
  SpinorFieldType c;
  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;
  axpyG5Functor(
      const SpinorFieldType::value_type& alpha,
      const SpinorFieldType& x,
      const SpinorFieldType& y,
      SpinorFieldType& c,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : x(x), y(y), c(c), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = gamma5(y(Idcs...) + (alpha * x(Idcs...)));
  }
};
/// @brief Calculates alpha*x+y

/// @param alpha
/// @param x
/// @param y
/// @return c = alpha*x+y
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axpy(const complex_t& alpha,
     const typename DSpinorFieldType::type& x,
     const typename DSpinorFieldType::type& y) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpyFunctor<DSpinorFieldType> add(alpha, x, y, c, x.dimensions);

  KTune::parallel_for(
      "SpinorField_axpy" + x.getPrecision(),
      Policy<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>(
          start, x.dimensions),
      add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates alpha*x+y
/// @param alpha
/// @param x
/// @param y
/// @param c
/// @return c = alpha*x+y
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION axpy(const complex_t& alpha,
                                      const typename DSpinorFieldType::type& x,
                                      const typename DSpinorFieldType::type& y,
                                      typename DSpinorFieldType::type& c) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpyFunctor<DSpinorFieldType> add(alpha, x, y, c, x.dimensions);

  KTune::parallel_for(
      "SpinorField_axpy_inplace" + x.getPrecision(),
      Policy<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>(
          start, x.dimensions),
      add);
  Kokkos::fence();
}

/// @brief Calculates gamma5(alpha*x+y)

/// @param alpha
/// @param x
/// @param y
/// @return c = gamma5(alpha*x+y)
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axpyG5(const complex_t& alpha,
       const typename DSpinorFieldType::type& x,
       const typename DSpinorFieldType::type& y) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpyG5Functor<DSpinorFieldType> add(alpha, x, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_G5axpy" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates alpha*x+y
/// @param alpha
/// @param x
/// @param y
/// @param c
/// @return c = gamma5(alpha*x+y)
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION
axpyG5(const complex_t& alpha,
       const typename DSpinorFieldType::type& x,
       const typename DSpinorFieldType::type& y,
       typename DSpinorFieldType::type& c) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpyG5Functor<DSpinorFieldType> add(alpha, x, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5axpy_inplace" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
}
template <typename DSpinorFieldType>
struct axFunctor {
  using SpinorFieldType = typename DSpinorFieldType::type;
  const SpinorFieldType x;
  const complex_t alpha;
  SpinorFieldType c;
  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;
  axFunctor(
      const complex_t& alpha,
      const SpinorFieldType& x,
      SpinorFieldType& c,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : x(x), c(c), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = (alpha * x(Idcs...));
  }
};
/// @brief Calculates alpha*x
/// @param alpha
/// @param x
/// @return c = alpha*x
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
ax(const complex_t& alpha, const typename DSpinorFieldType::type& x) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axFunctor<DSpinorFieldType> add(alpha, x, c, x.dimensions);

  KTune::parallel_for("SpinorField_ax" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates alpha*x
/// @param alpha
/// @param x
/// @param y
/// @param c
/// @return c = alpha*x
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION ax(const complex_t& alpha,
                                    const typename DSpinorFieldType::type& x,
                                    typename DSpinorFieldType::type& c) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == c.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(c.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axFunctor<DSpinorFieldType> add(alpha, x, c, x.dimensions);

  KTune::parallel_for(
      "SpinorField_ax_inplace" + x.getPrecision(),
      Policy<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>(
          start, x.dimensions),
      add);
  Kokkos::fence();
}
template <typename DSpinorFieldType>
struct axG5Functor {
  using SpinorFieldType = typename DSpinorFieldType::type;
  const SpinorFieldType x;
  const complex_t alpha;
  SpinorFieldType c;
  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;
  axG5Functor(
      const complex_t& alpha,
      const SpinorFieldType& x,
      SpinorFieldType& c,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : x(x), c(c), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) = gamma5((alpha * x(Idcs...)));
  }
};
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axG5(const complex_t& alpha, const typename DSpinorFieldType::type& x) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axG5Functor<DSpinorFieldType> add(alpha, x, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5ax" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates alpha*x
/// @param alpha
/// @param x
/// @param y
/// @param c
/// @return c = alpha*x
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION axG5(const complex_t& alpha,
                                      const typename DSpinorFieldType::type& x,
                                      typename DSpinorFieldType::type& c) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == c.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(c.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or host-host
                                                                // interaction
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axG5Functor<DSpinorFieldType> add(alpha, x, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5ax_inplace" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
}
template <typename DSpinorFieldType>
struct axpbyG5Functor {
  using SpinorFieldType = typename DSpinorFieldType::type;
  const SpinorFieldType x;
  const SpinorFieldType y;
  const complex_t alpha;
  const complex_t beta;
  SpinorFieldType c;
  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;
  axpbyG5Functor(
      const complex_t& alpha,
      const SpinorFieldType& x,
      const complex_t& beta,
      const SpinorFieldType& y,
      SpinorFieldType& c,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : x(x), y(y), c(c), beta(beta), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) += gamma5(beta * y(Idcs...) + (alpha * x(Idcs...)));
  }
};
/// @brief Calculates gamma5(alpha*x+beta*y)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @return c = gamma5(alpha*x+betay)
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axpbyG5(const complex_t& alpha,
        const typename DSpinorFieldType::type& x,
        const complex_t& beta,
        const typename DSpinorFieldType::type& y) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpbyG5Functor<DSpinorFieldType> add(alpha, x, beta, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5axpby" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
  return c;
}

/// @brief Calculates gamma5(alpha*x+beta*y+c)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @return c += gamma5(alpha*x+betay)
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION
axpbyG5(const complex_t& alpha,
        const typename DSpinorFieldType::type& x,
        const complex_t& beta,
        const typename DSpinorFieldType::type& y,
        typename DSpinorFieldType::type& c) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpbyG5Functor<DSpinorFieldType> add(alpha, x, beta, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_g5axpby_inplace" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
}

/*------------------------------*/

template <typename DSpinorFieldType>
struct axpbyFunctor {
  using SpinorFieldType = typename DSpinorFieldType::type;
  const SpinorFieldType x;
  const SpinorFieldType y;
  const complex_t alpha;
  const complex_t beta;
  SpinorFieldType c;
  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;
  axpbyFunctor(
      const complex_t& alpha,
      const SpinorFieldType& x,
      const complex_t& beta,
      const SpinorFieldType& y,
      SpinorFieldType& c,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : x(x), y(y), c(c), beta(beta), alpha(alpha), dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) += (beta * y(Idcs...) + (alpha * x(Idcs...)));
  }
};
/// @brief Calculates (alpha*x+beta*y+c)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @return c += (alpha*x+betay)
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axpby(const complex_t& alpha,
      const typename DSpinorFieldType::type& x,
      const complex_t& beta,
      const typename DSpinorFieldType::type& y) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpbyFunctor<DSpinorFieldType> add(alpha, x, beta, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_axpby" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates (alpha*x+beta*y)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @return c = (alpha*x+betay)
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION axpby(const complex_t& alpha,
                                       const typename DSpinorFieldType::type& x,
                                       const complex_t& beta,
                                       const typename DSpinorFieldType::type& y,
                                       typename DSpinorFieldType::type& c) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpbyFunctor<DSpinorFieldType> add(alpha, x, beta, y, c, x.dimensions);

  KTune::parallel_for("SpinorField_axpby_inplace" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
}
/*%%%%%%%%%%%%%%%%%%%*/

template <typename DSpinorFieldType>
struct axpbypczFunctor {
  using SpinorFieldType = typename DSpinorFieldType::type;
  const SpinorFieldType x;
  const SpinorFieldType y;
  const SpinorFieldType z;
  const complex_t alpha;
  const complex_t beta;
  const complex_t gamma;
  SpinorFieldType c;
  const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>
      dimensions;
  axpbypczFunctor(
      const complex_t& alpha,
      const SpinorFieldType& x,
      const complex_t& beta,
      const SpinorFieldType& y,
      const complex_t gamma,
      const SpinorFieldType z,
      SpinorFieldType& c,
      const IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank>&
          dimensions)
      : x(x),
        y(y),
        z(z),
        c(c),
        beta(beta),
        alpha(alpha),
        gamma(gamma),
        dimensions(dimensions) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // axpy(alpha, x(Idcs...), y(Idcs...), c(Idcs...));
    c(Idcs...) =
        (beta * y(Idcs...) + (alpha * x(Idcs...)) + (gamma * z(Idcs...)));
  }
};
/// @brief Calculates (alpha*x+beta*y+c)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @param gamma
/// @param z
/// @return c += (alpha*x+beta*y+gamma*z)
template <typename DSpinorFieldType>
typename DSpinorFieldType::type KOKKOS_FORCEINLINE_FUNCTION
axpbypcz(const complex_t& alpha,
         const typename DSpinorFieldType::type& x,
         const complex_t& beta,
         const typename DSpinorFieldType::type& y,
         const complex_t& gamma,
         const typename DSpinorFieldType::type& z) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction

  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType c(x.dimensions, complex_t(0.0, 0.0));
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpbypczFunctor<DSpinorFieldType> add(alpha, x, beta, y, gamma, z, c,
                                        x.dimensions);

  KTune::parallel_for("SpinorField_axpbypcz" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
  return c;
}
/// @brief Calculates (alpha*x+beta*y)

/// @param alpha
/// @param x
/// @param beta
/// @param y
/// @return c = (alpha*x+betay)
template <typename DSpinorFieldType>
void KOKKOS_FORCEINLINE_FUNCTION
axpbypcz(const complex_t& alpha,
         const typename DSpinorFieldType::type& x,
         const complex_t& beta,
         const typename DSpinorFieldType::type& y,
         const complex_t& gamma,
         const typename DSpinorFieldType::type& z,
         typename DSpinorFieldType::type& c) {
  constexpr static size_t Rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  assert(x.dimensions == y.dimensions);
  static_assert(
      Kokkos::SpaceAccessibility<
          typename decltype(x.field)::execution_space,
          typename decltype(y.field)::memory_space>::accessible,
      "Execution space of A cannot access memory space of B");  // allow only
                                                                // device-device
                                                                // or
                                                                // host-host
                                                                // interaction
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank> start{};
  axpbypczFunctor<DSpinorFieldType> add(alpha, x, beta, y, gamma, z, c,
                                        x.dimensions);

  KTune::parallel_for("SpinorField_axpbypcz_inplace" + x.getPrecision(),
                      Policy<Rank>(start, x.dimensions), add);
  Kokkos::fence();
}

}  // namespace klft
