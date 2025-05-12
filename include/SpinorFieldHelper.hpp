#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Spinor.hpp"
#include "Tuner.hpp"

//  For  now in an external file, should be in SpinorField.hpp
namespace klft {

template <size_t rank, size_t Nc, size_t RepDim>
struct SpinorDotProduct {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType a;
  const SpinorFieldType b;
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType dot_product_per_site;

  const IndexArray<rank> dimensions;
  SpinorDotProduct(const SpinorFieldType& a,
                   const SpinorFieldType& b,
                   FieldType& dot_product_per_site,
                   const IndexArray<rank>& dimensions)
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

template <size_t rank, size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION complex_t spinor_dot_product(
    const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& a,
    const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& b) {
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
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = a.dimensions[i];
  }
  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType dot_product_per_site(end, complex_t(0.0, 0.0));
  SpinorDotProduct<rank, Nc, RepDim> SDP(a, b, dot_product_per_site, end);

  tune_and_launch_for<rank>("SpinorField_dot_product", start, end, SDP);
  Kokkos::fence();
  result = dot_product_per_site.sum();
  Kokkos::fence();
  return result;
}

template <size_t rank, size_t Nc, size_t RepDim>
struct SpinorNorm {
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType a;
  using FieldType = typename DeviceScalarFieldType<rank>::type;
  FieldType norm_per_site;
  const IndexArray<rank> dimensions;

  SpinorNorm(const SpinorFieldType& a,
             FieldType& norm_per_site,
             const IndexArray<rank>& dimensions)
      : a(a), norm_per_site(norm_per_site), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    real_t temp_result = sqnorm(a(Idcs...));
    norm_per_site(Idcs...) = temp_result;
  }
};

template <size_t rank, size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION real_t
spinor_norm(const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& a) {
  real_t result = 0.0;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = a.dimensions[i];
  }
  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceScalarFieldType<rank>::type;
  real_t init = 0;
  FieldType norm_per_site(end, init);
  SpinorNorm<rank, Nc, RepDim> norm(a, norm_per_site, end);

  tune_and_launch_for<rank>("SpinorField_norm", start, end, norm);
  Kokkos::fence();
  result = norm_per_site.sum();
  Kokkos::fence();
  return result;
}

}  // namespace klft