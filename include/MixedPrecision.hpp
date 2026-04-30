#include <KTune/KTune.hpp>
#include "GLOBAL.hpp"
namespace klft {

template <typename DSpinorFieldTypePrecision1,
          typename DSpinorFieldTypePrecision2>
struct changePrecisionSpinorFieldFunktor {
  using SpinorFieldTypePrecision1 = typename DSpinorFieldTypePrecision1::type;
  using SpinorFieldTypePrecision2 = typename DSpinorFieldTypePrecision2::type;
  static constexpr index_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::RepDim;
  static constexpr index_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::Nc;
  const IndexArray<
      DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::Rank>
      dimensions;
  using new_precision = typename DeviceFermionFieldTypeTraits<
      DSpinorFieldTypePrecision1>::value_type;
  SpinorFieldTypePrecision1 dest;
  SpinorFieldTypePrecision2 src;
  changePrecisionSpinorFieldFunktor(
      SpinorFieldTypePrecision1& dest,
      const SpinorFieldTypePrecision2& src,
      const IndexArray<
          DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::Rank>&
          dimensions)
      : dest(dest), src(src), dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
#pragma unroll
    for (index_t c1 = 0; c1 < RepDim; ++c1) {
#pragma unroll
      for (index_t c2 = 0; c2 < Nc; ++c2) {
        dest(Idcs...)[c1][c2] =
            static_cast<new_precision>(src(Idcs...)[c1][c2]);
      }
    }
  }
};

template <typename DSpinorFieldTypePrecision1,
          typename DSpinorFieldTypePrecision2>
void changePrecisionSpinorField(
    typename DSpinorFieldTypePrecision1::type& dest,
    const typename DSpinorFieldTypePrecision2::type& src) {
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::Rank>
      start{};
  changePrecisionSpinorFieldFunktor<DSpinorFieldTypePrecision1,
                                    DSpinorFieldTypePrecision2>
      funktor(dest, src, start);
  KTune::parallel_for(
      "changePrecisionSpinorField",
      Policy<DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::Rank>(
          start, dest.dimensions),
      funktor);
  Kokkos::fence();
}

template <typename DGaugeFieldTypePrecision1,
          typename DGaugeFieldTypePrecision2>
struct changePrecisionGaugeFieldFunktor {
  using GaugeFieldTypePrecision1 = typename DGaugeFieldTypePrecision1::type;
  using GaugeFieldTypePrecision2 = typename DGaugeFieldTypePrecision2::type;

  static constexpr index_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldTypePrecision1>::Nc;
  static constexpr index_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldTypePrecision1>::Rank;
  using new_precision = typename DeviceGaugeFieldTypeTraits<
      DGaugeFieldTypePrecision1>::value_type;
  GaugeFieldTypePrecision1 dest;
  GaugeFieldTypePrecision2 src;
  changePrecisionGaugeFieldFunktor(GaugeFieldTypePrecision1& dest,
                                   const GaugeFieldTypePrecision2& src)
      : dest(dest), src(src) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
#pragma unroll
      for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
        for (index_t c2 = 0; c2 < Nc; ++c2) {
          dest(Idcs..., mu)[c1][c2] =
              static_cast<new_precision>(src(Idcs..., mu)[c1][c2]);
        }
      }
    }
  }
};
template <typename DGaugeFieldTypePrecision1,
          typename DGaugeFieldTypePrecision2>
void changePrecisionGaugeField(
    typename DGaugeFieldTypePrecision1::type& dest,
    const typename DGaugeFieldTypePrecision2::type& src) {
  IndexArray<DeviceGaugeFieldTypeTraits<DGaugeFieldTypePrecision1>::Rank>
      start{};
  changePrecisionGaugeFieldFunktor<DGaugeFieldTypePrecision1,
                                   DGaugeFieldTypePrecision2>
      funktor(dest, src);
  KTune::parallel_for(
      "changePrecisionGaugeField",
      Policy<DeviceGaugeFieldTypeTraits<DGaugeFieldTypePrecision1>::Rank>(
          start, dest.dimensions),
      funktor);
  Kokkos::fence();
}

template <typename DSpinorFieldTypePrecision1,
          typename DSpinorFieldTypePrecision2>
struct xpyMixedFunctor {
  using SpinorFieldTypePrecision1 = typename DSpinorFieldTypePrecision1::type;
  using SpinorFieldTypePrecision2 = typename DSpinorFieldTypePrecision2::type;
  static constexpr index_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::RepDim;
  static constexpr index_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::Nc;

  using new_Precision = typename DeviceFermionFieldTypeTraits<
      DSpinorFieldTypePrecision1>::value_type;
  SpinorFieldTypePrecision1 x;
  SpinorFieldTypePrecision2 y;
  xpyMixedFunctor(SpinorFieldTypePrecision1& x,
                  const SpinorFieldTypePrecision2& y)
      : x(x), y(y) {}
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
#pragma unroll
    for (index_t c1 = 0; c1 < RepDim; ++c1) {
#pragma unroll
      for (index_t c2 = 0; c2 < Nc; ++c2) {
        x(Idcs...)[c1][c2] += static_cast<new_Precision>(y(Idcs...)[c1][c2]);
      }
    }
  }
};
template <typename DSpinorFieldTypePrecision1,
          typename DSpinorFieldTypePrecision2>
void xpyMixed(typename DSpinorFieldTypePrecision1::type& x,
              const typename DSpinorFieldTypePrecision2::type& y) {
  IndexArray<DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::Rank>
      start{};
  xpyMixedFunctor<DSpinorFieldTypePrecision1, DSpinorFieldTypePrecision2>
      funktor(x, y);
  KTune::parallel_for(
      "xpyMixed",
      Policy<DeviceFermionFieldTypeTraits<DSpinorFieldTypePrecision1>::Rank>(
          start, x.dimensions),
      funktor);
  Kokkos::fence();
}

}  // namespace klft