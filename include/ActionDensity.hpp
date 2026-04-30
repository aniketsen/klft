#pragma once
#include "FieldStrengthTensor.hpp"
#include "GaugePlaquette.hpp"

namespace klft {

template <typename DGaugeFieldType,
          typename FSTTag =
              typename FieldStrengthTensor<DGaugeFieldType>::CloverDef>
struct ActionDensityFunctor {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static size_t Nd = rank;
  // constexpr static GaugeFieldKind kind =
  // DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;

  using GaugeField = typename DGaugeFieldType::type;
  using FieldType = typename DeviceFieldType<rank>::type;

  using RealMatrix = Kokkos::Array<Kokkos::Array<real_t, Nc>, Nc>;
  using ComplexMatrix = Kokkos::Array<Kokkos::Array<complex_t, Nc>, Nc>;

  GaugeField gauge_field;
  FieldType density_per_site;
  const FieldStrengthTensor<DGaugeFieldType> fst;

  ActionDensityFunctor(const GaugeField _gauge_field)
      : gauge_field(_gauge_field),
        density_per_site(_gauge_field.dimensions, real_t(0.0)),
        fst(_gauge_field) {}

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const indexType i0,
                                              const indexType i1,
                                              const indexType i2,
                                              const indexType i3) const {
    Kokkos::Array<Kokkos::Array<SUNAdj<Nc>, Nd>, Nd> C;

    for (int mu = 0; mu < Nd; ++mu) {
      for (int nu = mu + 1; nu < Nd; ++nu) {
        // get the clover C_munu
        C[mu][nu] = fst(FSTTag{}, i0, i1, i2, i3, mu, nu);
        C[nu][mu] = -1.0 * C[mu][nu];  // fst(FSTTag{}, i0, i1, i2, i3, nu, mu);
      }
    }

    real_t local_density = 0.0;
    // use (2.4) in 1304.0533 for renormalization
#pragma unroll
    for (int mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (int nu = 0; nu < Nd; ++nu) {
        if (mu != nu) {
          local_density +=
              0.5 *
              tr<Nc>(C[mu][nu],
                     C[mu][nu]);  // 0.5*: from E=1/4 G^a_{mu,nu}G^a_{mu,nu}
                                  // (one factor 1/2 is contained in tr)
        }
      }
      density_per_site(i0, i1, i2, i3) = local_density;
    }
  }
};

template <typename DGaugeFieldType>
real_t getActionDensity(const typename DGaugeFieldType::type g_in) {
  // calculate the density E according to (3.1) in
  // https://arxiv.org/pdf/1006.4518
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>(),
                "action density requires a device gauge field type.");
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using precision = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::value_type;
  using FieldType = typename DeviceFieldType<Nd>::type;
  FieldType density_per_site(g_in.dimensions, complex_t(0.0, 0.0));
  GaugePlaq<Nd, Nc, precision, kind> gaugePlaquette(g_in, density_per_site,
                                                    g_in.dimensions);
  KTune::parallel_for("Calculate GaugePlaquette",
                      Policy<Nd>(IndexArray<Nd>{0}, g_in.dimensions),
                      gaugePlaquette);
  // ActionDensityFunctor<DGaugeFieldType> actionDensity(g_in);
  // define the functor
  Kokkos::fence();
  // KTune::parallel_for("Calculate ActionDensity",Policy<Nd>( IndexArray<Nd>{0,
  // 0, 0, 0},
  //                         g_in.dimensions), actionDensity);
  // Kokkos::fence();

  // real_t density = Kokkos::real(actionDensity.density_per_site.avg());
  real_t density =
      1 - Kokkos::real(gaugePlaquette.plaq_per_site.avg()) / ((Nd * (Nd - 1)));
  Kokkos::fence();

  return density;
}

template <typename DGaugeFieldType>
real_t getActionDensity_clover(const typename DGaugeFieldType::type g_in) {
  // calculate the density E according to (3.1) in
  // https://arxiv.org/pdf/1006.4518
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>(),
                "action density requires a device gauge field type.");
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  // constexpr static const size_t Nc =
  //     DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  // constexpr static const GaugeFieldKind kind =
  //     DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;

  // using FieldType = typename DeviceFieldType<Nd>::type;
  // FieldType density_per_site(g_in.dimensions, complex_t(0.0, 0.0));
  // GaugePlaq<Nd, Nc, kind> gaugePlaquette(g_in, density_per_site,
  //                                        g_in.dimensions);
  // KTune::parallel_for("Calculate GaugePlaquette",Policy<Nd>(
  // IndexArray<Nd>{0},
  //                         g_in.dimensions, gaugePlaquette);
  ActionDensityFunctor<DGaugeFieldType> actionDensity(g_in);
  // define the functor
  Kokkos::fence();
  KTune::parallel_for("Calculate ActionDensity",
                      Policy<Nd>(IndexArray<Nd>{0, 0, 0, 0}, g_in.dimensions),
                      actionDensity);
  Kokkos::fence();

  real_t density = Kokkos::real(actionDensity.density_per_site.avg());
  Kokkos::fence();

  return density;
}

template <typename DGaugeFieldType>
real_t getActionDensity_rect(const typename DGaugeFieldType::type g_in) {
  // calculate the density E according to (3.1) in
  // https://arxiv.org/pdf/1006.4518
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>(),
                "action density requires a device gauge field type.");
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;

  ActionDensityFunctor<DGaugeFieldType, typename FieldStrengthTensor<
                                            DGaugeFieldType>::RectangleDef>
      actionDensity(g_in);
  // define the functor
  Kokkos::fence();
  KTune::parallel_for("Calculate ActionDensity",
                      Policy<Nd>(IndexArray<Nd>{0, 0, 0, 0}, g_in.dimensions),
                      actionDensity);
  Kokkos::fence();

  real_t density = Kokkos::real(actionDensity.density_per_site.avg());
  Kokkos::fence();

  return density;
}
}  // namespace klft
