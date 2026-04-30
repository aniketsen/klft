#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugePlaquette.hpp"

namespace klft {
template <typename DGaugeFieldType>
real_t WilsonAction(typename DGaugeFieldType::type deviceGaugeField,
                    const real_t beta) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);

  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using precision = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::value_type;
  real_t plaq = GaugePlaquette<Nd, Nc, precision, k>(deviceGaugeField, false);
  return -(beta / static_cast<real_t>(Nc)) * plaq;
}

// use the full 1 - plaq definition
template <typename DGaugeFieldType>
real_t WilsonAction_full(typename DGaugeFieldType::type deviceGaugeField,
                         const real_t beta,
                         const bool normalize = false) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);

  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using precision = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::value_type;
  real_t plaq =
      GaugePlaquette<Nd, Nc, precision, k>(deviceGaugeField, normalize);
  return (beta / static_cast<real_t>(Nc)) * (1 - plaq);
}
}  // namespace klft
