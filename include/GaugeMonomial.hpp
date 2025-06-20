#pragma once
#include "GLOBAL.hpp"
#include "GaugePlaquette.hpp"
#include "HamiltonianField.hpp"
#include "Monomial.hpp"

namespace klft {

template <typename DGaugeFieldType, typename DAdjFieldType>
class GaugeMonomial : public Monomial<DGaugeFieldType, DAdjFieldType> {
  // template argument deduction and safety
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeFieldType = typename DGaugeFieldType::type;

public:
  real_t beta;

  GaugeMonomial(real_t _beta, unsigned int _time_scale)
      : Monomial<DGaugeFieldType, DAdjFieldType>::Monomial(_time_scale) {
    beta = _beta;
    Monomial<DGaugeFieldType, DAdjFieldType>::monomial_type =
        KLFT_MONOMIAL_GAUGE;
  }

  void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Monomial<DGaugeFieldType, DAdjFieldType>::H_old =
        -(beta / static_cast<real_t>(Nc)) *
        GaugePlaquette<rank, Nc>(h.gauge_field, false);
  }

  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Monomial<DGaugeFieldType, DAdjFieldType>::H_new =
        -(beta / static_cast<real_t>(Nc)) *
        GaugePlaquette<rank, Nc>(h.gauge_field, false);
  }
};

} // namespace klft
