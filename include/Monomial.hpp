#pragma once
#include "GLOBAL.hpp"
#include "HamiltonianField.hpp"

namespace klft {

typedef enum MonomialType_s {
  KLFT_MONOMIAL_GAUGE = 0,
  KLFT_MONOMIAL_FERMION,
  KLFT_MONOMIAL_KINETIC
} MonomialType;

template <typename DGaugeFieldType, typename DAdjFieldType> class Monomial {
public:
  // template argument deduction and safety
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeFieldType = typename DGaugeFieldType::type;
  using AdjFieldType = typename DAdjFieldType::type;

  MonomialType monomial_type;
  real_t H_old, H_new;
  unsigned int time_scale;

  Monomial() : H_old(0.0), H_new(0.0), time_scale(0) {}

  Monomial(unsigned int _time_scale)
      : H_old(0.0), H_new(0.0), time_scale(_time_scale) {}

  virtual MonomialType get_monomial_type() { return monomial_type; }
  virtual void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) = 0;
  virtual void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) = 0;

  void reset() {
    H_old = 0.0;
    H_new = 0.0;
  }

  void set_time_scale(const unsigned int &_time_scale) {
    time_scale = _time_scale;
  }

  unsigned int get_time_scale() { return time_scale; }

  real_t get_delta_H() { return H_new - H_old; }
};

template <typename DGaugeFieldType, typename DAdjFieldType>
class KineticMonomial : public Monomial<DGaugeFieldType, DAdjFieldType> {
public:
  KineticMonomial(unsigned int _time_scale)
      : Monomial<DGaugeFieldType, DAdjFieldType>(_time_scale) {
    Monomial<DGaugeFieldType, DAdjFieldType>::monomial_type =
        KLFT_MONOMIAL_KINETIC;
  }
  void heatbath(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Monomial<DGaugeFieldType, DAdjFieldType>::H_old = h.kinetic_energy();
  }
  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Monomial<DGaugeFieldType, DAdjFieldType>::H_new = h.kinetic_energy();
  }
};

} // namespace klft
