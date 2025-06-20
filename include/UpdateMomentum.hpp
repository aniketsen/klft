
#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Gauge_Util.hpp"

namespace klft {

class UpdateMomentum : public std::enable_shared_from_this<UpdateMomentum> {
public:
  UpdateMomentum() = delete;
  virtual ~UpdateMomentum() = default;

  // update the momentum of the system
  // using the given step size
  virtual void update(const real_t step_size) = 0;

protected:
  explicit UpdateMomentum(int Tag) {}
};

template <typename DGaugeFieldType, typename DAdjFieldType>
class UpdateMomentumGauge : public UpdateMomentum {
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
  GaugeFieldType &gauge_field;
  AdjFieldType &adjoint_field;
  real_t beta;

  real_t eps;
  ConstGaugeFieldType<rank, Nc> staple_field;

  UpdateMomentumGauge() = delete;
  ~UpdateMomentumGauge() = default;

  UpdateMomentumGauge(GaugeFieldType &gauge_field_,
                      AdjFieldType &adjoint_field_, const real_t &beta_)
      : UpdateMomentum(0), gauge_field(gauge_field_),
        adjoint_field(adjoint_field_), beta(beta_), eps(0.0) {}
  // todo: Add Force as a function instead of it being incorporated into the
  // functor.

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
// Update the momentum field
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      adjoint_field(Idcs..., mu) -=
          this->eps *
          ((this->beta / this->Nc) * (traceT(this->gauge_field(Idcs..., mu) *
                                             this->staple_field(Idcs..., mu))));
    }
  }

  void update(const real_t step_size) override {
    eps = step_size;
    IndexArray<rank> start;
    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }
    // launch the kernels
    staple_field = stapleField<DGaugeFieldType>(gauge_field);
    tune_and_launch_for<rank>("UpdateMomentumGauge", start,
                              gauge_field.dimensions, *this);
    Kokkos::fence();
  }
};

} // namespace klft
