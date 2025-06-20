#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"

namespace klft {

class UpdatePosition : public std::enable_shared_from_this<UpdatePosition> {
public:
  UpdatePosition() = delete;
  virtual ~UpdatePosition() = default;

  // update the position of the system
  // using the given step size
  virtual void update(const real_t step_size) = 0;

protected:
  explicit UpdatePosition(int tag) {}
};

template <size_t rank, size_t Nc>
class UpdatePositionGauge : public UpdatePosition {
public:
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  using AdjFieldType = typename DeviceAdjFieldType<rank, Nc>::type;
  GaugeFieldType &gauge_field;
  AdjFieldType &adjoint_field;
  real_t eps;

  UpdatePositionGauge() = delete;
  ~UpdatePositionGauge() = default;

  UpdatePositionGauge(GaugeFieldType &gauge_field_,
                      AdjFieldType &adjoint_field_)
      : UpdatePosition(0), gauge_field(gauge_field_),
        adjoint_field(adjoint_field_), eps(0.0) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      gauge_field(Idcs..., mu) =
          expoSUN(eps * adjoint_field(Idcs..., mu)) * gauge_field(Idcs..., mu);
    }
  }

  void update(const real_t step_size) override {
    eps = step_size;
    IndexArray<rank> start;
    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }
    // launch the kernel
    tune_and_launch_for<rank>("UpdatePositionGauge", start,
                              gauge_field.dimensions, *this);
    Kokkos::fence();
  }
};
} // namespace klft
