
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
#include "Gauge_Util.hpp"
#include "Kokkos_Complex.hpp"

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
  GaugeFieldType gauge_field;
  typename DeviceGaugeFieldType<rank, Nc>::type staple_field;
  AdjFieldType adjoint_field;
  real_t beta;

  real_t eps;
  // ConstGaugeFieldType<rank, Nc> staple_field;

  UpdateMomentumGauge() = delete;
  ~UpdateMomentumGauge() = default;

  UpdateMomentumGauge(GaugeFieldType& gauge_field_,
                      AdjFieldType& adjoint_field_,
                      const real_t& beta_)
      : UpdateMomentum(0),
        gauge_field(gauge_field_),
        adjoint_field(adjoint_field_),
        beta(beta_),
        eps(0.0) {
    this->staple_field = typename DeviceGaugeFieldType<rank, Nc>::type(
        gauge_field.dimensions, complex_t(0.0, 0.0));
  }
  // todo: Add Force as a function instead of it being incorporated into the
  // functor.

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
// Update the momentum field
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      adjoint_field(Idcs..., mu) -=
          this->eps * ((this->beta / this->Nc) *  // 0.5 *
                       (traceT((this->gauge_field(Idcs..., mu) *
                                (this->staple_field(Idcs..., mu))))));
    }
  }

  void update(const real_t step_size) override {
    Kokkos::Profiling::pushRegion("updateMomentumGauge");
    eps = step_size;
    IndexArray<rank> start;
    for (size_t i = 0; i < rank; ++i) {
      start[i] = 0;
    }

    // launch the kernels
    stapleField<DGaugeFieldType>(gauge_field, staple_field);
    Kokkos::fence();
    KTune::parallel_for("UpdateMomentumGauge",
                        Policy<rank>(start, gauge_field.dimensions), *this);
    Kokkos::fence();
    Kokkos::Profiling::popRegion();
  }
};

}  // namespace klft
