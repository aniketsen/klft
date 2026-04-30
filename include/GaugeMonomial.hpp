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
#include "GLOBAL.hpp"
#include "GaugeAction.hpp"
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
  constexpr static GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;

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
    Kokkos::Profiling::pushRegion("GaugeHeatbath");
    Monomial<DGaugeFieldType, DAdjFieldType>::H_old =

        WilsonAction<DGaugeFieldType>(h.gauge_field, beta);
    Kokkos::Profiling::popRegion();
  }

  void accept(HamiltonianField<DGaugeFieldType, DAdjFieldType> h) override {
    Kokkos::Profiling::pushRegion("GaugeAccept");
    Monomial<DGaugeFieldType, DAdjFieldType>::H_new =
        WilsonAction<DGaugeFieldType>(h.gauge_field, beta);
    Kokkos::Profiling::popRegion();
  }
  void print() override {
    printf("Gauge Monomial:   %.20f\n", this->get_delta_H());
  }
};

}  // namespace klft
