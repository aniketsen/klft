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
#include "AdjointFieldHelper.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Kokkos_Macros.hpp"

namespace klft {

template <typename DGaugeFieldType, typename DAdjFieldType>
struct HamiltonianField {
  // template argument deduction and safety
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;

  GaugeField gauge_field;
  AdjointField adjoint_field;
  struct EKin {};

  HamiltonianField() = delete;

  HamiltonianField(GaugeField& _gauge_field, AdjointField& _adjoint_field)
      : gauge_field(_gauge_field), adjoint_field(_adjoint_field) {}

  // Rank-4 version
  KOKKOS_INLINE_FUNCTION void operator()(EKin,
                                         index_t i0,
                                         index_t i1,
                                         index_t i2,
                                         index_t i3,
                                         real_t& rtn) const {
    real_t tmp = 0.0;
    for (index_t mu = 0; mu < 4; ++mu) {
      tmp += 0.5 * norm2<Nc>(adjoint_field(i0, i1, i2, i3, mu));
    }
    rtn += tmp;
  }

  // Rank-3 version
  KOKKOS_INLINE_FUNCTION void operator()(EKin,
                                         index_t i0,
                                         index_t i1,
                                         index_t i2,
                                         real_t& rtn) const {
    real_t tmp = 0.0;
    for (index_t mu = 0; mu < 3; ++mu) {
      tmp += 0.5 * norm2<Nc>(adjoint_field(i0, i1, i2, mu));
    }
    rtn += tmp;
  }

  // Rank-2 version
  KOKKOS_INLINE_FUNCTION void operator()(EKin,
                                         index_t i0,
                                         index_t i1,
                                         real_t& rtn) const {
    real_t tmp = 0.0;
    for (index_t mu = 0; mu < 2; ++mu) {
      tmp += 0.5 * norm2<Nc>(adjoint_field(i0, i1, mu));
    }
    rtn += tmp;
  }

  real_t kinetic_energy() {
    real_t kinetic_energy = 0.0;
    auto rp = Kokkos::MDRangePolicy<EKin, Kokkos::Rank<rank>>(
        IndexArray<rank>{0}, adjoint_field.dimensions);
    KTune::parallel_reduce("kinetic_energy", rp, *this, kinetic_energy);
    Kokkos::fence();
    return kinetic_energy;
  }

  template <class RNG>
  void randomize_momentum(RNG& rng) {
    randomize_field<DAdjFieldType, RNG>(adjoint_field, rng);
  }
};

}  // namespace klft
