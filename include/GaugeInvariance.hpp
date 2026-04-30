
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

// this file defines the necessary functions to calculate
// different gauge observables to be measured during simulations

#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "IndexHelper.hpp"

namespace klft {

template <typename DGaugeFieldType>
struct GaugeInv {
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind Kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  static_assert(rank == 4);  // The wilson flow is only defined for 4D Fields

  using GaugeFieldT = typename DGaugeFieldType::type;
  GaugeFieldT source_field;
  GaugeFieldT field;    // the transformed field
  GaugeFieldT t_field;  // the randomzed field

  GaugeInv() = delete;

  template <class RNG>
  GaugeInv(const GaugeFieldT& _field, RNG& rng, const real_t delta = 1.0)
      : source_field(_field),
        field(_field.field),
        t_field(_field.dimensions, rng, delta) {
    Kokkos::fence();
  }

  void transform() {
    KTune::parallel_for(
        "gauge_inv_transform",
        Policy<rank>(IndexArray<rank>{0, 0, 0, 0}, field.dimensions), *this);
    Kokkos::fence();
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void operator()(const indexType i0,
                                         const indexType i1,
                                         const indexType i2,
                                         const indexType i3) const {
#pragma unroll
    for (index_t mu = 0; mu < 4; ++mu) {
      IndexArray<rank> idx{static_cast<int>(i0), static_cast<int>(i1),
                           static_cast<int>(i2), static_cast<int>(i3)};
      IndexArray<rank> idx_plus =
          shift_index_plus<rank>(idx, mu, 1, field.dimensions);
      field(i0, i1, i2, i3, mu) =
          t_field.field(i0, i1, i2, i3, 1) *
          source_field.field(i0, i1, i2, i3, mu) *
          conj(t_field.field(idx_plus[0], idx_plus[1], idx_plus[2], idx_plus[3],
                             1));
    }
  }
};

}  // namespace klft
