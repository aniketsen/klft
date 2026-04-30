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
#include "AdjointField.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
namespace klft {
template <typename DAdjFieldType, class RNG>
void randomize_field(typename DAdjFieldType::type& field, RNG& rng) {
  size_t constexpr Nd = DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank;
  size_t constexpr Nc = DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc;
  if constexpr (Nd == 4) {
    KTune::parallel_for(
        "randomize_adj_field", Policy<Nd>(IndexArray<Nd>{0}, field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          auto generator = rng.get_state();
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUNAdj<Nc>(field(i0, i1, i2, i3, mu), generator);
          }
          rng.free_state(generator);
        });
  }
  if constexpr (Nd == 3) {
    KTune::parallel_for(
        "randomize_adj_field", Policy<Nd>(IndexArray<Nd>{0}, field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUNAdj<Nc>(field(i0, i1, i2, mu), generator);
          }
          rng.free_state(generator);
        });
  }
  if constexpr (Nd == 2) {
    KTune::parallel_for(
        "randomize_adj_field", Policy<Nd>(IndexArray<Nd>{0}, field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUNAdj<Nc>(field(i0, i1, mu), generator);
          }
          rng.free_state(generator);
        });
  }

  Kokkos::fence();
}

template <typename DAdjFieldType>
void flip_sign(typename DAdjFieldType::type& field) {
  size_t constexpr Nd = DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank;
  size_t constexpr Nc = DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc;
  if constexpr (Nd == 4) {
    KTune::parallel_for(
        "randomize_adj_field", Policy<Nd>(IndexArray<Nd>{0}, field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          for (index_t mu = 0; mu < Nd; ++mu) {
            flip_sign<Nc>(field(i0, i1, i2, i3, mu));
          }
        });
  }
  if constexpr (Nd == 3) {
    KTune::parallel_for(
        "randomize_adj_field", Policy<Nd>(IndexArray<Nd>{0}, field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          for (index_t mu = 0; mu < Nd; ++mu) {
            flip_sign<Nc>(field(i0, i1, i2, mu));
          }
        });
  }
  if constexpr (Nd == 2) {
    KTune::parallel_for(
        "randomize_adj_field", Policy<Nd>(IndexArray<Nd>{0}, field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          for (index_t mu = 0; mu < Nd; ++mu) {
            flip_sign<Nc>(field(i0, i1, mu));
          }
        });
  }

  Kokkos::fence();
}
}  // namespace klft
