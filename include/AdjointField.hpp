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

#include "AdjointSUN.hpp"
#include "GLOBAL.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_Macros.hpp"
#include "View/Kokkos_ViewCtor.hpp"

namespace klft {

template <size_t Nd, size_t Nc>
struct deviceAdjointField {
  deviceAdjointField() = delete;

  SUNAdjField<Nd, Nc> field;
  IndexArray<Nd> dimensions;

  deviceAdjointField(const index_t L0,
                     const index_t L1,
                     const index_t L2,
                     const index_t L3,
                     const SUNAdj<Nc>& init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(field, init);
  }

  void do_init(SUNAdjField<Nd, Nc>& V, const SUNAdj<Nc>& init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    Kokkos::fence();
    KTune::parallel_for(
        "init_DeviceAdjointField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, i3, mu) = init;
          }
        });
    Kokkos::fence();
  }

  template <class RNG>
  void randomize_field(RNG& rng) {
    auto self = field;
    KTune::parallel_for(
        "randomize_adj_field", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          auto generator = rng.get_state();
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUNAdj<Nc>(self(i0, i1, i2, i3, mu), generator);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  // define accessors for the field
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(const indexType i,
                                                     const indexType j,
                                                     const indexType k,
                                                     const indexType l,
                                                     const index_t mu) const {
    return field(i, j, k, l, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(const indexType i,
                                                     const indexType j,
                                                     const indexType k,
                                                     const indexType l,
                                                     const index_t mu) {
    return field(i, j, k, l, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(
      const Kokkos::Array<indexType, 4> site,
      const index_t mu) const {
    return field(site[0], site[1], site[2], site[3], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(
      const Kokkos::Array<indexType, 4> site,
      const index_t mu) {
    return field(site[0], site[1], site[2], site[3], mu);
  }
};

template <size_t Nd, size_t Nc>
struct deviceAdjointField3D {
  deviceAdjointField3D() = delete;

  SUNAdjField3D<Nd, Nc> field;
  IndexArray<Nd> dimensions;

  deviceAdjointField3D(const index_t L0,
                       const index_t L1,
                       const index_t L2,
                       const SUNAdj<Nc>& init)
      : dimensions({L0, L1, L2}) {
    do_init(field, init);
  }
  void do_init(SUNAdjField3D<Nd, Nc>& V, const SUNAdj<Nc>& init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    KTune::parallel_for(
        "init_DeviceAdjointField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = init;
          }
        });
  }

  template <class RNG>
  void randomize_field(RNG& rng) {
    auto self = field;
    KTune::parallel_for(
        "randomize_adj_field", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUNAdj<Nc>(self(i0, i1, i2, mu), generator);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  // define accessors for the field
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(const indexType i,
                                                     const indexType j,
                                                     const indexType k,
                                                     const index_t mu) const {
    return field(i, j, k, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(const indexType i,
                                                     const indexType j,
                                                     const indexType k,
                                                     const index_t mu) {
    return field(i, j, k, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(
      const Kokkos::Array<indexType, 3> site,
      const index_t mu) const {
    return field(site[0], site[1], site[2], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(
      const Kokkos::Array<indexType, 3> site,
      const index_t mu) {
    return field(site[0], site[1], site[2], mu);
  }
};

template <size_t Nd, size_t Nc>
struct deviceAdjointField2D {
  deviceAdjointField2D() = delete;

  SUNAdjField2D<Nd, Nc> field;
  IndexArray<Nd> dimensions;

  deviceAdjointField2D(const index_t L0,
                       const index_t L1,
                       const SUNAdj<Nc>& init)
      : dimensions({L0, L1}) {
    do_init(field, init);
  }
  void do_init(SUNAdjField2D<Nd, Nc>& V, const SUNAdj<Nc>& init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    KTune::parallel_for(
        "init_DeviceAdjointField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = init;
          }
        });
  }

  template <class RNG>
  void randomize_field(RNG& rng) {
    auto self = field;
    KTune::parallel_for(
        "randomize_adj_field", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUNAdj<Nc>(self(i0, i1, mu), generator);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  // define accessors for the field
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(const indexType i,
                                                     const indexType j,
                                                     const index_t mu) const {
    return field(i, j, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(const indexType i,
                                                     const indexType j,
                                                     const index_t mu) {
    return field(i, j, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(
      const Kokkos::Array<indexType, 2> site,
      const index_t mu) const {
    return field(site[0], site[1], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc>& operator()(
      const Kokkos::Array<indexType, 2> site,
      const index_t mu) {
    return field(site[0], site[1], mu);
  }
};
}  // namespace klft
