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

// define structs for initializing SUN field

#pragma once
#include "GLOBAL.hpp"
#include "Tuner.hpp"
// Nc number of colors
// RepDim Dimension of Gamma matrices, Nd = RepDim
namespace klft {
template <size_t Nc, size_t RepDim>
struct deviceSpinorField {
  deviceSpinorField() = delete;

  // initialize all sites to a given value

  deviceSpinorField(const index_t L0, const index_t L1, const index_t L2,
                    const index_t L3, const complex_t init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }

  deviceSpinorField(const IndexArray<4> &dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            init);
  }

  // initialize all sites to given Spinor
  deviceSpinorField(const index_t L0, const index_t L1, const index_t L2,
                    const index_t L3, const Spinor<Nc, RepDim> &init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }
  deviceSpinorField(const IndexArray<4> &dimensions,
                    const Spinor<Nc, RepDim> &init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            init);
  }

  // initialize all latice size to random value drawn from a Normal Distribution
  // N(mean,var)
  template <class RNG>
  deviceSpinorField(const IndexArray<4> &dimensions, RNG &rng,
                    const real_t &mean, const real_t &var)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            rng, mean, var);
  }

  // initialize all latice size to random value drawn from a Normal Distribution
  // N(mean,var)
  template <class RNG>
  deviceSpinorField(const index_t L0, const index_t L1, const index_t L2,
                    const index_t L3, RNG &rng, const real_t &mean,
                    const real_t &var)
      : dimensions({L0, L1, L2, L3}) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            rng, mean, var);
  }

  // Initialize SpinorField with Single Value
  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, SpinorField<Nc, RepDim> &V,
               const complex_t init) {
    if (KLFT_VERBOSITY > 4) {
      printf("Initialize SpinorField with Single Value\n");
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    tune_and_launch_for<4>(
        "init_deviceSpinorField", IndexArray<4>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < RepDim; ++c2) {
              V(i0, i1, i2, i3)[c1][c2] = init;
            }
          }
        });
    Kokkos::fence();
  }
  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, SpinorField<Nc, RepDim> &V,
               const Spinor<Nc, RepDim> &init) {
    if (KLFT_VERBOSITY > 4) {
      printf("Initialize SpinorField with Spinors\n");
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    tune_and_launch_for<4>(
        "init_deviceSpinorField", IndexArray<4>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) { V(i0, i1, i2, i3) = init; });
    Kokkos::fence();
  }
  template <class RNG>
  void do_init(const index_t L0, const index_t L1, const index_t L2,
               const index_t L3, SpinorField<Nc, RepDim> &V, RNG &rng,
               const real_t &mean, const real_t &var) {
    if (KLFT_VERBOSITY > 4) {
      printf("Initialize SpinorField with random normal distributed values\n");
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    tune_and_launch_for<4>(
        "init_deviceSpinorField", IndexArray<4>{0, 0, 0, 0},
        IndexArray<4>{L0, L1, L2, L3},
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < RepDim; ++c2) {
              V(i0, i1, i2, i3)
              [c1][c2] = complex_t(generator.normal(mean, var),
                                   generator.normal(mean, var));
            }
          }
          // free state here corrct in l.138 of GaugeField it isn't done.
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  SpinorField<Nc, RepDim> field;
  const IndexArray<4> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim> &operator()(
      const indexType i0, const indexType i1, const indexType i2,
      const indexType i3) const {
    return field(i0, i1, i2, i3);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim> &operator()(
      const indexType i0, const indexType i1, const indexType i2,
      const indexType i3) {
    return field(i0, i1, i2, i3);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim> &operator()(
      const Kokkos::Array<indexType, 4> site) const {
    return field(site[0], site[1], site[2], site[3]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim> &operator()(
      const Kokkos::Array<indexType, 4> site) {
    return field(site[0], site[1], site[2], site[3]);
  }
};

}  // namespace klft