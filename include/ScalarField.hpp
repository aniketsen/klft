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

// define structs for initializing scalar field

#pragma once

#include "GLOBAL.hpp"

namespace klft {

struct deviceScalarField {
  deviceScalarField() = default;

  // initialize all sites to a given value
  deviceScalarField(const index_t L0,
                    const index_t L1,
                    const index_t L2,
                    const index_t L3,
                    const real_t init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }

  deviceScalarField(const IndexArray<4>& dimensions, const real_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            init);
  }

  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               const index_t L3,
               ScalarField& V,
               const real_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    KTune::parallel_for(
        "init_deviceScalarField",
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) { V(i0, i1, i2, i3) = init; });
    Kokkos::fence();
  }

  ScalarField field;
  IndexArray<4> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const indexType i2,
                                                 const indexType i3) const {
    return field(i0, i1, i2, i3);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const indexType i2,
                                                 const indexType i3) {
    return field(i0, i1, i2, i3);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 4> site) const {
    return field(site[0], site[1], site[2], site[3]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 4> site) {
    return field(site[0], site[1], site[2], site[3]);
  }

  real_t sum() const {
    real_t sum = 0.0;
    KTune::parallel_reduce(
        "sum_deviceScalarField", Policy<4>({0, 0, 0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1,
                            const index_t i2, const index_t i3,
                            real_t& lsum) { lsum += field(i0, i1, i2, i3); },
        Kokkos::Sum<real_t>(sum));
    return sum;
  }
};

struct deviceScalarField3D {
  deviceScalarField3D() = default;

  // initialize all sites to a given value
  deviceScalarField3D(const index_t L0,
                      const index_t L1,
                      const index_t L2,
                      const real_t init)
      : dimensions({L0, L1, L2}) {
    do_init(L0, L1, L2, field, init);
  }

  deviceScalarField3D(const IndexArray<3>& dimensions, const real_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], field, init);
  }

  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               ScalarField3D& V,
               const real_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    KTune::parallel_for(
        "init_deviceScalarField3D",
        Policy<3>(IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          V(i0, i1, i2) = init;
        });
    Kokkos::fence();
  }

  ScalarField3D field;
  IndexArray<3> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const indexType i2) const {
    return field(i0, i1, i2);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1,
                                                 const indexType i2) {
    return field(i0, i1, i2);
  }

  // define accessors with 3D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 3> site) const {
    return field(site[0], site[1], site[2]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 3> site) {
    return field(site[0], site[1], site[2]);
  }

  real_t sum() const {
    real_t sum = 0.0;
    KTune::parallel_reduce(
        "sum_deviceScalarField3D", Policy<3>({0, 0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1,
                            const index_t i2,
                            real_t& lsum) { lsum += field(i0, i1, i2); },
        Kokkos::Sum<real_t>(sum));
    return sum;
  }
};

struct deviceScalarField2D {
  deviceScalarField2D() = default;

  // initialize all sites to a given value
  deviceScalarField2D(const index_t L0, const index_t L1, const real_t init)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, init);
  }

  deviceScalarField2D(const IndexArray<2>& dimensions, const real_t init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], field, init);
  }

  void do_init(const index_t L0,
               const index_t L1,
               ScalarField2D& V,
               const real_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    KTune::parallel_for(
        "init_deviceScalarField2D",
        Policy<2>(IndexArray<2>{0, 0}, IndexArray<2>{L0, L1}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          V(i0, i1) = init;
        });
    Kokkos::fence();
  }

  ScalarField2D field;
  IndexArray<2> dimensions;

  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1) const {
    return field(i0, i1);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(const indexType i0,
                                                 const indexType i1) {
    return field(i0, i1);
  }

  // define accessors with 2D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 2> site) const {
    return field(site[0], site[1]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION real_t& operator()(
      const Kokkos::Array<indexType, 2> site) {
    return field(site[0], site[1]);
  }

  real_t sum() const {
    real_t sum = 0.0;
    KTune::parallel_reduce(
        "sum_deviceScalarField2D", Policy<2>({0, 0}, dimensions),
        KOKKOS_CLASS_LAMBDA(const index_t i0, const index_t i1, real_t& lsum) {
          lsum += field(i0, i1);
        },
        Kokkos::Sum<real_t>(sum));
    return sum;
  }
};

}  // namespace klft
