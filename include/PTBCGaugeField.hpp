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

// define structs for initializing ptbc gauge fields

#pragma once
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "SUN.hpp"
#include "Tuner.hpp"

namespace klft {
template <size_t Nd> struct defectParams {
  // A struct that is used to hold the defect information for a given
  // devicePTBCGaugeField
  index_t defect_length{1};
  real_t defect_value{1.0};
  IndexArray<Nd - 1> defect_position{
      0}; // origin of the defect in mu = 1,2,3 directions
};

template <size_t Nd, size_t Nc> struct devicePTBCGaugeField {

  devicePTBCGaugeField() = delete;

  GaugeField<Nd, Nc> field;
  const IndexArray<Nd> dimensions;
  LinkScalarField<Nd> defectField;
  using deviceDefectParams = defectParams<Nd>;
  deviceDefectParams dParams;

  // copy constructor from a given devicePTBCGaugeField
  devicePTBCGaugeField(const devicePTBCGaugeField<Nd, Nc> &dPTBCGaugeField)
      : dimensions(dPTBCGaugeField.dimensions) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    Kokkos::deep_copy(field, dPTBCGaugeField.field);
    do_init_defect(defectField, dPTBCGaugeField.dParams);
  }

  // 'copy' constructor from a given deviceGaugeField
  devicePTBCGaugeField(const deviceGaugeField<Nd, Nc> &dGaugeField)
      : dimensions(dGaugeField.dimensions) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    Kokkos::deep_copy(field, dGaugeField.field);
  }

  // 'copy' constructor from a given GaugeField
  devicePTBCGaugeField(const GaugeField<Nd, Nc> &dGaugeField)
      : dimensions(dGaugeField.extend(0), dGaugeField.extend(1),
                   dGaugeField.extend(2), dGaugeField.extend(3)) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    Kokkos::deep_copy(field, dGaugeField);
  }

  // should defect_length and cr be encompassed in a defect struct?
  devicePTBCGaugeField(const index_t L0, const index_t L1, const index_t L2,
                       const index_t L3, const complex_t init,
                       const deviceDefectParams dParam)
      : dimensions({L0, L1, L2, L3}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // initialize all links to a given SUN matrix
  devicePTBCGaugeField(const index_t L0, const index_t L1, const index_t L2,
                       const index_t L3, const SUN<Nc> &init,
                       const deviceDefectParams dParam)
      : dimensions({L0, L1, L2, L3}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized with a given delta
  template <class RNG>
  devicePTBCGaugeField(const index_t L0, const index_t L1, const index_t L2,
                       const index_t L3, RNG &rng, const real_t delta,
                       const deviceDefectParams dParam)
      : dimensions({L0, L1, L2, L3}) {
    do_init(field, rng, delta);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized
  template <class RNG>
  devicePTBCGaugeField(const index_t L0, const index_t L1, const index_t L2,
                       const index_t L3, RNG &rng,
                       const deviceDefectParams dParam)
      : dimensions({L0, L1, L2, L3}) {
    do_init(field, rng);
    do_init_defect(defectField, dParam);
  }

  void do_init(GaugeField<Nd, Nc> &V, complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, i2, i3, mu)[c1][c2] = init;
              }
            }
          }
        });
    Kokkos::fence();
  }

  void do_init(GaugeField<Nd, Nc> &V, const SUN<Nc> &init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, i3, mu) = init;
          }
        });
    Kokkos::fence();
  }

  // without any defect information, set defect to be non-existent.
  void do_init_defect(LinkScalarField<Nd> &V) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, i3, mu) = 1.0;
          }
        });
    Kokkos::fence();
  }

  // This fixes the defect location according to (2.3) in 2404.14151
  void do_init_defect(LinkScalarField<Nd> &V, const deviceDefectParams dP) {
    do_init_defect(V);
    dParams = dP;
    set_defect(dParams.defect_value);
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(GaugeField<Nd, Nc> &V, RNG &rng, const real_t delta) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUN(V(i0, i1, i2, i3, mu), generator, delta);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  template <class RNG> void do_init(GaugeField<Nd, Nc> &V, RNG &rng) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, i2, i3, mu)
                [c1][c2] = complex_t(generator.drand(-1.0, 1.0),
                                     generator.drand(-1.0, 1.0));
              }
            }
          }
        });
    Kokkos::fence();
  }

  // Sets the defect value
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void set_defect(real_t cr) {
    dParams.defect_value = cr;
    tune_and_launch_for<Nd - 1>(
        "set_defect", IndexArray<Nd - 1>{0},
        IndexArray<Nd - 1>{dParams.defect_length, dParams.defect_length,
                           dParams.defect_length},
        KOKKOS_LAMBDA(const indexType i1, const indexType i2,
                      const indexType i3) {
          const indexType i1_shift =
              (i1 + dParams.defect_position[0]) % dimensions[1];
          const indexType i2_shift =
              (i2 + dParams.defect_position[1]) % dimensions[2];
          const indexType i3_shift =
              (i3 + dParams.defect_position[2]) % dimensions[3];
          defectField(0, i1_shift, i2_shift, i3_shift, 0) = cr;
        });
    Kokkos::fence();
  }

  void shift_defect(IndexArray<Nd - 1> new_position) {
    // set the current defect regions defect to 1.0, update the position of the
    // defect and set the defect value.
    real_t tmp = dParams.defect_value;
    set_defect(real_t(1.0));
    dParams.defect_position = new_position;
    set_defect(tmp);
  }

  // define accessors for the field
  template <typename indexType> // why do we template indexType here, when it is
                                // defined in GLOBAL.hpp?
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i, const indexType j, const indexType k,
             const indexType l, const index_t mu) const {
    return field(i, j, k, l, mu) * defectField(i, j, k, l, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i, const indexType j, const indexType k,
             const indexType l, const index_t mu) {
    return field(i, j, k, l, mu) * defectField(i, j, k, l, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 4> site, const index_t mu) const {
    return field(site[0], site[1], site[2], site[3], mu) *
           defectField(site[0], site[1], site[2], site[3], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 4> site, const index_t mu) {
    return field(site[0], site[1], site[2], site[3], mu) *
           defectField(site[0], site[1], site[2], site[3], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
  staple(const indexType i0, const indexType i1, const indexType i2,
         const indexType i3, const index_t mu) const {
    // this only works if Nd == 4
    assert(Nd == 4);
    // temporary SUN matrix to store the staple
    SUN<Nc> temp = zeroSUN<Nc>();
    // get the x + mu indices
    const indexType i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
    const indexType i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
    const indexType i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;
    const indexType i3pmu = mu == 3 ? (i3 + 1) % dimensions[3] : i3;
// positive directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      // do nothing for mu = nu
      if (nu == mu)
        continue;
      // get the x + nu indices
      const indexType i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const indexType i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;
      const indexType i2pnu = nu == 2 ? (i2 + 1) % dimensions[2] : i2;
      const indexType i3pnu = nu == 3 ? (i3 + 1) % dimensions[3] : i3;
      // get the staple
      temp +=
          this->operator()<indexType>(i0pmu, i1pmu, i2pmu, i3pmu, nu) *
          conj(this->operator()<indexType>(i0pnu, i1pnu, i2pnu, i3pnu, mu)) *
          conj(this->operator()<indexType>(i0, i1, i2, i3, nu));
    } // loop over nu
// negative directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      // do nothing for mu = nu
      if (nu == mu)
        continue;
      // get the x + mu - nu indices
      const indexType i0pmu_mnu =
          nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
      const indexType i1pmu_mnu =
          nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;
      const indexType i2pmu_mnu =
          nu == 2 ? (i2pmu - 1 + dimensions[2]) % dimensions[2] : i2pmu;
      const indexType i3pmu_mnu =
          nu == 3 ? (i3pmu - 1 + dimensions[3]) % dimensions[3] : i3pmu;
      // get the x - nu indices
      const indexType i0mnu =
          nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
      const indexType i1mnu =
          nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;
      const indexType i2mnu =
          nu == 2 ? (i2 - 1 + dimensions[2]) % dimensions[2] : i2;
      const indexType i3mnu =
          nu == 3 ? (i3 - 1 + dimensions[3]) % dimensions[3] : i3;
      // get the staple
      temp +=
          conj(this->operator()<indexType>(i0pmu_mnu, i1pmu_mnu, i2pmu_mnu,
                                           i3pmu_mnu, nu)) *
          conj(this->operator()<indexType>(i0mnu, i1mnu, i2mnu, i3mnu, mu)) *
          this->operator()<indexType>(i0mnu, i1mnu, i2mnu, i3mnu, nu);
    } // loop over nu
    return temp;
  }
};

template <size_t Nd, size_t Nc> struct devicePTBCGaugeField3D {

  devicePTBCGaugeField3D() = delete;

  GaugeField<Nd, Nc> field;
  const IndexArray<Nd> dimensions;
  LinkScalarField<Nd> defectField;
  using deviceDefectParams = defectParams<Nd>;
  deviceDefectParams dParams;

  // copy constructor from a given devicePTBCGaugeField
  devicePTBCGaugeField3D(const devicePTBCGaugeField<Nd, Nc> &dPTBCGaugeField)
      : dimensions(dPTBCGaugeField.dimensions) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1], dimensions[2]);
    Kokkos::deep_copy(field, dPTBCGaugeField.field);
    do_init_defect(defectField, dPTBCGaugeField.dParams);
  }

  // 'copy' constructor from a given deviceGaugeField
  devicePTBCGaugeField3D(const deviceGaugeField<Nd, Nc> &dGaugeField)
      : dimensions(dGaugeField.dimensions) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1], dimensions[2]);
    Kokkos::deep_copy(field, dGaugeField.field);
  }

  // should defect_length and cr be encompassed in a defect struct?
  devicePTBCGaugeField3D(const index_t L0, const index_t L1, const index_t L2,
                         const complex_t init, const deviceDefectParams dParam)
      : dimensions({L0, L1, L2}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // 'copy' constructor from a given GaugeField
  devicePTBCGaugeField3D(const GaugeField3D<Nd, Nc> &dGaugeField)
      : dimensions(dGaugeField.extend(0), dGaugeField.extend(1),
                   dGaugeField.extend(2)) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1], dimensions[2]);
    Kokkos::deep_copy(field, dGaugeField);
  }

  // initialize all links to a given SUN matrix
  devicePTBCGaugeField3D(const index_t L0, const index_t L1, const index_t L2,
                         const SUN<Nc> &init, const deviceDefectParams dParam)
      : dimensions({L0, L1, L2}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized with a given delta
  template <class RNG>
  devicePTBCGaugeField3D(const index_t L0, const index_t L1, const index_t L2,
                         RNG &rng, const real_t delta,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1, L2}) {
    do_init(field, rng, delta);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized
  template <class RNG>
  devicePTBCGaugeField3D(const index_t L0, const index_t L1, const index_t L2,
                         RNG &rng, const deviceDefectParams dParam)
      : dimensions({L0, L1, L2}) {
    do_init(field, rng);
    do_init_defect(defectField, dParam);
  }

  void do_init(GaugeField<Nd, Nc> &V, complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, i2, mu)[c1][c2] = init;
              }
            }
          }
        });
    Kokkos::fence();
  }

  void do_init(GaugeField<Nd, Nc> &V, const SUN<Nc> &init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = init;
          }
        });
    Kokkos::fence();
  }

  // without any defect information, set defect to be non-existent.
  void do_init_defect(LinkScalarField<Nd> &V) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = 1.0;
          }
        });
    Kokkos::fence();
  }

  // This fixes the defect location according to (2.3) in 2404.14151
  void do_init_defect(LinkScalarField<Nd> &V, const deviceDefectParams dP) {
    do_init_defect(V);
    dParams = dP;
    set_defect(dParams.defect_value);
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(GaugeField<Nd, Nc> &V, RNG &rng, const real_t delta) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUN(V(i0, i1, i2, mu), generator, delta);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  template <class RNG> void do_init(GaugeField<Nd, Nc> &V, RNG &rng) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, i2, mu)
                [c1][c2] = complex_t(generator.drand(-1.0, 1.0),
                                     generator.drand(-1.0, 1.0));
              }
            }
          }
        });
    Kokkos::fence();
  }

  // Sets the defect value
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void set_defect(real_t cr) {
    dParams.defect_value = cr;
    tune_and_launch_for<Nd - 1>(
        "set_defect", IndexArray<Nd - 1>{0},
        IndexArray<Nd - 1>{dParams.defect_length, dParams.defect_length},
        KOKKOS_LAMBDA(const indexType i1, const indexType i2) {
          const indexType i1_shift =
              (i1 + dParams.defect_position[0]) % dimensions[1];
          const indexType i2_shift =
              (i2 + dParams.defect_position[1]) % dimensions[2];
          defectField(0, i1_shift, i2_shift, 0) = cr;
        });
    Kokkos::fence();
  }

  void shift_defect(IndexArray<Nd - 1> new_position) {
    // set the current defect regions defect to 1.0, update the position of the
    // defect and set the defect value.
    real_t tmp = dParams.defect_value;
    set_defect(real_t(1.0));
    dParams.defect_position = new_position;
    set_defect(tmp);
  }

  // define accessors for the field
  template <typename indexType> // why do we template indexType here, when it is
                                // defined in GLOBAL.hpp?
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i, const indexType j, const indexType k,
             const index_t mu) const {
    return field(i, j, k, mu) * defectField(i, j, k, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i, const indexType j, const indexType k,
             const index_t mu) {
    return field(i, j, k, mu) * defectField(i, j, k, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 3> site, const index_t mu) const {
    return field(site[0], site[1], site[2], mu) *
           defectField(site[0], site[1], site[2], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 3> site, const index_t mu) {
    return field(site[0], site[1], site[2], mu) *
           defectField(site[0], site[1], site[2], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
  staple(const indexType i0, const indexType i1, const indexType i2,
         const index_t mu) const {
    // this only works if Nd == 3
    assert(Nd == 3);
    // temporary SUN matrix to store the staple
    SUN<Nc> temp = zeroSUN<Nc>();
    // get the x + mu indices
    const indexType i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
    const indexType i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
    const indexType i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;

// positive directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      if (nu == mu)
        continue; // skip if mu == nu
      const indexType i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const indexType i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;
      const indexType i2pnu = nu == 2 ? (i2 + 1) % dimensions[2] : i2;

      temp += this->operator()<indexType>(i0pmu, i1pmu, i2pmu, nu) *
              conj(this->operator()<indexType>(i0pnu, i1pnu, i2pnu, mu)) *
              conj(this->operator()<indexType>(i0, i1, i2, nu));
    }

// negative directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      if (nu == mu)
        continue; // skip if mu == nu
      const indexType i0pmu_mnu =
          nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
      const indexType i1pmu_mnu =
          nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;
      const indexType i2pmu_mnu =
          nu == 2 ? (i2pmu - 1 + dimensions[2]) % dimensions[2] : i2pmu;

      const indexType i0mnu =
          nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
      const indexType i1mnu =
          nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;
      const indexType i2mnu =
          nu == 2 ? (i2 - 1 + dimensions[2]) % dimensions[2] : i2;

      temp += conj(this->operator()<indexType>(i0pmu_mnu, i1pmu_mnu, i2pmu_mnu,
                                               nu)) *
              conj(this->operator()<indexType>(i0mnu, i1mnu, i2mnu, mu)) *
              this->operator()<indexType>(i0mnu, i1mnu, i2mnu, nu);
    }

    return temp;
  }
};

template <size_t Nd, size_t Nc> struct devicePTBCGaugeField2D {

  devicePTBCGaugeField2D() = delete;

  GaugeField<Nd, Nc> field;
  const IndexArray<Nd> dimensions;
  LinkScalarField<Nd> defectField;
  using deviceDefectParams = defectParams<Nd>;
  deviceDefectParams dParams;

  // copy constructor from a given devicePTBCGaugeField
  devicePTBCGaugeField2D(const devicePTBCGaugeField<Nd, Nc> &dPTBCGaugeField)
      : dimensions(dPTBCGaugeField.dimensions) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1]);
    Kokkos::deep_copy(field, dPTBCGaugeField.field);
    do_init_defect(defectField, dPTBCGaugeField.dParams);
  }

  // 'copy' constructor from a given deviceGaugeField
  devicePTBCGaugeField2D(const deviceGaugeField<Nd, Nc> &dGaugeField)
      : dimensions(dGaugeField.dimensions) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1]);
    Kokkos::deep_copy(field, dGaugeField.field);
  }

  // 'copy' constructor from a given GaugeField
  devicePTBCGaugeField2D(const GaugeField2D<Nd, Nc> &dGaugeField)
      : dimensions(
            IndexArray<Nd>{dGaugeField.extend(0), dGaugeField.extend(1)}) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1]);
    Kokkos::deep_copy(field, dGaugeField);
  }

  // should defect_length and cr be encompassed in a defect struct?
  devicePTBCGaugeField2D(const index_t L0, const index_t L1,
                         const complex_t init, const deviceDefectParams dParam)
      : dimensions({L0, L1}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // initialize all links to a given SUN matrix
  devicePTBCGaugeField2D(const index_t L0, const index_t L1,
                         const SUN<Nc> &init, const deviceDefectParams dParam)
      : dimensions({L0, L1}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized with a given delta
  template <class RNG>
  devicePTBCGaugeField2D(const index_t L0, const index_t L1, RNG &rng,
                         const real_t delta, const deviceDefectParams dParam)
      : dimensions({L0, L1}) {
    do_init(field, rng, delta);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized
  template <class RNG>
  devicePTBCGaugeField2D(const index_t L0, const index_t L1, RNG &rng,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1}) {
    do_init(field, rng);
    do_init_defect(defectField, dParam);
  }

  void do_init(GaugeField<Nd, Nc> &V, complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, mu)[c1][c2] = init;
              }
            }
          }
        });
    Kokkos::fence();
  }

  void do_init(GaugeField<Nd, Nc> &V, const SUN<Nc> &init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = init;
          }
        });
    Kokkos::fence();
  }

  // without any defect information, set defect to be non-existent.
  void do_init_defect(LinkScalarField<Nd> &V) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = 1.0;
          }
        });
    Kokkos::fence();
  }

  // This fixes the defect location according to (2.3) in 2404.14151
  void do_init_defect(LinkScalarField<Nd> &V, const deviceDefectParams dP) {
    do_init_defect(V);
    dParams = dP;
    set_defect(dParams.defect_value);
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(GaugeField<Nd, Nc> &V, RNG &rng, const real_t delta) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            randSUN(V(i0, i1, mu), generator, delta);
          }
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  template <class RNG> void do_init(GaugeField<Nd, Nc> &V, RNG &rng) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    tune_and_launch_for<Nd>(
        "init_PTBCdeviceGaugeField", IndexArray<Nd>{0}, dimensions,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          auto generator = rng.get_state();
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
            for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
              for (index_t c2 = 0; c2 < Nc; ++c2) {
                V(i0, i1, mu)
                [c1][c2] = complex_t(generator.drand(-1.0, 1.0),
                                     generator.drand(-1.0, 1.0));
              }
            }
          }
        });
    Kokkos::fence();
  }

  // Sets the defect value
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void set_defect(real_t cr) {
    dParams.defect_value = cr;
    tune_and_launch_for<Nd - 1>(
        "set_defect", IndexArray<Nd - 1>{0},
        IndexArray<Nd - 1>{dParams.defect_length},
        KOKKOS_LAMBDA(const indexType i1) {
          const indexType i1_shift =
              (i1 + dParams.defect_position[0]) % dimensions[1];
          defectField(0, i1_shift, 0) = cr;
        });
    Kokkos::fence();
  }

  void shift_defect(IndexArray<Nd - 1> new_position) {
    // set the current defect regions defect to 1.0, update the position of the
    // defect and set the defect value.
    real_t tmp = dParams.defect_value;
    set_defect(real_t(1.0));
    dParams.defect_position = new_position;
    set_defect(tmp);
  }

  // define accessors for the field
  template <typename indexType> // why do we template indexType here, when it is
                                // defined in GLOBAL.hpp?
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i, const indexType j, const index_t mu) const {
    return field(i, j, mu) * defectField(i, j, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const indexType i, const indexType j, const index_t mu) {
    return field(i, j, mu) * defectField(i, j, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 2> site, const index_t mu) const {
    return field(site[0], site[1], mu) * defectField(site[0], site[1], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> &
  operator()(const Kokkos::Array<indexType, 2> site, const index_t mu) {
    return field(site[0], site[1], mu) * defectField(site[0], site[1], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc>
  staple(const indexType i0, const indexType i1, const index_t mu) const {
    // this only works if Nd == 2
    assert(Nd == 2);
    // temporary SUN matrix to store the staple
    SUN<Nc> temp = zeroSUN<Nc>();
    // get the x + mu indices
    const indexType i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
    const indexType i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;

// positive directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      if (nu == mu)
        continue; // skip if mu == nu
      const indexType i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const indexType i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;

      temp += this->operator()<indexType>(i0pmu, i1pmu, nu) *
              conj(this->operator()<indexType>(i0pnu, i1pnu, mu)) *
              conj(this->operator()<indexType>(i0, i1, nu));
    }

// negative directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) { // loop over nu
      if (nu == mu)
        continue; // skip if mu == nu
      const indexType i0pmu_mnu =
          nu == 0 ? (i0pmu - 1 + dimensions[0]) % dimensions[0] : i0pmu;
      const indexType i1pmu_mnu =
          nu == 1 ? (i1pmu - 1 + dimensions[1]) % dimensions[1] : i1pmu;

      const indexType i0mnu =
          nu == 0 ? (i0 - 1 + dimensions[0]) % dimensions[0] : i0;
      const indexType i1mnu =
          nu == 1 ? (i1 - 1 + dimensions[1]) % dimensions[1] : i1;

      temp += conj(this->operator()<indexType>(i0pmu_mnu, i1pmu_mnu, nu)) *
              conj(this->operator()<indexType>(i0mnu, i1mnu, mu)) *
              this->operator()<indexType>(i0mnu, i1mnu, nu);
    }

    return temp;
  }
};

} // namespace klft
