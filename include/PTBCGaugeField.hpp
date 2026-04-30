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

#include "DefectParams.hpp"
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "IOHelperFunctions.hpp"
#include "SUN.hpp"

namespace klft {

template <size_t Nd, size_t Nc, typename precision = complex_t>
struct devicePTBCGaugeField {
  devicePTBCGaugeField() = default;

  GaugeField<Nd, Nc, precision> field;
  IndexArray<Nd> dimensions;
  LinkScalarField<Nd> defectField;
  using deviceDefectParams = defectParams<Nd>;
  deviceDefectParams dParams;

  // // copy constructor from a given devicePTBCGaugeField
  // devicePTBCGaugeField(const devicePTBCGaugeField<Nd, Nc,precision>
  // &dPTBCGaugeField)
  //     : dimensions(dPTBCGaugeField.dimensions) {
  //   Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
  //                   dimensions[1], dimensions[2], dimensions[3]);
  //   Kokkos::deep_copy(field, dPTBCGaugeField.field);
  //   do_init_defect(defectField, dPTBCGaugeField.dParams);
  // }

  // 'copy' constructor from a given deviceGaugeField
  // devicePTBCGaugeField(const deviceGaugeField<Nd, Nc,precision> &dGaugeField)
  //     : dimensions(dGaugeField.dimensions) {
  //   Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
  //                   dimensions[1], dimensions[2], dimensions[3]);
  //   Kokkos::deep_copy(field, dGaugeField.field);
  // }

  // 'copy' constructor from a given GaugeField
  devicePTBCGaugeField(const GaugeField<Nd, Nc, precision> dGaugeField)
      : dimensions({static_cast<index_t>(dGaugeField.extent(0)),
                    static_cast<index_t>(dGaugeField.extent(1)),
                    static_cast<index_t>(dGaugeField.extent(2)),
                    static_cast<index_t>(dGaugeField.extent(3))}) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    Kokkos::fence();
    Kokkos::deep_copy(field, dGaugeField);
    do_init_defect(defectField);
    Kokkos::fence();
  }

  operator deviceGaugeField<Nd, Nc, precision>() const {
    return deviceGaugeField<Nd, Nc, precision>(this->field, this->dimensions);
  }
  operator deviceGaugeField<Nd, Nc, precision>() {
    return deviceGaugeField<Nd, Nc, precision>(this->field, this->dimensions);
  }
  // should defect_length and cr be encompassed in a defect struct?
  devicePTBCGaugeField(const index_t L0,
                       const index_t L1,
                       const index_t L2,
                       const index_t L3,
                       const complex_t init,
                       const deviceDefectParams dParam)
      : dimensions({L0, L1, L2, L3}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // initialize all links to a given SUN matrix
  devicePTBCGaugeField(const index_t L0,
                       const index_t L1,
                       const index_t L2,
                       const index_t L3,
                       const SUN<Nc>& init,
                       const deviceDefectParams dParam)
      : dimensions({L0, L1, L2, L3}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized with a given delta
  template <class RNG>
  devicePTBCGaugeField(const index_t L0,
                       const index_t L1,
                       const index_t L2,
                       const index_t L3,
                       RNG& rng,
                       const real_t delta,
                       const deviceDefectParams dParam)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, rng, delta);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized
  template <class RNG>
  devicePTBCGaugeField(const index_t L0,
                       const index_t L1,
                       const index_t L2,
                       const index_t L3,
                       RNG& rng,
                       const deviceDefectParams dParam)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, rng);
    do_init_defect(defectField, dParam);
  }

  devicePTBCGaugeField(const IndexArray<4>& dimensions, const SUN<Nc>& init)
      : dimensions(dimensions) {
    do_init(field, init);
    do_init_defect(defectField);
  }

  devicePTBCGaugeField(const IndexArray<4>& dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(field, init);
    do_init_defect(defectField);
  }

  void do_init(GaugeField<Nd, Nc, precision>& V, complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    KTune::parallel_for(
        "init_PTBCdeviceGaugeField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
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

  void do_init(GaugeField<Nd, Nc, precision>& V,
               const GaugeField<Nd, Nc, precision>& f_in) {
    if (!V.is_allocated()) {
      V = GaugeField<Nd, Nc, precision>("gauge_field_tmp", 0, 0, 0, 0);
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    Kokkos::fence();
    Kokkos::deep_copy(V, f_in);
    Kokkos::fence();
  }

  void do_init(GaugeField<Nd, Nc, precision>& V, const SUN<Nc>& init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    KTune::parallel_for(
        "init_PTBCdeviceGaugeField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
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
  void do_init_defect(LinkScalarField<Nd>& V) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
    KTune::parallel_for(
        "init_PTBCdeviceGaugeField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
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
  void do_init_defect(LinkScalarField<Nd>& V, const deviceDefectParams dP) {
    do_init_defect(V);
    this->dParams = dP;
    set_defect<index_t>(this->dParams.defect_value);
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               const index_t L3,
               GaugeField<Nd, Nc, precision>& V,
               RNG& rng,
               const real_t delta) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    KTune::parallel_for(
        "init_deviceGaugeField",
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3}),
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

  template <class RNG>
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               const index_t L3,
               GaugeField<Nd, Nc, precision>& V,
               RNG& rng) {
    if (!V.is_allocated()) {
      V = GaugeField<Nd, Nc, precision>("gauge_field_tmp", 0, 0, 0, 0);
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    KTune::parallel_for(
        "init_deviceGaugeField",
        Policy<4>(IndexArray<4>{0, 0, 0, 0}, IndexArray<4>{L0, L1, L2, L3}),
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
  void set_defect(real_t cr) {
    this->dParams.defect_value = cr;
    auto dimensions_local = this->dimensions;
    auto defect_position_local = this->dParams.defect_position;
    // DEBUG_MPI_PRINT("Setting defect at position (%d, %d, %d) with value %f",
    //                 defect_position_local[0], defect_position_local[1],
    //                 defect_position_local[2], cr);
    // DEBUG_MPI_PRINT("Defect length: %d", this->dParams.defect_length);
    auto defectField_local = this->defectField;

    KTune::parallel_for(
        "set_defect",
        Policy<Nd - 1>(IndexArray<Nd - 1>{0},
                       IndexArray<Nd - 1>{this->dParams.defect_length,
                                          this->dParams.defect_length,
                                          this->dParams.defect_length}),
        KOKKOS_LAMBDA(const indexType i1, const indexType i2,
                      const indexType i3) {
          const indexType i1_shift =
              (i1 + defect_position_local[0]) % dimensions_local[1];
          const indexType i2_shift =
              (i2 + defect_position_local[1]) % dimensions_local[2];
          const indexType i3_shift =
              (i3 + defect_position_local[2]) % dimensions_local[3];
          defectField_local(0, i1_shift, i2_shift, i3_shift, 0) = cr;
        });
    Kokkos::fence();
  }

  // void check_defect_application() {
  //   int rank, size;
  //   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //   MPI_Comm_size(MPI_COMM_WORLD, &size);
  //   // Raw field entry at site (0,0,0,0,mu=0)
  //   SUN<Nc> field_value = field(0, 0, 0, 0, 0);
  //
  //   // Defect-applied entry via operator()
  //   SUN<Nc> defect_SUN = this->operator()(0, 0, 0, 0, 0);
  //
  //   print_SUN<Nc>(field_value, "[Rank " + std::to_string(rank) +
  //                                  "] Field value at (0,0,0,0,mu=0): ");
  //   print_SUN<Nc>(defect_SUN, "[Rank " + std::to_string(rank) +
  //                                 "] Defect-applied value at (0,0,0,0,mu=0):
  //                                 ");
  // }

  void shift_defect(IndexArray<Nd - 1> new_position) {
    // set the current defect regions defect to 1.0, update the position of the
    // defect and set the defect value.
    real_t tmp = this->dParams.defect_value;
    set_defect<index_t>(real_t(1.0));
    this->dParams.defect_position = new_position;
    set_defect<index_t>(tmp);
  }

  real_t get_defect() const {
    // return the defect value
    return this->dParams.defect_value;
  }

  // TODO: return as deviceGaugeField

  template <class FieldView, class DefectView>
  struct PTBCLinkRef {
    FieldView field;
    DefectView defect;
    index_t i, j, k, l, mu;

    // read: multiply by defect
    KOKKOS_INLINE_FUNCTION
    operator SUN<Nc>() const {
      return field(i, j, k, l, mu) * defect(i, j, k, l, mu);
    }

    // write: raw write (no defect factor)
    KOKKOS_INLINE_FUNCTION
    PTBCLinkRef& operator=(const SUN<Nc>& v) {
      field(i, j, k, l, mu) = v;
      return *this;
    }

    // optional: compound op
    KOKKOS_INLINE_FUNCTION
    PTBCLinkRef& operator*=(const SUN<Nc>& rhs) {
      field(i, j, k, l, mu) = field(i, j, k, l, mu) * rhs;
      return *this;
    }

    // assign from another ref
    KOKKOS_INLINE_FUNCTION
    PTBCLinkRef& operator=(const PTBCLinkRef& rhs) {
      return (*this = static_cast<SUN<Nc>>(rhs));
    }
  };
  // READ (const): return value = field * defect
  template <typename I>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(I i, I j, I k, I l, index_t mu)
      const {
    return field(i, j, k, l, mu) * defectField(i, j, k, l, mu);
  }

  // READ/WRITE (non-const): return proxy
  template <typename I>
  KOKKOS_FORCEINLINE_FUNCTION auto operator()(I i, I j, I k, I l, index_t mu) {
    return PTBCLinkRef<decltype(field), decltype(defectField)>{
        field, defectField, (index_t)i, (index_t)j, (index_t)k, (index_t)l, mu};
  }

  // Array overloads
  template <typename I>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(const Kokkos::Array<I, 4>& s,
                                                 index_t mu) const {
    return (*this)(s[0], s[1], s[2], s[3], mu);
  }
  template <typename I>
  KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::Array<I, 4>& s,
                                              index_t mu) {
    return (*this)(s[0], s[1], s[2], s[3], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void set(const indexType i,
                                       const indexType j,
                                       const indexType k,
                                       const indexType l,
                                       const index_t mu,
                                       const SUN<Nc>& value) const {
    field(i, j, k, l, mu) = (value);  // raw write
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> staple(
      const Kokkos::Array<indexType, 4> site,
      const index_t mu) const {
    // this only works if Nd == 4
    assert(Nd == 4);
    // get the indices
    const index_t i0 = site[0];
    const index_t i1 = site[1];
    const index_t i2 = site[2];
    const index_t i3 = site[3];
    // temporary SUN matrix to store the staple
    SUN<Nc> temp = zeroSUN<Nc>();
    // get the x + mu indices
    const indexType i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
    const indexType i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
    const indexType i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;
    const indexType i3pmu = mu == 3 ? (i3 + 1) % dimensions[3] : i3;
// positive directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) {  // loop over nu
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
    }  // loop over nu
// negative directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) {  // loop over nu
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
    }  // loop over nu
    return temp;
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> staple_rect(const IndexArray<Nd> site,
                                                  const index_t mu) const {
    // this only works if Nd == 4
    assert(Nd == 4);
    IndexArray<Nd> temp_site1 = site;
    IndexArray<Nd> temp_site2 = site;
    IndexArray<Nd> temp_site3 = site;
    IndexArray<Nd> temp_site4 = site;

    SUN<Nc> temp = zeroSUN<Nc>();

    // Helper lambda for modular arithmetic
    auto mod = [&](index_t s, index_t dir) {
      return (s + this->dimensions[dir]) % this->dimensions[dir];
    };

#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) {  // loop over nu
      // do nothing for mu = nu
      if (nu == mu)
        continue;
      // first do the 1x2 rectangles
      temp_site1[mu] = (temp_site1[mu] + 1) % this->dimensions[mu];
      temp_site2[nu] = (temp_site2[nu] + 2) % this->dimensions[nu];
      temp_site3[mu] = (temp_site3[mu] + 1) % this->dimensions[mu];
      temp_site3[nu] = (temp_site3[nu] + 1) % this->dimensions[nu];
      temp_site4[nu] = (temp_site4[nu] + 1) % this->dimensions[nu];
      temp += this->operator()<indexType>(temp_site1, nu) *
              this->operator()<indexType>(temp_site2, nu) *
              conj(this->operator()<indexType>(temp_site3, mu)) *
              conj(this->operator()<indexType>(temp_site4, nu)) *
              conj(this->operator()<indexType>(site, nu));
      temp_site1 = site;
      temp_site2 = site;
      temp_site3 = site;
      temp_site4 = site;

      temp_site1[nu] = (temp_site1[nu] + 1) % this->dimensions[nu];
      temp_site2[mu] = (temp_site2[mu] + 2) % this->dimensions[mu];
      temp_site2[nu] = mod(temp_site2[nu] - 1, nu);
      temp_site3[mu] = (temp_site3[mu] + 1) % this->dimensions[mu];
      temp_site3[nu] = mod(temp_site3[nu] - 1, nu);
      temp_site4[nu] = mod(temp_site4[nu] - 1, nu);
      temp += this->operator()<indexType>(temp_site1, mu) *
              conj(this->operator()<indexType>(temp_site2, nu)) *
              conj(this->operator()<indexType>(temp_site3, mu)) *
              conj(this->operator()<indexType>(temp_site4, mu)) *
              this->operator()<indexType>(temp_site4, nu);
      temp_site1 = site;
      temp_site2 = site;
      temp_site3 = site;
      temp_site4 = site;

      temp_site1[mu] = (temp_site1[mu] + 1) % this->dimensions[mu];
      temp_site1[nu] = mod(temp_site1[nu] - 1, nu);
      temp_site2[nu] = mod(temp_site2[nu] - 1, nu);
      temp_site3[mu] = mod(temp_site3[mu] - 1, mu);
      temp_site3[nu] = mod(temp_site3[nu] - 1, nu);
      temp_site4[mu] = mod(temp_site4[mu] - 1, mu);
      temp += conj(this->operator()<indexType>(temp_site1, nu)) *
              conj(this->operator()<indexType>(temp_site2, mu)) *
              conj(this->operator()<indexType>(temp_site3, mu)) *
              this->operator()<indexType>(temp_site3, nu) *
              this->operator()<indexType>(temp_site4, mu);
      temp_site1 = site;
      temp_site2 = site;
      temp_site3 = site;
      temp_site4 = site;

      // now do the 2x1 rectangles
      temp_site1[mu] = (temp_site1[mu] + 1) % this->dimensions[mu];
      temp_site2[mu] = (temp_site2[mu] + 2) % this->dimensions[mu];
      temp_site3[mu] = (temp_site3[mu] + 1) % this->dimensions[mu];
      temp_site3[nu] = (temp_site3[nu] + 1) % this->dimensions[nu];
      temp_site4[nu] = (temp_site4[nu] + 1) % this->dimensions[nu];
      temp += this->operator()<indexType>(temp_site1, mu) *
              this->operator()<indexType>(temp_site2, nu) *
              conj(this->operator()<indexType>(temp_site3, mu)) *
              conj(this->operator()<indexType>(temp_site4, mu)) *
              conj(this->operator()<indexType>(site, nu));
      temp_site1 = site;
      temp_site2 = site;
      temp_site3 = site;
      temp_site4 = site;

      temp_site1[mu] = (temp_site1[mu] + 1) % this->dimensions[mu];
      temp_site2[nu] = (temp_site2[nu] + 1) % this->dimensions[nu];
      temp_site3[nu] = (temp_site3[nu] + 1) % this->dimensions[nu];
      temp_site3[mu] = mod(temp_site3[mu] - 1, mu);
      temp_site4[mu] = mod(temp_site4[mu] - 1, mu);
      temp += this->operator()<indexType>(temp_site1, nu) *
              conj(this->operator()<indexType>(temp_site2, mu)) *
              conj(this->operator()<indexType>(temp_site3, mu)) *
              conj(this->operator()<indexType>(temp_site4, nu)) *
              this->operator()<indexType>(temp_site4, mu);
      temp_site1 = site;
      temp_site2 = site;
      temp_site3 = site;
      temp_site4 = site;

      temp_site1[mu] = (temp_site1[mu] + 1) % this->dimensions[mu];
      temp_site1[nu] = mod(temp_site1[nu] - 1, nu);
      temp_site2[nu] = mod(temp_site2[nu] - 2, nu);
      temp_site2[mu] = (temp_site2[mu] + 1) % this->dimensions[mu];
      temp_site3[nu] = mod(temp_site3[nu] - 2, nu);
      temp_site4[nu] = mod(temp_site4[nu] - 1, nu);
      temp += conj(this->operator()<indexType>(temp_site1, nu)) *
              conj(this->operator()<indexType>(temp_site2, nu)) *
              conj(this->operator()<indexType>(temp_site3, mu)) *
              this->operator()<indexType>(temp_site3, nu) *
              this->operator()<indexType>(temp_site4, nu);
      temp_site1 = site;
      temp_site2 = site;
      temp_site3 = site;
      temp_site4 = site;
    }
    return temp / 2;
  }
  void save(std::string filename) const {
    auto h_field = Kokkos::create_mirror_view(field);
    Kokkos::deep_copy(h_field, field);
    std::ofstream file;
    try {
      file.open(filename, std::ios::out | std::ios::binary);
      file.write(reinterpret_cast<const char*>(h_field.data()),
                 h_field.size() * sizeof(SUN<Nc>));
      file.close();
    } catch (const std::exception& e) {
      printf("Error saving PTBC field to file %s: %s\n", filename.c_str(),
             e.what());
    }
  }
  void load(std::string filename) {
    try {
      set_defect<index_t>(1.0);  // reset defect to 1.0 before loading
      this->dParams = parse<Nd>(filename);
      set_defect<index_t>(this->dParams.defect_value);
      auto h_field = Kokkos::create_mirror_view(field);
      std::ifstream file;
      file.open(filename, std::ios::in | std::ios::binary);
      file.read(reinterpret_cast<char*>(h_field.data()),
                h_field.size() * sizeof(SUN<Nc>));
      file.close();
      Kokkos::deep_copy(field, h_field);
    } catch (const std::exception& e) {
      printf("Error loading PTBC field from file %s: %s\n", filename.c_str(),
             e.what());
    }
  }
};

template <size_t Nd, size_t Nc, typename precision = complex_t>
struct devicePTBCGaugeField3D {
  devicePTBCGaugeField3D() = default;

  GaugeField3D<Nd, Nc, precision> field;
  IndexArray<Nd> dimensions;
  LinkScalarField3D<Nd> defectField;
  using deviceDefectParams = defectParams<Nd>;
  deviceDefectParams dParams;

  // // copy constructor from a given devicePTBCGaugeField
  // devicePTBCGaugeField3D(const devicePTBCGaugeField<Nd, Nc,precision>
  // &dPTBCGaugeField)
  //     : dimensions(dPTBCGaugeField.dimensions) {
  //   Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
  //                   dimensions[1], dimensions[2]);
  //   Kokkos::deep_copy(field, dPTBCGaugeField.field);
  //   do_init_defect(defectField, dPTBCGaugeField.dParams);
  // }
  //
  // // 'copy' constructor from a given deviceGaugeField
  // devicePTBCGaugeField3D(const deviceGaugeField<Nd, Nc,precision>
  // &dGaugeField)
  //     : dimensions(dGaugeField.dimensions) {
  //   Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
  //                   dimensions[1], dimensions[2]);
  //   Kokkos::deep_copy(field, dGaugeField.field);
  // }

  // should defect_length and cr be encompassed in a defect struct?
  devicePTBCGaugeField3D(const index_t L0,
                         const index_t L1,
                         const index_t L2,
                         const complex_t init,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1, L2}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // 'copy' constructor from a given GaugeField
  devicePTBCGaugeField3D(const GaugeField3D<Nd, Nc, precision>& dGaugeField)
      : dimensions({static_cast<index_t>(dGaugeField.extent(0)),
                    static_cast<index_t>(dGaugeField.extent(1)),
                    static_cast<index_t>(dGaugeField.extent(2))}) {
    Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
                    dimensions[1], dimensions[2]);
    Kokkos::fence();
    Kokkos::deep_copy(field, dGaugeField);
    Kokkos::fence();
  }
  devicePTBCGaugeField3D(const IndexArray<3>& dimensions, const SUN<Nc>& init)
      : dimensions(dimensions) {
    do_init(field, init);
    do_init_defect(defectField);
  }

  operator deviceGaugeField3D<Nd, Nc, precision>() const {
    return deviceGaugeField3D<Nd, Nc, precision>(this->field, this->dimensions);
  }

  devicePTBCGaugeField3D(const IndexArray<3>& dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(field, init);
    do_init_defect(defectField);
  }

  // initialize all links to a given SUN matrix
  devicePTBCGaugeField3D(const index_t L0,
                         const index_t L1,
                         const index_t L2,
                         const SUN<Nc>& init,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1, L2}) {
    printf("[DEBUG] Initializing PTBCGaugeField with SUN init Nd=3\n");
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized with a given delta
  template <class RNG>
  devicePTBCGaugeField3D(const index_t L0,
                         const index_t L1,
                         const index_t L2,
                         RNG& rng,
                         const real_t delta,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1, L2}) {
    printf("[DEBUG] Initializing PTBCGaugeField with RNG Nd=3\n");
    do_init(L0, L1, L2, field, rng, delta);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized
  template <class RNG>
  devicePTBCGaugeField3D(const index_t L0,
                         const index_t L1,
                         const index_t L2,
                         RNG& rng,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1, L2}) {
    do_init(L0, L1, L2, field, rng);
    do_init_defect(defectField, dParam);
  }

  void do_init(GaugeField3D<Nd, Nc, precision>& V, complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    KTune::parallel_for(
        "init_PTBCdeviceGaugeField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
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

  void do_init(GaugeField3D<Nd, Nc, precision>& V, const SUN<Nc>& init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    KTune::parallel_for(
        "init_PTBCdeviceGaugeField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = init;
          }
        });
    Kokkos::fence();
  }

  void do_init(GaugeField3D<Nd, Nc, precision>& V,
               const GaugeField3D<Nd, Nc, precision>& f_in) {
    if (!V.is_allocated()) {
      V = GaugeField3D<Nd, Nc, precision>("gauge_field_tmp", 0, 0, 0);
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    Kokkos::deep_copy(V, f_in);
  }

  // without any defect information, set defect to be non-existent.
  void do_init_defect(LinkScalarField3D<Nd>& V) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1], dimensions[2]);
    KTune::parallel_for(
        "init_PTBCdeviceGaugeField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, i2, mu) = 1.0;
          }
        });
    Kokkos::fence();
  }

  // This fixes the defect location according to (2.3) in 2404.14151
  void do_init_defect(LinkScalarField3D<Nd>& V, const deviceDefectParams dP) {
    do_init_defect(V);
    this->dParams = dP;
    set_defect<index_t>(this->dParams.defect_value);
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               GaugeField3D<Nd, Nc, precision>& V,
               RNG& rng,
               const real_t delta) {
    if (!V.is_allocated()) {
      V = GaugeField3D<Nd, Nc, precision>("gauge_field_tmp", 0, 0, 0);
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    KTune::parallel_for(
        "init_deviceGaugeField3D",
        Policy<3>(IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2}),
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

  template <class RNG>
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               GaugeField3D<Nd, Nc, precision>& V,
               RNG& rng) {
    if (!V.is_allocated()) {
      V = GaugeField3D<Nd, Nc, precision>("gauge_field_tmp", 0, 0, 0);
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    KTune::parallel_for(
        "init_deviceGaugeField3D",
        Policy<3>(IndexArray<3>{0, 0, 0}, IndexArray<3>{L0, L1, L2}),
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
  void set_defect(real_t cr) {
    this->dParams.defect_value = cr;
    auto dimensions_local = this->dimensions;
    auto defect_position_local = this->dParams.defect_position;
    auto defectField_local = this->defectField;
    KTune::parallel_for(
        "set_defect",
        Policy<Nd - 1>(IndexArray<Nd - 1>{0},
                       IndexArray<Nd - 1>{this->dParams.defect_length,
                                          this->dParams.defect_length}),
        KOKKOS_LAMBDA(const indexType i1, const indexType i2) {
          const indexType i1_shift =
              (i1 + defect_position_local[0]) % dimensions_local[1];
          const indexType i2_shift =
              (i2 + defect_position_local[1]) % dimensions_local[2];
          defectField_local(0, i1_shift, i2_shift, 0) = cr;
        });
    Kokkos::fence();
  }

  void shift_defect(IndexArray<Nd - 1> new_position) {
    // set the current defect regions defect to 1.0, update the position of the
    // defect and set the defect value.
    real_t tmp = this->dParams.defect_value;
    set_defect<index_t>(real_t(1.0));
    this->dParams.defect_position = new_position;
    set_defect<index_t>(tmp);
  }

  real_t get_defect() const {
    // return the defect value
    return this->dParams.defect_value;
  }

  // define accessors for the field
  template <typename indexType>  // why do we template indexType here, when it
                                 // is defined in GLOBAL.hpp?
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(const indexType i,
                                                 const indexType j,
                                                 const indexType k,
                                                 const index_t mu) const {
    return field(i, j, k, mu) * defectField(i, j, k, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(const indexType i,
                                                 const indexType j,
                                                 const indexType k,
                                                 const index_t mu) {
    return field(i, j, k, mu) * defectField(i, j, k, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(
      const Kokkos::Array<indexType, 3> site,
      const index_t mu) const {
    return field(site[0], site[1], site[2], mu) *
           defectField(site[0], site[1], site[2], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(
      const Kokkos::Array<indexType, 3> site,
      const index_t mu) {
    return field(site[0], site[1], site[2], mu) *
           defectField(site[0], site[1], site[2], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void set(const indexType i,
                                       const indexType j,
                                       const indexType k,
                                       const index_t mu,
                                       const SUN<Nc>& value) const {
    field(i, j, k, mu) = restoreSUN(value);  // raw write
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> staple(
      const Kokkos::Array<indexType, 3> site,
      const index_t mu) const {
    // this only works if Nd == 3
    assert(Nd == 3);
    // get the indices
    const index_t i0 = site[0];
    const index_t i1 = site[1];
    const index_t i2 = site[2];
    // temporary SUN matrix to store the staple
    SUN<Nc> temp = zeroSUN<Nc>();
    // get the x + mu indices
    const indexType i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
    const indexType i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;
    const indexType i2pmu = mu == 2 ? (i2 + 1) % dimensions[2] : i2;

// positive directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) {  // loop over nu
      if (nu == mu)
        continue;  // skip if mu == nu
      const indexType i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const indexType i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;
      const indexType i2pnu = nu == 2 ? (i2 + 1) % dimensions[2] : i2;

      temp += this->operator()<indexType>(i0pmu, i1pmu, i2pmu, nu) *
              conj(this->operator()<indexType>(i0pnu, i1pnu, i2pnu, mu)) *
              conj(this->operator()<indexType>(i0, i1, i2, nu));
    }

// negative directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) {  // loop over nu
      if (nu == mu)
        continue;  // skip if mu == nu
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
  void save(std::string filename) const {
    auto h_field = Kokkos::create_mirror_view(field);
    Kokkos::deep_copy(h_field, field);
    std::ofstream file;
    try {
      file.open(filename, std::ios::out | std::ios::binary);
      file.write(reinterpret_cast<const char*>(h_field.data()),
                 h_field.size() * sizeof(SUN<Nc>));
      file.close();
    } catch (const std::exception& e) {
      printf("Error saving PTBC field to file %s: %s\n", filename.c_str(),
             e.what());
    }
  }
  void load(std::string filename) {
    try {
      set_defect<index_t>(1.0);  // reset defect to 1.0 before loading
      this->dParams = parse<Nd>(filename);
      set_defect<index_t>(this->dParams.defect_value);
      auto h_field = Kokkos::create_mirror_view(field);
      std::ifstream file;
      file.open(filename, std::ios::in | std::ios::binary);
      file.read(reinterpret_cast<char*>(h_field.data()),
                h_field.size() * sizeof(SUN<Nc>));
      file.close();
      Kokkos::deep_copy(field, h_field);
    } catch (const std::exception& e) {
      printf("Error loading PTBC field from file %s: %s\n", filename.c_str(),
             e.what());
    }
  }
};

template <size_t Nd, size_t Nc, typename precision = complex_t>
struct devicePTBCGaugeField2D {
  devicePTBCGaugeField2D() = default;

  GaugeField2D<Nd, Nc, precision> field;
  IndexArray<Nd> dimensions;
  LinkScalarField2D<Nd> defectField;
  using deviceDefectParams = defectParams<Nd>;
  deviceDefectParams dParams;

  // // copy constructor from a given devicePTBCGaugeField
  // devicePTBCGaugeField2D(const devicePTBCGaugeField<Nd, Nc,precision>
  // &dPTBCGaugeField)
  //     : dimensions(dPTBCGaugeField.dimensions) {
  //   Kokkos::realloc(Kokkos::WithoutInitializing, field, dimensions[0],
  //                   dimensions[1]);
  //   Kokkos::deep_copy(field, dPTBCGaugeField.field);
  //   do_init_defect(defectField, dPTBCGaugeField.dParams);
  // }

  // 'copy' constructor from a given deviceGaugeField
  devicePTBCGaugeField2D(const GaugeField2D<Nd, Nc, precision>& f_in)
      : field("gauge_field",
              f_in.extent(0),
              f_in.extent(1)),  // Allocate directly in constructor
        dimensions({static_cast<index_t>(f_in.extent(0)),
                    static_cast<index_t>(f_in.extent(1))}) {
    do_init(field, f_in);
  }

  // should defect_length and cr be encompassed in a defect struct?
  devicePTBCGaugeField2D(const index_t L0,
                         const index_t L1,
                         const complex_t init,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1}) {
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  // initialize all links to a given SUN matrix
  devicePTBCGaugeField2D(const index_t L0,
                         const index_t L1,
                         const SUN<Nc>& init,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1}) {
    printf("[DEBUG] Initializing PTBCGaugeField with SUN init Nd=2\n");
    do_init(field, init);
    do_init_defect(defectField, dParam);
  }

  devicePTBCGaugeField2D(const IndexArray<2>& dimensions, const SUN<Nc>& init)
      : dimensions(dimensions) {
    do_init(field, init);
    do_init_defect(defectField);
  }

  devicePTBCGaugeField2D(const IndexArray<2>& dimensions, const complex_t init)
      : dimensions(dimensions) {
    do_init(field, init);
    do_init_defect(defectField);
  }

  // initialize all links randomized with a given delta
  template <class RNG>
  devicePTBCGaugeField2D(const index_t L0,
                         const index_t L1,
                         RNG& rng,
                         const real_t delta,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1}) {
    printf("[DEBUG] Initializing PTBCGaugeField with RNG Nd=2\n");
    do_init(L0, L1, field, rng, delta);
    do_init_defect(defectField, dParam);
  }

  // initialize all links randomized
  template <class RNG>
  devicePTBCGaugeField2D(const index_t L0,
                         const index_t L1,
                         RNG& rng,
                         const deviceDefectParams dParam)
      : dimensions({L0, L1}) {
    do_init(L0, L1, field, rng);
    do_init_defect(defectField, dParam);
  }

  void do_init(GaugeField2D<Nd, Nc, precision>& V, complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    KTune::parallel_for(
        "init_PTBCdeviceGaugeField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
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

  void do_init(GaugeField2D<Nd, Nc, precision>& V, const SUN<Nc>& init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    KTune::parallel_for(
        "init_PTBCdeviceGaugeField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = init;
          }
        });
    Kokkos::fence();
  }

  void do_init(GaugeField2D<Nd, Nc, precision>& V,
               const GaugeField2D<Nd, Nc, precision>& f_in) {
    if (!V.is_allocated()) {
      V = GaugeField2D<Nd, Nc, precision>("gauge_field_tmp", 0, 0);
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    Kokkos::deep_copy(V, f_in);
  }

  // without any defect information, set defect to be non-existent.
  void do_init_defect(LinkScalarField2D<Nd>& V) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, dimensions[0],
                    dimensions[1]);
    KTune::parallel_for(
        "init_PTBCdeviceGaugeField", Policy<Nd>(IndexArray<Nd>{0}, dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
#pragma unroll
          for (index_t mu = 0; mu < Nd; ++mu) {
            V(i0, i1, mu) = 1.0;
          }
        });
    Kokkos::fence();
  }

  // This fixes the defect location according to (2.3) in 2404.14151
  void do_init_defect(LinkScalarField2D<Nd>& V, const deviceDefectParams dP) {
    do_init_defect(V);
    this->dParams = dP;
    set_defect<index_t>(this->dParams.defect_value);
    Kokkos::fence();
  }

  template <class RNG>
  void do_init(const index_t L0,
               const index_t L1,
               GaugeField2D<Nd, Nc, precision>& V,
               RNG& rng,
               const real_t delta) {
    if (!V.is_allocated()) {
      V = GaugeField2D<Nd, Nc, precision>("gauge_field", 0, 0);
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    KTune::parallel_for(
        "init_deviceGaugeField2D",
        Policy<2>(IndexArray<2>{0, 0}, IndexArray<2>{L0, L1}),
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

  template <class RNG>
  void do_init(const index_t L0,
               const index_t L1,
               GaugeField2D<Nd, Nc, precision>& V,
               RNG& rng) {
    if (!V.is_allocated()) {
      V = GaugeField2D<Nd, Nc, precision>("gauge_field", 0, 0);
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    KTune::parallel_for(
        "init_deviceGaugeField2D",
        Policy<2>(IndexArray<2>{0, 0}, IndexArray<2>{L0, L1}),
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
          rng.free_state(generator);
        });
    Kokkos::fence();
  }

  // Sets the defect value
  template <typename indexType>
  void set_defect(real_t cr) {
    this->dParams.defect_value = cr;
    auto dimensions_local = this->dimensions;
    auto defect_position_local = this->dParams.defect_position;
    auto defectField_local = this->defectField;
    auto policy = Policy1D<>(0, this->dParams.defect_length);
    KTune::parallel_for(
        "set_defect", policy, KOKKOS_LAMBDA(const indexType i1) {
          const indexType i1_shift =
              (i1 + defect_position_local[0]) % dimensions_local[1];
          defectField_local(0, i1_shift, 0) = cr;
        });
    Kokkos::fence();
  }

  void shift_defect(IndexArray<Nd - 1> new_position) {
    // set the current defect regions defect to 1.0, update the position of the
    // defect and set the defect value.
    real_t tmp = this->dParams.defect_value;
    this->set_defect<index_t>(real_t(1.0));
    this->dParams.defect_position = new_position;
    this->set_defect<index_t>(tmp);
  }

  real_t get_defect() const {
    // return the defect value
    return this->dParams.defect_value;
  }

  operator deviceGaugeField2D<Nd, Nc, precision>() const {
    return deviceGaugeField2D<Nd, Nc, precision>(this->field, this->dimensions);
  }

  // define accessors for the field
  template <
      typename indexType>  // why do we template indexType here, when it is
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(const indexType i,
                                                 const indexType j,
                                                 const index_t mu) const {
    return field(i, j, mu) * defectField(i, j, mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(const indexType i,
                                                 const indexType j,
                                                 const index_t mu) {
    return field(i, j, mu) * defectField(i, j, mu);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(
      const Kokkos::Array<indexType, 2> site,
      const index_t mu) const {
    return field(site[0], site[1], mu) * defectField(site[0], site[1], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> operator()(
      const Kokkos::Array<indexType, 2> site,
      const index_t mu) {
    return field(site[0], site[1], mu) * defectField(site[0], site[1], mu);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION void set(const indexType i,
                                       const indexType j,
                                       const index_t mu,
                                       const SUN<Nc>& value) const {
    field(i, j, mu) = restoreSUN(value);  // raw write
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> staple(
      const Kokkos::Array<indexType, 2> site,
      const index_t mu) const {
    // this only works if Nd == 2
    assert(Nd == 2);
    // get the indices
    const index_t i0 = site[0];
    const index_t i1 = site[1];
    // temporary SUN matrix to store the staple
    SUN<Nc> temp = zeroSUN<Nc>();
    // get the x + mu indices
    const indexType i0pmu = mu == 0 ? (i0 + 1) % dimensions[0] : i0;
    const indexType i1pmu = mu == 1 ? (i1 + 1) % dimensions[1] : i1;

// positive directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) {  // loop over nu
      if (nu == mu)
        continue;  // skip if mu == nu
      const indexType i0pnu = nu == 0 ? (i0 + 1) % dimensions[0] : i0;
      const indexType i1pnu = nu == 1 ? (i1 + 1) % dimensions[1] : i1;

      temp += this->operator()<indexType>(i0pmu, i1pmu, nu) *
              conj(this->operator()<indexType>(i0pnu, i1pnu, mu)) *
              conj(this->operator()<indexType>(i0, i1, nu));
    }

// negative directions
#pragma unroll
    for (index_t nu = 0; nu < Nd; ++nu) {  // loop over nu
      if (nu == mu)
        continue;  // skip if mu == nu
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
  void save(std::string filename) const {
    auto h_field = Kokkos::create_mirror_view(field);
    Kokkos::deep_copy(h_field, field);
    std::ofstream file;
    try {
      file.open(filename, std::ios::out | std::ios::binary);
      file.write(reinterpret_cast<const char*>(h_field.data()),
                 h_field.size() * sizeof(SUN<Nc>));
      file.close();
    } catch (const std::exception& e) {
      printf("Error saving PTBC field to file %s: %s\n", filename.c_str(),
             e.what());
    }
  }
  void load(std::string filename) {
    try {
      set_defect<index_t>(1.0);  // reset defect to 1.0 before loading
      this->dParams = parse<Nd>(filename);
      set_defect<index_t>(this->dParams.defect_value);
      auto h_field = Kokkos::create_mirror_view(field);
      std::ifstream file;
      file.open(filename, std::ios::in | std::ios::binary);
      file.read(reinterpret_cast<char*>(h_field.data()),
                h_field.size() * sizeof(SUN<Nc>));
      file.close();
      Kokkos::deep_copy(field, h_field);
    } catch (const std::exception& e) {
      printf("Error loading PTBC field from file %s: %s\n", filename.c_str(),
             e.what());
    }
  }
};

}  // namespace klft
