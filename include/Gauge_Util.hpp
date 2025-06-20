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

// utility functions for gauge fields
#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "Kokkos_Macros.hpp"
#include "PTBCGaugeField.hpp"
#include "SUN.hpp"

namespace klft {

template <size_t Nd, typename FieldA, typename FieldB>
struct GaugeFieldMultFunctor;

template <typename FieldA, typename FieldB>
struct GaugeFieldMultFunctor<2, FieldA, FieldB> {
  FieldA a;
  FieldB b;

  KOKKOS_INLINE_FUNCTION
  void operator()(index_t i0, index_t i1) {
    for (index_t mu = 0; mu < 2; ++mu) {
      a(i0, i1, mu) *= b(i0, i1, mu);
    }
  }
};

template <typename FieldA, typename FieldB>
struct GaugeFieldMultFunctor<3, FieldA, FieldB> {
  FieldA a;
  FieldB b;

  KOKKOS_INLINE_FUNCTION
  void operator()(index_t i0, index_t i1, index_t i2) {
    for (index_t mu = 0; mu < 3; ++mu) {
      a(i0, i1, i2, mu) *= b(i0, i1, i2, mu);
    }
  }
};

template <typename FieldA, typename FieldB>
struct GaugeFieldMultFunctor<4, FieldA, FieldB> {
  FieldA a;
  FieldB b;

  KOKKOS_INLINE_FUNCTION
  void operator()(index_t i0, index_t i1, index_t i2, index_t i3) {
    for (index_t mu = 0; mu < 4; ++mu) {
      a(i0, i1, i2, i3, mu) *= b(i0, i1, i2, i3, mu);
    }
  }
};

// TODO: fix the multiplication operators
//
// define multiplication between GaugeFields for now only define *= operators,
// as I have yet to decide on a heuristic for the output type when mixing
// different GaugeFieldKinds
// KOKKOS_FORCEINLINE_FUNCTION
// template <typename DGaugeFieldType1, typename DGaugeFieldType2>
// typename DGaugeFieldType1::type &
// operator*=(typename DGaugeFieldType1::type &a,
//            typename DGaugeFieldType2::type const &b) {
//   tune_and_launch_for<Nd>(
//       "operator*=DeviceGaugeFieldType<Nd, Nc, "
//       "k1>::type*DeviceGaugeFieldType<Nd, Nc, k2>::type",
//       IndexArray<Nd>{0}, a.dimensions,
//       KOKKOS_LAMBDA(index_t i0, index_t i1, index_t i2, index_t i3) {
// #pragma unroll
//         for (index_t mu = 0; mu < Nd; ++mu) {
//           a.field(i0, i1, i2, i3, mu) *= b(i0, i1, i2, i3, mu);
//         }
//       });
//   return a;
// }

// template<size_t Nd, size_t Nc, GaugeFieldKind k>
// GaugeField<Nd, Nc>& operator*=(GaugeField<Nd, Nc> &a,
//   typename DeviceGaugeFieldType<Nd, Nc, k>::type const &b){
//   assert(a.layout() == b.field.layout());
//   assert(a.memory_space == b.field.memory_space); //only allow device-device
//   and host-host operations
//   tune_and_launch_for<Nd>("operator*=GaugeField*DeviceGaugeFieldType<Nd, Nc,
//   k>::type",IndexArray<Nd>{0}, a.dimensions,
//     KOKKOS_LAMBDA(index_t i0, index_t i1, index_t i2, index_t i3){
//       #pragma unroll
//       for (index_t mu = 0;mu<Nd;++mu){
//         a(i0, i1, i2, i3, mu) *= b(i0, i1, i2, i3, mu);
//       }
//     });
//   return a;
// }
//
// template<size_t Nd, size_t Nc, GaugeFieldKind k>
// typename DeviceGaugeFieldType<Nd, Nc, k>::type& operator*=(
//   typename DeviceGaugeFieldType<Nd, Nc, k>::type &a,
//   GaugeField<Nd, Nc> const &b){
//   assert(a.field.layout() == b.layout());
//   assert(a.field.memory_space == b.memory_space); //only allow device-device
//   and host-host operations
//   tune_and_launch_for<Nd>("operator*=DeviceGaugeFieldType<Nd, Nc,
//   k>::type*GaugeField",IndexArray<Nd>{0}, a.dimensions,
//     KOKKOS_LAMBDA(index_t i0, index_t i1, index_t i2, index_t i3){
//       #pragma unroll
//       for (index_t mu = 0;mu<Nd;++mu){
//         a.field(i0, i1, i2, i3, mu) *= b(i0, i1, i2, i3, mu);
//       }
//     });
//   return a;
// }
//
// template<size_t Nd, size_t Nc, GaugeFieldKind k>
// typename DeviceGaugeFieldType<Nd, Nc, k>::type& operator*=(
//   typename DeviceGaugeFieldType<Nd, Nc, k>::type &a,
//   constGaugeField<Nd, Nc> const &b){
//   assert(a.field.layout() == b.layout());
//   assert(a.field.memory_space == b.memory_space); //only allow device-device
//   and host-host operations
//   tune_and_launch_for<Nd>("operator*=DeviceGaugeFieldType<Nd, Nc,
//   k>::type*constGaugeField",IndexArray<Nd>{0}, a.dimensions,
//     KOKKOS_LAMBDA(index_t i0, index_t i1, index_t i2, index_t i3){
//       #pragma unroll
//     for (index_t mu = 0;mu<Nd;++mu){
//         a.field(i0, i1, i2, i3, mu) *= b(i0, i1, i2, i3, mu);
//       }
//     });
//   return a;
// }
// // should the multiplication between GaugeField<...> types also be defined?
//
// // now define the multiplication with scalars
// template<size_t Nd, size_t Nc, GaugeFieldKind K1>
// typename DeviceGaugeFieldType<Nd, Nc, K1>::type&
// operator*=(
//   typename DeviceGaugeFieldType<Nd, Nc, K1>::type &a,
//   real_t const b){
//   tune_and_launch_for<Nd>("operator*=DeviceGaugeFieldType<Nd, Nc,
//   k1>::type*Scalar",IndexArray<Nd>{0}, a.dimensions, KOKKOS_LAMBDA(index_t
//   i0, index_t i1, index_t i2, index_t i3){
//     #pragma unroll
//     for (index_t mu = 0;mu<Nd;++mu){
//       a.field(i0, i1, i2, i3, mu) *= b;
//     }
//   });
//   return a;
// }

// template <size_t Nd, size_t Nc>
// GaugeField<Nd, Nc> &operator*=(GaugeField<Nd, Nc> &a, real_t const b) {
//   tune_and_launch_for<Nd>(
//       "operator*=GaugeField*Scalar", IndexArray<Nd>{0}, a.dimensions,
//       KOKKOS_LAMBDA(index_t i0, index_t i1, index_t i2, index_t i3) {
// #pragma unroll
//         for (index_t mu = 0; mu < Nd; ++mu) {
//           a(i0, i1, i2, i3, mu) *= b;
//         }
//       });
//   return a;
// }

// // function to conjugate a given DeviceGaugeField in place.
// template <size_t Nd, size_t Nc, GaugeFieldKind K = GaugeFieldKind::Standard>
// void conj_field(typename DeviceGaugeFieldType<Nd, Nc, K>::type &field) {
//   tune_and_launch_for<Nd>(
//       "conj_field", IndexArray<Nd>{0}, field.dimensions,
//       KOKKOS_LAMBDA(index_t i0, index_t i1, index_t i2, index_t i3) {
// #pragma unroll
//         for (index_t mu = 0; mu < Nd; ++mu) {
//           field.field(i0, i1, i2, i3, mu) = conj(field(i0, i1, i2, i3, mu));
//         }
//       });
//   Kokkos::fence();
// }
//

// calculate staple per site and store in another gauge
template <typename DGaugeFieldType>
auto stapleField(const typename DGaugeFieldType::type &g_in)
    -> ConstGaugeFieldType<DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank,
                           DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc> {
  // initialize the output field
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;

  typename DGaugeFieldType::type g_out(g_in.field);

  // get the start and end indices
  const auto &dimensions = g_in.field.layout().dimension;
  IndexArray<Nd> start;
  IndexArray<Nd> end;
  for (index_t i = 0; i < Nd; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  // It would be trivial to add a stapleField return into each
  // DeviceGaugeFieldType, though as already done with .staple, shouldn't the
  // definition and calculations be seperated?
  if constexpr (Nd == 4) {
    tune_and_launch_for<4>(
        "stapleField_GaugeField", start, end,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          for (index_t mu = 0; mu < Nd; ++mu) {
            g_out.field(i0, i1, i2, i3, mu) =
                g_in.staple(IndexArray<4>{i0, i1, i2, i3}, mu);
          }
        });
  } else if constexpr (Nd == 3) {
    tune_and_launch_for<3>(
        "stapleField_GaugeField3D", start, end,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          for (index_t mu = 0; mu < Nd; ++mu) {
            g_out.field(i0, i1, i2, mu) =
                g_in.staple(IndexArray<3>{i0, i1, i2}, mu);
          }
        });
  } else if constexpr (Nd == 2) {
    tune_and_launch_for<2>(
        "stapleField_GaugeField3D", start, end,
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          for (index_t mu = 0; mu < Nd; ++mu) {
            g_out.field(i0, i1, mu) = g_in.staple(IndexArray<2>{i0, i1}, mu);
          }
        });
  } else {
    static_assert(Nd == 2 || Nd == 3 || Nd == 4, "Unsupported Nd");
  }

  Kokkos::fence();
  // return the output field
  return ConstGaugeFieldType<Nd, Nc>(g_out.field);
}

} // namespace klft
