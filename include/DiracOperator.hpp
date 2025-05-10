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

// this file defines various versions of the Wilson-Dirac (WD) operator

#pragma once
#include "FieldTypeHelper.hpp"
#include "GammaMatrix.hpp"
#include "IndexHelper.hpp"

namespace klft {
// Define a functor for the normal WD operator:
template <size_t rank, size_t Nc, size_t RepDim>
struct apply_D {
  constexpr static const size_t Nd = rank;

  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType s_in;
  SpinorFieldType s_out;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  const GaugeFieldType g_in;
  using VecGammaMatrix = Kokkos::Array<GammaMat<RepDim>, 4>;
  const VecGammaMatrix gammas;
  const IndexArray<rank> dimensions;
  const real_t mass;
  apply_D(SpinorFieldType &s_out, const SpinorFieldType &s_in,
          const GaugeFieldType &g_in, const VecGammaMatrix &gammas,
          const IndexArray<rank> &dimensions, const real_t &mass)
      : s_out(s_out),
        s_in(s_in),
        g_in(g_in),
        gammas(gammas),
        dimensions(dimensions),
        mass(mass) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp = zeroSpinor<Nc, RepDim>();
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);
      auto xp = shift_index_plus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);

      temp += 0.5 * (1 - gammas[mu]) * g_in(Idcs..., mu) * s_in(xp);
      temp += 0.5 * (1 + gammas[mu]) * conj(g_in(xm, mu)) * s_in(xm);
    }
    // Is the +4 correct? Instead of += only = depending on how s_out is
    // initialized or used!
    s_out(Idcs...) += (mass + 4) * s_in(Idcs...) - temp;
  }
};

template <size_t rank, size_t Nc, size_t RepDim>
struct apply_Q {
  constexpr static const size_t Nd = rank;

  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType s_in;
  SpinorFieldType s_out;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  const GaugeFieldType g_in;
  using VecGammaMatrix = Kokkos::Array<GammaMat<RepDim>, 4>;
  const VecGammaMatrix gammas;
  const GammaMat<RepDim> gamma5;
  const IndexArray<rank> dimensions;
  const real_t mass;
  apply_Q(SpinorFieldType &s_out, const SpinorFieldType &s_in,
          const GaugeFieldType &g_in, const VecGammaMatrix &gammas,
          const GammaMat<RepDim> gamma5, const IndexArray<rank> &dimensions,
          const real_t &mass)
      : s_out(s_out),
        s_in(s_in),
        g_in(g_in),
        gammas(gammas),
        gamma5(gamma5),
        dimensions(dimensions),
        mass(mass) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp = zeroSpinor<Nc, RepDim>();
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);
      auto xp = shift_index_plus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);

      temp += 0.5 * (1 - gammas[mu]) * g_in(Idcs..., mu) * s_in(xp);
      temp += 0.5 * (1 + gammas[mu]) * conj(g_in(xm, mu)) * s_in(xm);
    }
    // Is the +4 correct? Instead of += only = depending on how s_out is
    // initialized or used!
    s_out(Idcs...) += gamma5 * ((mass + 4) * s_in(Idcs...) - temp);
  }
};

}  // namespace klft
