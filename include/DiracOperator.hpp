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

// this file defines various versions of the Wilson-Dirac (WD) operator, in
// lattice units following Gattringer2010 (5.55) f. and absorbing the constant C
// into the field definition
#pragma once
#include "FieldTypeHelper.hpp"
#include "GammaMatrix.hpp"
#include "IndexHelper.hpp"
#include "Spinor.hpp"

namespace klft {
// Define a functor for the normal WD operator:
template <size_t rank, size_t Nc, size_t RepDim>
struct DiracOperator {
  constexpr static const size_t Nd = rank;

  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  const SpinorFieldType s_in;
  SpinorFieldType s_out;
  using GaugeFieldType = typename DeviceGaugeFieldType<rank, Nc>::type;
  const GaugeFieldType g_in;
  using VecGammaMatrix = Kokkos::Array<GammaMat<RepDim>, 4>;
  const VecGammaMatrix gammas;
  const GammaMat<RepDim> gamma_id = get_identity<RepDim>();
  const IndexArray<rank> dimensions;
  const real_t kappa;
  DiracOperator(SpinorFieldType& s_out,
                const SpinorFieldType& s_in,
                const GaugeFieldType& g_in,
                const VecGammaMatrix& gammas,
                const IndexArray<rank>& dimensions,
                const real_t& kappa)
      : s_out(s_out),
        s_in(s_in),
        g_in(g_in),
        gammas(gammas),

        dimensions(dimensions),
        kappa(kappa) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp = zeroSpinor<Nc, RepDim>();
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);
      auto xp = shift_index_plus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);

      temp += (gamma_id - gammas[mu]) * 0.5 * (g_in(Idcs..., mu) * s_in(xp));
      temp += (gamma_id + gammas[mu]) * 0.5 * (conj(g_in(xm, mu)) * s_in(xm));
    }
    // Is the +4 correct? Instead of += only = depending on how s_out is
    // initialized or used!
    s_out(Idcs...) += s_in(Idcs...) - kappa * temp;
  }
};
template <size_t rank, size_t Nc, size_t RepDim>
typename DeviceSpinorFieldType<rank, Nc, RepDim>::type apply_D(
    const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& s_in,
    const typename DeviceGaugeFieldType<rank, Nc>::type& g_in,
    const Kokkos::Array<GammaMat<RepDim>, 4>& gammas,
    const real_t& kappa) {
  const auto& dimensions = s_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  SpinorFieldType s_out(end, complex_t(0.0, 0.0));

  // Define the functor
  DiracOperator<rank, Nc, RepDim> D(s_out, s_in, g_in, gammas, end, kappa);

  tune_and_launch_for<rank>("Apply_Dirac_Operator", start, end, D);
  Kokkos::fence();
  return s_out;
}

template <size_t rank, size_t Nc, size_t RepDim>
struct HDiracOperator {
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
  const GammaMat<RepDim> gamma_id = get_identity<RepDim>();
  const IndexArray<rank> dimensions;
  const real_t kappa;
  HDiracOperator(SpinorFieldType& s_out,
                 const SpinorFieldType& s_in,
                 const GaugeFieldType& g_in,
                 const VecGammaMatrix& gammas,
                 const GammaMat<RepDim> gamma5,
                 const IndexArray<rank>& dimensions,
                 const real_t& kappa)
      : s_out(s_out),
        s_in(s_in),
        g_in(g_in),
        gammas(gammas),
        gamma5(gamma5),
        dimensions(dimensions),
        kappa(kappa) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp = zeroSpinor<Nc, RepDim>();
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);
      auto xp = shift_index_plus<rank, size_t>(
          Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions);

      temp += (gamma_id - gammas[mu]) * 0.5 * (g_in(Idcs..., mu) * s_in(xp));
      temp += (gamma_id + gammas[mu]) * 0.5 * (conj(g_in(xm, mu)) * s_in(xm));
    }
    // Is the +4 correct? Instead of += only = depending on how s_out is
    // initialized or used!
    s_out(Idcs...) += gamma5 * (s_in(Idcs...) - kappa * temp);
  }
};

template <size_t rank, size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION
    typename DeviceSpinorFieldType<rank, Nc, RepDim>::type
    apply_HD(const typename DeviceSpinorFieldType<rank, Nc, RepDim>::type& s_in,
             const typename DeviceGaugeFieldType<rank, Nc>::type& g_in,
             const Kokkos::Array<GammaMat<RepDim>, 4>& gammas,
             const GammaMat<RepDim>& gamma5,
             const real_t& kappa) {
  const auto& dimensions = s_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }
  using SpinorFieldType =
      typename DeviceSpinorFieldType<rank, Nc, RepDim>::type;
  SpinorFieldType s_out(end, complex_t(0.0, 0.0));

  // Define the functor
  HDiracOperator<rank, Nc, RepDim> HD(s_out, s_in, g_in, gammas, gamma5, end,
                                      kappa);

  tune_and_launch_for<rank>("Apply_Dirac_Operator", start, end, HD);
  Kokkos::fence();
  return s_out;
}

}  // namespace klft
