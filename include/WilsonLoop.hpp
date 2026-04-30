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

// this file defines functions to calculate
// Wilson Loops for different gauge fields
// and different gauge groups

#pragma once
#include "FieldTypeHelper.hpp"
#include "IndexHelper.hpp"

namespace klft {

// alright...!!! Lets give the same treatment of the plaquette to the Wilson
// loop first we define a functor this functor builds the Wilson Line along the
// mu and nu directions
template <size_t rank,
          size_t Nc,
          typename precision = complex_t,
          GaugeFieldKind k = GaugeFieldKind::Standard>
struct WLmunu {
  // all of this works strictly for the case Nd = rank
  constexpr static const size_t Nd = rank;
  // define the gauge field type
  using GaugeFieldType =
      typename DeviceGaugeFieldType<rank, Nc, complex_t, k>::type;
  // input gauge field
  const GaugeFieldType g_in;
  // define the SUN field type
  using SUNFieldType = typename DeviceSUNFieldType<rank, Nc>::type;
  // SUN fields to store the Wilson lines along mu and nu
  SUNFieldType WLmu, WLnu;
  // mu and nu directions
  const index_t mu, nu;
  // length of the Wilson lines
  index_t Lmu_old, Lnu_old;
  index_t Lmu, Lnu;
  // dimensions of the field
  const IndexArray<rank> dimensions;
  WLmunu(const GaugeFieldType& g_in,
         const index_t mu,
         const index_t nu,
         const index_t Lmu,
         const index_t Lnu,
         SUNFieldType& WLmu,
         SUNFieldType& WLnu,
         const IndexArray<rank>& dimensions)
      : g_in(g_in),
        mu(mu),
        nu(nu),
        Lmu(Lmu),
        Lnu(Lnu),
        Lmu_old(0),
        Lnu_old(0),
        WLmu(WLmu),
        WLnu(WLnu),
        dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // temp SUN matrices to store products
    SUN<Nc> lmu = WLmu(Idcs...);
    SUN<Nc> lnu = WLnu(Idcs...);
    // build Lmu
    for (index_t i = Lmu_old; i < Lmu; ++i) {
      lmu *= g_in(shift_index_plus<rank, size_t>(
                      Kokkos::Array<size_t, rank>{Idcs...}, mu, i, dimensions),
                  mu);
    }
    // build Lnu
    for (index_t i = Lnu_old; i < Lnu; ++i) {
      lnu *= g_in(shift_index_plus<rank, size_t>(
                      Kokkos::Array<size_t, rank>{Idcs...}, nu, i, dimensions),
                  nu);
    }
    // store the lines
    WLmu(Idcs...) = lmu;
    WLnu(Idcs...) = lnu;
  }

  void update_Lmu_Lnu(const index_t Lmu_new, const index_t Lnu_new) {
    // update the Wilson lines
    Lmu_old = Lmu;
    Lnu_old = Lnu;
    Lmu = Lmu_new;
    Lnu = Lnu_new;
  }

  void reset_Lmu_Lnu() {
    // reset the Wilson lines
    Lmu_old = 0;
    Lnu_old = 0;
    Lmu = 0;
    Lnu = 0;
    // reset the SUN fields
    Kokkos::deep_copy(WLmu.field,
                      SUNFieldType(dimensions, identitySUN<Nc>()).field);
    Kokkos::deep_copy(WLnu.field,
                      SUNFieldType(dimensions, identitySUN<Nc>()).field);
  }
};

// define a second functor to build the Wilson loop out of WLmu and WLnu
template <size_t rank, size_t Nc>
struct WLoop_munu {
  // all of this works strictly for the case Nd = rank
  constexpr static const size_t Nd = rank;
  // define the SUN field type
  using SUNFieldType = typename DeviceSUNFieldType<rank, Nc>::type;
  // input SUN fields
  const SUNFieldType WLmu, WLnu;
  // mu and nu directions
  const index_t mu, nu;
  // length of the Wilson lines
  const index_t Lmu, Lnu;
  // define the field type
  using FieldType = typename DeviceFieldType<rank>::type;
  // output field to store loop per site
  FieldType Wmunu_per_site;
  // dimensions of the field
  const IndexArray<rank> dimensions;
  WLoop_munu(const SUNFieldType& WLmu,
             const SUNFieldType& WLnu,
             const index_t mu,
             const index_t nu,
             const index_t Lmu,
             const index_t Lnu,
             FieldType& Wmunu_per_site,
             const IndexArray<rank>& dimensions)
      : WLmu(WLmu),
        WLnu(WLnu),
        mu(mu),
        nu(nu),
        Lmu(Lmu),
        Lnu(Lnu),
        Wmunu_per_site(Wmunu_per_site),
        dimensions(dimensions) {}

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // similar to the plaquette, we first build two half loops
    // lmu = WL_mu (x) * WL_nu (x + mu)
    //              |
    //              ^
    //              |
    //    ----->-----
    SUN<Nc> lmu = WLmu(Idcs...) * WLnu(shift_index_plus<rank, size_t>(
                                      Kokkos::Array<size_t, rank>{Idcs...}, mu,
                                      Lmu, dimensions));
    // lnu = WL_nu (x) * WL_mu (x + nu)
    //    ----->-----
    //    |
    //    ^
    //    |
    SUN<Nc> lnu = WLnu(Idcs...) * WLmu(shift_index_plus<rank, size_t>(
                                      Kokkos::Array<size_t, rank>{Idcs...}, nu,
                                      Lnu, dimensions));
    // finally, construct the loop Wmunu = lmu * lnu^dagger
    // only need to take trace
    complex_t tmunu(0.0, 0.0);
#pragma unroll
    for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
      for (index_t c2 = 0; c2 < Nc; ++c2) {
        tmunu += lmu[c1][c2] * Kokkos::conj(lnu[c1][c2]);
      }
    }
    // store the Wilson loop per site
    Wmunu_per_site(Idcs...) = tmunu;
  }
};

// define a function to calculate Wilson Loop of
// sets of lengths Lmu and Lnu
// in the mu - nu plane
// return is normalized based on bool normalize
// the Lmu and Lnu pairs must be strictly non-decreasing
template <size_t rank,
          size_t Nc,
          typename precision = complex_t,
          GaugeFieldKind k = GaugeFieldKind::Standard>
void WilsonLoop_mu_nu(
    const typename DeviceGaugeFieldType<rank, Nc, precision, k>::type& g_in,
    const index_t mu,
    const index_t nu,
    const std::vector<Kokkos::Array<index_t, 2>>& Lmu_nu_pairs,
    std::vector<Kokkos::Array<real_t, 5>>& Wmunu_vals,
    const bool normalize = true) {
  // this kernel is defined for Nd = rank
  constexpr static const size_t Nd = rank;
  // temp variable to store the Wilson loop
  complex_t Wmunu;

  // get the start and end indices
  const auto& dimensions = g_in.field.layout().dimension;
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < Nd; ++i) {
    start[i] = 0;
    end[i] = dimensions[i];
  }

  // define 2 temporary SUNFields for storing the 2 Wilson lines
  // of length Lmu and Lnu along mu and nu directions respectively
  // initialize to identity SUN matrix
  using SUNFieldType = typename DeviceSUNFieldType<rank, Nc>::type;
  SUNFieldType WLmu(end, identitySUN<Nc>());
  SUNFieldType WLnu(end, identitySUN<Nc>());

  // temporary field for storing the Wilson loop per site
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType Wmunu_per_site(end, complex_t(0.0, 0.0));

  // initialize the WLmunu functor
  WLmunu<rank, Nc, precision, k> wlmunu(g_in, mu, nu, 0, 0, WLmu, WLnu, end);

  // iterate over all pairs of Lmu and Lnu
  for (const auto& Lmu_nu : Lmu_nu_pairs) {
    // get the lengths
    const index_t Lmu = Lmu_nu[0];
    const index_t Lnu = Lmu_nu[1];
    // chck if the new lengths are greater than
    // the old lengths, if not, reset the
    // Wilson lines
    if (Lmu < wlmunu.Lmu || Lnu < wlmunu.Lnu) {
      wlmunu.reset_Lmu_Lnu();
    }
    // the following is an extra bit of check
    // most likely unnecessary, but keep it for now
    // check if the lengths are valid
    assert(Lmu >= wlmunu.Lmu);
    assert(Lnu >= wlmunu.Lnu);
    // update the WLmunu functor
    wlmunu.update_Lmu_Lnu(Lmu, Lnu);
    // launch the kernel to build the Wilson lines
    KTune::parallel_for("WilsonLoop_GaugeField_WLmunu",
                        Policy<rank>(start, end), wlmunu);
    Kokkos::fence();
    // launch the kernel to build the Wilson loop
    KTune::parallel_for("WilsonLoop_GaugeField_WLoop_munu",
                        Policy<rank>(start, end),
                        WLoop_munu<rank, Nc>(WLmu, WLnu, mu, nu, Lmu, Lnu,
                                             Wmunu_per_site, end));
    Kokkos::fence();
    // reduce the Wilson loop over all sites
    Wmunu = Wmunu_per_site.sum();
    Kokkos::fence();
    // normalize the Wilson loop
    if (normalize) {
#pragma unroll
      for (index_t i = 0; i < rank; ++i) {
        Wmunu /= static_cast<real_t>(end[i]);
      }
      Wmunu /= static_cast<real_t>(Nc);
    }
    // store the Wilson loop in the output vector
    Wmunu_vals.push_back(Kokkos::Array<real_t, 5>{
        static_cast<real_t>(mu), static_cast<real_t>(nu),
        static_cast<real_t>(Lmu), static_cast<real_t>(Lnu), Wmunu.real()});
  }
}

// define a function to calculate the Wilson Loop of length L
// in the spatial directions and T in the temporal direction
// result is normalized based on bool normalize
template <size_t rank,
          size_t Nc,
          typename precision = complex_t,
          GaugeFieldKind k = GaugeFieldKind::Standard>
void WilsonLoop_temporal(
    const typename DeviceGaugeFieldType<rank, Nc, precision, k>::type& g_in,
    const std::vector<Kokkos::Array<index_t, 2>>& L_T_pairs,
    std::vector<Kokkos::Array<real_t, 3>>& Wtemporal_vals,
    const bool normalize = true) {
  // this kernel is defined for Nd = rank
  constexpr static const size_t Nd = rank;

  // temp variable to store the Wilson loop
  std::vector<Kokkos::Array<real_t, 5>> Wmunu_vals;
  // run the kernel for mu = 0
  WilsonLoop_mu_nu<rank, Nc, precision, k>(g_in, 0, Nd - 1, L_T_pairs,
                                           Wmunu_vals, normalize);
  // push the results to the output vector
  for (const auto& Wmunu : Wmunu_vals) {
    Wtemporal_vals.push_back(
        Kokkos::Array<real_t, 3>{Wmunu[2], Wmunu[3], Wmunu[4]});
  }
  // repeat this for remaining spatial dimensions
  if constexpr (Nd > 2) {
    // clear the Wmunu_vals vector
    Wmunu_vals.clear();
    WilsonLoop_mu_nu<rank, Nc, precision, k>(g_in, 1, Nd - 1, L_T_pairs,
                                             Wmunu_vals, normalize);
    // need to average over the spatial dimensions
    // push the results to the output vector
    for (index_t i = 0; i < Wmunu_vals.size(); ++i) {
      if (Wmunu_vals[i][2] == Wtemporal_vals[i][0] &&
          Wmunu_vals[i][3] == Wtemporal_vals[i][1]) {
        Wtemporal_vals[i][2] += Wmunu_vals[i][4];
      } else {
        // this should not happen
        throw std::runtime_error(
            "WilsonLoop_temporal: dimensions do not match");
      }
    }
  }
  if constexpr (Nd > 3) {
    // clear the Wmunu_vals vector
    Wmunu_vals.clear();
    WilsonLoop_mu_nu<rank, Nc, precision, k>(g_in, 2, Nd - 1, L_T_pairs,
                                             Wmunu_vals, normalize);
    // need to average over the spatial dimensions
    // push the results to the output vector
    for (index_t i = 0; i < Wmunu_vals.size(); ++i) {
      if (Wmunu_vals[i][2] == Wtemporal_vals[i][0] &&
          Wmunu_vals[i][3] == Wtemporal_vals[i][1]) {
        Wtemporal_vals[i][2] += Wmunu_vals[i][4];
      } else {
        // this should not happen
        throw std::runtime_error(
            "WilsonLoop_temporal: dimensions do not match");
      }
    }
  }
  // now we need to average over the spatial dimensions
  // this is done by dividing the Wilson loop by the number of spatial
  // dimensions
  for (index_t i = 0; i < Wtemporal_vals.size(); ++i) {
    Wtemporal_vals[i][2] /= static_cast<real_t>(Nd - 1);
  }
}

}  // namespace klft
