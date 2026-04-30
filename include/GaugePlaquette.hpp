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

// define plaquette functions for different gauge fields

#pragma once

#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "Kokkos_Macros.hpp"

namespace klft {

// define a function to calculate the gauge plaquette
// U_{mu nu} (x) = Tr[ U_mu(x) U_nu(x+mu) U_mu^dagger(x+nu) U_nu^dagger(x) ]
// for SU(N) gauge group

// first define the necessary functor
template <size_t rank,
          size_t Nc,
          typename precision = complex_t,
          GaugeFieldKind k = GaugeFieldKind::Standard>
struct GaugePlaq {
  // this kernel is defined for rank = Nd
  constexpr static const size_t Nd = rank;
  // define the gauge field type
  using GaugeFieldType =
      typename DeviceGaugeFieldType<rank, Nc, precision, k>::type;
  const GaugeFieldType g_in;
  // define the field type
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType plaq_per_site;
  const IndexArray<rank> dimensions;
  GaugePlaq(const GaugeFieldType& g_in,
            FieldType& plaq_per_site,
            const IndexArray<rank>& dimensions)
      : g_in(g_in),
        plaq_per_site(plaq_per_site),
        dimensions(dimensions) {
  }  // TODO: g_in does copy construction (does it?), this needs to be changed

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION complex_t
  operator()(const index_t mu, const index_t nu, const Indices... Idcs) const {
    // return the trace of a given plaquette (mu,nu) at site Idcs...
    //  temp SUN matrices to store products
    SUN<Nc> lmu, lnu;
    // reduction variable for all mu and nu
    complex_t tmunu(0.0, 0.0);

    lmu = g_in(Idcs..., mu) *
          g_in(shift_index_plus<rank, size_t>(
                   Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions),
               nu);
    // lnu = U_nu(x) * U_mu(x+nu)
    lnu = g_in(Idcs..., nu) *
          g_in(shift_index_plus<rank, size_t>(
                   Kokkos::Array<size_t, rank>{Idcs...}, nu, 1, dimensions),
               mu);
// multiply the 2 half plaquettes
// lmu * lnu^dagger
// take the trace
#pragma unroll
    for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
      for (index_t c2 = 0; c2 < Nc; ++c2) {
        tmunu += lmu[c1][c2] * Kokkos::conj(lnu[c1][c2]);
      }
    }
    return tmunu;
  }

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Indices... Idcs) const {
    // temp SUN matrices to store products
    SUN<Nc> lmu, lnu;
    // reduction variable for all mu and nu
    complex_t tmunu(0.0, 0.0);

#pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
#pragma unroll
      for (index_t nu = 0; nu < Nd; ++nu) {
        if (nu > mu) {
          // build plaquette in two halves
          // U_{mu nu} (x) = Tr[ lmu * lnu^dagger ]
          // lmu = U_mu(x) * U_nu(x+mu)
          lmu =
              g_in(Idcs..., mu) *
              g_in(shift_index_plus<rank, size_t>(
                       Kokkos::Array<size_t, rank>{Idcs...}, mu, 1, dimensions),
                   nu);
          // lnu = U_nu(x) * U_mu(x+nu)
          lnu =
              g_in(Idcs..., nu) *
              g_in(shift_index_plus<rank, size_t>(
                       Kokkos::Array<size_t, rank>{Idcs...}, nu, 1, dimensions),
                   mu);
// multiply the 2 half plaquettes
// lmu * lnu^dagger
// take the trace
#pragma unroll
          for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < Nc; ++c2) {
              tmunu += lmu[c1][c2] * Kokkos::conj(lnu[c1][c2]);
            }
          }
        }
      }
    }
    // store the result in the temporary field
    plaq_per_site(Idcs...) = tmunu;
  }
};

template <size_t rank,
          size_t Nc,
          typename precision = complex_t,
          GaugeFieldKind k = GaugeFieldKind::Standard>
real_t GaugePlaquette(
    const typename DeviceGaugeFieldType<rank, Nc, precision, k>::type& g_in,
    const bool normalize = true) {
  // this kernel is defined for rank = Nd
  constexpr static const size_t Nd = rank;
  // final return variable
  complex_t plaq = 0.0;
  // get the start and end indices
  // this is temporary solution
  // ideally, we want to have a policy factory
  IndexArray<rank> start;
  IndexArray<rank> end;
  for (index_t i = 0; i < rank; ++i) {
    start[i] = 0;
    end[i] = g_in.dimensions[i];
  }

  // temporary field for storing results per site
  // direct reduction is slow
  // this field will be summed over in the end
  using FieldType = typename DeviceFieldType<rank>::type;
  FieldType plaq_per_site(end, complex_t(0.0, 0.0));

  // define the functor
  GaugePlaq<rank, Nc, precision, k> gaugePlaquette(g_in, plaq_per_site, end);

  // tune and launch the kernel
  KTune::parallel_for("GaugePlaquette_GaugeField", Policy<rank>(start, end),
                      gaugePlaquette);
  Kokkos::fence();

  // sum over all sites
  plaq = plaq_per_site.sum();
  Kokkos::fence();

  // normalization
  if (normalize) {
    real_t norm = 1.0;
    for (index_t i = 0; i < rank; ++i) {
      norm *= static_cast<real_t>(end[i]);
    }
    norm *= static_cast<real_t>(((Nd - 1) * Nd / 2.0) * Nc);
    plaq /= norm;
  }

  return Kokkos::real(plaq);
}

// return the maximum value of ReTr(1 - plaquette) of the lattice
template <typename DGaugeFieldType>
real_t get_spmax(const typename DGaugeFieldType::type gauge_field) {
  static const size_t Nd = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(Nd == 4);
  static const size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static const GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using precision = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::value_type;
  using GaugeFieldType = typename DGaugeFieldType::type;
  using FieldType = typename DeviceFieldType<Nd>::type;
  FieldType plaq_per_site(gauge_field.dimensions, complex_t(0.0, 0.0));

  // number of bins
  GaugePlaq<Nd, Nc, precision, k> GPlaq(gauge_field, plaq_per_site,
                                        gauge_field.dimensions);

  real_t rtn = 0.0;
  auto policy = Policy<Nd>({0, 0, 0, 0}, gauge_field.dimensions);
  KTune::parallel_reduce(
      "get h (sp_max)", policy,
      KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3,
                    real_t& local_max) {
        // GPlaq(i0, i1, i2, i3);
        real_t s = 0.0;
        for (index_t mu = 0; mu < Nd; ++mu) {
          for (index_t nu = 0; nu < Nd; ++nu) {
            if (nu > mu) {
              real_t tmp = Kokkos::real(2 - GPlaq(mu, nu, i0, i1, i2, i3));
              s = tmp > s ? tmp : s;
            }
          }
        }
        local_max = Kokkos::max(local_max, s);
      },
      Kokkos::Max<real_t>(rtn));
  Kokkos::fence();

  return rtn;
}

// return the avg value of ReTr(1 - plaquette) of the lattice
template <typename DGaugeFieldType>
real_t get_spavg(const typename DGaugeFieldType::type gauge_field) {
  static const size_t Nd = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(Nd == 4);
  static const size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static const GaugeFieldKind k =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using precision = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::value_type;
  using GaugeFieldType = typename DGaugeFieldType::type;
  using FieldType = typename DeviceFieldType<Nd>::type;
  FieldType plaq_per_site(gauge_field.dimensions, complex_t(0.0, 0.0));

  GaugePlaq<Nd, Nc, precision, k> GPlaq(gauge_field, plaq_per_site,
                                        gauge_field.dimensions);

  real_t rtn = 0.0;
  auto policy = Policy<Nd>({0, 0, 0, 0}, gauge_field.dimensions);
  KTune::parallel_reduce(
      "get h (sp_max)", policy,
      KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2, size_t i3, real_t& local) {
        // GPlaq(i0, i1, i2, i3);
        real_t local_avg = 0.0;
        int tmp = 0;
        for (index_t mu = 0; mu < Nd; ++mu) {
          for (index_t nu = 0; nu < Nd; ++nu) {
            if (nu > mu) {
              local_avg += Kokkos::real(2 - GPlaq(mu, nu, i0, i1, i2, i3));
              tmp++;
            }
          }
        }
        local += local_avg / tmp;
      },
      rtn);
  Kokkos::fence();

  return rtn / gauge_field.field.size();
}
}  // namespace klft
