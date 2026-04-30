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
#include <unistd.h>

#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"

namespace klft {

// TODO: move these operators somewhere more appropriate

template <typename T, size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<T, N>, N> operator-(
    const Kokkos::Array<Kokkos::Array<T, N>, N>& a) {
  Kokkos::Array<Kokkos::Array<T, N>, N> c;
#pragma unroll
  for (size_t i = 0; i < N; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      c[i][j] = -a[i][j];
    }
  }
  return c;
}

// get the imaginary parts of an SUN matrix
template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<real_t, N>, N> imag(
    const SUN<N>& in) {
  Kokkos::Array<Kokkos::Array<real_t, N>, N> out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      out[i][j] = in[i][j].imag();
    }
  }
  return out;
}

// get the real parts of an SUN matrix
template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<real_t, N>, N> real(
    const SUN<N>& in) {
  Kokkos::Array<Kokkos::Array<real_t, N>, N> out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      out[i][j] = in[i][j].real();
    }
  }
  return out;
}

template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<real_t, N>, N> imag(
    const Kokkos::Array<Kokkos::Array<complex_t, N>, N>& in) {
  Kokkos::Array<Kokkos::Array<real_t, N>, N> out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      out[i][j] = in[i][j].imag();
    }
  }
  return out;
}

// get the real parts of an SUN matrix
template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<real_t, N>, N> real(
    const Kokkos::Array<Kokkos::Array<complex_t, N>, N>& in) {
  Kokkos::Array<Kokkos::Array<real_t, N>, N> out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      out[i][j] = in[i][j].real();
    }
  }
  return out;
}

// return the trace of a RealMatrix
template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION real_t
trace(const Kokkos::Array<Kokkos::Array<real_t, N>, N>& in) {
  real_t out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
    out += in[i][i];
  }
  return out;
}

template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION complex_t
trace(const Kokkos::Array<Kokkos::Array<complex_t, N>, N>& in) {
  complex_t out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
    out += in[i][i];
  }
  return out;
}

template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION real_t
retrace(const Kokkos::Array<Kokkos::Array<complex_t, N>, N>& in) {
  complex_t out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
    out += in[i][i];
  }
  return out.real();
}

template <size_t N>
KOKKOS_FORCEINLINE_FUNCTION real_t
imtrace(const Kokkos::Array<Kokkos::Array<complex_t, N>, N>& in) {
  complex_t out{0};
#pragma unroll
  for (int i = 0; i < N; ++i) {
    out += in[i][i];
  }
  return out.imag();
}

template <typename DGaugeFieldType>
struct FieldStrengthTensor {
  // this kernel is defined for rank = Nd
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  static_assert(
      Nd == 4,
      "FieldStrengthTensor is only defined for 4D gauge fields. (for now)");
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  // define the gauge field type
  using GaugeFieldType = typename DGaugeFieldType::type;
  const GaugeFieldType g_in;
  // define the field type
  using FieldType = typename DeviceScalarFieldType<Nd>::type;

  struct CloverDef {};
  struct RectangleDef {};

  using RealMatrix = Kokkos::Array<Kokkos::Array<real_t, Nc>, Nc>;
  using ComplexMatrix = Kokkos::Array<Kokkos::Array<complex_t, Nc>, Nc>;

  // define the dimensions of the given Field
  const IndexArray<Nd> dimensions;

  FieldStrengthTensor(const GaugeFieldType g_in)
      : g_in(g_in), dimensions(g_in.dimensions) {}

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc> operator()(CloverDef,
                                                    const indexType i0,
                                                    const indexType i1,
                                                    const indexType i2,
                                                    const indexType i3,
                                                    index_t mu,
                                                    index_t nu) const {
    // implemented according to
    // https://doi.org/10.1140/epjc/s10052-020-7984-9 (21)

    SUN<Nc> P_munu = zeroSUN<Nc>();
    const IndexArray<Nd> x{static_cast<index_t>(i0), static_cast<index_t>(i1),
                           static_cast<index_t>(i2), static_cast<index_t>(i3)};

    // Helper lambda for modular arithmetic
    auto mod = [&](index_t s, index_t dir) {
      return (s + this->dimensions[dir]) % this->dimensions[dir];
    };

    // 1. Plaquette in (+mu, +nu) plane starting at x
    IndexArray<Nd> x_p_mu = x;
    x_p_mu[mu] = (x_p_mu[mu] + 1) % dimensions[mu];
    IndexArray<Nd> x_p_nu = x;
    x_p_nu[nu] = (x_p_nu[nu] + 1) % dimensions[nu];
    P_munu += g_in(x, mu) * g_in(x_p_mu, nu) * conj(g_in(x_p_nu, mu)) *
              conj(g_in(x, nu));

    // 2. Plaquette in (+mu, -nu) plane starting at x
    IndexArray<Nd> x_p_mu_m_nu = x_p_mu;
    x_p_mu_m_nu[nu] = mod(x_p_mu_m_nu[nu] - 1, nu);
    IndexArray<Nd> x_m_nu = x;
    x_m_nu[nu] = mod(x[nu] - 1, nu);
    P_munu += conj(g_in(x_m_nu, nu)) * g_in(x_m_nu, mu) *
              g_in(x_p_mu_m_nu, nu) * conj(g_in(x, mu));

    // 3. Plaquette in (-mu, +nu) plane starting at x
    IndexArray<Nd> x_m_mu = x;
    x_m_mu[mu] = mod(x_m_mu[mu] - 1, mu);
    IndexArray<Nd> x_m_mu_p_nu = x_m_mu;
    x_m_mu_p_nu[nu] = (x_m_mu_p_nu[nu] + 1) % dimensions[nu];
    P_munu += g_in(x, nu) * conj(g_in(x_m_mu_p_nu, mu)) *
              conj(g_in(x_m_mu, nu)) * g_in(x_m_mu, mu);

    // 4. Plaquette in (-mu, -nu) plane starting at x
    IndexArray<Nd> x_m_mu_m_nu = x_m_mu;
    x_m_mu_m_nu[nu] = mod(x_m_mu_m_nu[nu] - 1, nu);
    P_munu += conj(g_in(x_m_mu, mu)) * conj(g_in(x_m_mu_m_nu, nu)) *
              g_in(x_m_mu_m_nu, mu) * g_in(x_m_nu, nu);

    return traceT(P_munu) * 0.25;
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc> operator()(RectangleDef,
                                                    const indexType i0,
                                                    const indexType i1,
                                                    const indexType i2,
                                                    const indexType i3,
                                                    index_t mu,
                                                    index_t nu) const {
    // implemented according to https://doi.org/10.1140/epjc/s10052-020-7984-9
    // (24)

    SUN<Nc> P_munu = zeroSUN<Nc>();
    const IndexArray<Nd> x{static_cast<index_t>(i0), static_cast<index_t>(i1),
                           static_cast<index_t>(i2), static_cast<index_t>(i3)};

    // Helper lambda for modular arithmetic
    auto mod = [&](index_t s, index_t dir) {
      return (s + this->dimensions[dir]) % this->dimensions[dir];
    };

    // 1. 1x2 Plaquette in (+mu, +nu) plane starting at x
    IndexArray<Nd> x_p_mu = x;
    IndexArray<Nd> x_p_mu_p_nu = x;
    IndexArray<Nd> x_p_nu = x;
    IndexArray<Nd> x_p_2nu = x;
    IndexArray<Nd> x_p_2mu = x;
    x_p_2mu[mu] = (x_p_2mu[mu] + 2) % dimensions[mu];
    x_p_2nu[nu] = (x_p_2nu[nu] + 2) % dimensions[nu];
    x_p_mu[mu] = (x_p_mu[mu] + 1) % dimensions[mu];
    x_p_mu_p_nu[mu] = (x_p_mu_p_nu[mu] + 1) % dimensions[mu];
    x_p_mu_p_nu[nu] = (x_p_mu_p_nu[nu] + 1) % dimensions[nu];
    x_p_nu[nu] = (x_p_nu[nu] + 1) % dimensions[nu];

    P_munu += g_in(x, mu) * g_in(x_p_mu, nu) * g_in(x_p_mu_p_nu, nu) *
              conj(g_in(x_p_2nu, mu)) * conj(g_in(x_p_nu, nu)) *
              conj(g_in(x, nu));

    // 1. 2x1 Plaquette in (+mu, +nu) plane starting at x
    P_munu += g_in(x, mu) * g_in(x_p_mu, mu) * g_in(x_p_2mu, nu) *
              conj(g_in(x_p_mu_p_nu, mu)) * conj(g_in(x_p_nu, mu)) *
              conj(g_in(x, nu));

    // // 2. 1x2 Plaquette in (+mu, -nu) plane starting at x
    IndexArray<Nd> x_p_2mu_m_nu = x_p_2mu;
    IndexArray<Nd> x_p_mu_m_nu = x_p_mu;
    IndexArray<Nd> x_m_nu = x;
    IndexArray<Nd> x_m_2nu = x;
    IndexArray<Nd> x_p_mu_m_2nu = x_p_mu;
    x_p_mu_m_2nu[nu] = mod(x_p_mu_m_2nu[nu] - 2, nu);
    x_p_2mu_m_nu[nu] = mod(x_p_2mu_m_nu[nu] - 1, nu);
    x_p_mu_m_nu[nu] = mod(x_p_mu_m_nu[nu] - 1, nu);
    x_m_nu[nu] = mod(x[nu] - 1, nu);
    x_m_2nu[nu] = mod(x[nu] - 2, nu);

    P_munu += conj(g_in(x_m_nu, nu)) * conj(g_in(x_m_2nu, nu)) *
              g_in(x_m_2nu, mu) * g_in(x_p_mu_m_2nu, nu) *
              g_in(x_p_mu_m_nu, nu) * conj(g_in(x, mu));

    // // 2. 2x1 Plaquette in (+mu, -nu) plane starting at x
    P_munu += conj(g_in(x_m_nu, nu)) * g_in(x_m_nu, mu) *
              g_in(x_p_mu_m_nu, mu) * g_in(x_p_2mu_m_nu, nu) *
              conj(g_in(x_p_mu, mu)) * conj(g_in(x, mu));

    // // 3. 1x2 Plaquette in (-mu, +nu) plane starting at x
    IndexArray<Nd> x_m_mu = x;
    x_m_mu[mu] = mod(x_m_mu[mu] - 1, mu);
    IndexArray<Nd> x_m_2mu = x;
    x_m_2mu[mu] = mod(x_m_2mu[mu] - 2, mu);
    IndexArray<Nd> x_m_mu_p_nu = x_m_mu;
    IndexArray<Nd> x_m_2mu_p_nu = x_m_2mu;
    x_m_mu_p_nu[nu] = mod(x_m_mu_p_nu[nu] + 1, nu);
    x_m_2mu_p_nu[nu] = mod(x_m_2mu_p_nu[nu] + 1, nu);
    P_munu += g_in(x, nu) * conj(g_in(x_m_mu_p_nu, mu)) *
              conj(g_in(x_m_2mu_p_nu, mu)) * conj(g_in(x_m_2mu, nu)) *
              g_in(x_m_2mu, mu) * g_in(x_m_mu, mu);
    //
    // // 3. 2x1 Plaquette in (-mu, +nu) plane starting at x
    IndexArray<Nd> x_m_mu_p_2nu = x_m_mu;
    x_m_mu_p_2nu[nu] = (x_m_mu_p_2nu[nu] + 2) % dimensions[nu];
    P_munu += g_in(x, nu) * g_in(x_p_nu, nu) * conj(g_in(x_m_mu_p_2nu, mu)) *
              conj(g_in(x_m_mu_p_nu, nu)) * conj(g_in(x_m_mu, nu)) *
              g_in(x_m_mu, mu);

    // // 4. 1x2 Plaquette in (-mu, -nu) plane starting at x
    IndexArray<Nd> x_m_2mu_m_nu = x_m_2mu;
    IndexArray<Nd> x_m_mu_m_nu = x_m_mu;
    IndexArray<Nd> x_m_mu_m_2nu = x_m_mu;
    x_m_mu_m_2nu[nu] = mod(x_m_mu_m_2nu[nu] - 2, nu);
    x_m_2mu_m_nu[nu] = mod(x_m_2mu_m_nu[nu] - 1, nu);
    x_m_mu_m_nu[nu] = mod(x_m_mu_m_nu[nu] - 1, nu);
    P_munu += conj(g_in(x_m_mu, mu)) * conj(g_in(x_m_mu_m_nu, nu)) *
              conj(g_in(x_m_mu_m_2nu, nu)) * g_in(x_m_mu_m_2nu, mu) *
              g_in(x_m_2nu, nu) * g_in(x_m_nu, nu);

    // 4. 1x2 Plaquette in (-mu, -nu) plane starting at x
    P_munu += conj(g_in(x_m_mu, mu)) * conj(g_in(x_m_2mu, mu)) *
              conj(g_in(x_m_2mu_m_nu, nu)) * g_in(x_m_2mu_m_nu, mu) *
              g_in(x_m_mu_m_nu, mu) * g_in(x_m_nu, nu);

    return traceT(P_munu) * 0.125;
  }
};

}  // namespace klft
