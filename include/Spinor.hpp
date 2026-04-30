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

// define Spinor operations

#pragma once
#include "GLOBAL.hpp"
#include "GammaMatrix.hpp"

namespace klft {
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator*(
    const SUN<Nc, precision_t>& U,
    const Spinor<Nc, Nd, precision_t>& spinor) {
  Spinor<Nc, Nd, precision_t> res{};
#pragma unroll
  for (size_t k = 0; k < Nd; k++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
#pragma unroll
      for (size_t j = 0; j < Nc; j++) {
        res[k][i] += U[i][j] * spinor[k][j];
      }
    }
  }
  return res;
}
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator*(
    const Spinor<Nc, Nd, precision_t>& spinor,
    const SUN<Nc, precision_t>& U) {
  Spinor<Nc, Nd, precision_t> res{};

#pragma unroll
  for (size_t k = 0; k < Nd; ++k) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
#pragma unroll
      for (size_t i = 0; i < Nc; ++i) {
        res[k][i] += spinor[k][j] * U[j][i];
      }
    }
  }

  return res;
}

// *= makes no sense f spinor gauge link

template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator*(
    const complex_t& scalar,
    const Spinor<Nc, Nd, precision_t>& spinor) {
  Spinor<Nc, Nd, precision_t> res;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      res[j][i] = spinor[j][i] * scalar;
    }
  }
  return res;
}
// this is for construction of the force matrix, no implicit conjugation,
// however this would be better for performance
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION SUN<Nc, precision_t> operator*(
    const Spinor<Nc, Nd, precision_t>& a,
    const Spinor<Nc, Nd, precision_t>& b) {
  SUN<Nc, precision_t> res{};
#pragma unroll
  for (size_t k = 0; k < Nd; ++k) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
#pragma unroll
      for (size_t j = 0; j < Nc; ++j) {
        res[i][j] += a[k][i] * (b[k][j]);
      }
    }
  }
  return res;
}

template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator*=(
    Spinor<Nc, Nd, precision_t>& spinor,
    const complex_t& scalar) {
  Spinor<Nc, Nd, precision_t> res = scalar * spinor;
  spinor = res;
  return spinor;
}

template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator*(
    const real_t& scalar,
    const Spinor<Nc, Nd, precision_t>& spinor) {
  Spinor<Nc, Nd, precision_t> res;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      res[j][i] = scalar * spinor[j][i];
    }
  }
  return res;
}
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator*=(
    Spinor<Nc, Nd, precision_t>& spinor,
    const real_t& scalar) {
  Spinor<Nc, Nd, precision_t> res = scalar * spinor;
  spinor = res;
  return spinor;
}

template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator+(
    const Spinor<Nc, Nd, precision_t>& spinor1,
    const Spinor<Nc, Nd, precision_t>& spinor2) {
  Spinor<Nc, Nd, precision_t> res;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      res[j][i] = spinor2[j][i] + spinor1[j][i];
    }
  }
  return res;
}
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator+=(
    Spinor<Nc, Nd, precision_t>& spinor1,
    const Spinor<Nc, Nd, precision_t>& spinor2) {
  Spinor<Nc, Nd, precision_t> res = spinor1 + spinor2;
  spinor1 = res;
  return spinor1;
}

template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> axpy(
    const complex_t& alpha,
    const Spinor<Nc, Nd, precision_t>& spinor1,
    const Spinor<Nc, Nd, precision_t>&
        spinor2) {  // returns alpha*spinor1 + spinor2
  Spinor<Nc, Nd, precision_t> res;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      res[j][i] = spinor2[j][i] + alpha * spinor1[j][i];
    }
  }
  return res;
}

template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION void axpy(
    const precision_t& alpha,
    const Spinor<Nc, Nd, precision_t>& spinor1,
    const Spinor<Nc, Nd, precision_t>& spinor2,
    Spinor<Nc, Nd, precision_t>& res) {
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      res[j][i] = spinor2[j][i] + alpha * spinor1[j][i];
    }
  }
}

template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator-(
    const Spinor<Nc, Nd, precision_t>& spinor1,
    const Spinor<Nc, Nd, precision_t>& spinor2) {
  Spinor<Nc, Nd, precision_t> res;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      res[j][i] = spinor1[j][i] - spinor2[j][i];
    }
  }
  return res;
}
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator-=(
    Spinor<Nc, Nd, precision_t>& spinor1,
    const Spinor<Nc, Nd, precision_t>& spinor2) {
  Spinor<Nc, Nd, precision_t> res = spinor1 - spinor2;
  spinor1 = res;
  return spinor1;
}
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION real_t
sqnorm(const Spinor<Nc, Nd, precision_t>& spinor) {
  real_t res = 0;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      if constexpr (std::is_same_v<precision_t, Kokkos::complex<real_t>>) {
        res += static_cast<real_t>(spinor[j][i].imag()) *
               static_cast<real_t>(spinor[j][i].imag());
        res += static_cast<real_t>(spinor[j][i].real()) *
               static_cast<real_t>(spinor[j][i].real());
      } else {
        res += spinor[j][i].imag() * spinor[j][i].imag() +
               spinor[j][i].real() * spinor[j][i].real();
      }
    }
  }
  return res;
}

// Define Gamma Spinor interaction
// Dirac index and gamma matrix have to have the same dimension

// This is ineficnet because of the sparsity of the gamma matrices
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator*(
    const GammaMat<Nd>& matrix,
    const Spinor<Nc, Nd, precision_t>& spinor) {
  Spinor<Nc, Nd, precision_t> c;
#pragma unroll
  for (size_t j = 0; j < Nd; j++) {
#pragma unroll
    for (size_t i = 0; i < Nc; i++) {
      precision_t val = 0.0;
#pragma unroll
      for (size_t k = 0; k < Nd; k++) {
        val += matrix(j, k) * spinor[k][i];
      }
      c[j][i] = val;
    }
  }
  return c;
}

template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> operator*(
    const Spinor<Nc, Nd, precision_t>& spinor,
    const GammaMat<Nd>& matrix) {
  Spinor<Nc, Nd, precision_t> c;
#pragma unroll
  for (size_t j = 0; j < Nd; ++j) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
      precision_t val = 0.0;
#pragma unroll
      for (size_t k = 0; k < Nd; ++k) {
        val += spinor[k][i] * matrix(k, j);
      }
      c[j][i] = val;
    }
  }
  return c;
}

// Random generation of Spinors
template <size_t Nc, size_t Nd, typename precision_t, class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSpinor(
    Spinor<Nc, Nd, precision_t>& r,
    RNG& generator,
    const typename precision_t::value_type& mean,
    const typename precision_t::value_type& var) {
#pragma unroll
  for (size_t j = 0; j < Nd; ++j) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
      r[j][i] =
          precision_t(generator.normal(mean, var), generator.normal(mean, var));
    }
  }
}

// calculate a^\dagger b
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION complex_t
spinor_inner_prod(const Spinor<Nc, Nd, precision_t>& a,
                  const Spinor<Nc, Nd, precision_t>& b) {
  complex_t res(0.0, 0.0);
#pragma unroll
  for (size_t j = 0; j < Nd; ++j) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
      if (std::is_same_v<precision_t, Kokkos::complex<real_t>>) {
        complex_t a_val(static_cast<real_t>(a[j][i].real()),
                        static_cast<real_t>(a[j][i].imag()));
        complex_t b_val(static_cast<real_t>(b[j][i].real()),
                        static_cast<real_t>(b[j][i].imag()));

        auto value = conj(a_val) * b_val;
        res += value;
      } else {
        auto value = conj(a[j][i]) * b[j][i];
        res.imag() += value.imag();
        res.real() += value.real();
      }
    }
  }
  return res;
}
template <size_t Nc, size_t Nd, typename precision_t>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> conj(
    const Spinor<Nc, Nd, precision_t>& a) {
  Spinor<Nc, Nd, precision_t> res;
#pragma unroll
  for (size_t j = 0; j < Nd; ++j) {
#pragma unroll
    for (size_t i = 0; i < Nc; ++i) {
      res[j][i] = conj(a[j][i]);
    }
  }
  return res;
}

template <size_t Nc, size_t Nd, typename precision_t = complex_t>
KOKKOS_INLINE_FUNCTION Spinor<Nc, Nd, precision_t> deltaSpinor(index_t i) {
  KOKKOS_ASSERT(i < Nc * Nd);
  KOKKOS_ASSERT(i >= 0);
  Spinor<Nc, Nd, precision_t> a;
  index_t dirac = i / Nc;
  index_t color = i % Nc;
  a[dirac][color] = 1;
  return a;
}
template <size_t Nc, size_t Nd, typename precision_t>
void print_spinor_int(const Spinor<Nc, Nd, precision_t>& s,
                      const char* name = "Spinor") {
  printf("%s:\n", name);
  for (size_t d = 0; d < Nd; ++d) {
    printf("  Spin %zu:\n", d);
    for (size_t c = 0; c < Nc; ++c) {
      double re = s[d][c].real();
      double im = s[d][c].imag();
      printf("    [%zu] = (% .20f, % .20f i)\n", c, re, im);
    }
  }
}
}  // namespace klft
