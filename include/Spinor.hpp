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
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const SUN<Nc> &U, const Spinor<Nc, Nd> &spinor) {
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t i = 0; i < Nc; i++) {
#pragma unroll
    for (size_t j = 0; j < Nc; j++) {
#pragma unroll
      for (size_t k = 0; k < Nd; k++) {
        res[i][k] += U[i][j] * spinor[j][k];
      }
    }
  }
  return res;
}
// *= makes no sense f spinor gauge link

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const complex_t &scalar, const Spinor<Nc, Nd> &spinor) {
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t i = 0; i < Nc; i++) {
#pragma unroll
    for (size_t j = 0; j < Nd; j++) {
      res[i, j] *= scalar;
    }
  }
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*=(
    const Spinor<Nc, Nd> &spinor, const complex_t &scalar) {
  Spinor<Nc, Nd> res = scalar * spinor;
  spinor = res;
  return spinor;
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const real_t &scalar, const Spinor<Nc, Nd> &spinor) {
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t i = 0; i < Nc; i++) {
#pragma unroll
    for (size_t j = 0; j < Nd; j++) {
      res[i, j] *= scalar;
    }
  }
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*=(
    const Spinor<Nc, Nd> &spinor, const real_t &scalar) {
  Spinor<Nc, Nd> res = scalar * spinor;
  spinor = res;
  return spinor;
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator+(
    const Spinor<Nc, Nd> &spinor1, const Spinor<Nc, Nd> &spinor2) {
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t i = 0; i < Nc; i++) {
#pragma unroll
    for (size_t j = 0; j < Nd; j++) {
      res[i, j] += scalar;
    }
  }
  return res;
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator+=(
    const Spinor<Nc, Nd> &spinor1, const Spinor<Nc, Nd> &spinor2) {
  Spinor<Nc, Nd> res = spinor1 + spinor2;
  spinor1 = res;
  return spinor1;
}

template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator-(
    const Spinor<Nc, Nd> &spinor1, const Spinor<Nc, Nd> &spinor2) {
  Spinor<Nc, Nd> res;
#pragma unroll
  for (size_t i = 0; i < Nc; i++) {
#pragma unroll
    for (size_t j = 0; j < Nd; j++) {
      res[i, j] -= scalar;
    }
  }
  return res;
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator-=(
    const Spinor<Nc, Nd> &spinor1, const Spinor<Nc, Nd> &spinor2) {
  Spinor<Nc, Nd> res = spinor1 - spinor2;
  spinor1 = res;
  return spinor1;
}
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION real_t sqnorm(const Spinor<Nc, Nd> &spinor) {
  real_t res = 0;
#pragma unroll
  for (size_t i = 0; i < Nc; i++) {
#pragma unroll
    for (size_t j = 0; j < Nd; j++) {
      res += spinor[i, j].imag() * spinor[i, j].imag() +
             spinor[i, j].real() * spinor[i, j].real();
    }
  }
  return res;
}

// Define Gamma Spinor interaction
// Dirac index and gamma matrix have to have the same dimension

// This is ineficnet because of the sparsity of the gamma matrices
template <size_t Nc, size_t Nd>
KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> operator*(
    const GammaMat<Nd> &matrix, const Spinor<Nc, Nd> &spinor) {
  Spinor<nc, Nd> c;
#pragma unroll
  for (size_t i = 0; i < Nc; i++) {
#pragma unroll
    for (size_t j = 0; j < Nd; j++) {
#pragma unroll
      for (size_t k = 0; k < Nd; k++) {
        c[i][j] += matrix(j, k) * spinor[i][k];
      }
    }
  }
  return c;
}

}  // namespace klft
