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

// Define the global types and views for Kokkos Lattice Field Theory (KLFT)
// This file contains the definitions for the types used in KLFT, including
// the real and complex types, gauge field types, and field view types.
// It also includes the definitions for the policies used in Kokkos parallel
// programming.

#pragma once
#include "GLOBAL.hpp"

// Implementation of Dirac Gamma Matrices
// The Idea is to initialise the gamma matrices once at the beginning as array
// and use it through out the simulation
namespace klft {
// using RepDim =size_t 4;
template <size_t RepDim>
struct GammaMat {
  Kokkos::Array<Kokkos::Array<complex_t, RepDim>, RepDim> matrix;

  GammaMat() = default;
  GammaMat(
      const Kokkos::Array<Kokkos::Array<complex_t, RepDim>, RepDim>& _mat) {
    matrix = _mat;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  complex_t& operator()(size_t i, size_t j) { return matrix[i][j]; }
  KOKKOS_FORCEINLINE_FUNCTION
  const complex_t& operator()(size_t i, size_t j) const { return matrix[i][j]; }
  KOKKOS_FORCEINLINE_FUNCTION
  GammaMat<RepDim> operator-(const GammaMat<RepDim>& b) const {
    GammaMat c{};
#pragma unroll
    for (size_t i = 0; i < RepDim; ++i) {
#pragma unroll
      for (size_t j = 0; j < RepDim; j++) {
        c(i, j) = matrix[i][j] - b(i, j);
      }
    }
    return c;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  GammaMat<RepDim> operator+(const GammaMat<RepDim>& b) const {
    GammaMat c{};
#pragma unroll
    for (size_t i = 0; i < RepDim; ++i) {
#pragma unroll
      for (size_t j = 0; j < RepDim; j++) {
        c(i, j) = matrix[i][j] + b(i, j);
      }
    }
    return c;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  GammaMat<RepDim> operator*(const GammaMat<RepDim>& b) const {
    GammaMat<RepDim> c;
#pragma unroll
    for (size_t i = 0; i < RepDim; ++i) {
#pragma unroll
      for (size_t j = 0; j < RepDim; ++j) {
        c(i, j) = matrix[i][0] * b(0, j);
#pragma unroll
        for (size_t k = 1; k < RepDim; ++k) {
          c(i, j) += matrix[i][k] * b(k, j);
        }
      }
    }
    return c;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  GammaMat<RepDim> operator*(const real_t& b) const {
    GammaMat<RepDim> c;
#pragma unroll
    for (size_t i = 0; i < RepDim; ++i) {
#pragma unroll
      for (size_t j = 0; j < RepDim; ++j) {
        c(i, j) = matrix[i][j] * b;
      }
    }
    return c;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  GammaMat<RepDim> operator*(const complex_t& b) const {
    GammaMat<RepDim> c;
#pragma unroll
    for (size_t i = 0; i < RepDim; ++i) {
#pragma unroll
      for (size_t j = 0; j < RepDim; ++j) {
        c(i, j) = matrix[i][j] * b;
      }
    }
    return c;
  }
  KOKKOS_FORCEINLINE_FUNCTION
  bool operator==(const GammaMat<RepDim>& b) { return matrix == b.matrix; }
  KOKKOS_FORCEINLINE_FUNCTION
  bool operator==(const GammaMat<RepDim>& b) const {
    return matrix == b.matrix;
  }
};

/* Following Gattinger:
           [0 0 0 -i
gamma_2 =   0 0 -i 0
            0 i 0 0
            i 0 0 0]

*/
// xdir
GammaMat<4> get_gamma0() {
  GammaMat<4> g{};
  g(0, 0) = complex_t(0, 0);
  g(0, 1) = complex_t(0, 0);
  g(0, 2) = complex_t(0, 0);
  g(0, 3) = complex_t(0, -1);
  g(1, 0) = complex_t(0, 0);
  g(1, 1) = complex_t(0, 0);
  g(1, 2) = complex_t(0, -1);
  g(1, 3) = complex_t(0, 0);
  g(2, 0) = complex_t(0, 0);
  g(2, 1) = complex_t(0, 1);
  g(2, 2) = complex_t(0, 0);
  g(2, 3) = complex_t(0, 0);
  g(3, 0) = complex_t(0, 1);
  g(3, 1) = complex_t(0, 0);
  g(3, 2) = complex_t(0, 0);
  g(3, 3) = complex_t(0, 0);
  return g;
}

/* Following Gattinger:
           [0 0 0 -1
gamma_2 =   0 0 1 0
            0 1 0 0
           -1 0 0 0]

*/
// y dir
GammaMat<4> get_gamma1() {
  GammaMat<4> g{};
  g(0, 0) = complex_t(0, 0);
  g(0, 1) = complex_t(0, 0);
  g(0, 2) = complex_t(0, 0);
  g(0, 3) = complex_t(-1, 0);
  g(1, 0) = complex_t(0, 0);
  g(1, 1) = complex_t(0, 0);
  g(1, 2) = complex_t(1, 0);
  g(1, 3) = complex_t(0, 0);
  g(2, 0) = complex_t(0, 0);
  g(2, 1) = complex_t(1, 0);
  g(2, 2) = complex_t(0, 0);
  g(2, 3) = complex_t(0, 0);
  g(3, 0) = complex_t(-1, 0);
  g(3, 1) = complex_t(0, 0);
  g(3, 2) = complex_t(0, 0);
  g(3, 3) = complex_t(0, 0);
  return g;
}

/* Following Gattinger:
           [0 0 -i 0
gamma_3 =   0 0 0 i
            i 0 0 0
            0 -i 0 0]

*/
// z  dir
GammaMat<4> get_gamma2() {
  GammaMat<4> g{};
  g(0, 0) = complex_t(0, 0);
  g(0, 1) = complex_t(0, 0);
  g(0, 2) = complex_t(0, -1);
  g(0, 3) = complex_t(0, 0);
  g(1, 0) = complex_t(0, 0);
  g(1, 1) = complex_t(0, 0);
  g(1, 2) = complex_t(0, 0);
  g(1, 3) = complex_t(0, 1);
  g(2, 0) = complex_t(0, 1);
  g(2, 1) = complex_t(0, 0);
  g(2, 2) = complex_t(0, 0);
  g(2, 3) = complex_t(0, 0);
  g(3, 0) = complex_t(0, 0);
  g(3, 1) = complex_t(0, -1);
  g(3, 2) = complex_t(0, 0);
  g(3, 3) = complex_t(0, 0);
  return g;
}
/* Following Gattinger:
           [0 0 1 0
gamma_4 =   0 0 0 1
            1 0 0 0
            0 1 0 0]

*/
// t dir
GammaMat<4> get_gamma3() {
  GammaMat<4> g{};
  g(0, 0) = complex_t(0, 0);
  g(0, 1) = complex_t(0, 0);
  g(0, 2) = complex_t(1, 0);
  g(0, 3) = complex_t(0, 0);
  g(1, 0) = complex_t(0, 0);
  g(1, 1) = complex_t(0, 0);
  g(1, 2) = complex_t(0, 0);
  g(1, 3) = complex_t(1, 0);
  g(2, 0) = complex_t(1, 0);
  g(2, 1) = complex_t(0, 0);
  g(2, 2) = complex_t(0, 0);
  g(2, 3) = complex_t(0, 0);
  g(3, 0) = complex_t(0, 0);
  g(3, 1) = complex_t(1, 0);
  g(3, 2) = complex_t(0, 0);
  g(3, 3) = complex_t(0, 0);
  return g;
}

GammaMat<4> get_gamma5() {
  GammaMat<4> g{};
  g(0, 0) = complex_t(1, 0);
  g(0, 1) = complex_t(0, 0);
  g(0, 2) = complex_t(0, 0);
  g(0, 3) = complex_t(0, 0);
  g(1, 0) = complex_t(0, 0);
  g(1, 1) = complex_t(1, 0);
  g(1, 2) = complex_t(0, 0);
  g(1, 3) = complex_t(0, 0);
  g(2, 0) = complex_t(0, 0);
  g(2, 1) = complex_t(0, 0);
  g(2, 2) = complex_t(-1, 0);
  g(2, 3) = complex_t(0, 0);
  g(3, 0) = complex_t(0, 0);
  g(3, 1) = complex_t(0, 0);
  g(3, 2) = complex_t(0, 0);
  g(3, 3) = complex_t(-1, 0);
  return g;
}
template <size_t RepDim>
const Kokkos::Array<GammaMat<RepDim>, 4> get_gammas() {
  Kokkos::Array<GammaMat<RepDim>, 4> c;
  c[0] = get_gamma0();
  c[1] = get_gamma1();
  c[2] = get_gamma2();
  c[3] = get_gamma3();
  return c;
}
template <size_t RepDim>
const GammaMat<RepDim> get_identity() {
  GammaMat<RepDim> c;
  for (size_t i = 0; i < RepDim; ++i) {
    for (size_t j = 0; j < RepDim; ++j) {
      c(i, j) = complex_t(0.0, 0.0);
      if (i == j) {
        c(i, j) = complex_t(1.0, 0.0);
      }
    }
  }
  return c;
}

template <size_t Nc, size_t RepDim, typename precision_t>
/// @brief Calculates the projection (1+ sign*gamma_dim)*spinor
/// @param dim
/// @param sign
/// @return result spinor
static constexpr KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim, precision_t>
project(size_t dim,
        index_t sign,
        const Spinor<Nc, RepDim, precision_t>& spinor) {
  constexpr auto id = get_identity<RepDim, precision_t>();

  switch (dim) {
    case 0:
      switch (sign) {
        case +1:
          return (id + get_gamma0()) * spinor;
          break;

        case -1:

          return (id - get_gamma0()) * spinor;
          break;
      }
      break;
    case 1:
      switch (sign) {
        case +1:
          return (id + get_gamma1()) * spinor;

          break;

        case -1:
          return (id - get_gamma1()) * spinor;
      }
      break;
    case 2:
      switch (sign) {
        case +1:
          return (id + get_gamma2()) * spinor;
          break;
        case -1:
          return (id - get_gamma2()) * spinor;
          break;
      }
      break;
    case 3:
      switch (sign) {
        case 1:
          return (id + get_gamma3()) * spinor;

          break;

        case -1:
          return (id - get_gamma3()) * spinor;

          break;
      }
      break;
  }
  return spinor;
}
template <size_t Nc, size_t RepDim, typename precision_t>
/// @brief Calculates the projection spinor*(1+ sign*gamma_dim)
/// @param dim
/// @param sign
/// @return result spinor
static constexpr KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim, precision_t>
project_alt(size_t dim,
            index_t sign,
            const Spinor<Nc, RepDim, precision_t>& spinor) {
  constexpr auto id = get_identity<RepDim>();

  switch (dim) {
    case 0:
      switch (sign) {
        case +1:
          return spinor * (id + get_gamma0());
          break;

        case -1:

          return spinor * (id - get_gamma0());
          break;
      }
      break;
    case 1:
      switch (sign) {
        case +1:
          return spinor * (id + get_gamma1());

          break;

        case -1:
          return spinor * (id - get_gamma1());
      }
      break;
    case 2:
      switch (sign) {
        case +1:
          return spinor * (id + get_gamma2());
          break;
        case -1:
          return spinor * (id - get_gamma2());
          break;
      }
      break;
    case 3:
      switch (sign) {
        case 1:
          return spinor * (id + get_gamma3());

          break;

        case -1:
          return spinor * (id - get_gamma3());

          break;
      }
      break;
  }
  return spinor;
}
template <size_t Nc, typename precision_t>
/// @brief Calculates the projection (1+ sign*gamma_dim)*spinor
/// @param dim
/// @param sign
/// @return result spinor
static constexpr KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, 2, precision_t>
project(size_t dim, index_t sign, const Spinor<Nc, 4, precision_t>& spinor) {
  Spinor<Nc, 2, precision_t> result;
  switch (dim) {
    case 0:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] - precision_t(0, 1) * spinor[3][i];
            result[1][i] = spinor[1][i] - precision_t(0, 1) * spinor[2][i];
          }
          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + precision_t(0, 1) * spinor[3][i];
            result[1][i] = spinor[1][i] + precision_t(0, 1) * spinor[2][i];
          }
          break;
      }
      break;
    case 1:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] - spinor[3][i];
            result[1][i] = spinor[1][i] + spinor[2][i];
          }

          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + spinor[3][i];
            result[1][i] = spinor[1][i] - spinor[2][i];
          }
      }
      break;
    case 2:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] - precision_t(0, 1) * spinor[2][i];
            result[1][i] = spinor[1][i] + precision_t(0, 1) * spinor[3][i];
          }
          break;
        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + precision_t(0, 1) * spinor[2][i];
            result[1][i] = spinor[1][i] - precision_t(0, 1) * spinor[3][i];
          }
          break;
      }
      break;
    case 3:
      switch (sign) {
        case 1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + spinor[2][i];
            result[1][i] = spinor[1][i] + spinor[3][i];
          }

          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] - spinor[2][i];
            result[1][i] = spinor[1][i] - spinor[3][i];
          }

          break;
      }
      break;
  }
  return result;
}
template <size_t Nc, typename precision_t>
/// @brief Calculates the projection spinor*(1+ sign*gamma_dim)
/// @param dim
/// @param sign
/// @return result spinor
static constexpr KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, 2, precision_t>
project_alt(size_t dim,
            index_t sign,
            const Spinor<Nc, 4, precision_t>& spinor) {
  Spinor<Nc, 2> result;
  switch (dim) {
    case 0:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + precision_t(0, 1) * spinor[3][i];
            result[1][i] = spinor[1][i] + precision_t(0, 1) * spinor[2][i];
          }
          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + precision_t(0, -1) * spinor[3][i];
            result[1][i] = spinor[1][i] + precision_t(0, -1) * spinor[2][i];
          }
          break;
      }
      break;
    case 1:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] - spinor[3][i];
            result[1][i] = spinor[1][i] + spinor[2][i];
          }

          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + spinor[3][i];
            result[1][i] = spinor[1][i] - spinor[2][i];
          }
      }
      break;
    case 2:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + precision_t(0, 1) * spinor[2][i];
            result[1][i] = spinor[1][i] + precision_t(0, -1) * spinor[3][i];
          }
          break;
        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + precision_t(0, -1) * spinor[2][i];
            result[1][i] = spinor[1][i] + precision_t(0, 1) * spinor[3][i];
          }
          break;
      }
      break;
    case 3:
      switch (sign) {
        case 1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] + spinor[2][i];
            result[1][i] = spinor[1][i] + spinor[3][i];
          }

          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i] - spinor[2][i];
            result[1][i] = spinor[1][i] - spinor[3][i];
          }

          break;
      }
      break;
  }
  return result;
}

template <size_t Nc, size_t RepDim, typename precision_t>
/// @brief Reconstruct the projection (1+ sign*gamma_dim)*spinor
/// @param dim
/// @param sign
/// @return result spinor
static constexpr KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim, precision_t>
reconstruct(size_t dim,
            index_t sign,
            const Spinor<Nc, RepDim, precision_t>& spinor) {
  return spinor;
}
template <size_t Nc, size_t RepDim, typename precision_t>
/// @brief Reconstruct the projection spinor*(1+ sign*gamma_dim)
/// @param dim
/// @param sign
/// @return result spinor
constexpr KOKKOS_FORCEINLINE_FUNCTION
    Spinor<Nc, RepDim, precision_t> static reconstruct_alt(
        size_t dim,
        index_t sign,
        const Spinor<Nc, RepDim, precision_t>& spinor) {
  return spinor;
}
template <size_t Nc, typename precision_t>
/// @brief Reconstruct the projection (1+ sign*gamma_dim)*spinor
/// @param dim
/// @param sign
/// @return result spinor
static constexpr KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, 4, precision_t>
reconstruct(size_t dim,
            index_t sign,
            const Spinor<Nc, 2, precision_t>& spinor) {
  Spinor<Nc, 4, precision_t> result;
  switch (dim) {
    case 0:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = precision_t(0, 1) * result[1][i];
            result[3][i] = precision_t(0, 1) * result[0][i];
          }
          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = precision_t(0, -1) * result[1][i];
            result[3][i] = precision_t(0, -1) * result[0][i];
          }
          break;
      }
      break;
    case 1:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = result[1][i];
            result[3][i] = -1 * result[0][i];
          }

          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = -1 * result[1][i];
            result[3][i] = result[0][i];
          }
      }
      break;
    case 2:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = precision_t(0, 1) * result[0][i];
            result[3][i] = precision_t(0, -1) * result[1][i];
          }
          break;
        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = precision_t(0, -1) * result[0][i];
            result[3][i] = precision_t(0, 1) * result[1][i];
          }
          break;
      }
      break;
    case 3:
      switch (sign) {
        case 1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = result[0][i];
            result[3][i] = result[1][i];
          }

          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = -1 * result[0][i];
            result[3][i] = -1 * result[1][i];
          }

          break;
      }
      break;
  }
  return result;
}
template <size_t Nc, typename precision_t>
/// @brief Reconstruct the projection spinor*(1+ sign*gamma_dim)
/// @param dim
/// @param sign
/// @return result spinor
constexpr KOKKOS_FORCEINLINE_FUNCTION
    Spinor<Nc, 4, precision_t> static reconstruct_alt(
        size_t dim,
        index_t sign,
        const Spinor<Nc, 2, precision_t>& spinor) {
  Spinor<Nc, 4> result;
  switch (dim) {
    case 0:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = precision_t(0, -1) * result[1][i];
            result[3][i] = precision_t(0, -1) * result[0][i];
          }
          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = precision_t(0, 1) * result[1][i];
            result[3][i] = precision_t(0, 1) * result[0][i];
          }
          break;
      }
      break;
    case 1:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = result[1][i];
            result[3][i] = -1 * result[0][i];
          }

          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = -1 * result[1][i];
            result[3][i] = result[0][i];
          }
      }
      break;
    case 2:
      switch (sign) {
        case +1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = -precision_t(0, 1) * result[0][i];
            result[3][i] = precision_t(0, 1) * result[1][i];
          }
          break;
        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = precision_t(0, 1) * result[0][i];
            result[3][i] = -precision_t(0, 1) * result[1][i];
          }
          break;
      }
      break;
    case 3:
      switch (sign) {
        case 1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = result[0][i];
            result[3][i] = result[1][i];
          }

          break;

        case -1:
#pragma unroll
          for (size_t i = 0; i < Nc; i++) {
            result[0][i] = spinor[0][i];
            result[1][i] = spinor[1][i];
            result[2][i] = -1 * result[0][i];
            result[3][i] = -1 * result[1][i];
          }

          break;
      }
      break;
  }
  return result;
}
template <size_t Nc, size_t RepDim, typename precision_t>
constexpr KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim, precision_t> gamma5(
    const Spinor<Nc, RepDim, precision_t>& spinor) {
  Spinor<Nc, RepDim, precision_t> result;
#pragma unroll
  for (size_t i = 0; i < Nc; i++) {
    result[0][i] = spinor[0][i];
    result[1][i] = spinor[1][i];
    result[2][i] = -spinor[2][i];
    result[3][i] = -spinor[3][i];
  }
  return result;
}
}  // namespace klft
