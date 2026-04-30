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
#include <mpi.h>

#include <KTune/KTune.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <type_traits>
#ifndef STREAM_NTIMES
#define STREAM_NTIMES 20
#endif
#ifdef ENABLE_DEBUG
#include <iostream>
#define DEBUG_LOG(msg)             \
  do {                             \
    std::cout << msg << std::endl; \
  } while (0)
#else
#define DEBUG_LOG(msg) \
  do {                 \
  } while (0)
#endif

#ifdef DEBUG_MPI
#include <mpi.h>
#include <stdio.h>

#define DEBUG_MPI_PRINT(...)                                             \
  do {                                                                   \
    int _rank;                                                           \
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);                               \
    fprintf(stderr, "[Rank %d] %s:%d (%s): ", _rank, __FILE__, __LINE__, \
            __func__);                                                   \
    fprintf(stderr, __VA_ARGS__);                                        \
    fprintf(stderr, "\n");                                               \
    fflush(stderr);                                                      \
  } while (0)
#else
#define DEBUG_MPI_PRINT(...) \
  do {                       \
  } while (0)
#endif

namespace klft {

// define the global types
// is this a good idea?
// if we want different fields with different precision (e.g. coarse grid)
// might be a lot of work to redefine, but on the other hand, if we want to
// be precision agnostic, everything has to be templated with given precision
// let's start like this and worry about mixed precision later
using real_t = double;
using single_t = float;

constexpr const real_t REAL_T_EPSILON = std::numeric_limits<real_t>::epsilon();
constexpr const real_t PI = Kokkos::numbers::pi_v<real_t>;

// use int for the index type
using index_t = int;

using complex_t = Kokkos::complex<real_t>;
using complexsingle_t = Kokkos::complex<single_t>;
// define index_arrays
template <size_t rank>
using IndexArray = Kokkos::Array<index_t, rank>;

// maybe these should be somewhere else
//  element‐wise addition
template <size_t rank>
KOKKOS_INLINE_FUNCTION IndexArray<rank> operator+(IndexArray<rank> const& a,
                                                  IndexArray<rank> const& b) {
  IndexArray<rank> c;
  for (size_t i = 0; i < rank; ++i)
    c[i] = a[i] + b[i];
  return c;
}

// element‐wise subtraction
template <size_t rank>
KOKKOS_INLINE_FUNCTION IndexArray<rank> operator-(IndexArray<rank> const& a,
                                                  IndexArray<rank> const& b) {
  IndexArray<rank> c;
  for (size_t i = 0; i < rank; ++i)
    c[i] = a[i] - b[i];
  return c;
}

// element‐wise modulo (array % array)
template <size_t rank>
KOKKOS_INLINE_FUNCTION IndexArray<rank> operator%(IndexArray<rank> const& a,
                                                  IndexArray<rank> const& b) {
  IndexArray<rank> c;
  for (size_t i = 0; i < rank; ++i)
    c[i] = a[i] % b[i];
  return c;
}

// optionally: array % scalar
template <size_t rank>
KOKKOS_INLINE_FUNCTION IndexArray<rank> operator%(IndexArray<rank> const& a,
                                                  index_t m) {
  IndexArray<rank> c;
  for (size_t i = 0; i < rank; ++i)
    c[i] = a[i] % m;
  return c;
}

// and scalar % array
template <size_t rank>
KOKKOS_INLINE_FUNCTION IndexArray<rank> operator%(index_t m,
                                                  IndexArray<rank> const& a) {
  IndexArray<rank> c;
  for (size_t i = 0; i < rank; ++i)
    c[i] = m % a[i];
  return c;
}

// define groups for gauge fields
template <typename T>
struct Wrapper {
  T data;

  // Implicit conversion to T&
  KOKKOS_INLINE_FUNCTION
  operator T&() { return data; }

  KOKKOS_INLINE_FUNCTION
  operator const T&() const { return data; }

  // Optional: pointer-style access (if T is a View or Array)
  KOKKOS_INLINE_FUNCTION
  auto operator->() { return &data; }

  KOKKOS_INLINE_FUNCTION
  auto operator->() const { return &data; }

  // Add custom operations
  // KOKKOS_INLINE_FUNCTION
  // Wrapper operator+(const Wrapper& other) const {
  //     Wrapper result;
  //     result.data = this->data + other.data; // requires T supports +
  //     return result;
  // }

  // operator[] forwarding
  template <typename Index>
  constexpr KOKKOS_INLINE_FUNCTION auto& operator[](Index i) {
    return data[i];
  }

  template <typename Index>
  KOKKOS_INLINE_FUNCTION constexpr const auto& operator[](Index i) const {
    return data[i];
  }

  // Optional: operator() forwarding (for Views)
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto operator()(Indices... indices)
      -> decltype(data(indices...)) {
    return data(indices...);
  }

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto operator()(Indices... indices) const
      -> decltype(data(indices...)) {
    return data(indices...);
  }
};

template <size_t Nc, typename presicion = complex_t>
// using SUN = Kokkos::Array<Kokkos::Array<complex_t, Nc>, Nc>;

using SUN = Wrapper<Kokkos::Array<Kokkos::Array<presicion, Nc>, Nc>>;

// define Spinor Type
// info correct dispatch is only guaranteed for    Nd != Nc ! -> Conflicts
// with SUN.hpp version Maybe via class to make it safe
template <typename T>
struct WrapperSpinor {
  T data;

  // Implicit conversion to T&
  KOKKOS_INLINE_FUNCTION
  operator T&() { return data; }

  KOKKOS_INLINE_FUNCTION
  operator const T&() const { return data; }

  // Optional: pointer-style access (if T is a View or Array)
  KOKKOS_INLINE_FUNCTION
  auto operator->() { return &data; }

  KOKKOS_INLINE_FUNCTION
  auto operator->() const { return &data; }

  // Add custom operations
  // KOKKOS_INLINE_FUNCTION
  // Wrapper operator+(const Wrapper& other) const {
  //     Wrapper result;
  //     result.data = this->data + other.data; // requires T supports +
  //     return result;
  // }

  // operator[] forwarding
  template <typename Index>
  constexpr KOKKOS_INLINE_FUNCTION auto& operator[](Index i) {
    return data[i];
  }

  template <typename Index>
  KOKKOS_INLINE_FUNCTION constexpr const auto& operator[](Index i) const {
    return data[i];
  }

  // Optional: operator() forwarding (for Views)
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto operator()(Indices... indices)
      -> decltype(data(indices...)) {
    return data(indices...);
  }

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto operator()(Indices... indices) const
      -> decltype(data(indices...)) {
    return data(indices...);
  }
};
template <size_t Nc, size_t Nd, typename precision = complex_t>
using Spinor = WrapperSpinor<Kokkos::Array<Kokkos::Array<precision, Nc>, Nd>>;
template <size_t Nc, size_t RepDim>
using PropagatorMatrix =
    Kokkos::Array<Kokkos::Array<complex_t, RepDim * Nc>, RepDim * Nc>;
// define field view types
// by default all views are 4D
// some dimensions are set to 1 for lower dimensions
// I'm still not sure if this is the best way to do it
// Nd here is templated, but for a 4D gauge field,
// shouldn't Nd always be 4?
// Nc is the number of colors
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using SpinorField = Kokkos::View<Spinor<Nc, RepDim, precision>****,
                                 Kokkos::MemoryTraits<Kokkos::Restrict>>;
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using SpinorField3D = Kokkos::View<Spinor<Nc, RepDim, precision>***,
                                   Kokkos::MemoryTraits<Kokkos::Restrict>>;
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using SpinorField2D = Kokkos::View<Spinor<Nc, RepDim, precision>**,
                                   Kokkos::MemoryTraits<Kokkos::Restrict>>;

// define adjoint groups
template <size_t Nc>
constexpr size_t NcAdj = (Nc * Nc > 1) ? Nc * Nc - 1 : 1;

template <size_t Nc>
struct SUNAdj {
  Kokkos::Array<real_t, NcAdj<Nc>> data;

  KOKKOS_INLINE_FUNCTION
  auto operator->() { return &data; }

  KOKKOS_INLINE_FUNCTION
  auto operator->() const { return &data; }

  // operator[] forwarding
  template <typename Index>
  KOKKOS_INLINE_FUNCTION constexpr auto& operator[](Index i) {
    return data[i];
  }

  template <typename Index>
  KOKKOS_INLINE_FUNCTION constexpr const auto& operator[](Index i) const {
    return data[i];
  }
};
// template <size_t Nc> using SUNAdj = Wrapper<Kokkos::Array<real_t,
// NcAdj<Nc>>>;

// define field view types
// by default all views are 4D
// some dimensions are set to 1 for lower dimensions
// I'm still not sure if this is the best way to do it
// Nd here is templated, but for a 4D gauge field,
// shouldn't Nd always be 4?
// Nc is the number of colors
template <size_t Nd, size_t Nc, typename precision_t = complex_t>
using GaugeField = Kokkos::View<SUN<Nc, precision_t>**** [Nd],
                                Kokkos::MemoryTraits<Kokkos::Restrict>>;
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using Propagator = Kokkos::View<PropagatorMatrix<Nc, RepDim>****,
                                Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc, typename precision_t = complex_t>
using GaugeField3D = Kokkos::View<SUN<Nc, precision_t>*** [Nd],
                                  Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc, typename precision_t = complex_t>
using GaugeField2D = Kokkos::View<SUN<Nc, precision_t>** [Nd],
                                  Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using SUNAdjField =
    Kokkos::View<SUNAdj<Nc>**** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using SUNAdjField3D =
    Kokkos::View<SUNAdj<Nc>*** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using SUNAdjField2D =
    Kokkos::View<SUNAdj<Nc>** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using SUNField =
    Kokkos::View<SUN<Nc>****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using SUNField3D =
    Kokkos::View<SUN<Nc>***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using SUNField2D =
    Kokkos::View<SUN<Nc>**, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field =
    Kokkos::View<complex_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field3D =
    Kokkos::View<complex_t***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field2D =
    Kokkos::View<complex_t**, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using Field1D =
    Kokkos::View<complex_t*, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField =
    Kokkos::View<real_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField3D =
    Kokkos::View<real_t***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField2D =
    Kokkos::View<real_t**, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using ScalarField1D =
    Kokkos::View<real_t*, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using LinkScalarField =
    Kokkos::View<real_t**** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using LinkScalarField3D =
    Kokkos::View<real_t*** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using LinkScalarField2D =
    Kokkos::View<real_t** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

// define corresponding constant fields
#if defined(KOKKOS_ENABLE_CUDA)
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using constSpinorField =
    Kokkos::View<const Spinor<Nc, RepDim, precision>****,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using constSpinorField3D =
    Kokkos::View<const Spinor<Nc, RepDim, precision>***,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using constSpinorField2D =
    Kokkos::View<const Spinor<Nc, RepDim, precision>**,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc, typename precision_t = complex_t>
using constGaugeField =
    Kokkos::View<const SUN<Nc, precision_t>**** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc, typename precision_t = complex_t>
using constGaugeField3D =
    Kokkos::View<const SUN<Nc, precision_t>*** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc, typename precision_t = complex_t>
using constGaugeField2D =
    Kokkos::View<const SUN<Nc, precision_t>** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc>
using constSUNAdjField =
    Kokkos::View<const SUNAdj<Nc>**** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc>
using constSUNAdjField3D =
    Kokkos::View<const SUNAdj<Nc>**** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc>
using constSUNAdjField2D =
    Kokkos::View<const SUNAdj<Nc>**** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nc>
using constSUNField =
    Kokkos::View<const SUN<Nc>****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nc>
using constSUNField3D =
    Kokkos::View<const SUN<Nc>***, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nc>
using constSUNField2D =
    Kokkos::View<const SUN<Nc>**, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constField = Kokkos::View<const complex_t****,
                                Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constField3D = Kokkos::View<const complex_t***,
                                  Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constField2D =
    Kokkos::View<const complex_t**, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constField1D =
    Kokkos::View<const complex_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constScalarField =
    Kokkos::View<const real_t****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constScalarField3D =
    Kokkos::View<const real_t***, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constScalarField2D =
    Kokkos::View<const real_t**, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constScalarField1D =
    Kokkos::View<const real_t*, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd>
using constLinkScalarField =
    Kokkos::View<const real_t**** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd>
using constLinkScalarField3D =
    Kokkos::View<const real_t*** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd>
using constLinkScalarField2D =
    Kokkos::View<const real_t** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

#else
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using constSpinorField = Kokkos::View<const Spinor<Nc, RepDim, precision>****,
                                      Kokkos::MemoryTraits<Kokkos::Restrict>>;
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using constSpinorField3D = Kokkos::View<const Spinor<Nc, RepDim, precision>***,
                                        Kokkos::MemoryTraits<Kokkos::Restrict>>;
template <size_t Nc, size_t RepDim, typename precision = complex_t>
using constSpinorField2D = Kokkos::View<const Spinor<Nc, RepDim, precision>**,
                                        Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc, typename precision_t = complex_t>
using constGaugeField = Kokkos::View<const SUN<Nc, precision_t>**** [Nd],
                                     Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc, typename precision_t = complex_t>
using constGaugeField3D = Kokkos::View<const SUN<Nc, precision_t>*** [Nd],
                                       Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc, typename precision_t = complex_t>
using constGaugeField2D = Kokkos::View<const SUN<Nc, precision_t>** [Nd],
                                       Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using constSUNAdjField = Kokkos::View<const SUNAdj<Nc>**** [Nd],
                                      Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using constSUNAdjField3D = Kokkos::View<const SUNAdj<Nc>**** [Nd],
                                        Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using constSUNAdjField2D = Kokkos::View<const SUNAdj<Nc>**** [Nd],
                                        Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using constSUNField =
    Kokkos::View<const SUN<Nc>****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using constSUNField3D =
    Kokkos::View<const SUN<Nc>***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nc>
using constSUNField2D =
    Kokkos::View<const SUN<Nc>**, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constField =
    Kokkos::View<const complex_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constField3D =
    Kokkos::View<const complex_t***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constField2D =
    Kokkos::View<const complex_t**, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constField1D =
    Kokkos::View<const complex_t*, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constScalarField =
    Kokkos::View<const real_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constScalarField3D =
    Kokkos::View<const real_t***, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constScalarField2D =
    Kokkos::View<const real_t**, Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constScalarField1D =
    Kokkos::View<const real_t*, Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using constLinkScalarField =
    Kokkos::View<const real_t**** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using constLinkScalarField3D =
    Kokkos::View<const real_t*** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd>
using constLinkScalarField2D =
    Kokkos::View<const real_t** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

#endif

// define policy as mdrange
template <size_t rank, class WorkTag = void>
using Policy = Kokkos::MDRangePolicy<WorkTag, Kokkos::Rank<rank>>;

// special case for 1D
template <class WorkTag = void>
using Policy1D = Kokkos::RangePolicy<WorkTag>;

// define a global zero field generator
// for the color x color matrix
template <size_t Nc, typename precision_t = complex_t>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<Nc, precision_t> zeroSUN() {
  SUN<Nc, precision_t> zero;
#pragma unroll
  for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
    for (index_t c2 = 0; c2 < Nc; ++c2) {
      zero[c1][c2] = complex_t(0.0, 0.0);
    }
  }
  return zero;
}
// define a global zero generator
// for spinor
template <size_t Nc, size_t Nd, typename precision_t = complex_t>
constexpr KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd, precision_t> zeroSpinor() {
  Spinor<Nc, Nd, precision_t> zero;
#pragma unroll
  for (size_t i = 0; i < Nd; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      zero[i][j] = complex_t(0.0, 0.0);
    }
  }
  return zero;
}
// define a global identity field generator
// for the color x color matrix
template <size_t Nc, typename precision_t = complex_t>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<Nc, precision_t> identitySUN() {
  SUN<Nc, precision_t> id = zeroSUN<Nc, precision_t>();
#pragma unroll
  for (index_t c1 = 0; c1 < Nc; ++c1) {
    id[c1][c1] = complex_t(1.0, 0.0);
  }
  return id;
}

template <typename T, size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<T, N>, N> operator*(
    const Kokkos::Array<Kokkos::Array<T, N>, N>& a,
    const Kokkos::Array<Kokkos::Array<T, N>, N>& b) {
  Kokkos::Array<Kokkos::Array<T, N>, N> c;
#pragma unroll
  for (size_t i = 0; i < N; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      c[i][j] = a[i][0] * b[0][j];
#pragma unroll
      for (size_t k = 1; k < N; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

template <typename T, typename U, size_t N>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Kokkos::Array<T, N>, N> operator*(
    const Kokkos::Array<Kokkos::Array<T, N>, N>& a,
    const U& b) {
  Kokkos::Array<Kokkos::Array<T, N>, N> c;
#pragma unroll
  for (size_t i = 0; i < N; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      c[i][j] = a[i][j] * b;
    }
  }
  return c;
}

// define a global one generator
// for spinor
template <size_t Nc, size_t Nd>
constexpr KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, Nd> oneSpinor() {
  Spinor<Nc, Nd> id = zeroSpinor<Nc, Nd>();
#pragma unroll
  for (size_t i = 0; i < Nd; ++i) {
#pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      id[i][j] = complex_t(1.0, 0.0);
    }
  }
  return id;
}
// --- real_t ---
template <typename T>
inline MPI_Datatype mpi_real_type() {
  using base_t = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<base_t, double>)
    return MPI_DOUBLE;
  else if constexpr (std::is_same_v<base_t, float>)
    return MPI_FLOAT;
  else if constexpr (std::is_same_v<base_t, long double>)
    return MPI_LONG_DOUBLE;
  else {
    static_assert(!sizeof(base_t), "Unsupported real_t for MPI");
    return MPI_DATATYPE_NULL;
  }
}

// --- index_t ---
template <typename T>
inline MPI_Datatype mpi_index_type() {
  using base_t = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<base_t, int>)
    return MPI_INT;
  else if constexpr (std::is_same_v<base_t, long>)
    return MPI_LONG;
  else if constexpr (std::is_same_v<base_t, long long>)
    return MPI_LONG_LONG;
  else if constexpr (std::is_same_v<base_t, short>)
    return MPI_SHORT;
  else {
    static_assert(!sizeof(base_t), "Unsupported index_t for MPI");
    return MPI_DATATYPE_NULL;
  }
}

// --- size_t ---
template <typename T>
inline MPI_Datatype mpi_size_type() {
  using base_t = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<base_t, unsigned int>)
    return MPI_UNSIGNED;
  else if constexpr (std::is_same_v<base_t, unsigned long>)
    return MPI_UNSIGNED_LONG;
  else if constexpr (std::is_same_v<base_t, unsigned long long>)
    return MPI_UNSIGNED_LONG_LONG;
  else {
    static_assert(!sizeof(base_t), "Unsupported size_t for MPI");
    return MPI_DATATYPE_NULL;
  }
}

// --- complex_t ---
template <typename T>
inline MPI_Datatype mpi_complex_type() {
  using base_t = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<base_t, std::complex<float>> ||
                std::is_same_v<base_t, Kokkos::complex<float>>) {
    return MPI_CXX_FLOAT_COMPLEX;
  } else if constexpr (std::is_same_v<base_t, std::complex<double>> ||
                       std::is_same_v<base_t, Kokkos::complex<double>>) {
    return MPI_CXX_DOUBLE_COMPLEX;
  } else if constexpr (std::is_same_v<base_t, std::complex<long double>> ||
                       std::is_same_v<base_t, Kokkos::complex<long double>>) {
    return MPI_CXX_LONG_DOUBLE_COMPLEX;
  } else {
    static_assert(!sizeof(base_t), "Unsupported complex_t for MPI");
    return MPI_DATATYPE_NULL;
  }
}

// Concrete instantiations for your typedefs:
inline MPI_Datatype mpi_real_t() {
  return mpi_real_type<real_t>();
}
inline MPI_Datatype mpi_index_t() {
  return mpi_index_type<index_t>();
}
inline MPI_Datatype mpi_size_t() {
  return mpi_size_type<size_t>();
}
inline MPI_Datatype mpi_complex_t() {
  return mpi_complex_type<complex_t>();
}

// global verbosity level
// 0 = silent
// 1 = normal
// 2 = verbose
// 3 = very verbose
// 4 = debug
// 5 = trace
inline int KLFT_VERBOSITY = 0;

inline void setVerbosity(int v) {
  KLFT_VERBOSITY = v;
}

// variable that enables tuning
// 0 = no tuning
// 1 = tuning enabled
inline int KLFT_TUNING = 0;

inline void setTuning(int t) {
  KLFT_TUNING = t;
}

}  // namespace klft
