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
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace klft {

// define the global types
// is this a good idea?
// if we want different fields with different precision (e.g. coarse grid)
// might be a lot of work to redefine, but on the other hand, if we want to
// be precision agnostic, everything has to be templated with given precision
// let's start like this and worry about mixed precision later
using real_t = double;
using complex_t = Kokkos::complex<real_t>;

// use int for the index type
using index_t = int;

// define index_arrays
template <size_t rank>
using IndexArray = Kokkos::Array<index_t, rank>;

// define groups for gauge fields
template <size_t Nc>
using SUN = Kokkos::Array<Kokkos::Array<complex_t, Nc>, Nc>;

// define Spinor Type
// info correct dispatch is only guaranteed for    Nd != Nc ! -> Conflicts with
// SUN.hpp version Maybe via class to make it safe
template <size_t Nc, size_t Nd>
using Spinor = Kokkos::Array<Kokkos::Array<complex_t, Nd>, Nc>;

// define field view types
// by default all views are 4D
// some dimensions are set to 1 for lower dimensions
// I'm still not sure if this is the best way to do it
// Nd here is templated, but for a 4D gauge field,
// shouldn't Nd always be 4?
// Nc is the number of colors
template <size_t Nd, size_t RepDim, size_t Nc>
using SpinorField = Kokkos::View<Spinor<Nc, RepDim>**** [Nd],
                                 Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using GaugeField =
    Kokkos::View<SUN<Nc>**** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using GaugeField3D =
    Kokkos::View<SUN<Nc>*** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using GaugeField2D =
    Kokkos::View<SUN<Nc>** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

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
template <size_t Nd, size_t RepDim, size_t Nc>
using constSpinorField =
    Kokkos::View<const Spinor<Nc, RepDim>**** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc>
using constGaugeField =
    Kokkos::View<const SUN<Nc>**** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc>
using constGaugeField3D =
    Kokkos::View<const SUN<Nc>*** [Nd],
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

template <size_t Nd, size_t Nc>
using constGaugeField2D =
    Kokkos::View<const SUN<Nc>** [Nd],
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

template <size_t Nd, size_t RepDim, size_t Nc>
using constSpinorField = Kokkos::View<const Spinor<Nc, RepDim>**** [Nd],
                                      Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using constGaugeField = Kokkos::View<const SUN<Nc>**** [Nd],
                                     Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using constGaugeField3D =
    Kokkos::View<const SUN<Nc>*** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

template <size_t Nd, size_t Nc>
using constGaugeField2D =
    Kokkos::View<const SUN<Nc>** [Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;

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
template <size_t rank>
using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

// special case for 1D
using Policy1D = Kokkos::RangePolicy<>;

// define a global zero field generator
// for the color x color matrix
template <size_t Nc>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> zeroSUN() {
  SUN<Nc> zero;
#pragma unroll
  for (index_t c1 = 0; c1 < Nc; ++c1) {
#pragma unroll
    for (index_t c2 = 0; c2 < Nc; ++c2) {
      zero[c1][c2] = complex_t(0.0, 0.0);
    }
  }
  return zero;
}

// define a global identity field generator
// for the color x color matrix
template <size_t Nc>
constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<Nc> identitySUN() {
  SUN<Nc> id = zeroSUN<Nc>();
#pragma unroll
  for (index_t c1 = 0; c1 < Nc; ++c1) {
    id[c1][c1] = complex_t(1.0, 0.0);
  }
  return id;
}

// global verbosity level
// 0 = silent
// 1 = normal
// 2 = verbose
// 3 = very verbose
// 4 = debug
// 5 = trace
inline int KLFT_VERBOSITY = 0;

inline void setVerbosity(int v) { KLFT_VERBOSITY = v; }

// variable that enables tuning
// 0 = no tuning
// 1 = tuning enabled
inline int KLFT_TUNING = 0;

inline void setTuning(int t) { KLFT_TUNING = t; }

}  // namespace klft