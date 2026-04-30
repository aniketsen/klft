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

// this is a helper file to define the field types based on dimension
// for the gauge fields here we assume that Nd = rank (dimensionality)
// this would work for most use cases. But if you have a different
// dimensionality from Nd, you can not use the definitions here

#pragma once
#include <cstddef>

#include "AdjointField.hpp"
#include "Field.hpp"
#include "GaugeField.hpp"
#include "PTBCGaugeField.hpp"
#include "Propagator.hpp"
#include "SUNField.hpp"
#include "ScalarField.hpp"
#include "SpinorField.hpp"
#include "SpinorPointSource.hpp"

namespace klft {
// define GaugeFieldKinds
enum class GaugeFieldKind { Standard, PTBC };

enum class SpinorFieldKind { Standard, Staggered, PointSource };
enum class SpinorFieldLayout { FULL, Checkerboard };
// define a function to get the gauge field type based on the rank,
// with the default Field being the default GaugeField
template <size_t rank,
          size_t Nc,
          typename precision_t = complex_t,
          GaugeFieldKind k = GaugeFieldKind::Standard>
struct DeviceGaugeFieldType;

// now define the specializations
template <size_t Nc, typename precision_t>
struct DeviceGaugeFieldType<2, Nc, precision_t, GaugeFieldKind::Standard> {
  using type = deviceGaugeField2D<2, Nc, precision_t>;
};

template <size_t Nc, typename precision_t>
struct DeviceGaugeFieldType<3, Nc, precision_t, GaugeFieldKind::Standard> {
  using type = deviceGaugeField3D<3, Nc, precision_t>;
};

template <size_t Nc, typename precision_t>
struct DeviceGaugeFieldType<4, Nc, precision_t, GaugeFieldKind::Standard> {
  using type = deviceGaugeField<4, Nc, precision_t>;
};

template <size_t rank, size_t Nc, size_t RepDim>
struct DevicePropagator;

template <size_t Nc, size_t RepDim>
struct DevicePropagator<4, Nc, RepDim> {
  using type = devicePropagator<Nc, RepDim>;
};

// now do the same for the SpinorField field types
template <size_t rank,
          size_t Nc,
          size_t RepDim,
          typename precision_t,
          SpinorFieldKind k = SpinorFieldKind::Standard,
          SpinorFieldLayout l = SpinorFieldLayout::FULL>
struct DeviceSpinorFieldType;

template <size_t Nc, typename precision_t>
struct DeviceSpinorFieldType<4,
                             Nc,
                             4,
                             precision_t,
                             SpinorFieldKind::Standard,
                             SpinorFieldLayout::FULL> {
  using type = deviceSpinorField<Nc, 4, precision_t>;
};
template <size_t Nc, typename precision_t>
struct DeviceSpinorFieldType<3,
                             Nc,
                             4,
                             precision_t,
                             SpinorFieldKind::Standard,
                             SpinorFieldLayout::FULL> {
  using type = deviceSpinorField3D<Nc, 4, precision_t>;
};

template <size_t Nc, typename precision_t>
struct DeviceSpinorFieldType<2,
                             Nc,
                             4,
                             precision_t,
                             SpinorFieldKind::Standard,
                             SpinorFieldLayout::FULL> {
  using type = deviceSpinorField2D<Nc, 4, precision_t>;
};

// Checkerboarded spinor fields

template <size_t Nc, typename precision_t>
struct DeviceSpinorFieldType<4,
                             Nc,
                             4,
                             precision_t,
                             SpinorFieldKind::Standard,
                             SpinorFieldLayout::Checkerboard> {
  using type = deviceSpinorField<Nc, 4, precision_t>;
};
template <size_t Nc>
struct DeviceSpinorFieldType<4,
                             Nc,
                             4,
                             complex_t,
                             SpinorFieldKind::PointSource> {
  using type = deviceSpinorPointSource<Nc, 4>;
};
template <size_t Nc, typename precision_t>
struct DeviceSpinorFieldType<3,
                             Nc,
                             4,
                             precision_t,
                             SpinorFieldKind::Standard,
                             SpinorFieldLayout::Checkerboard> {
  using type = deviceSpinorField3D<Nc, 4, precision_t>;
};
template <size_t Nc>
struct DeviceSpinorFieldType<3,
                             Nc,
                             4,
                             complex_t,
                             SpinorFieldKind::PointSource> {
  using type = deviceSpinorPointSource3D<Nc, 4>;
};

template <size_t Nc, typename precision_t>
struct DeviceSpinorFieldType<2,
                             Nc,
                             4,
                             precision_t,
                             SpinorFieldKind::Standard,
                             SpinorFieldLayout::Checkerboard> {
  using type = deviceSpinorField2D<Nc, 4, precision_t>;
};
template <size_t Nc>
struct DeviceSpinorFieldType<2,
                             Nc,
                             4,
                             complex_t,
                             SpinorFieldKind::PointSource> {
  using type = deviceSpinorPointSource2D<Nc, 4>;
};

// now do the same for the PTBC gauge field types
template <size_t Nc, typename precision_t>
struct DeviceGaugeFieldType<4, Nc, precision_t, GaugeFieldKind::PTBC> {
  using type = devicePTBCGaugeField<4, Nc,precision_t>;
};

template <size_t Nc, typename precision_t>
struct DeviceGaugeFieldType<3, Nc, precision_t, GaugeFieldKind::PTBC> {
  using type = devicePTBCGaugeField3D<3, Nc,precision_t>;
};

template <size_t Nc, typename precision_t>
struct DeviceGaugeFieldType<2, Nc, precision_t, GaugeFieldKind::PTBC> {
  using type = devicePTBCGaugeField2D<2, Nc,precision_t>;
};
template <typename T>
struct DeviceGaugeFieldConverter;

template <size_t rank, size_t Nc>
struct DeviceGaugeFieldConverter<
    DeviceGaugeFieldType<rank, Nc, complex_t, GaugeFieldKind::PTBC>> {
  using type =
      DeviceGaugeFieldType<rank, Nc, complex_t, GaugeFieldKind::Standard>;
};
template <size_t rank, size_t Nc, typename precision_t>
struct DeviceGaugeFieldConverter<
    DeviceGaugeFieldType<rank, Nc, precision_t, GaugeFieldKind::Standard>> {
  using type =
      DeviceGaugeFieldType<rank, Nc, precision_t, GaugeFieldKind::Standard>;
};
// define Traits to extract the rank, Nc and GaugeFieldKind at a later point
template <typename T>
struct DeviceGaugeFieldTypeTraits;

template <size_t _rank, size_t _Nc, typename precision_t, GaugeFieldKind _k>
struct DeviceGaugeFieldTypeTraits<
    DeviceGaugeFieldType<_rank, _Nc, precision_t, _k>> {
  static constexpr size_t Rank = _rank;
  static constexpr size_t Nc = _Nc;
  using value_type = precision_t;
  static constexpr GaugeFieldKind Kind = _k;
};

// compile time check for the appropriate types
template <typename T>
struct isDeviceGaugeFieldType : std::false_type {};

template <typename T>
struct isDeviceFermionFieldType : std::false_type {};

template <size_t rank, size_t Nc, typename precision_t, GaugeFieldKind k>
struct isDeviceGaugeFieldType<DeviceGaugeFieldType<rank, Nc, precision_t, k>>
    : std::true_type {};

template <size_t rank,
          size_t Nc,
          size_t RepDim,
          typename precision_t,
          SpinorFieldKind k,
          SpinorFieldLayout l>
struct isDeviceFermionFieldType<
    DeviceSpinorFieldType<rank, Nc, RepDim, precision_t, k, l>>
    : std::true_type {};

template <typename T>
struct DeviceFermionFieldTypeTraits;

template <size_t _rank,
          size_t _Nc,
          size_t _RepDim,
          typename precision_t,
          SpinorFieldKind _k,
          SpinorFieldLayout _l>
struct DeviceFermionFieldTypeTraits<
    DeviceSpinorFieldType<_rank, _Nc, _RepDim, precision_t, _k, _l>> {
  static constexpr size_t Rank = _rank;
  static constexpr size_t Nc = _Nc;
  static constexpr size_t RepDim = _RepDim;
  using value_type = precision_t;
  static constexpr SpinorFieldKind Kind = _k;
  static constexpr SpinorFieldLayout Layout = _l;
};

// define a function to get the gauge field type based on the rank,
// with the default Field being the default GaugeField
template <size_t rank, size_t Nc>
struct DeviceAdjFieldType;

template <size_t Nc>
struct DeviceAdjFieldType<4, Nc> {
  using type = deviceAdjointField<4, Nc>;
};

template <size_t Nc>
struct DeviceAdjFieldType<3, Nc> {
  using type = deviceAdjointField3D<3, Nc>;
};

template <size_t Nc>
struct DeviceAdjFieldType<2, Nc> {
  using type = deviceAdjointField2D<2, Nc>;
};

// define Traits to extract the rank, Nc and GaugeFieldKind at a later point
template <typename T>
struct DeviceAdjFieldTypeTraits;

template <size_t _rank, size_t _Nc>
struct DeviceAdjFieldTypeTraits<DeviceAdjFieldType<_rank, _Nc>> {
  static constexpr size_t Rank = _rank;
  static constexpr size_t Nc = _Nc;
};

// compile time check for the appropriate types
template <typename T>
struct isDeviceAdjFieldType : std::false_type {};

template <size_t rank, size_t Nc>
struct isDeviceAdjFieldType<DeviceAdjFieldType<rank, Nc>> : std::true_type {};

// define the same thing for SUN fields
template <size_t rank, size_t Nc>
struct DeviceSUNFieldType;

template <size_t Nc>
struct DeviceSUNFieldType<2, Nc> {
  using type = deviceSUNField2D<Nc>;
};

template <size_t Nc>
struct DeviceSUNFieldType<3, Nc> {
  using type = deviceSUNField3D<Nc>;
};

template <size_t Nc>
struct DeviceSUNFieldType<4, Nc> {
  using type = deviceSUNField<Nc>;
};

// repeat for field
template <size_t rank>
struct DeviceFieldType;

template <>
struct DeviceFieldType<2> {
  using type = deviceField2D;
};

template <>
struct DeviceFieldType<3> {
  using type = deviceField3D;
};

template <>
struct DeviceFieldType<4> {
  using type = deviceField;
};

// define the same for the scalar fields
template <size_t rank>
struct DeviceScalarFieldType;

template <>
struct DeviceScalarFieldType<2> {
  using type = deviceScalarField2D;
};

template <>
struct DeviceScalarFieldType<3> {
  using type = deviceScalarField3D;
};

template <>
struct DeviceScalarFieldType<4> {
  using type = deviceScalarField;
};

// Type selector
template <size_t Nd, size_t Nc>
struct ConstGaugeFieldSelector;

template <size_t Nc>
struct ConstGaugeFieldSelector<4, Nc> {
  using type = constGaugeField<4, Nc>;
};

template <size_t Nc>
struct ConstGaugeFieldSelector<3, Nc> {
  using type = constGaugeField3D<3, Nc>;
};

template <size_t Nc>
struct ConstGaugeFieldSelector<2, Nc> {
  using type = constGaugeField2D<2, Nc>;
};
template <typename T, SpinorFieldKind NewKind>
struct WithSpinorFieldKind {
  static_assert(isDeviceFermionFieldType<T>::value);
  using type = typename DeviceSpinorFieldType<
      DeviceFermionFieldTypeTraits<T>::Rank,
      DeviceFermionFieldTypeTraits<T>::Nc,
      DeviceFermionFieldTypeTraits<T>::RepDim,
      typename DeviceFermionFieldTypeTraits<T>::value_type,
      NewKind,
      DeviceFermionFieldTypeTraits<T>::Layout>::type;
};
template <typename T, typename precision>
struct WithPrecisionSpinorField {
  using type = DeviceSpinorFieldType<DeviceFermionFieldTypeTraits<T>::Rank,
                                     DeviceFermionFieldTypeTraits<T>::Nc,
                                     DeviceFermionFieldTypeTraits<T>::RepDim,
                                     precision,
                                     DeviceFermionFieldTypeTraits<T>::Kind,
                                     DeviceFermionFieldTypeTraits<T>::Layout>;
};
template <typename T, typename precision>
struct WithPrecisionGaugeField {
  using type = DeviceGaugeFieldType<DeviceGaugeFieldTypeTraits<T>::Rank,
                                    DeviceGaugeFieldTypeTraits<T>::Nc,
                                    precision,
                                    DeviceGaugeFieldTypeTraits<T>::Kind>;
};
// Type alias for convenience
template <size_t Nd, size_t Nc>
using ConstGaugeFieldType = typename ConstGaugeFieldSelector<Nd, Nc>::type;

// add the same for scalar fields here when needed
}  // namespace klft
