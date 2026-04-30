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
#include "DiracOperator.hpp"
#include "WilsonDiracOperator.hpp"
namespace klft {
template <typename T>
struct DiracOpFieldTypeTraits;
template <template <typename, typename, bool> class _Derived,
          typename _DSpinorFieldType,
          typename _DGaugeFieldType,
          bool HasMassShift>
struct DiracOpFieldTypeTraits<DiracOperator<_Derived,
                                            _DSpinorFieldType,
                                            _DGaugeFieldType,
                                            HasMassShift>> {
  using Derived = _Derived<_DSpinorFieldType, _DGaugeFieldType, HasMassShift>;
  using DSpinorFieldType = _DSpinorFieldType;
  using DGaugeFieldType = _DGaugeFieldType;
};
template <typename _DSpinorFieldType, typename _DGaugeFieldType>
struct DiracOpFieldTypeTraits<
    WilsonDiracOperator<_DSpinorFieldType, _DGaugeFieldType, false>> {
  using Derived = WilsonDiracOperator<_DSpinorFieldType, _DGaugeFieldType>;
  using DSpinorFieldType = _DSpinorFieldType;
  using DGaugeFieldType = _DGaugeFieldType;
};
template <typename _DSpinorFieldType, typename _DGaugeFieldType>
struct DiracOpFieldTypeTraits<
    HWilsonDiracOperator<_DSpinorFieldType, _DGaugeFieldType, false>> {
  using Derived =
      HWilsonDiracOperator<_DSpinorFieldType, _DGaugeFieldType, false>;
  using DSpinorFieldType = _DSpinorFieldType;
  using DGaugeFieldType = _DGaugeFieldType;
};

template <typename T>
struct GetUnshiftedDiracOpType;

}  // namespace klft
