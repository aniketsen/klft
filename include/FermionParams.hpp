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
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GammaMatrix.hpp"
#include "HMC_Params.hpp"

namespace klft {

// Parameters specific to the Dirac operator

struct diracParams {
  real_t kappa;
  real_t massShift;
  diracParams() = default;
  // kappa_tilde is the reduced kappa i.e kappa_tilde<kappa as in
  // arXiv:hep-lat/0107019v1
  diracParams(const real_t& _kappa) : kappa(_kappa), massShift(0.0) {};
  diracParams(const real_t& _kappa, const real_t& _massShift)
      : kappa(_kappa), massShift(_massShift) {};
};
auto getDiracParams(const FermionMonomial_Params& fparams) {
  diracParams dParams(fparams.kappa);
  return dParams;
}
auto getDiracParams(const Hasenbusch_Params fparams) {
  diracParams dParams(fparams.kappa, fparams.massShift);
  return dParams;
}

}  // namespace klft
