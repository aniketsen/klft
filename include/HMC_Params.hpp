
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

// this file defines metropolis parameters for simulating different fields and
// actions

#pragma once
#include "GLOBAL.hpp"

namespace klft {

struct HMCParams {

  // general parameters
  index_t Ndims; // number of dimensions of the simulated system
  index_t L0;    // length of the first dimension
  index_t L1;    // length of the second dimension
  index_t L2;    // length of the third dimension
  index_t L3;    // length of the fourth dimension
  // hard cutoff at 4 for now, since we have not
  // implemented any 5D cases yet
  // more dimensions can and will be added when necessary
  real_t tau;          // integration "time" for the hmc
  real_t rngDelta;     // integration "time" for the hmc
  index_t seed;        // seed for the random number generator
  index_t nsteps;      // number of hmc steps
  index_t nstepsGauge; // number of integration steps in gaugefield integration
  bool coldStart;

  // parameters specific to the GaugeField
  size_t Nd; // number of mu degrees of freedom
  size_t Nc; // number of color degrees of freedom

  // parameters specific to the Wilson action
  real_t beta; // inverse coupling constant

  // add more parameters above this line as needed
  // ...

  // default parameters
  // in practice, user should set these
  // parameters after constructing the object
  HMCParams() {
    Ndims = 4;
    L0 = 4;
    L1 = 4;
    L2 = 4;
    L3 = 4;
    tau = 1.0;
    seed = 1234;
    nsteps = 10;
    nstepsGauge = 20;
    rngDelta = 1.0;
    coldStart = false;

    Nd = 4;
    Nc = 2;

    beta = 1.0;

    // set default values to newly added parameters
    // above this line
    // ...
  }

  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("HMC Parameters:\n");
      printf("General Parameters:\n");
      printf("Ndims: %d\n", Ndims);
      printf("L0: %d\n", L0);
      printf("L1: %d\n", L1);
      printf("L2: %d\n", L2);
      printf("L3: %d\n", L3);
      printf("tau: %.3f\n", tau);
      printf("coldStart: %b", coldStart);
      printf("rngDelta: %.3f\n", rngDelta);
      printf("seed: %d\n", seed);
      printf("nsteps: %d\n", nsteps);
      printf("nstepsGauge: %d\n", nstepsGauge);
      printf("GaugeField Parameters:\n");
      printf("Nd: %zu\n", Nd);
      printf("Nc: %zu\n", Nc);
      printf("Wilson Action Parameters:\n");
      printf("beta: %.4f\n", beta);
    }
  }
};
} // namespace klft
