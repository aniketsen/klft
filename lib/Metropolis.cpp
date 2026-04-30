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

// this file defines the main function to run the metropolis
// for 2D, 3D and 4D SU(N) gauge fields

#include "Metropolis.hpp"

#include "InputParser.hpp"

// we are hard coding the RNG now to use Kokkos::Random_XorShift64_Pool
// we might want to use our own RNG or allow the user to choose from
// different RNGs in the future
#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

int Metropolis(const std::string& input_file,
               const std::string& output_directory) {
  // get verbosity from environment
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);
  // get tuning from environment
  const int tuning =
      std::getenv("KLFT_TUNING") ? std::atoi(std::getenv("KLFT_TUNING")) : 0;
  setTuning(tuning);
  KTune::initialize();
  // if tuning is enbled, check if the user has set the
  // KLFT_CACHE_FILE environment variable

  // parse input file
  MetropolisParams metropolisParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, output_directory, metropolisParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, output_directory, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  // print the parameters
  metropolisParams.print();
  // initialize RNG
  RNGType rng(metropolisParams.seed);
  // initialize gauge field and run metropolis
  // based on the system parameters
  // case 4D
  if (metropolisParams.Ndims == 4) {
    // case U(1)
    if (metropolisParams.Nc == 1) {
      deviceGaugeField<4, 1> dev_g_U1_4D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          metropolisParams.L3, identitySUN<1>());
      run_metropolis<4, 1>(dev_g_U1_4D, metropolisParams, gaugeObsParams, rng);
    }
    // case SU(2)
    else if (metropolisParams.Nc == 2) {
      deviceGaugeField<4, 2> dev_g_SU2_4D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          metropolisParams.L3, identitySUN<2>());
      run_metropolis<4, 2>(dev_g_SU2_4D, metropolisParams, gaugeObsParams, rng);
    }
    // case SU(3)
    else if (metropolisParams.Nc == 3) {
      deviceGaugeField<4, 3> dev_g_SU3_4D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          metropolisParams.L3, identitySUN<3>());
      run_metropolis<4, 3>(dev_g_SU3_4D, metropolisParams, gaugeObsParams, rng);
    }
    // case SU(N)
    else {
      printf("Error: Unsupported gauge group\n");
      return -1;
    }
  }
  // case 3D
  else if (metropolisParams.Ndims == 3) {
    // case U(1)
    if (metropolisParams.Nc == 1) {
      deviceGaugeField3D<3, 1> dev_g_U1_3D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          identitySUN<1>());
      run_metropolis<3, 1>(dev_g_U1_3D, metropolisParams, gaugeObsParams, rng);
    }
    // case SU(2)
    else if (metropolisParams.Nc == 2) {
      deviceGaugeField3D<3, 2> dev_g_SU2_3D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          identitySUN<2>());
      run_metropolis<3, 2>(dev_g_SU2_3D, metropolisParams, gaugeObsParams, rng);
    }
    // case SU(3)
    else if (metropolisParams.Nc == 3) {
      deviceGaugeField3D<3, 3> dev_g_SU3_3D(
          metropolisParams.L0, metropolisParams.L1, metropolisParams.L2,
          identitySUN<3>());
      run_metropolis<3, 3>(dev_g_SU3_3D, metropolisParams, gaugeObsParams, rng);
    }
    // case SU(N)
    else {
      printf("Error: Unsupported gauge group\n");
      return -1;
    }
  }
  // case 2D
  else if (metropolisParams.Ndims == 2) {
    // case U(1)
    if (metropolisParams.Nc == 1) {
      deviceGaugeField2D<2, 1> dev_g_U1_2D(
          metropolisParams.L0, metropolisParams.L1, identitySUN<1>());
      run_metropolis<2, 1>(dev_g_U1_2D, metropolisParams, gaugeObsParams, rng);
    }
    // case SU(2)
    else if (metropolisParams.Nc == 2) {
      deviceGaugeField2D<2, 2> dev_g_SU2_2D(
          metropolisParams.L0, metropolisParams.L1, identitySUN<2>());
      run_metropolis<2, 2>(dev_g_SU2_2D, metropolisParams, gaugeObsParams, rng);
    }
    // case SU(3)
    else if (metropolisParams.Nc == 3) {
      deviceGaugeField2D<2, 3> dev_g_SU3_2D(
          metropolisParams.L0, metropolisParams.L1, identitySUN<3>());
      run_metropolis<2, 3>(dev_g_SU3_2D, metropolisParams, gaugeObsParams, rng);
    }
    // case SU(N)
    else {
      printf("Error: Unsupported gauge group\n");
      return -1;
    }
  }
  // if tuning is enabled, write the cache file

  return 0;
}

}  // namespace klft
