
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
#include <cstddef>

#include "FermionObservable.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "SimulationLogging.hpp"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

template <typename HMCType>
int run_HMC(HMCType& hmc,
            const Integrator_Params& integratorParams,
            GaugeObservableParams& gaugeObsParams,
            SimulationLoggingParams& simLogParams,
            FermionObservableParams& fermionObsParams) {
  // initiate and execute the HMC with the given parameters
  printf("Executing HMC ...\n");
  static_assert(isHMCClass<HMCType>::value,
                "HMCType must be a valid HMC class");

  // timer to measure the time per step
  Kokkos::Timer timer;
  bool accept;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};
  // hmc loop
  for (size_t step = 0; step < integratorParams.nsteps; ++step) {
    timer.reset();

    // perform hmc_step
    accept = hmc.hmc_step();

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);

    if (KLFT_VERBOSITY > 0) {
      time_t now = std::time(nullptr);
      char time_str[26];
      std::strftime(time_str, sizeof(time_str), "%Y %b %d %H:%M:%S",
                    std::localtime(&now));
      Kokkos::printf(
          "%s - Step: %zu, accepted: %d, Acceptance rate: %f, Time: %f\n",
          time_str, step, accept, acc_rate, time);
    }
    timer.reset();
    measureGaugeObservables<typename HMCType::DeviceGaugeFieldType>(
        hmc.hamiltonian_field.gauge_field, gaugeObsParams, step);
    // For now fix fermion measurment stuff:
    measureFermionObservables<
        std::mt19937,
        DeviceSpinorFieldType<HMCType::rank, HMCType::Nc, 4, complex_t>,
        typename HMCType::DeviceGaugeFieldType, CGSolver,
        EOWilsonDiracOperator>(hmc.hamiltonian_field.gauge_field,
                               fermionObsParams, step, hmc.mt);
    const real_t obs_time = timer.seconds();
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time,
               obs_time);
    flushSimulationLogs(simLogParams, step, true);
    flushAllGaugeObservables(gaugeObsParams, step, true);
    flushIO<DeviceGaugeFieldType<HMCType::rank, HMCType::Nc>>(
        hmc.ioParams, step, hmc.hamiltonian_field.gauge_field);
    if (fermionObsParams.flush != 0 && step % fermionObsParams.flush == 0) {
      flushAllFermionObservables(fermionObsParams, step == simLogParams.flush);
      clearAllFermionObservables(fermionObsParams);
    }
    // flush the measurements to the files
    // if flush is set to 0, we flush with the  header at the end of
    // the simulation
  }

  forceflushSimulationLogs(simLogParams, true);
  forceflushAllGaugeObservables(gaugeObsParams, true);
  flushAllFermionObservables(fermionObsParams, fermionObsParams.flush == 0);
  flushIO<DeviceGaugeFieldType<HMCType::rank, HMCType::Nc>>(
      hmc.ioParams, integratorParams.nsteps, hmc.hamiltonian_field.gauge_field,
      true);

  printf("Total Acceptance rate: %f, Accept %f Configs", acc_rate, acc_sum);
  return 0;
}

}  // namespace klft
