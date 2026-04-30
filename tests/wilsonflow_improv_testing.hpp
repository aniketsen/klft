#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeAction.hpp"
#include "GaugeObservable.hpp"
#include "GaugePlaquette.hpp"
#include "Kokkos_Atomic.hpp"
#include "SimulationLogging.hpp"

namespace klft {

template <typename DGaugeFieldType, typename HMCType>
index_t do_wilsonflow_improv_test(HMCType& hmc,
                                  GaugeObservableParams& gaugeObsParams,
                                  SimulationLoggingParams& simLogParams,
                                  std::string output_directory) {
  static const size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  // Construct the output filename. Each MPI rank will get its own file.

  real_t b1 = -1.0 / 12;
  b1 = -1.4088;
  const real_t factors[]{1.0, 1.3, 1.5, 1.7, 2.0, 3.0, 4.0, 5.0};
  std::string output_filename_action_density_clover =
      output_directory + "action_densities_clover_cumulative.txt";
  std::ofstream output_file_actiondensity_clover(
      output_filename_action_density_clover);

  if (!output_file_actiondensity_clover.is_open()) {
    fprintf(stderr, "Error: Could not open output file %s\n",
            output_filename_action_density_clover.c_str());
    return -1;  // Or handle the error as appropriate
  }

  // Set precision for floating point numbers in the output file
  output_file_actiondensity_clover << std::fixed << std::setprecision(12);
  bool header_written = false;

  Kokkos::Timer timer;
  bool accept;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};

  std::vector<real_t> action_densities_clover;
  // Write the header only once, before the first line of data
  if (!header_written) {
    output_file_actiondensity_clover << "hmc_step";
    output_file_actiondensity_clover << ",W_normal";
    for (const auto& t : factors) {
      output_file_actiondensity_clover << "," << t;
    }
    output_file_actiondensity_clover << "\n";
    header_written = true;
  }

  // hmc loop
  for (size_t step = 0; step < hmc.params.nsteps; ++step) {
    timer.reset();
    action_densities_clover.clear();

    // perform hmc_step
    accept = hmc.hmc_step();

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);

    if (KLFT_VERBOSITY > 0) {
      printf("Step: %ld, accepted: %ld, Acceptance rate: %f, Time: %f\n", step,
             static_cast<size_t>(accept), acc_rate, time);
    }

    // Simulation logging: log step, delta_H, acceptance rate, accept, time
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time);
    flushSimulationLogs(simLogParams, step, true);

    if (accept) {
      WilsonFlow<DGaugeFieldType> wilson_flow(
          hmc.hamiltonian_field.gauge_field, gaugeObsParams.wilson_flow_params);
      action_densities_clover.push_back(
          getActionDensity_clover<DGaugeFieldType>(
              hmc.hamiltonian_field.gauge_field));

      // perform wilson flow step
      wilson_flow.flow();

      action_densities_clover.push_back(
          getActionDensity_clover<DGaugeFieldType>(wilson_flow.field));

      for (const auto& t : factors) {
        auto tmp_params = gaugeObsParams.wilson_flow_params;
        tmp_params.eps *= static_cast<real_t>(t);
        tmp_params.n_steps = static_cast<int>(tmp_params.tau / tmp_params.eps);
        WilsonFlow<DGaugeFieldType>(hmc.hamiltonian_field.gauge_field,
                                    tmp_params);
        wilson_flow.flow_impr(b1);
        action_densities_clover.push_back(
            getActionDensity_clover<DGaugeFieldType>(wilson_flow.field));
      }

      // Write the data for the current step
      output_file_actiondensity_clover << step;
      for (const auto& density : action_densities_clover) {
        output_file_actiondensity_clover << "," << density;
      }
      output_file_actiondensity_clover << "\n";
    }
  }
  // Final flush of simulation logs
  forceflushSimulationLogs(simLogParams, true);
  return 0;
}

}  // namespace klft
