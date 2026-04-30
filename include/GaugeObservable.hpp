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

// this file defines the necessary functions to calculate
// different gauge observables to be measured during simulations

#pragma once
#include <mpi.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "ActionDensity.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugePlaquette.hpp"
#include "TopoCharge.hpp"
#include "WilsonFlow.hpp"
#include "WilsonLoop.hpp"

namespace klft {
// define a struct to hold parameters related to the gauge observables
struct GaugeObservableParams {
  size_t thermalization_steps;        // number of thermalization steps
  size_t measurement_interval;        // interval between measurements
  bool measure_plaquette;             // whether to measure the plaquette
  bool measure_wilson_loop_temporal;  // whether to measure the temporal Wilson
                                      // loop
  bool measure_wilson_loop_mu_nu;   // whether to measure the mu-nu Wilson loop
  bool measure_topological_charge;  // whether to measure the topological charge
  // TODO: measuring the density also needs to save the flowtime that was used
  bool measure_action_density;  // whether to measure the gauge density
  bool measure_sp_max;  // whether to measure the max of Re Tr (1 - Plaquette)

  std::vector<Kokkos::Array<index_t, 2>>
      W_temp_L_T_pairs;  // pairs of (L,T) for the temporal Wilson loop
  std::vector<Kokkos::Array<index_t, 2>>
      W_mu_nu_pairs;  // pairs of (mu,nu) for the mu-nu Wilson loop
  std::vector<Kokkos::Array<index_t, 2>>
      W_Lmu_Lnu_pairs;  // pairs of (Lmu,Lnu) for the Wilson loop

  // we also define vectors to hold the measurements
  std::vector<size_t>
      measurement_steps;  // steps at which measurements are taken
  std::vector<real_t> plaquette_measurements;  // measurements of the plaquette
  std::vector<std::vector<Kokkos::Array<real_t, 3>>>
      W_temp_measurements;  // L, T and corresponding W_temp
  std::vector<std::vector<Kokkos::Array<real_t, 5>>>
      W_mu_nu_measurements;  // mu, nu, Lmu, Lnu and corresponding W_mu_nu
  std::vector<real_t> topological_charge_measurements;  // measurements of the
                                                        // topological charge
  std::vector<real_t>
      action_density_measurements;  // measurements of the gauge density
  std::vector<real_t>
      sp_max_measurements;  // measurements of the max of Re Tr (1 - Plaquette)

  // finally, some filenames where the measurements will be flushed
  std::string plaquette_filename;  // filename for the plaquette measurements
  std::string
      W_temp_filename;  // filename for the temporal Wilson loop measurements
  std::string
      W_mu_nu_filename;  // filename for the mu-nu Wilson loop measurements
  std::string
      topological_charge_filename;  // filename for the topological charge
  std::string
      action_density_filename;  // filename for the gauge density measurements
  std::string sp_max_filename;  // filename for the max of Re Tr (1 - Plaquette)

  // boolean flag to indicate if the measurements are to be flushed
  bool write_to_file;

  WilsonFlowParams wilson_flow_params;  // parameters for the Wilson flow
  bool do_wilson_flow;                  // whether to perform the Wilson flow

  //
  size_t flush;  // interval to flush measurements to file, 0 to flush at the
                 // end of the simulation
  bool flushed;  // check if the measurements were flushed at least once -> used
                 // to add the header line to the file

  // constructor to initialize the parameters
  // by default nothing is measured
  GaugeObservableParams()
      : thermalization_steps(0),
        measurement_interval(1),
        measure_plaquette(false),
        measure_sp_max(false),
        measure_wilson_loop_temporal(false),
        measure_wilson_loop_mu_nu(false),
        measure_topological_charge(false),
        measure_action_density(false),
        flush(25),
        flushed(false),
        wilson_flow_params() {}
};

typedef enum {
  MPI_GAUGE_OBSERVABLES_PLAQUETTE = 0,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU = 1,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL = 2,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU_SIZE = 3,
  MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL_SIZE = 4,
  MPI_GAUGE_OBSERVABLES_TOPOLOGICAL_CHARGE = 5,
  MPI_GAUGE_OBSERVABLES_ACTION_DENSITY = 6,
  MPI_GAUGE_OBSERVABLES_SP_MAX = 7,
  MPI_GAUGE_OBSERVABLES_WILSONFLOW_DETAILS = 8,
  MPI_GAUGE_OBSERVABLES_WILSONFLOW_DETAILS_SIZE = 9
} MPI_GaugeObservableTags;
template <typename DGaugeFieldType>
void measureGaugeObservablesPTBC(const typename DGaugeFieldType::type& g_in,
                                 GaugeObservableParams& params,
                                 const size_t step,
                                 const int compute_rank,
                                 const bool do_compute = false) {
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind gkind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  using precision = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::value_type;

  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0) ||
      (step < params.thermalization_steps)) {
    return;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  real_t TopologicalCharge;
  real_t Plaquette;
  real_t ActionDensity;
  real_t SP_max;
  std::vector<Kokkos::Array<real_t, 5>> WilsonLoop_meas;
  std::vector<Kokkos::Array<real_t, 3>> WilsonTemp_measurements;

  if (do_compute) {
    if (rank == 0) {
      params.measurement_steps.push_back(step);
    }

    // otherwise, carry out the measurements
    if (KLFT_VERBOSITY > 1) {
      printf("Measurement of Gauge Observables\n");
      printf("step: %zu\n", step);
    }

    if constexpr (Nd == 4) {
      // Wilson flow is only defined for 4D gauge fields
      WilsonFlowParams wfparams = params.wilson_flow_params;
      using DGaugeFieldType_Standard =
          DeviceGaugeFieldType<Nd, Nc, complex_t, GaugeFieldKind::Standard>;
      WilsonFlow<DGaugeFieldType_Standard> wf(g_in, wfparams);
      if (params.do_wilson_flow) {
        if (KLFT_VERBOSITY > 1) {
          printf("Performing Wilson flow...\n");
        }
        wf.flow();
        if (KLFT_VERBOSITY > 1) {
          printf("Wilson flow completed.\n");
        }
      }

      if (params.measure_action_density) {
        // measure the gauge density if requested
        if (params.do_wilson_flow) {
          // perform the Wilson flow if requested
          ActionDensity =
              getActionDensity_clover<DGaugeFieldType_Standard>(wf.field);
        } else {
          ActionDensity =
              getActionDensity_clover<DGaugeFieldType_Standard>(g_in);
        }
        if (rank != 0) {
          MPI_Send(&ActionDensity, 1, mpi_real_t(), 0,
                   MPI_GAUGE_OBSERVABLES_ACTION_DENSITY, MPI_COMM_WORLD);
        } else {
          params.action_density_measurements.push_back(ActionDensity);
        }
        if (KLFT_VERBOSITY > 1) {
          printf("gauge density: %11.6f\n", ActionDensity);
        }
      }

      if (params.measure_sp_max) {
        // measure the max of Re Tr (1 - Plaquette) if requested
        if (params.do_wilson_flow) {
          // perform the Wilson flow if requested
          SP_max = get_spmax<DGaugeFieldType_Standard>(wf.field);
        } else {
          SP_max = get_spmax<DGaugeFieldType_Standard>(g_in);
        }
        if (rank != 0) {
          MPI_Send(&SP_max, 1, mpi_real_t(), 0, MPI_GAUGE_OBSERVABLES_SP_MAX,
                   MPI_COMM_WORLD);
        } else {
          params.sp_max_measurements.push_back(SP_max);
        }
        if (KLFT_VERBOSITY > 1) {
          printf("SP max: %11.6f\n", SP_max);
        }
      }

      if (params.measure_topological_charge && Nd == 4) {
        // measure the topological charge if requested
        if (params.do_wilson_flow) {
          // perform the Wilson flow if requested
          TopologicalCharge =
              get_topological_charge<DGaugeFieldType_Standard>(wf.field);
        } else {
          TopologicalCharge =
              get_topological_charge<DGaugeFieldType_Standard>(g_in);
        }
        if (rank != 0) {
          MPI_Send(&TopologicalCharge, 1, mpi_real_t(), 0,
                   MPI_GAUGE_OBSERVABLES_TOPOLOGICAL_CHARGE, MPI_COMM_WORLD);
        } else {
          params.topological_charge_measurements.push_back(TopologicalCharge);
        }
        if (KLFT_VERBOSITY > 1) {
          printf("topological charge: %11.6f\n", TopologicalCharge);
        }
      }
    }

    // measure the plaquette if requested
    if (params.measure_plaquette) {
      Plaquette = GaugePlaquette<Nd, Nc, precision, GaugeFieldKind::PTBC>(g_in);
      if (KLFT_VERBOSITY > 1) {
        printf("plaquette: %11.6f\n", Plaquette);
      }
      if (rank != 0) {
        MPI_Send(&Plaquette, 1, mpi_real_t(), 0,
                 MPI_GAUGE_OBSERVABLES_PLAQUETTE, MPI_COMM_WORLD);
      } else {
        params.plaquette_measurements.push_back(Plaquette);
      }
    }
    if (params.measure_wilson_loop_mu_nu) {
      if (KLFT_VERBOSITY > 1) {
        printf("Wilson loop in the mu-nu plane:\n");
        printf("mu, nu, Lmu, Lnu, W_mu_nu\n");
      }
      for (const auto& pair_mu_nu : params.W_mu_nu_pairs) {
        const index_t mu = pair_mu_nu[0];
        const index_t nu = pair_mu_nu[1];
        WilsonLoop_mu_nu<Nd, Nc, precision, GaugeFieldKind::Standard>(
            g_in, mu, nu, params.W_Lmu_Lnu_pairs, WilsonLoop_meas);
        if (KLFT_VERBOSITY > 1) {
          for (const auto& measure : WilsonLoop_meas) {
            printf("%d, %d, %d, %d, %11.6f\n", static_cast<index_t>(measure[0]),
                   static_cast<index_t>(measure[1]),
                   static_cast<index_t>(measure[2]),
                   static_cast<index_t>(measure[3]), measure[4]);
          }
        }
      }
      if (rank != 0) {
        size_t WilsonLoop_meas_size = WilsonLoop_meas.size();
        MPI_Send(&WilsonLoop_meas_size, 1, mpi_size_t(), 0,
                 MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU_SIZE, MPI_COMM_WORLD);
        if (WilsonLoop_meas_size > 0) {
          MPI_Send(WilsonLoop_meas.data(),
                   WilsonLoop_meas_size * sizeof(Kokkos::Array<real_t, 5>),
                   MPI_BYTE, 0, MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU,
                   MPI_COMM_WORLD);
        }
      } else {
        params.W_mu_nu_measurements.push_back(WilsonLoop_meas);
      }
    }

    if (params.measure_wilson_loop_temporal) {
      // measure the Wilson loop in the temporal direction
      WilsonLoop_temporal<Nd, Nc, precision, GaugeFieldKind::Standard>(
          g_in, params.W_temp_L_T_pairs, WilsonTemp_measurements);
      if (rank != 0) {
        size_t WilsonTemp_measurements_size = WilsonTemp_measurements.size();
        if (KLFT_VERBOSITY > 1) {
          printf(
              "Rank %d: Sending temporal Wilson loop measurements of size "
              "%zu\n",
              static_cast<index_t>(rank), WilsonTemp_measurements_size);
        }
        if (KLFT_VERBOSITY > 2) {
          printf("temporal Wilson loop:\n");
          printf("Rank, L, T, W_temp\n");
          for (const auto& measure : WilsonTemp_measurements) {
            printf("Rank: %d , %d, %d, %11.6f\n", static_cast<index_t>(rank),
                   static_cast<index_t>(measure[0]),
                   static_cast<index_t>(measure[1]), measure[2]);
          }
        }
        MPI_Send(&WilsonTemp_measurements_size, 1, mpi_size_t(), 0,
                 MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL_SIZE,
                 MPI_COMM_WORLD);
        if (WilsonTemp_measurements_size > 0) {
          MPI_Request send_request;
          MPI_Isend(
              WilsonTemp_measurements.data(),
              WilsonTemp_measurements_size * sizeof(Kokkos::Array<real_t, 3>),
              MPI_BYTE, 0, MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL,
              MPI_COMM_WORLD, &send_request);

          // Wait for send to complete
          MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        }
      } else {
        params.W_temp_measurements.push_back(WilsonTemp_measurements);
      }
      if (KLFT_VERBOSITY > 1) {
        printf("Rank %d: Send temporal Wilson loop measurements\n",
               static_cast<index_t>(rank));
      }
    }
  }

  if (rank == 0 && compute_rank != 0) {
    // only the computing rank measures the observables
    params.measurement_steps.push_back(step);

    // ... inside if (rank == 0) { ...
    if (compute_rank != 0 &&
        params.wilson_flow_params.dynamicParams.log_details) {
      DEBUG_MPI_PRINT(
          "Receiving wilson flow log details size from compute rank: %d\n",
          compute_rank);
      size_t size{0};

      MPI_Recv(&size, 1, mpi_size_t(), compute_rank,
               MPI_GAUGE_OBSERVABLES_WILSONFLOW_DETAILS_SIZE, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      char* buffer = new char[size + 1];
      MPI_Recv(buffer, size, MPI_CHAR, compute_rank,
               MPI_GAUGE_OBSERVABLES_WILSONFLOW_DETAILS, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      std::string log_string(buffer, size);
      params.wilson_flow_params.dynamicParams.log_strings.push_back(log_string);
      delete[] buffer;  // <-- Added memory cleanup
    }
    // ...

    if constexpr (Nd == 4) {
      if (params.measure_action_density) {
        // send the gauge density measurement to the compute rank
        MPI_Recv(&ActionDensity, 1, mpi_real_t(), compute_rank,
                 MPI_GAUGE_OBSERVABLES_ACTION_DENSITY, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        params.action_density_measurements.push_back(ActionDensity);
      }

      if (params.measure_topological_charge && Nd == 4) {
        // measure the topological charge if requested
        MPI_Recv(&TopologicalCharge, 1, mpi_real_t(), compute_rank,
                 MPI_GAUGE_OBSERVABLES_TOPOLOGICAL_CHARGE, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        params.topological_charge_measurements.push_back(TopologicalCharge);
      }
      if (params.measure_plaquette) {
        // send the plaquette measurement to the compute rank
        MPI_Recv(&Plaquette, 1, mpi_real_t(), compute_rank,
                 MPI_GAUGE_OBSERVABLES_PLAQUETTE, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        params.plaquette_measurements.push_back(Plaquette);
      }
    }

    if (params.measure_wilson_loop_mu_nu) {
      // MPI_Recv(&WilsonLoop_meas, 1, mpi_real_t(), compute_rank, 0,
      //  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      size_t WilsonLoop_meas_size;
      MPI_Recv(&WilsonLoop_meas_size, 1, mpi_size_t(), compute_rank,
               MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU_SIZE, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      if (WilsonLoop_meas_size > 0) {
        WilsonLoop_meas.resize(WilsonLoop_meas_size);
        MPI_Recv(WilsonLoop_meas.data(),
                 WilsonLoop_meas_size * sizeof(Kokkos::Array<real_t, 5>),
                 MPI_BYTE, compute_rank,
                 MPI_GAUGE_OBSERVABLES_WILSON_LOOP_MU_NU, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
      params.W_mu_nu_measurements.push_back(WilsonLoop_meas);
    }

    if (params.measure_wilson_loop_temporal) {
      // measure the Wilson loop in the temporal direction
      size_t WilsonTemp_measurements_size;
      MPI_Recv(&WilsonTemp_measurements_size, 1, mpi_size_t(), compute_rank,
               MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL_SIZE, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      if (KLFT_VERBOSITY > 1) {
        printf(
            "Rank 0: Receiving temporal Wilson loop measurements of size %zu\n",
            WilsonTemp_measurements_size);
      }
      if (KLFT_VERBOSITY > 2) {
        printf("temporal Wilson loop:\n");
        printf("Rank, L, T, W_temp\n");
        for (const auto& measure : WilsonTemp_measurements) {
          printf("Rank: %d , %d, %d, %11.6f\n", static_cast<index_t>(rank),
                 static_cast<index_t>(measure[0]),
                 static_cast<index_t>(measure[1]), measure[2]);
        }
      }
      if (WilsonTemp_measurements_size > 0) {
        WilsonTemp_measurements.resize(WilsonTemp_measurements_size);
        MPI_Recv(
            WilsonTemp_measurements.data(),
            WilsonTemp_measurements_size * sizeof(Kokkos::Array<real_t, 3>),
            MPI_BYTE, compute_rank, MPI_GAUGE_OBSERVABLES_WILSON_LOOP_TEMPORAL,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      if (KLFT_VERBOSITY > 1) {
        printf("Rank 0: Received temporal Wilson loop measurements\n");
      }

      params.W_temp_measurements.push_back(WilsonTemp_measurements);
    }

    if (params.measure_sp_max) {
      // send the plaquette measurement to the compute rank
      MPI_Recv(&SP_max, 1, mpi_real_t(), compute_rank,
               MPI_GAUGE_OBSERVABLES_SP_MAX, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      params.sp_max_measurements.push_back(SP_max);
    }
  }
}

// define a function to measure the gauge observables
template <typename DGaugeFieldType>
void measureGaugeObservables(const typename DGaugeFieldType::type& g_in,
                             GaugeObservableParams& params,
                             const size_t step) {
  constexpr static const size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  // check if the step is a measurement step
  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0) ||
      (step < params.thermalization_steps)) {
    return;
  }
  // otherwise, carry out the measurements
  if (KLFT_VERBOSITY > 1) {
    printf("Measurement of Gauge Observables\n");
    printf("step: %zu\n", step);
  }

  if constexpr (Nd == 4) {
    WilsonFlowParams& wfparams = params.wilson_flow_params;
    WilsonFlow<DGaugeFieldType> wf(g_in, wfparams);
    if (params.do_wilson_flow) {
      if (KLFT_VERBOSITY > 1) {
        printf("Performing Wilson flow...\n");
      }
      wf.flow();
    }

    if (params.measure_action_density) {
      // measure the gauge density if requested
      real_t Density_E;
      if (params.do_wilson_flow) {
        // perform the Wilson flow if requested
        Density_E = getActionDensity_clover<DGaugeFieldType>(wf.field);
      } else {
        Density_E = getActionDensity_clover<DGaugeFieldType>(g_in);
      }
      params.action_density_measurements.push_back(Density_E);
      if (KLFT_VERBOSITY > 1) {
        printf("gauge density: %11.6f\n", Density_E);
      }
    }
    if (params.measure_topological_charge) {
      // now calculate the topological charge
      real_t TopologicalCharge;
      if (params.do_wilson_flow) {
        // perform the Wilson flow if requested
        TopologicalCharge = get_topological_charge<DGaugeFieldType>(wf.field);
      } else {
        TopologicalCharge = get_topological_charge<DGaugeFieldType>(g_in);
      }
      params.topological_charge_measurements.push_back(TopologicalCharge);
      if (KLFT_VERBOSITY > 1) {
        printf("topological charge: %11.6f\n", TopologicalCharge);
      }
    }
    if (params.measure_sp_max) {
      // measure the max of Re Tr (1 - Plaquette) if requested
      real_t SP_max;
      if (params.do_wilson_flow) {
        // perform the Wilson flow if requested
        SP_max = get_spmax<DGaugeFieldType>(wf.field);
      } else {
        SP_max = get_spmax<DGaugeFieldType>(g_in);
      }
      params.sp_max_measurements.push_back(SP_max);
      if (KLFT_VERBOSITY > 1) {
        printf("SP max: %11.6f\n", SP_max);
      }
    }
  }
  // measure the plaquette if requested
  if (params.measure_plaquette) {
    real_t P = GaugePlaquette<Nd, Nc>(g_in);
    params.plaquette_measurements.push_back(P);
    if (KLFT_VERBOSITY > 1) {
      printf("plaquette: %11.6f\n", P);
    }
  }
  // measure the Wilson loop in the temporal direction
  if (params.measure_wilson_loop_temporal) {
    if (KLFT_VERBOSITY > 1) {
      printf("temporal Wilson loop:\n");
      printf("L, T, W_temp\n");
    }
    std::vector<Kokkos::Array<real_t, 3>> temp_measurements;
    WilsonLoop_temporal<Nd, Nc>(g_in, params.W_temp_L_T_pairs,
                                temp_measurements);
    if (KLFT_VERBOSITY > 1) {
      for (const auto& measure : temp_measurements) {
        printf("%d, %d, %11.6f\n", static_cast<index_t>(measure[0]),
               static_cast<index_t>(measure[1]), measure[2]);
      }
    }
    params.W_temp_measurements.push_back(temp_measurements);
  }
  // measure the Wilson loop in the mu-nu plane
  if (params.measure_wilson_loop_mu_nu) {
    if (KLFT_VERBOSITY > 1) {
      printf("Wilson loop in the mu-nu plane:\n");
      printf("mu, nu, Lmu, Lnu, W_mu_nu\n");
    }
    std::vector<Kokkos::Array<real_t, 5>> temp_measurements;
    for (const auto& pair_mu_nu : params.W_mu_nu_pairs) {
      const index_t mu = pair_mu_nu[0];
      const index_t nu = pair_mu_nu[1];
      WilsonLoop_mu_nu<Nd, Nc>(g_in, mu, nu, params.W_Lmu_Lnu_pairs,
                               temp_measurements);
      if (KLFT_VERBOSITY > 1) {
        for (const auto& measure : temp_measurements) {
          printf("%d, %d, %d, %d, %11.6f\n", static_cast<index_t>(measure[0]),
                 static_cast<index_t>(measure[1]),
                 static_cast<index_t>(measure[2]),
                 static_cast<index_t>(measure[3]), measure[4]);
        }
      }
    }
    params.W_mu_nu_measurements.push_back(temp_measurements);
  }
  // add the step to the measurement steps
  params.measurement_steps.push_back(step);
  return;
}

inline void flushSPMax(std::ofstream& file,
                       const GaugeObservableParams& params,
                       const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if sp_max measurements are available
  if (!params.measure_sp_max) {
    printf("Error: no sp_max measurements available\n");
    return;
  }
  if (HEADER)
    file << "# step, sp_max\n";
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", " << params.sp_max_measurements[i]
         << "\n";
  }
}

inline void flushTopologicalCharge(std::ofstream& file,
                                   const GaugeObservableParams& params,
                                   const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if topological charge measurements are available
  if (!params.measure_topological_charge) {
    printf("Error: no topological charge measurements available\n");
    return;
  }
  if (HEADER)
    file << "step,topological_charge\n";
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", "
         << params.topological_charge_measurements[i] << "\n";
  }
}

inline void flushActionDensity(std::ofstream& file,
                               const GaugeObservableParams& params,
                               const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if density_E measurements are available
  if (!params.measure_action_density) {
    printf("Error: no density_E measurements available\n");
    return;
  }
  if (HEADER)
    file << "step, action_density,tsquaredxaction_density\n";
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", "
         << params.action_density_measurements[i] << ", "
         << params.action_density_measurements[i] *
                params.wilson_flow_params.tau * params.wilson_flow_params.tau
         << "\n";
  }
}

// flush the plaquette measurements to disk
inline void flushPlaquette(std::ofstream& file,
                           const GaugeObservableParams& params,
                           const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if plaquette measurements are available
  if (!params.measure_plaquette) {
    printf("Error: no plaquette measurements available\n");
    return;
  }
  if (HEADER)
    file << "step,plaquette\n";
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", "
         << params.plaquette_measurements[i] << "\n";
  }
}

// flush the temporal Wilson loop measurements to disk
inline void flushWilsonLoopTemporal(std::ofstream& file,
                                    const GaugeObservableParams& params,
                                    const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if temporal Wilson loop measurements are available
  if (!params.measure_wilson_loop_temporal) {
    printf("Error: no temporal Wilson loop measurements available\n");
    return;
  }
  if (HEADER)
    file << "step,L,T,W_temp\n";
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto& measurement : params.W_temp_measurements[i]) {
      file << params.measurement_steps[i] << ", " << measurement[0] << ", "
           << measurement[1] << ", " << measurement[2] << "\n";
    }
  }
}

// flush the mu-nu Wilson loop measurements to disk
inline void flushWilsonLoopMuNu(std::ofstream& file,
                                const GaugeObservableParams& params,
                                const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if mu-nu Wilson loop measurements are available
  if (!params.measure_wilson_loop_mu_nu) {
    printf("Error: no mu-nu Wilson loop measurements available\n");
    return;
  }
  if (HEADER)
    file << "step,mu,nu,Lmu,Lnu,W_mu_nu\n";
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    for (const auto& measurement : params.W_mu_nu_measurements[i]) {
      file << params.measurement_steps[i] << ", " << measurement[0] << ", "
           << measurement[1] << ", " << measurement[2] << ", " << measurement[3]
           << ", " << measurement[4] << "\n";
    }
  }
}

inline void flushWilsonFlowDetails(std::ofstream& file,
                                   GaugeObservableParams& params,
                                   const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  WilsonFlowParams& wfparams = params.wilson_flow_params;
  if (HEADER) {
    file << "# step, flow_step, flow_time, sp_max_init, sp_max, sp_max_deriv, "
            "tsquaredxaction_density(old), next_measure_t^E_step, "
            "tsquaredxaction_density\n";
  }
  for (size_t i = 0; i < params.measurement_steps.size(); ++i) {
    file << params.measurement_steps[i] << ", "
         << wfparams.dynamicParams.log_strings[i] << "\n";
  }
}

// function to clear all measurements
inline void clearAllGaugeObservables(GaugeObservableParams& params) {
  params.measurement_steps.clear();
  params.topological_charge_measurements.clear();
  params.plaquette_measurements.clear();
  params.W_temp_measurements.clear();
  params.W_mu_nu_measurements.clear();
  params.action_density_measurements.clear();
  params.sp_max_measurements.clear();

  // ...
  // add more clear functions for other observables here
}

// define a global function to flush all measurements
inline void forceflushAllGaugeObservables(
    GaugeObservableParams& params,
    const bool clear_after_flush = false,
    const int& p = std::cout.precision()) {
  auto _ = std::setprecision(p);
  // check if write_to_file is enabled
  if (!params.write_to_file) {
    printf("write_to_file is not enabled\n");
    return;
  }
  bool HEADER = !params.flushed;  // write header only once
  // flush plaquette measurements
  if (params.measure_topological_charge &&
      params.topological_charge_filename != "") {
    std::ofstream file(params.topological_charge_filename, std::ios::app);
    flushTopologicalCharge(file, params, HEADER);
    file.close();
  }

  if (params.measure_sp_max && params.sp_max_filename != "") {
    std::ofstream file(params.sp_max_filename, std::ios::app);
    flushSPMax(file, params, HEADER);
    file.close();
  }

  if (params.measure_action_density && params.action_density_filename != "") {
    std::ofstream file(params.action_density_filename, std::ios::app);
    flushActionDensity(file, params, HEADER);
    file.close();
  }

  if (params.measure_plaquette && params.plaquette_filename != "") {
    std::ofstream file(params.plaquette_filename, std::ios::app);
    flushPlaquette(file, params, HEADER);
    file.close();
  }
  // flush temporal Wilson loop measurements
  if (params.measure_wilson_loop_temporal && params.W_temp_filename != "") {
    std::ofstream file(params.W_temp_filename, std::ios::app);
    flushWilsonLoopTemporal(file, params, HEADER);
    file.close();
  }
  // flush mu-nu Wilson loop measurements
  if (params.measure_wilson_loop_mu_nu && params.W_mu_nu_filename != "") {
    std::ofstream file(params.W_mu_nu_filename, std::ios::app);
    flushWilsonLoopMuNu(file, params, HEADER);
    file.close();
  }

  // ...
  // add more flush functions for other observables here
  if (clear_after_flush) {
    clearAllGaugeObservables(params);
  }
  params.flushed = true;  // set flushed to true after flushing
}

// check if the current step should be flushed, if so call
// forceflushAllGaugeObservables
inline void flushAllGaugeObservables(GaugeObservableParams& params,
                                     const size_t step,
                                     const bool clear_after_flush = false,
                                     const int& p = std::cout.precision()) {
  if (params.flush != 0 && step % params.flush == 0) {
    forceflushAllGaugeObservables(params, clear_after_flush, p);
  }
}

}  // namespace klft
