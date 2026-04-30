#pragma once
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <random>
#include <sstream>

#include "FermionObservable.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HMC.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "IndexHelper.hpp"
#include "SimulationLogging.hpp"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

struct PTBCParams {
  // Define parameters for the PTBC algorithm
  index_t n_sims;
  std::vector<real_t>
      defects;  // a vector that hold the different defect values
  std::vector<real_t> prev_defects;  // used for logging purposes
  index_t defect_length;             // size of the defect on the lattice

  GaugeMonomial_Params gauge_params;  // HMC parameters´
  PTBCSimulationLoggingParams ptbcSimLogParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;
  FermionObservableParams fermionObsParams;
  std::string to_string() const {
    std::ostringstream oss;
    oss << "PTBCParams: n_sims = " << n_sims
        << ", defect_length = " << defect_length << ", defects = [";
    for (const auto& defect : defects) {
      oss << defect << " ";  // This will now save the rank correponding to each
                             // defect value, so opposite as before
    }
    oss << "]";
    return oss.str();
  }
};

template <typename DGaugeFieldType, typename DAdjFieldType, class RNG>
class PTBC {  // do I need the AdjFieldType here?
  // template argument deduction and safety
  static_assert(DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind ==
                GaugeFieldKind::PTBC);
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);

 public:
  constexpr static size_t Nd =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((Nd == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;
  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  using HMCType = HMC<DGaugeFieldType, DAdjFieldType, RNG>;

  PTBCParams& params;  // parameters for the PTBC algorithm
  HMCType& hmc;
  // std::shared_ptr<LeapFrog> integrator; // TODO: make integrator agnostic
  RNG& rng;
  std::mt19937 mt;
  std::uniform_real_distribution<real_t> dist;

  index_t current_index;

  std::vector<bool> swap_accepts;   // a vector that hold the last values shown
                                    // if a given swap was accepted
  std::vector<real_t> swap_deltas;  // a vector that holds the partial Delta_S
                                    // values for each swap
  bool starting_defect_value;

  typedef enum {
    TAG_DELTAS = 0,
    TAG_ACCEPT = 1,
    TAG_INDEX = 2,
    TAG_SWAPSTART = 3,
    TAG_SHIFTDEFECT = 4
  } MPI_Tags;

  PTBC() = delete;  // default constructor is not allowed

  PTBC(PTBCParams& params_,
       HMCType& hmc_,
       RNG& rng_,
       std::uniform_real_distribution<real_t> dist_,
       std::mt19937 mt_)
      : params(params_), rng(rng_), dist(dist_), mt(mt_), hmc(hmc_) {
    device_id = Kokkos::device_id();  // default device id
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    current_index = rank;

    swap_accepts.resize(std::max<size_t>(params.defects.size() - 1, 1));
    swap_deltas.resize(std::max<size_t>(params.defects.size() - 1, 1));

    hmc.hamiltonian_field.gauge_field.template set_defect<index_t>(
        params.defects[current_index]);
    Kokkos::fence();

    // host only fallback
    if (device_id == -1) {
      device_id = 0;
      partner_device_id = 0;  // default to first two devices
    } else {
      partner_device_id =
          (device_id + 1) % Kokkos::num_devices();  // default partner device id
    }
  }
  void load(const std::string path_to_file) {
    size_t pos = path_to_file.find_last_of("/");
    std::string folder;

    if (pos != std::string::npos) {
      folder = path_to_file.substr(0, pos);
    } else {
      folder = "";
    }
    std::cout << "Folder found:" << folder << " . \n";
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::string substr = "rank" + std::to_string(rank);
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
      if (entry.path().string().find(substr) != std::string::npos) {
        // s contains substr
        hmc.hamiltonian_field.gauge_field.load(entry.path());
        MPI_Allgather(&hmc.hamiltonian_field.gauge_field.dParams.defect_value,
                      1, mpi_real_t(), params.defects.data(), 1, mpi_real_t(),
                      MPI_COMM_WORLD);
        params.defect_length =
            hmc.hamiltonian_field.gauge_field.dParams.defect_length;
        if (KLFT_VERBOSITY > 1) {
          /* code */

          std::cout << "Rank " << rank << " loaded from path:" << entry.path()
                    << "with the following paramets:\n";
          std::cout << "Rank " << rank << " Pares: "
                    << hmc.hamiltonian_field.gauge_field.dParams.format()
                    << "\n";
          std::cout << "Rank " << rank << " PTBCParams:" << params.to_string()
                    << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        return;
      }
    }
    std::string err = "FATAL ERROR: Rank " + std::to_string(rank) +
                      "could't load gauge config";
    throw std::runtime_error(err);
  }

  real_t getDefectValue() const {
    // return the defect value for the current index
    return params.defects[current_index];
  }

  void measure(PTBCSimulationLoggingParams& ptbcSimLogParams,
               const size_t step) {
    // measure the simulation logging observables
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
      addPTBCLogData(ptbcSimLogParams, step, starting_defect_value,
                     &swap_accepts, &swap_deltas, &params.defects,
                     &params.prev_defects);  // add the data to the log
    }
  }

  void measure(GaugeObservableParams& gaugeObsParams, size_t step) {
    // measure the gauge observables
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int compute_rank{0};
    int dummy_rank{0};

    if (rank == 0 || getDefectValue() >= 0.99999) {
      // only the computing rank or the defect rank measures the observables
      MPI_Reduce(
          &rank, &compute_rank, 1, MPI_INT, MPI_SUM, 0,
          MPI_COMM_WORLD);  // sum the ranks to get the computing rank in rank 0
      if (getDefectValue() >= 0.99999) {
        // if the defect value is close to 1, we measure the observables
        // for the PTBC case
        if (KLFT_VERBOSITY > 1) {
          printf("Measuring PTBC observables at step %zu\n", step);
        }
        measureGaugeObservablesPTBC<DGaugeFieldType>(
            hmc.hamiltonian_field.gauge_field, gaugeObsParams, step, rank,
            true);
      } else {
        measureGaugeObservablesPTBC<DGaugeFieldType>(
            hmc.hamiltonian_field.gauge_field, gaugeObsParams, step,
            compute_rank, false);
      }
    } else {
      MPI_Reduce(&dummy_rank, &compute_rank, 1, MPI_INT, MPI_SUM, 0,
                 MPI_COMM_WORLD);
      return;  // skip measurement for other ranks
    }
  }

  void measure(FermionObservableParams& fermionObsParam, size_t step) {
    // measure the gauge observables
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int compute_rank{0};
    int dummy_rank{0};

    if (rank == 0 || getDefectValue() >= 0.99999) {
      // only the computing rank or the defect rank measures the observables
      MPI_Reduce(
          &rank, &compute_rank, 1, MPI_INT, MPI_SUM, 0,
          MPI_COMM_WORLD);  // sum the ranks to get the computing rank in rank 0
      if (getDefectValue() >= 0.99999) {
        // if the defect value is close to 1, we measure the observables
        // for the PTBC case
        if (KLFT_VERBOSITY > 1) {
          printf("Measuring   Fermionic observables at step %zu\n", step);
        }
        measureFermionObservablesPTBC<
            std::mt19937,
            DeviceSpinorFieldType<HMCType::rank, HMCType::Nc, 4, complex_t>,
            DeviceGaugeFieldType<Nd, Nc, complex_t, GaugeFieldKind::Standard>,
            CGSolver, EOWilsonDiracOperator>(hmc.hamiltonian_field.gauge_field,
                                             fermionObsParam, step, 0, hmc.mt,
                                             true);
      } else {
        measureFermionObservablesPTBC<
            std::mt19937,
            DeviceSpinorFieldType<HMCType::rank, HMCType::Nc, 4, complex_t>,
            DeviceGaugeFieldType<Nd, Nc, complex_t, GaugeFieldKind::Standard>,
            CGSolver, EOWilsonDiracOperator>(hmc.hamiltonian_field.gauge_field,
                                             fermionObsParam, step,
                                             compute_rank, hmc.mt, false);
      }
    } else {
      MPI_Reduce(&dummy_rank, &compute_rank, 1, MPI_INT, MPI_SUM, 0,
                 MPI_COMM_WORLD);
      return;  // skip measurement for other ranks
    }
  }

  void measure(SimulationLoggingParams& simLogParams,
               index_t step,
               real_t acc_rate,
               bool accept,
               real_t time,
               real_t obs_time) {
    // measure the simulation logging observables
    addLogData(simLogParams, step, hmc.delta_H, acc_rate, accept, time,
               obs_time);
  }

  int step() {
    auto rtn = hmc.hmc_step();
    return rtn;
  }

  // TODO ist that correct?
  real_t swap_partner(index_t partner_rank) {
    // swaps the Defect with the partner rank and returns the partial Delta_S

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    real_t S_ss = getActionAroundDefect();
    // DEBUG_MPI_PRINT("S_ss = %f", S_ss);
    // now do the index/defect swapping
    hmc.hamiltonian_field.gauge_field.template set_defect<index_t>(
        params.defects[partner_rank]);  // set the defect value of the current
    //
    real_t S_sr = getActionAroundDefect();
    // DEBUG_MPI_PRINT("S_sr = %f", S_sr);
    return S_sr - S_ss;  // return the partial Delta_S
  }

  void reverse_swap(bool accept) {
    if (!accept) {
      hmc.hamiltonian_field.gauge_field.template set_defect<index_t>(
          params
              .defects[current_index]);  // set the defect value of the current
    }
  }

  void checkForNANs() {
    if constexpr (Nd == 4) {
#ifdef DEBUG_MPI
      // Create a host mirror and copy data from device to host
      auto host_field = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), hmc.hamiltonian_field.gauge_field.field);
      auto dimensions = hmc.hamiltonian_field.gauge_field.dimensions;

      for (index_t i = 0; i < dimensions[0]; ++i)
        for (index_t j = 0; j < dimensions[1]; ++j)
          for (index_t k = 0; k < dimensions[2]; ++k)
            for (index_t l = 0; l < dimensions[3]; ++l)
              for (index_t mu = 0; mu < Nd; ++mu) {
                auto link = host_field(i, j, k, l, mu);
                for (index_t c1 = 0; c1 < Nc; ++c1)
                  for (index_t c2 = 0; c2 < Nc; ++c2) {
                    if (std::isnan(Kokkos::real(link[c1][c2]))) {
                      DEBUG_MPI_PRINT("NaN at (%d,%d,%d,%d,mu=%d)", i, j, k, l,
                                      mu);
                    }
                  }
              }
#endif
    }
  }
  void shift_defect() {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // shift[0] is the direction, shift [1] is the amount (1 or -1)
    int shift[2];
    if (rank == 0) {
      shift[0] =
          int(dist(mt) *
              (Nd - 1));  // random direction but not in t direction ( so 0,1,2)
      shift[1] = (dist(mt) > 0.5) ? 1 : -1;
    }
    MPI_Bcast(shift, 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (getDefectValue() == 1) {
      auto old_position =
          hmc.hamiltonian_field.gauge_field.dParams.defect_position;
      auto new_position = old_position;
      new_position[shift[0]] =
          (old_position[shift[0]] + shift[1] +
           (old_position[shift[0]] == 0) * (shift[1] < 0) *
               hmc.hamiltonian_field.gauge_field.dimensions[shift[0]]) %
          hmc.hamiltonian_field.gauge_field.dimensions[shift[0]];
      hmc.hamiltonian_field.gauge_field.shift_defect(new_position);
    }
  }

  std::vector<index_t> argsort(const std::vector<real_t>& vec,
                               bool descending = true) {
    // Create a vector of indices
    std::vector<index_t> indices(vec.size());
    for (index_t i = 0; i < vec.size(); ++i) {
      indices[i] = i;
    }

    if (!descending) {
      // Sort the indices based on the values in vec
      std::sort(indices.begin(), indices.end(),
                [&vec](real_t a, real_t b) { return vec[a] < vec[b]; });
    } else {
      // Sort the indices based on the values in vec in descending order
      std::sort(indices.begin(), indices.end(),
                [&vec](real_t a, real_t b) { return vec[a] > vec[b]; });
    }
    return indices;
  }

  int swap() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    params.prev_defects = params.defects;  // store previous defects for logging

    bool accept = false;
    real_t Delta_S{0};
    starting_defect_value = true;

    // Rank 0 determines swap_start and broadcasts
    if (rank == 0) {
      starting_defect_value = bool(int(dist(mt) * 2));
    }
    MPI_Bcast(&starting_defect_value, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    for (index_t i = 0; i < size - 1; ++i) {  // interpret as loop over c values
      // assumption: the defects vector is up to  date and synchronized across
      // ranks
      std::vector<index_t> sorted_indices = argsort(
          params.defects, starting_defect_value);  // get the sorted indices

      auto swap_rank = sorted_indices[i];
      auto partner_rank = sorted_indices[i + 1];

      // SWAP RANK sends its Delta_S
      if (rank == swap_rank) {
        // DEBUG_MPI_PRINT(
        //     "%s, DefectLength(Field): %d", params.to_string().c_str(),
        //     hmc.hamiltonian_field.gauge_field.dParams.defect_length);
        DEBUG_MPI_PRINT(
            "Iteration %d: swap_rank=%d, defect(PTBC)=%f , "
            "defect(Field)=%f \n\t "
            "partner_rank = % d defect(PTBC) = %f , starting_defect_value: %d ",
            i, swap_rank, params.defects[swap_rank],
            hmc.hamiltonian_field.gauge_field.get_defect(), partner_rank,
            params.defects[partner_rank], int(starting_defect_value));
        real_t temp = swap_partner(partner_rank);

        MPI_Send(&temp, 1, mpi_real_t(), 0, TAG_DELTAS, MPI_COMM_WORLD);
      }

      // PARTNER RANK sends its Delta_S
      if (rank == partner_rank) {
        real_t temp = swap_partner(swap_rank);
        MPI_Send(&temp, 1, mpi_real_t(), 0, TAG_DELTAS, MPI_COMM_WORLD);
      }

      // RANK 0 gathers both values
      if (rank == 0) {
        real_t Delta_S_partner1, Delta_S_partner2;
        MPI_Recv(&Delta_S_partner1, 1, mpi_real_t(), swap_rank, TAG_DELTAS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(&Delta_S_partner2, 1, mpi_real_t(), partner_rank, TAG_DELTAS,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Delta_S = Delta_S_partner1 + Delta_S_partner2;

        // Decide accept/reject
        accept = true;
        if (Delta_S > 0) {
          if (dist(mt) > Kokkos::exp(-Delta_S)) {
            accept = false;
          }
        }

        MPI_Send(&accept, 1, MPI_C_BOOL, swap_rank, TAG_ACCEPT, MPI_COMM_WORLD);
        MPI_Send(&accept, 1, MPI_C_BOOL, partner_rank, TAG_ACCEPT,
                 MPI_COMM_WORLD);

        swap_accepts[i] = accept;
        swap_deltas[i] = Delta_S;
      }

      // Swap ranks receive accept flag
      if (rank == swap_rank || rank == partner_rank) {
        MPI_Recv(&accept, 1, MPI_C_BOOL, 0, TAG_ACCEPT, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        reverse_swap(accept);
      }

      // Rank 0 performs the swap if accepted
      if (rank == 0 && accept) {
        std::swap(params.defects[swap_rank], params.defects[partner_rank]);
      }

      MPI_Barrier(MPI_COMM_WORLD);  // synchronize all ranks after each swap
      MPI_Bcast(params.defects.data(), params.defects.size(), mpi_real_t(), 0,
                MPI_COMM_WORLD);

      if (rank == 0) {
        std::ostringstream oss;
        oss << "Defects after broadcast: [";
        for (size_t i = 0; i < params.defects.size(); ++i) {
          oss << params.defects[i];
          if (i + 1 < params.defects.size())
            oss << ", ";
        }
        oss << "]";
        // DEBUG_MPI_PRINT("%s", oss.str().c_str());
      }
      MPI_Barrier(MPI_COMM_WORLD);  // synchronize all ranks after each swap
    }
    // TODO: shift the defect by one lattice spacing in a random direction
    shift_defect();
    MPI_Barrier(MPI_COMM_WORLD);
    //
    return 0;
  }

  real_t getActionAroundDefect() {
    // calculate the plaquette around the defect
    // this is a placeholder function,
    // TODO: implement the actual action around
    // the defect

    auto action = WilsonAction<DGaugeFieldType>(
        hmc.hamiltonian_field.gauge_field, params.gauge_params.beta);

    return action;
  }

 private:
  int device_id;
  int partner_device_id;
};

// below: Functions used to dispatch the PTBC algorithm

template <typename PTBCType>
int run_PTBC(PTBCType& ptbc, Integrator_Params& int_params) {
  Kokkos::Timer timer;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (size_t step = 0; step < int_params.nsteps; ++step) {
    timer.reset();

    // DEBUG_MPI_PRINT("Enter ptbc(hmc) step: %zu", step);

    int accept = ptbc.step();
    MPI_Barrier(MPI_COMM_WORLD);  // synchronize all ranks after step
    // DEBUG_MPI_PRINT("HMC Step %zu: accept = %d", step, accept);

    int ptbc_accept = ptbc.swap();

    const real_t time = timer.seconds();
    timer.reset();
    // Gauge observables
    // TODO:only measure on rank 0 and rank with cval = 1.0 , find via
    // defect_positions[std::find(params.defect_positions.begin(),params.defect_positions.end(),1.0)-params.defect_positions.begin()],
    // alt define one ore do it once at the beginning because cval will be
    // always the same index, since params.defect will not change
    ptbc.measure(ptbc.params.gaugeObsParams, step);

    const real_t obs_time = timer.seconds();
    ptbc.measure(ptbc.params.ptbcSimLogParams, step);
    ptbc.measure(ptbc.params.fermionObsParams, step);

    if (rank == 0) {
      flushAllGaugeObservables(ptbc.params.gaugeObsParams, step, true);
      flushPTBCSimulationLogs(ptbc.params.ptbcSimLogParams, step, true);
      flushAllFermionObservables(ptbc.params.fermionObsParams, step, true);
    }
    // PTBC swap/accept

    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);
    ptbc.measure(ptbc.params.simLogParams, step, acc_rate, accept, time,
                 obs_time);
    if (rank == 0) {
      time_t now = std::time(nullptr);
      char time_str[26];
      std::strftime(time_str, sizeof(time_str), "%Y %b %d %H:%M:%S",
                    std::localtime(&now));
      Kokkos::printf(
          "%s - Step: %zu, accepted: %d, Acceptance rate: %f, Time: %f\n",
          time_str, step, accept, acc_rate, time);
    }
    flushSimulationLogs(ptbc.params.simLogParams, step, true);
    flushIOPTBC<DeviceGaugeFieldType<PTBCType::Nd, PTBCType::Nc, complex_t,
                                     GaugeFieldKind::PTBC>>(
        ptbc.hmc.ioParams, rank, step, ptbc.hmc.hamiltonian_field.gauge_field);
  }

  MPI_Barrier(MPI_COMM_WORLD);  // synchronize all ranks after the loop
  if (rank == 0) {
    forceflushAllGaugeObservables(ptbc.params.gaugeObsParams, true);
    forceflushPTBCSimulationLogs(ptbc.params.ptbcSimLogParams, true);
    forceflushAllFermionObservables(ptbc.params.fermionObsParams, true);
  }
  forceflushSimulationLogs(ptbc.params.simLogParams, true);
  flushIOPTBC<DeviceGaugeFieldType<PTBCType::Nd, PTBCType::Nc, complex_t,
                                   GaugeFieldKind::PTBC>>(
      ptbc.hmc.ioParams, rank, int_params.nsteps,
      ptbc.hmc.hamiltonian_field.gauge_field, true);
  return 0;
}

// #define INITIALIZE_PTBCPREPARE(ND, NC, RNG) \
//   template int run_PTBC<DeviceGaugeFieldType<ND, NC, GaugeFieldKind::PTBC>, \
//                         DeviceAdjFieldType<ND, NC>, RNG>( \
//       PTBCParams, RNG &, std::uniform_real_distribution<real_t>, \
//       std::mt19937);
// // INITIALIZE_PTBCPREPARE(2, 1, RNGType)
// // INITIALIZE_PTBCPREPARE(2, 2, RNGType)
// // // INITIALIZE_PTBCPREPARE(2, 3, RNGType)
// // INITIALIZE_PTBCPREPARE(3, 1, RNGType)
// // INITIALIZE_PTBCPREPARE(3, 2, RNGType)
// // INITIALIZE_PTBCPREPARE(3, 3, RNGType)
// INITIALIZE_PTBCPREPARE(4, 1, RNGType)
// INITIALIZE_PTBCPREPARE(4, 2, RNGType)
// // INITIALIZE_PTBCPREPARE(4, 3, RNGType)
// #undef INITIALIZE_PTBCPREPARE
//

}  // namespace klft
