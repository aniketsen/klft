#pragma once
#include <fstream>
#include <iomanip>

#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "PionCorrelator.hpp"

namespace klft {
struct FermionObservableParams {
  size_t measurement_interval;
  bool measure_pion_correlator;
  std::vector<std::vector<real_t>> pion_correlator;
  std::string pion_correlator_filename;
  std::vector<size_t> measurement_steps;
  real_t tol;
  real_t kappa;
  index_t n_sources;
  size_t RepDim;
  bool write_to_file;
  bool flushed;
  bool preconditioning;

  //
  size_t flush;  // interval to flush measurements to file, 0 to flush at the
  // end of the simulation

  void print() const {
    printf("FermionObservableParams:\n");
    printf("  measurement_interval: %zu\n", measurement_interval);
    printf("  measure_pion_correlator: %s\n",
           measure_pion_correlator ? "true" : "false");
    printf("  pion_correlator_filename: %s\n",
           pion_correlator_filename.c_str());
    printf("  tol: %e\n", tol);
    printf("  kappa: %f\n", kappa);
    printf("  RepDim: %zu\n", RepDim);
    printf("  write_to_file: %s\n", write_to_file ? "true" : "false");
    printf("  flush: %zu\n", flush);
    printf("  n_sources: %d\n", n_sources);
  }
};

auto getDiracParams(const FermionObservableParams& fparams) {
  if (fparams.RepDim == 4) {
    diracParams dParams(fparams.kappa);
    return dParams;

  } else {
    printf("Warning: Unsupported Gamma Matrix Representation\n");
    printf("Warning: Fallback RepDim = 4\n");

    diracParams dParams(fparams.kappa);
    return dParams;
  }
}

template <typename RNG,
          typename DSpinorFieldType,
          typename DGaugeFieldType,
          template <typename> class _Solver,
          template <typename, typename, bool> class DiracOpT>
void measureFermionObservables(const typename DGaugeFieldType::type& g_in,
                               FermionObservableParams& params,
                               const size_t step,
                               RNG& rng) {
  // using DiracOperator = DiracOpT<DSpinorFieldType, DGaugeFieldType, false>;

  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0)) {
    return;
  }
  if (KLFT_VERBOSITY > 1) {
    printf("Measurement of Fermion Observables\n");
    printf("step: %zu\n", step);
  }
  if (params.measure_pion_correlator) {
    if constexpr (std::is_same_v<
                      typename DiracOpT<DSpinorFieldType, DGaugeFieldType,
                                        false>::Base,
                      DiracOperator<DiracOpT, DGaugeFieldType, bool>>) {
      printf("Computing Pion Correlator FULL layout\n");
      auto PC =
          PionCorrelator<RNG, CGSolver,
                         DiracOpT<DSpinorFieldType, DGaugeFieldType, false>>(
              g_in, getDiracParams(params), params.tol, rng, params.n_sources);
      params.pion_correlator.push_back(PC);
      if (KLFT_VERBOSITY > 1) {
        printf("Pion Correlator:\n");
        for (auto&& i : PC) {
          printf("%f, ", i);
        }
        printf("\n");
      }
    } else {
      printf("Computing Pion Correlator Checkerboard layout\n");
      auto dims = g_in.dimensions;
      dims[0] /= 2;
      auto PC =
          PionCorrelatorEO<RNG, BiCGStabMultiP,
                           DiracOpT<DSpinorFieldType, DGaugeFieldType, false>>(
              g_in, getDiracParams(params), dims, params.tol, rng,
              params.n_sources);
      params.pion_correlator.push_back(PC);
      if (KLFT_VERBOSITY > 1) {
        printf("Pion Correlator:\n");
        for (auto&& i : PC) {
          printf("%f, ", i);
        }
        printf("\n");
      }
    }
  }
  params.measurement_steps.push_back(step);
  return;
}
inline void flushPionCorrelator(std::ofstream& file,
                                const FermionObservableParams& params,
                                const bool HEADER = true) {
  // check if the file is open
  if (!file.is_open()) {
    printf("Error: file is not open\n");
    return;
  }
  // check if plaquette measurements are available
  if (!params.measure_pion_correlator) {
    printf("Error: no plaquette measurements available\n");
    return;
  }
  if (HEADER)
    file << "# step, pion correlator\n";
  for (size_t i = 0; i < params.pion_correlator.size(); ++i) {
    file << params.measurement_steps[i] << ", ";
    for (int j = 0; j < params.pion_correlator[i].size() - 1; ++j) {
      file << params.pion_correlator[i][j] << ",";
    }
    file << params.pion_correlator[i].back();
    file << "\n";
  }
}

inline void clearAllFermionObservables(FermionObservableParams& params) {
  params.measurement_steps.clear();
  params.pion_correlator.clear();
}
inline void forceflushAllFermionObservables(
    FermionObservableParams& params,
    const bool clear_after_flush = false,
    const int& p = std::cout.precision()) {
  auto _ = std::setprecision(p);
  // check if write_to_file is enabled
  if (!params.write_to_file) {
    printf("write_to_file is not enabled\n");
    return;
  }
  bool HEADER = !params.flushed;  // write header only once

  // TODO : flush similar to gauge obs
  if (params.measure_pion_correlator && params.pion_correlator_filename != "") {
    std::ofstream file(params.pion_correlator_filename, std::ios::app);
    flushPionCorrelator(file, params, HEADER);
    file.close();
  }
  if (clear_after_flush) {
    clearAllFermionObservables(params);
  }
  params.flushed = true;
}

inline void flushAllFermionObservables(FermionObservableParams& params,

                                       const size_t step,
                                       const bool clear_after_flush = false,
                                       const int& p = std::cout.precision()) {
  if (params.flush != 0 && step % params.flush == 0) {
    forceflushAllFermionObservables(params, clear_after_flush, p);
  }
}
typedef enum {
  MPI_FERMION_OBSERVABLE_PION_CORRELATOR_SIZE = 0,
  MPI_FERMION_OBSERVABLE_PION_CORRELATOR = 1

} MPI_FermionObservableTypes;
template <typename RNG,
          typename DSpinorFieldType,
          typename DGaugeFieldType,
          template <typename> class _Solver,
          template <typename, typename, bool> class DiracOpT>
void measureFermionObservablesPTBC(const typename DGaugeFieldType::type& g_in,
                                   FermionObservableParams& params,

                                   const size_t step,
                                   const int compute_rank,
                                   RNG& rng,
                                   const bool do_compute = false) {
  if ((params.measurement_interval == 0) ||
      (step % params.measurement_interval != 0) || (step == 0)) {
    return;
  }
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (KLFT_VERBOSITY > 1) {
    printf("Measurement of Fermion Observables\n");
    printf("step: %zu\n", step);
  }
  if (do_compute) {
    /* code */
    if (params.measure_pion_correlator) {
      if constexpr (DeviceFermionFieldTypeTraits<DSpinorFieldType>::Layout ==
                    SpinorFieldLayout::FULL) {
        if (KLFT_VERBOSITY > 1) {
          printf("Computing Pion Correlator FULL layout\n");
        }
        printf("Computing Pion Correlator Checkerboard layout\n");
        auto dims = g_in.dimensions;
        dims[0] /= 2;
        auto PC = PionCorrelatorEO<
            RNG, CGSolver, DiracOpT<DSpinorFieldType, DGaugeFieldType, false>>(
            g_in, getDiracParams(params), dims, params.tol, rng,
            params.n_sources);
        index_t size = PC.size();
        MPI_Send(&size, 1, mpi_index_t(), 0,
                 MPI_FERMION_OBSERVABLE_PION_CORRELATOR_SIZE, MPI_COMM_WORLD);
        printf("Sent Pion Correlator size");
        MPI_Send(PC.data(), size, mpi_real_t(), 0,
                 MPI_FERMION_OBSERVABLE_PION_CORRELATOR, MPI_COMM_WORLD);
      } else {
        if (KLFT_VERBOSITY > 1) {
          printf("Computing Pion Correlator Checkerboard layout\n");
        }
        auto dims = g_in.dimensions;
        dims[0] /= 2;
        auto PC = PionCorrelatorEO<RNG, BiCGStabMultiP, DiracOpT>(
            g_in, getDiracParams(params), dims, params.tol, rng,
            params.n_sources);
        index_t size = PC.size();
        MPI_Send(&size, 1, mpi_index_t(), 0,
                 MPI_FERMION_OBSERVABLE_PION_CORRELATOR_SIZE, MPI_COMM_WORLD);
        printf("Sent Pion Correlator size");
        MPI_Send(PC.data(), size, mpi_real_t(), 0,
                 MPI_FERMION_OBSERVABLE_PION_CORRELATOR, MPI_COMM_WORLD);
      }
    }
    // if (KLFT_VERBOSITY > 1) {
    //   printf("Pion Correlator:\n");
    //   for (auto&& i : PC) {
    //     printf("%f, ", i);
    //   }
    //   printf("\n");
    // }
  }

  if (rank == 0) {
    params.measurement_steps.push_back(step);
    if (params.measure_pion_correlator) {
      index_t PC_size;
      MPI_Recv(&PC_size, 1, mpi_index_t(), compute_rank,
               MPI_FERMION_OBSERVABLE_PION_CORRELATOR_SIZE, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      std::vector<real_t> PC(PC_size);
      MPI_Recv(PC.data(), PC_size, mpi_real_t(), compute_rank,
               MPI_FERMION_OBSERVABLE_PION_CORRELATOR, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      params.pion_correlator.push_back(PC);
      if (KLFT_VERBOSITY > 1) {
        printf("Pion Correlator:\n");
        for (auto&& i : PC) {
          printf("%f, ", i);
        }
        printf("\n");
      }
    }
  }

  return;
}

}  // namespace klft
