#include "PTBC.hpp"

#include <mpi.h>

#include <KTune/KTune.hpp>
#include "FermionObservable.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeObservable.hpp"
#include "HamiltonianField.hpp"
#include "IOParams.hpp"
#include "InputParser.hpp"
#include "SimulationLogging.hpp"
#include "updateMomentumFermionEO.hpp"

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

std::string ranked_filename(const std::string& base_filename, int rank) {
  auto pos = base_filename.rfind('.');
  if (pos == std::string::npos) {
    // No extension → just append
    return base_filename + ".rank" + std::to_string(rank);
  } else {
    // Insert before extension
    return base_filename.substr(0, pos) + ".rank" + std::to_string(rank) +
           base_filename.substr(pos);
  }
}

int PTBC_execute(const std::string& input_file,
                 const std::string& output_directory) {
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);
  // get tuning from environment
  KTune::initialize();
  const int tuning =
      std::getenv("KLFT_TUNING") ? std::atoi(std::getenv("KLFT_TUNING")) : 1;
  setTuning(tuning);
  // if tuning is enbled, check if the user has set the
  // KLFT_CACHE_FILE environment variable

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Rank %d of %d\n", rank, size);

  PTBCParams ptbcParams;
  HMCParams hmcParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;
  PTBCSimulationLoggingParams ptbcSimLogParams;
  Integrator_Params integratorParams;
  FermionMonomial_Params fermionParams;
  auto resParsef = parseInputFile(input_file, output_directory, fermionParams);
  GaugeMonomial_Params gaugeMonomialParams;
  FermionObservableParams fermionObsParams;
  IOParams ioParams;
  Hasenbusch_Params hbparams;

  bool inputFileParsedCorrectly =
      (parseInputFile(input_file, output_directory, gaugeObsParams) &&
       parseInputFile(input_file, output_directory, hmcParams) &&
       parseInputFile(input_file, output_directory, simLogParams) &&
       parseInputFile(input_file, output_directory, ptbcParams) &&
       parseInputFile(input_file, output_directory, integratorParams) &&
       abs(resParsef) &&
       parseInputFile(input_file, output_directory, gaugeMonomialParams) &&
       parseInputFile(input_file, output_directory, ptbcSimLogParams) &&
       parseInputFile(input_file, output_directory, fermionObsParams) &&
       parseInputFile(input_file, output_directory, ioParams)) &&
      parseInputFile(input_file, output_directory, hbparams);
  if (!inputFileParsedCorrectly) {
    printf("Error parsing input file\n");
    return -1;
  }
  simLogParams.log_filename = ranked_filename(simLogParams.log_filename, rank);

  if (ptbcParams.defects.size() != size) {
    printf("Error: Number of defects (%zu) does not match size (%d)\n",
           ptbcParams.defects.size(), size);
    return -1;
  }
  if (resParsef < 0) {
    fermionObsParams.measure_pion_correlator = false;
  }
  ioParams.print();
  ptbcParams.gaugeObsParams = gaugeObsParams;
  ptbcParams.simLogParams = simLogParams;
  ptbcParams.ptbcSimLogParams = ptbcSimLogParams;
  ptbcParams.gauge_params = gaugeMonomialParams;
  ptbcParams.fermionObsParams = fermionObsParams;

  // DEBUG_MPI_PRINT("%s", gaugeMonomialParams.to_string().c_str());
  RNGType rng(hmcParams.seed + rank);
  std::mt19937 mt(hmcParams.seed);
  std::uniform_real_distribution<real_t> dist(0.0, 1.0);

  if (hmcParams.coldStart) {
    if (hmcParams.Ndims == 4) {
      defectParams<4> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];
      if (hmcParams.Nc == 1) {
        if (fermionParams.preconditioning == true) {
          using DGaugeFieldType =
              DeviceGaugeFieldType<4, 1, complex_t, GaugeFieldKind::PTBC>;
          using DAdjFieldType = DeviceAdjFieldType<4, 1>;
          using DSpinorFieldType =
              DeviceSpinorFieldType<4, 1, 4, complex_t,
                                    SpinorFieldKind::Standard,
                                    SpinorFieldLayout::Checkerboard>;
          typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, hmcParams.L3,
                                                identitySUN<1>(), dParams);
          typename DAdjFieldType::type a_4_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, hmcParams.L3,
                                              traceT(identitySUN<1>()));
          typename DSpinorFieldType::type s_4_U1(hmcParams.L0 / 2, hmcParams.L1,
                                                 hmcParams.L2, hmcParams.L3, 0);
          typename DSpinorFieldType::type s_4_U1_HB(
              hmcParams.L0 / 2, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
                                             DSpinorFieldType>(
              g_4_U1, a_4_U1, s_4_U1, s_4_U1_HB, integratorParams,
              gaugeMonomialParams, fermionParams, hbparams, resParsef);
          using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
          HField hamiltonian_field = HField(g_4_U1, a_4_U1);

          const auto& dimensions = g_4_U1.dimensions;

          using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
          HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator,
                  rng, dist, mt);
          hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
          hmc.add_kinetic_monomial(0);
          if (resParsef > 0 && hbparams.level < 0) {
            auto diracParams = getDiracParams(fermionParams);
            if (fermionParams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEO<
                  BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEO<
                  BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomialEO<
                  CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEO<
                  CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            }
          } else if (resParsef > 0 && hbparams.level >= 0) {
            auto diracParams = getDiracParams(fermionParams);
            auto diracParams_light = getDiracParams(hbparams);  // light
            if (fermionParams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEO<
                  BiCGStab, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_U1, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEO<
                  BiCGStabMultiP, EOWilsonDiracOperator<DSpinorFieldType,
                                                        DGaugeFieldType, true>>(
                  s_4_U1, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomialEO<
                  CGSolver, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_U1, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEO<
                  CGMultiP, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_U1, diracParams_light, fermionParams.tol_accept, rng, 0);
            }
            printf("Using Hasenbusch preconditioning with level %d\n",
                   hbparams.level);
            if (hbparams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEOHasenbusch<
                  BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else if (hbparams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEOHasenbusch<
                  BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else if (hbparams.Solver == "CG") {
              hmc.add_fermion_monomialEOHasenbusch<
                  CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEOHasenbusch<
                  CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            }
          }
          using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
          PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
          if (hmcParams.loadfile != "") {
            try {
              ptbc.load(hmcParams.loadfile);
            } catch (const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
            }
          }

          run_PTBC(ptbc, integratorParams);
        } else {
          using DGaugeFieldType =
              DeviceGaugeFieldType<4, 1, complex_t, GaugeFieldKind::PTBC>;
          using DAdjFieldType = DeviceAdjFieldType<4, 1>;
          using DSpinorFieldType = DeviceSpinorFieldType<4, 1, 4, complex_t>;
          typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, hmcParams.L3,
                                                identitySUN<1>(), dParams);
          typename DAdjFieldType::type a_4_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, hmcParams.L3,
                                              traceT(identitySUN<1>()));
          typename DSpinorFieldType::type s_4_U1(hmcParams.L0, hmcParams.L1,
                                                 hmcParams.L2, hmcParams.L3, 0);
          typename DSpinorFieldType::type s_4_U1_HB(
              hmcParams.L0 / 2, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
                                             DSpinorFieldType>(
              g_4_U1, a_4_U1, s_4_U1, s_4_U1_HB, integratorParams,
              gaugeMonomialParams, fermionParams, hbparams, resParsef);
          using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
          HField hamiltonian_field = HField(g_4_U1, a_4_U1);

          using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
          HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator,
                  rng, dist, mt);
          hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
          hmc.add_kinetic_monomial(0);
          if (resParsef > 0) {
            auto diracParams = getDiracParams(fermionParams);
            if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomial<
                  CGSolver,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomial<
                  CGMultiP,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            }
          }

          using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
          PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
          if (hmcParams.loadfile != "") {
            try {
              ptbc.load(hmcParams.loadfile);
            } catch (const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
            }
          }

          run_PTBC(ptbc, integratorParams);
        }

      } else if (hmcParams.Nc == 2) {
        if (fermionParams.preconditioning) {
          using DGaugeFieldType =
              DeviceGaugeFieldType<4, 2, complex_t, GaugeFieldKind::PTBC>;
          using DAdjFieldType = DeviceAdjFieldType<4, 2>;
          using DSpinorFieldType =
              DeviceSpinorFieldType<4, 2, 4, complex_t,
                                    SpinorFieldKind::Standard,
                                    SpinorFieldLayout::Checkerboard>;
          typename DGaugeFieldType::type g_4_SU2(hmcParams.L0, hmcParams.L1,
                                                 hmcParams.L2, hmcParams.L3,
                                                 identitySUN<2>(), dParams);
          typename DAdjFieldType::type a_4_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3,
                                               traceT(identitySUN<2>()));
          typename DSpinorFieldType::type s_4_SU2(
              hmcParams.L0 / 2, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          typename DSpinorFieldType::type s_4_SU2_HB(
              hmcParams.L0 / 2, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
                                             DSpinorFieldType>(
              g_4_SU2, a_4_SU2, s_4_SU2, s_4_SU2_HB, integratorParams,
              gaugeMonomialParams, fermionParams, hbparams, resParsef);
          using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
          HField hamiltonian_field = HField(g_4_SU2, a_4_SU2);

          const auto& dimensions = g_4_SU2.dimensions;

          using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
          HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator,
                  rng, dist, mt);
          hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
          hmc.add_kinetic_monomial(0);
          if (resParsef > 0 && hbparams.level < 0) {
            auto diracParams = getDiracParams(fermionParams);
            if (fermionParams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEO<
                  BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEO<
                  BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomialEO<
                  CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEO<
                  CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            }
          } else if (resParsef > 0 && hbparams.level >= 0) {
            auto diracParams = getDiracParams(fermionParams);
            auto diracParams_light = getDiracParams(hbparams);
            if (fermionParams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEO<
                  BiCGStab, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_SU2, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEO<
                  BiCGStabMultiP, EOWilsonDiracOperator<DSpinorFieldType,
                                                        DGaugeFieldType, true>>(
                  s_4_SU2, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomialEO<
                  CGSolver, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_SU2, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEO<
                  CGMultiP, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_SU2, diracParams_light, fermionParams.tol_accept, rng, 0);
            }
            printf("Using Hasenbusch preconditioning with level %d\n",
                   hbparams.level);  // light
            if (hbparams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEOHasenbusch<
                  BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else if (hbparams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEOHasenbusch<
                  BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else if (hbparams.Solver == "CG") {
              hmc.add_fermion_monomialEOHasenbusch<
                  CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEOHasenbusch<
                  CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            }
          }

          using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
          PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
          if (hmcParams.loadfile != "") {
            try {
              ptbc.load(hmcParams.loadfile);
            } catch (const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
            }
          }

          run_PTBC(ptbc, integratorParams);
        } else {
          using DGaugeFieldType =
              DeviceGaugeFieldType<4, 2, complex_t, GaugeFieldKind::PTBC>;
          using DAdjFieldType = DeviceAdjFieldType<4, 2>;
          using DSpinorFieldType = DeviceSpinorFieldType<4, 2, 4, complex_t>;
          typename DGaugeFieldType::type g_4_SU2(hmcParams.L0, hmcParams.L1,
                                                 hmcParams.L2, hmcParams.L3,
                                                 identitySUN<2>(), dParams);
          typename DAdjFieldType::type a_4_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3,
                                               traceT(identitySUN<2>()));
          typename DSpinorFieldType::type s_4_SU2(
              hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          typename DSpinorFieldType::type s_4_SU2_HB(
              hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
                                             DSpinorFieldType>(
              g_4_SU2, a_4_SU2, s_4_SU2, s_4_SU2_HB, integratorParams,
              gaugeMonomialParams, fermionParams, hbparams, resParsef);
          using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
          HField hamiltonian_field = HField(g_4_SU2, a_4_SU2);

          using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
          HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator,
                  rng, dist, mt);
          hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
          hmc.add_kinetic_monomial(0);
          if (resParsef > 0) {
            auto diracParams = getDiracParams(fermionParams);
            if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomial<
                  CGSolver,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomial<
                  CGMultiP,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            }
          }

          using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
          PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
          if (hmcParams.loadfile != "") {
            try {
              ptbc.load(hmcParams.loadfile);
            } catch (const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
            }
          }

          run_PTBC(ptbc, integratorParams);
        }

      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;
        // if (fermionParams.preconditioning) {
        // using DGaugeFieldType =
        //     DeviceGaugeFieldType<4, 1, complex_t,GaugeFieldKind::PTBC>;
        //   using DAdjFieldType = DeviceAdjFieldType<4, 3>;
        //   using DSpinorFieldType =
        //       DeviceSpinorFieldType<4, 3, complex_t,4>;
        //   typename DGaugeFieldType::type g_4_SU3(hmcParams.L0, hmcParams.L1,
        //                                          hmcParams.L2, hmcParams.L3,
        //                                          identitySUN<1>());
        //   typename DAdjFieldType::type a_4_SU3(hmcParams.L0, hmcParams.L1,
        //                                        hmcParams.L2, hmcParams.L3,
        //                                        traceT(identitySUN<3>()));
        //   typename DSpinorFieldType::type s_4_SU3(
        //       hmcParams.L0/2, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
        //                 typename DSpinorFieldType::type s_4_SU3_HB(
        //     hmcParams.L0 / 2, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
        // auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
        //                                    DSpinorFieldType>(
        //     g_4_SU3, a_4_SU3, s_4_SU3,s_4_SU3_HB, integratorParams,
        //     gaugeMonomialParams, fermionParams,hbparams, resParsef);
        // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        // HField hamiltonian_field = HField(g_4_SU3, a_4_SU3);

        // const auto& dimensions = g_4_SU3.dimensions;

        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator,
        //   rng, dist, mt);
        //   hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        //   hmc.add_kinetic_monomial(0);
        // if (resParsef > 0 && hbparams.level < 0) {
        //     auto diracParams = getDiracParams(fermionParams);
        //     if (fermionParams.Solver ==BiCGStabCG"){
        //     hmc.add_fermion_monBiCGStab
        //         CGSolverif ( fermionParams.Solver == "BiCGStabMultiP"),
        //         EOWilsonDiracOperator<DSpinorFBiCGStabMultiP,
        //         DGaugeFieldType>>( s_4_SU3, diracParams,
        //         fermionParams.tol_accept, rng, 0);}
        //        else //     if (fermionParams.Solver == "CG"){
        //     hmc.add_fermion_monomialEO<
        //         CGSolver,
        //         EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
        //         s_4_SU3, diracParams, fermionParams.tol_accept, rng, 0);}
        //         else {
        //
        //     hmc.add_fermion_monomialEO<
        //         CGMultiP,
        //         EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
        //         s_4_SU3, diracParams, fermionParams.tol_accept, rng, 0);
        //         }
        //   } else if (resParsef > 0 && hbparams.level >= 0) {
        //     auto diracParams_light = getDiracParams(hbparams);
        //     auto diracParams = getDiracParams(fermionParams);
        //     if (fermionParams.Solver ==BiCGStabCG"){
        //     hmc.add_fermion_monBiCGStab
        //         CGSolverif ( fermionParams.Solver == "BiCGStabMultiP"),
        //         EOWilsonDiracOperator<DSpinorFBiCGStabMultiP,
        //         DGaugeFieldType, true>>( s_4_SU3, diracParams_light,
        // else //     if (fermionParams.Solver == "CG"){
        //     hmc.add_fermion_monomialEO<
        //         CGSolver,
        //         EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
        //         true>>( s_4_SU3, diracParams_light,
        //         fermionParams.tol_accept,} else {
        //
        //     hmc.add_fermion_monomialEO<
        //         CGMultiP,
        //         EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
        //         true>>( s_4_SU3, diracParams_light, fermionParams.tol_accept,
        //         }
        //         rng, 0);
        //     printf("Using Hasenbusch preconditioning with level %d\n",
        //            hbparams.level);  // light
        //            if (hbparams.Solver ==BiCGStabCG") {
        //     hmc.add_fermion_monomialEOHasBiCGStab
        //         CGSolverif ( fermionParams.Solver == "BiCGStabMultiP"),
        //         EOWilsonDiracOperator<DSpinorFBiCGStabMultiP,
        //         DGaugeFieldType, true>,
        //         EOWilsonDiracOperator<DSpinorFieldType,
        // else //            if (hbparams.Solver == "CG") {
        //     hmc.add_fermion_monomialEOHasenbusch<
        //         CGSolver,
        //         EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
        //         true>, EOWilsonDiracOperator<DSpinorFieldType,
        //         DGaugeFieldType>>( s_4_SU3_HB, diracParams_light,
        //         } else {
        //
        //     hmc.add_fermion_monomialEOHasenbusch<
        //         CGMultiP,
        //         EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
        //         true>, EOWilsonDiracOperator<DSpinorFieldType,
        //         DGaugeFieldType>>( s_4_SU3_HB, diracParams_light,
        //         }
        //         hbparams.tol_accept, rng, 0);
        //   }

        //
        // using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        //  PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        // if (hmcParams.loadfile != "") {
        //   try {
        //     ptbc.load(hmcParams.loadfile);
        //   } catch (const std::exception& e) {
        //     std::cerr << e.what() << "\n";
        return 1;
        //   }
        // }
        //
        // run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
        //          simLogParams);
        //} else { // using DGaugeFieldType =
        //     DeviceGaugeFieldType<4, 1, complex_t,GaugeFieldKind::PTBC>;
        // using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        // using DSpinorFieldType = DeviceSpinorFieldType<4, 1, complex_t,4>;
        // typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
        //                                       hmcParams.L2, hmcParams.L3,
        //                                       identitySUN<1>(), dParams);
        // typename DAdjFieldType::type a_4_U1(hmcParams.L0, hmcParams.L1,
        //                                     hmcParams.L2, hmcParams.L3,
        //                                     traceT(identitySUN<1>()));
        // typename DSpinorFieldType::type s_4_U1(hmcParams.L0, hmcParams.L1,
        //                                        hmcParams.L2, hmcParams.L3,
        //                                        0);
        // typename DSpinorFieldType::type s_4_SU3_HB(
        //     hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
        // auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
        //                                    DSpinorFieldType>(
        //     g_4_SU3, a_4_SU3, s_4_SU3,s_4_SU3_HB, integratorParams,
        //     gaugeMonomialParams, fermionParams,hbparams, resParsef);
        // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        // HField hamiltonian_field = HField(g_4_U1, a_4_U1);
        //
        //
        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams,ioParams, hamiltonian_field, integrator,
        // rng, dist, mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);
        // if (resParsef > 0) {
        //   auto diracParams =
        //       getDiracParams( fermionParams);
        //       if (fermionParams.Solver == "CG"){
        //   hmc.add_fermion_monomial<CGSolver,
        //   WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>
        //                           (s_4_U1, diracParams,
        //                                              fermionParams.tol_accept,}
        //                                              else {
        //
        //   hmc.add_fermion_monomial<CGSolver,
        //   WilsonDiracOperator<DSpinorFieldType, DGaugeFiCGMultiP>
        //                           (s_4_U1, diracParams,
        //                                              fermionParams.tol_accept,
        //                                              }
        //                                              rng, 0);
        // }
        //
        // using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        //  PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        // if (hmcParams.loadfile != "") {
        //   try {
        //     ptbc.load(hmcParams.loadfile);
        //   } catch (const std::exception& e) {
        //     std::cerr << e.what() << "\n";
        return 1;
        //   }
        // }
        //
        // run_PTBC(ptbc, integratorParams, gaugeObsParams, ptbcSimLogParams,
        //          simLogParams);}
      }
    } else if (hmcParams.Ndims == 3) {
      if (resParsef > 0) {
        printf("Error: Fermions are currently not supported in 3D\n");
        return 1;
      }

      defectParams<3> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 1, complex_t, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 1, 4, complex_t>;
        typename DGaugeFieldType::type g_3_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, identitySUN<1>(),
                                              dParams);
        typename DAdjFieldType::type a_3_U1(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<1>()));
        typename DSpinorFieldType::type s_3_U1(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_3_U1, a_3_U1, s_3_U1, s_3_U1, integratorParams,
                gaugeMonomialParams, fermionParams, hbparams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_3_U1, a_3_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
                dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams = getDiracParams(fermionParams);
          hmc.add_fermion_monomial<
              CGSolver, WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
              s_3_U1, diracParams, fermionParams.tol_accept, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        if (hmcParams.loadfile != "") {
          try {
            ptbc.load(hmcParams.loadfile);
          } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
          }
        }

        run_PTBC(ptbc, integratorParams);

      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 2, complex_t, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 2, 4, complex_t>;
        typename DGaugeFieldType::type g_3_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, identitySUN<2>(),
                                               dParams);
        typename DAdjFieldType::type a_3_SU2(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_3_SU2(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_3_SU2, a_3_SU2, s_3_SU2, s_3_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, hbparams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_3_SU2, a_3_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
                dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams = getDiracParams(fermionParams);
          if (fermionParams.Solver == "CG") {
            hmc.add_fermion_monomial<
                CGSolver,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_3_SU2, diracParams, fermionParams.tol_accept, rng, 0);
          } else {
            hmc.add_fermion_monomial<
                CGMultiP,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_3_SU2, diracParams, fermionParams.tol_accept, rng, 0);
          }
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        if (hmcParams.loadfile != "") {
          try {
            ptbc.load(hmcParams.loadfile);
          } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
          }
        }

        run_PTBC(ptbc, integratorParams);

      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;

        // if (resParsef > 0) {
        //   printf("Error: Fermions are currently not supported in 3D\n");
        //   return 1;
        // }
        // using DGaugeFieldType = DeviceGaugeFieldType<3, 3>;
        // using DAdjFieldType = DeviceAdjFieldType<3, 3>;
        // using DSpinorFieldType = DeviceSpinorFieldType<4, 3, complex_t,4>;
        // typename DGaugeFieldType::type g_3_SU3(hmcParams.L0, hmcParams.L1,
        //                                        hmcParams.L2,
        //                                        identitySUN<3>());
        // typename DAdjFieldType::type a_3_SU3(
        //     hmcParams.L0, hmcParams.L1, hmcParams.L2,
        //     traceT(identitySUN<3>()));
        // typename DSpinorFieldType::type s_3_SU3(hmcParams.L0, hmcParams.L1,
        //                                         hmcParams.L2, 1, 0);
        // auto integrator =
        //     createIntegrator<DGaugeFieldType, DAdjFieldType,
        //     DSpinorFieldType>(
        //         g_3_SU3, a_3_SU3, s_3_SU3, integratorParams,
        //         gaugeMonomialParams, fermionParams, resParsef);
        // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        // HField hamiltonian_field = HField(g_3_SU3, a_3_SU3);
        // if (hmcParams.loadfile != "") {

        // const auto& dimensions = g_3_SU3.dimensions;

        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams,ioParams, hamiltonian_field, integrator,
        // rng, dist, mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);

        // run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams
        //         );
      }
    } else if (hmcParams.Ndims == 2) {
      if (resParsef > 0) {
        printf("Error: Fermions are currently not supported in 2D\n");
        return 1;
      }

      defectParams<2> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 1, complex_t, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 1, 4, complex_t>;
        typename DGaugeFieldType::type g_2_U1(hmcParams.L0, hmcParams.L1,
                                              identitySUN<1>(), dParams);
        typename DAdjFieldType::type a_2_U1(hmcParams.L0, hmcParams.L1,
                                            traceT(identitySUN<1>()));
        typename DSpinorFieldType::type s_2_U1(hmcParams.L0, hmcParams.L2, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_2_U1, a_2_U1, s_2_U1, s_2_U1, integratorParams,
                gaugeMonomialParams, fermionParams, hbparams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_2_U1, a_2_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
                dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams = getDiracParams(fermionParams);
          hmc.add_fermion_monomial<
              CGSolver, WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
              s_2_U1, diracParams, fermionParams.tol_accept, rng, 0);
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        if (hmcParams.loadfile != "") {
          try {
            ptbc.load(hmcParams.loadfile);
          } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
          }
        }

        run_PTBC(ptbc, integratorParams);

      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 2, complex_t, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 2, 4, complex_t>;
        typename DGaugeFieldType::type g_2_SU2(hmcParams.L0, hmcParams.L1,
                                               identitySUN<2>(), dParams);
        typename DAdjFieldType::type a_2_SU2(hmcParams.L0, hmcParams.L1,
                                             traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_2_SU2(hmcParams.L0, hmcParams.L1, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_2_SU2, a_2_SU2, s_2_SU2, s_2_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, hbparams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_2_SU2, a_2_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
                dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams = getDiracParams(fermionParams);
          if (fermionParams.Solver == "CG") {
            hmc.add_fermion_monomial<
                CGSolver,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_2_SU2, diracParams, fermionParams.tol_accept, rng, 0);
          } else {
            hmc.add_fermion_monomial<
                CGMultiP,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_2_SU2, diracParams, fermionParams.tol_accept, rng, 0);
          }
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        if (hmcParams.loadfile != "") {
          try {
            ptbc.load(hmcParams.loadfile);
          } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
          }
        }

        run_PTBC(ptbc, integratorParams);

      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;

        // if (resParsef > 0) {
        //   printf("Error: Fermions are currently not supported in 3D\n");
        //   return 1;
        // }
        // using DGaugeFieldType = DeviceGaugeFieldType<2, 3>;
        // using DAdjFieldType = DeviceAdjFieldType<2, 3>;
        // using DSpinorFieldType = DeviceSpinorFieldType<4, 3, complex_t,4>;
        // typename DGaugeFieldType::type g_2_SU3(hmcParams.L0, hmcParams.L1,
        //                                        identitySUN<3>());
        // typename DAdjFieldType::type a_2_SU3(hmcParams.L0, hmcParams.L1,
        //                                      traceT(identitySUN<3>()));
        // typename DSpinorFieldType::type s_2_SU3(hmcParams.L0, hmcParams.L1,
        // 1,
        //                                         1, 0);
        // auto integrator =
        //     createIntegrator<DGaugeFieldType, DAdjFieldType,
        //     DSpinorFieldType>(
        //         g_2_SU3, a_2_SU3, s_2_SU3, integratorParams,
        //         gaugeMonomialParams, fermionParams, resParsef);
        // using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        // HField hamiltonian_field = HField(g_2_SU3, a_2_SU3);

        // const auto& dimensions = g_2_SU3.dimensions;

        // using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        // HMC hmc(integratorParams,ioParams, hamiltonian_field, integrator,
        // rng, dist, mt); hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        // hmc.add_kinetic_monomial(0);

        // run_HMC(hmc, integratorParams, gaugeObsParams, simLogParams
        //         );
      }
    }
  } else {  // Hotstart
    if (hmcParams.Ndims == 4) {
      defectParams<4> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        if (fermionParams.preconditioning) {
          using DGaugeFieldType =
              DeviceGaugeFieldType<4, 1, complex_t, GaugeFieldKind::PTBC>;
          using DAdjFieldType = DeviceAdjFieldType<4, 1>;
          using DSpinorFieldType =
              DeviceSpinorFieldType<4, 1, 4, complex_t,
                                    SpinorFieldKind::Standard,
                                    SpinorFieldLayout::Checkerboard>;
          typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, hmcParams.L3, rng,
                                                hmcParams.rngDelta, dParams);
          typename DAdjFieldType::type a_4_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, hmcParams.L3,
                                              traceT(identitySUN<1>()));
          typename DSpinorFieldType::type s_4_U1(hmcParams.L0 / 2, hmcParams.L1,
                                                 hmcParams.L2, hmcParams.L3, 0);
          typename DSpinorFieldType::type s_4_U1_HB(
              hmcParams.L0 / 2, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
                                             DSpinorFieldType>(
              g_4_U1, a_4_U1, s_4_U1, s_4_U1_HB, integratorParams,
              gaugeMonomialParams, fermionParams, hbparams, resParsef);
          using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
          HField hamiltonian_field = HField(g_4_U1, a_4_U1);

          const auto& dimensions = g_4_U1.dimensions;

          using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
          HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator,
                  rng, dist, mt);
          hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
          hmc.add_kinetic_monomial(0);
          if (resParsef > 0 && hbparams.level < 0) {
            auto diracParams = getDiracParams(fermionParams);
            if (fermionParams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEO<
                  BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEO<
                  BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomialEO<
                  CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEO<
                  CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            }
          } else if (resParsef > 0 && hbparams.level >= 0) {
            auto diracParams = getDiracParams(fermionParams);
            auto diracParams_light = getDiracParams(hbparams);  // light
            if (fermionParams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEO<
                  BiCGStab, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_U1, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEO<
                  BiCGStabMultiP, EOWilsonDiracOperator<DSpinorFieldType,
                                                        DGaugeFieldType, true>>(
                  s_4_U1, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomialEO<
                  CGSolver, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_U1, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEO<
                  CGMultiP, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_U1, diracParams_light, fermionParams.tol_accept, rng, 0);
            }
            printf("Using Hasenbusch preconditioning with level %d\n",
                   hbparams.level);
            if (hbparams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEOHasenbusch<
                  BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else if (hbparams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEOHasenbusch<
                  BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else if (hbparams.Solver == "CG") {
              hmc.add_fermion_monomialEOHasenbusch<
                  CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEOHasenbusch<
                  CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            }
          }

          using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
          PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
          if (hmcParams.loadfile != "") {
            try {
              ptbc.load(hmcParams.loadfile);
            } catch (const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
            }
          }

          run_PTBC(ptbc, integratorParams);
        } else {
          using DGaugeFieldType =
              DeviceGaugeFieldType<4, 1, complex_t, GaugeFieldKind::PTBC>;
          using DAdjFieldType = DeviceAdjFieldType<4, 1>;
          using DSpinorFieldType = DeviceSpinorFieldType<4, 1, 4, complex_t>;
          typename DGaugeFieldType::type g_4_U1(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, hmcParams.L3, rng,
                                                hmcParams.rngDelta, dParams);
          typename DAdjFieldType::type a_4_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, hmcParams.L3,
                                              traceT(identitySUN<1>()));
          typename DSpinorFieldType::type s_4_U1(hmcParams.L0, hmcParams.L1,
                                                 hmcParams.L2, hmcParams.L3, 0);
          typename DSpinorFieldType::type s_4_U1_HB(
              hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
                                             DSpinorFieldType>(
              g_4_U1, a_4_U1, s_4_U1, s_4_U1_HB, integratorParams,
              gaugeMonomialParams, fermionParams, hbparams, resParsef);
          using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
          HField hamiltonian_field = HField(g_4_U1, a_4_U1);

          using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
          HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator,
                  rng, dist, mt);
          hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
          hmc.add_kinetic_monomial(0);
          if (resParsef > 0) {
            auto diracParams = getDiracParams(fermionParams);
            if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomial<
                  CGSolver,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomial<
                  CGMultiP,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_U1, diracParams, fermionParams.tol_accept, rng, 0);
            }
          }

          using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
          PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
          if (hmcParams.loadfile != "") {
            try {
              ptbc.load(hmcParams.loadfile);
            } catch (const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
            }
          }

          run_PTBC(ptbc, integratorParams);
        }
      } else if (hmcParams.Nc == 2) {
        if (fermionParams.preconditioning) {
          using DGaugeFieldType =
              DeviceGaugeFieldType<4, 2, complex_t, GaugeFieldKind::PTBC>;
          using DAdjFieldType = DeviceAdjFieldType<4, 2>;
          using DSpinorFieldType =
              DeviceSpinorFieldType<4, 2, 4, complex_t,
                                    SpinorFieldKind::Standard,
                                    SpinorFieldLayout::Checkerboard>;
          typename DGaugeFieldType::type g_4_SU2(
              hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3, rng,
              hmcParams.rngDelta, dParams);
          typename DAdjFieldType::type a_4_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3,
                                               traceT(identitySUN<2>()));
          typename DSpinorFieldType::type s_4_SU2(
              hmcParams.L0 / 2, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          typename DSpinorFieldType::type s_4_SU2_HB(
              hmcParams.L0 / 2, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
                                             DSpinorFieldType>(
              g_4_SU2, a_4_SU2, s_4_SU2, s_4_SU2_HB, integratorParams,
              gaugeMonomialParams, fermionParams, hbparams, resParsef);
          using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
          HField hamiltonian_field = HField(g_4_SU2, a_4_SU2);

          const auto& dimensions = g_4_SU2.dimensions;

          using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
          HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator,
                  rng, dist, mt);
          hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
          hmc.add_kinetic_monomial(0);
          if (resParsef > 0 && hbparams.level < 0) {
            auto diracParams = getDiracParams(fermionParams);
            if (fermionParams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEO<
                  BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEO<
                  BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomialEO<
                  CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEO<
                  CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            }
          } else if (resParsef > 0 && hbparams.level >= 0) {
            auto diracParams = getDiracParams(fermionParams);
            auto diracParams_light = getDiracParams(hbparams);
            if (fermionParams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEO<
                  BiCGStab, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_SU2, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEO<
                  BiCGStabMultiP, EOWilsonDiracOperator<DSpinorFieldType,
                                                        DGaugeFieldType, true>>(
                  s_4_SU2, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomialEO<
                  CGSolver, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_SU2, diracParams_light, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEO<
                  CGMultiP, EOWilsonDiracOperator<DSpinorFieldType,
                                                  DGaugeFieldType, true>>(
                  s_4_SU2, diracParams_light, fermionParams.tol_accept, rng, 0);
            }
            printf("Using Hasenbusch preconditioning with level %d\n",
                   hbparams.level);  // light
            if (hbparams.Solver == "BiCGStab") {
              hmc.add_fermion_monomialEOHasenbusch<
                  BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else if (hbparams.Solver == "BiCGStabMultiP") {
              hmc.add_fermion_monomialEOHasenbusch<
                  BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else if (hbparams.Solver == "CG") {
              hmc.add_fermion_monomialEOHasenbusch<
                  CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomialEOHasenbusch<
                  CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2_HB, diracParams_light, hbparams.tol_accept, rng, 0);
            }
          }

          using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
          PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
          if (hmcParams.loadfile != "") {
            try {
              ptbc.load(hmcParams.loadfile);
            } catch (const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
            }
          }

          if (KLFT_VERBOSITY > 1) {
            printf(
                "Running PTBC with Nc = %zu, Ndims = %d, L0 = %d, L1 = %d, "
                "L2 = %d, L3 = %d\n",
                hmcParams.Nc, hmcParams.Ndims, hmcParams.L0, hmcParams.L1,
                hmcParams.L2, hmcParams.L3);
          }

          run_PTBC(ptbc, integratorParams);
        } else {
          using DGaugeFieldType =
              DeviceGaugeFieldType<4, 2, complex_t, GaugeFieldKind::PTBC>;
          using DAdjFieldType = DeviceAdjFieldType<4, 2>;
          using DSpinorFieldType = DeviceSpinorFieldType<4, 2, 4, complex_t>;
          typename DGaugeFieldType::type g_4_SU2(
              hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3, rng,
              hmcParams.rngDelta, dParams);
          typename DAdjFieldType::type a_4_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, hmcParams.L3,
                                               traceT(identitySUN<2>()));
          typename DSpinorFieldType::type s_4_SU2(
              hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          typename DSpinorFieldType::type s_4_SU2_HB(
              hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3, 0);
          auto integrator = createIntegrator<DGaugeFieldType, DAdjFieldType,
                                             DSpinorFieldType>(
              g_4_SU2, a_4_SU2, s_4_SU2, s_4_SU2_HB, integratorParams,
              gaugeMonomialParams, fermionParams, hbparams, resParsef);
          using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
          HField hamiltonian_field = HField(g_4_SU2, a_4_SU2);

          using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
          HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator,
                  rng, dist, mt);
          hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
          hmc.add_kinetic_monomial(0);
          if (resParsef > 0) {
            auto diracParams = getDiracParams(fermionParams);
            if (fermionParams.Solver == "CG") {
              hmc.add_fermion_monomial<
                  CGSolver,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            } else {
              hmc.add_fermion_monomial<
                  CGMultiP,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                  s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
            }
          }

          using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
          PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
          if (hmcParams.loadfile != "") {
            try {
              ptbc.load(hmcParams.loadfile);
            } catch (const std::exception& e) {
              std::cerr << e.what() << "\n";
              return 1;
            }
          }

          if (KLFT_VERBOSITY > 1) {
            printf(
                "Running PTBC with Nc = %zu, Ndims = %d, L0 = %d, L1 = %d, "
                "L2 = %d, L3 = %d\n",
                hmcParams.Nc, hmcParams.Ndims, hmcParams.L0, hmcParams.L1,
                hmcParams.L2, hmcParams.L3);
          }

          run_PTBC(ptbc, integratorParams);
        }
      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;
      }
    } else if (hmcParams.Ndims == 3) {
      if (resParsef > 0) {
        printf("Error: Fermions are currently not supported in 3D\n");
        return 1;
      }
      defectParams<3> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 1, complex_t, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 1, 4, complex_t>;
        typename DGaugeFieldType::type g_3_U1(hmcParams.L0, hmcParams.L1,
                                              hmcParams.L2, rng,
                                              hmcParams.rngDelta, dParams);
        typename DAdjFieldType::type a_3_U1(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<1>()));
        typename DSpinorFieldType::type s_3_U1(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_3_U1, a_3_U1, s_3_U1, s_3_U1, integratorParams,
                gaugeMonomialParams, fermionParams, hbparams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_3_U1, a_3_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
                dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams = getDiracParams(fermionParams);
          if (fermionParams.Solver == "CG") {
            hmc.add_fermion_monomial<
                CGSolver,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_3_U1, diracParams, fermionParams.tol_accept, rng, 0);
          } else {
            hmc.add_fermion_monomial<
                CGMultiP,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_3_U1, diracParams, fermionParams.tol_accept, rng, 0);
          }
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        if (hmcParams.loadfile != "") {
          try {
            ptbc.load(hmcParams.loadfile);
          } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
          }
        }

        run_PTBC(ptbc, integratorParams);
      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<3, 2, complex_t, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<3, 2, 4, complex_t>;
        typename DGaugeFieldType::type g_3_SU2(hmcParams.L0, hmcParams.L1,
                                               hmcParams.L2, rng,
                                               hmcParams.rngDelta, dParams);
        typename DAdjFieldType::type a_3_SU2(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_3_SU2(hmcParams.L0, hmcParams.L1,
                                                hmcParams.L2, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_3_SU2, a_3_SU2, s_3_SU2, s_3_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, hbparams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_3_SU2, a_3_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
                dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams = getDiracParams(fermionParams);
          if (fermionParams.Solver == "CG") {
            hmc.add_fermion_monomial<
                CGSolver,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_3_SU2, diracParams, fermionParams.tol_accept, rng, 0);
          } else {
            hmc.add_fermion_monomial<
                CGMultiP,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_3_SU2, diracParams, fermionParams.tol_accept, rng, 0);
          }
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        if (hmcParams.loadfile != "") {
          try {
            ptbc.load(hmcParams.loadfile);
          } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
          }
        }

        run_PTBC(ptbc, integratorParams);
      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;
      }
    } else if (hmcParams.Ndims == 2) {
      if (resParsef > 0) {
        printf("Error: Fermions are currently not supported in 2D\n");
        return 1;
      }
      defectParams<2> dParams;
      dParams.defect_length = ptbcParams.defect_length;
      dParams.defect_value = ptbcParams.defects[rank];

      if (hmcParams.Nc == 1) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 1, complex_t, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 1, 4, complex_t>;
        typename DGaugeFieldType::type g_2_U1(hmcParams.L0, hmcParams.L1, rng,
                                              hmcParams.rngDelta, dParams);
        typename DAdjFieldType::type a_2_U1(hmcParams.L0, hmcParams.L1,
                                            traceT(identitySUN<1>()));
        typename DSpinorFieldType::type s_2_U1(hmcParams.L0, hmcParams.L1, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_2_U1, a_2_U1, s_2_U1, s_2_U1, integratorParams,
                gaugeMonomialParams, fermionParams, hbparams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_2_U1, a_2_U1);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
                dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams = getDiracParams(fermionParams);
          if (fermionParams.Solver == "CG") {
            hmc.add_fermion_monomial<
                CGSolver,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_2_U1, diracParams, fermionParams.tol_accept, rng, 0);
          } else {
            hmc.add_fermion_monomial<
                CGMultiP,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_2_U1, diracParams, fermionParams.tol_accept, rng, 0);
          }
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        if (hmcParams.loadfile != "") {
          try {
            ptbc.load(hmcParams.loadfile);
          } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
          }
        }

        run_PTBC(ptbc, integratorParams);
      } else if (hmcParams.Nc == 2) {
        using DGaugeFieldType =
            DeviceGaugeFieldType<2, 2, complex_t, GaugeFieldKind::PTBC>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        using DSpinorFieldType = DeviceSpinorFieldType<2, 2, 4, complex_t>;
        typename DGaugeFieldType::type g_2_SU2(hmcParams.L0, hmcParams.L1, rng,
                                               hmcParams.rngDelta, dParams);
        typename DAdjFieldType::type a_2_SU2(hmcParams.L0, hmcParams.L1,
                                             traceT(identitySUN<2>()));
        typename DSpinorFieldType::type s_2_SU2(hmcParams.L0, hmcParams.L1, 0);
        auto integrator =
            createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
                g_2_SU2, a_2_SU2, s_2_SU2, s_2_SU2, integratorParams,
                gaugeMonomialParams, fermionParams, hbparams, resParsef);
        using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
        HField hamiltonian_field = HField(g_2_SU2, a_2_SU2);

        using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
        HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
                dist, mt);
        hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
        hmc.add_kinetic_monomial(0);
        if (resParsef > 0) {
          auto diracParams = getDiracParams(fermionParams);
          if (fermionParams.Solver == "CG") {
            hmc.add_fermion_monomial<
                CGSolver,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_2_SU2, diracParams, fermionParams.tol_accept, rng, 0);
          } else {
            hmc.add_fermion_monomial<
                CGMultiP,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
                s_2_SU2, diracParams, fermionParams.tol_accept, rng, 0);
          }
        }

        using PTBC = PTBC<DGaugeFieldType, DAdjFieldType, RNGType>;
        PTBC ptbc(ptbcParams, hmc, rng, dist, mt);
        if (hmcParams.loadfile != "") {
          try {
            ptbc.load(hmcParams.loadfile);
          } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
          }
        }

        run_PTBC(ptbc, integratorParams);
      } else if (hmcParams.Nc == 3) {
        printf("Error: SU(3) isn't supported yet");
        return 1;
      }
    }
  }
  // if tuning is enabled, write the cache file
  // if (KLFT_TUNING) {
  //   const char *cache_file = std::getenv("KLFT_CACHE_FILE");
  //   if (cache_file) {
  //     writeTuneCache(cache_file);
  //   } else {
  //     printf("KLFT_CACHE_FILE not set\n");
  //   }
  // }
  return 0;
}
}  // namespace klft
