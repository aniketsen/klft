#include <getopt.h>

#include <cassert>
#include <cstddef>
#include <memory>

#include "AdjointSUN.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "GaugeFieldHelper.hpp"
#include "HMC.hpp"
#include "HamiltonianField.hpp"
#include "IOParams.hpp"
#include "InputParser.hpp"
#include "Integrator.hpp"
#include "SimulationLogging.hpp"
#include "UpdateMomentum.hpp"
#include "UpdatePosition.hpp"

using namespace klft;

#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

#define HLINE \
  "====================================================================\n"

int parse_args(int argc, char** argv, std::string& input_file) {
  // Defaults
  input_file = "input.yaml";

  const std::string help_string =
      "  -f <file_name> --filename <file_name>\n"
      "     Name of the input file.\n"
      "     Default: input.yaml\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"filename", required_argument, NULL, 'f'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "f:h", long_options, &option_index)) !=
         -1)
    switch (c) {
      case 'f':
        input_file = optarg;
        break;
      case 'h':
        printf("%s", help_string.c_str());
        return -2;
        break;
      case 0:
        break;
      default:
        printf("%s", help_string.c_str());
        return -1;
        break;
    }
  return 0;
}

int test_reversability(const std::string& input_file,
                       const std::string& output_directory) {
  PTBCParams ptbcParams;
  HMCParams hmcParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;
  PTBCSimulationLoggingParams ptbcSimLogParams;
  Integrator_Params integratorParams;
  FermionMonomial_Params fermionParams;
  auto resParsef = parseInputFile(input_file, output_directory, fermionParams);
  GaugeMonomial_Params gaugeMonomialParams;
  IOParams ioParams;
  bool inputFileParsedCorrectly =
      (parseInputFile(input_file, output_directory, gaugeObsParams) &&
       parseInputFile(input_file, output_directory, hmcParams) &&
       parseInputFile(input_file, output_directory, simLogParams) &&
       parseInputFile(input_file, output_directory, ptbcParams) &&
       parseInputFile(input_file, output_directory, integratorParams) &&
       abs(resParsef) &&
       parseInputFile(input_file, output_directory, gaugeMonomialParams) &&
       parseInputFile(input_file, output_directory, ptbcSimLogParams) &&
       parseInputFile(input_file, output_directory, ioParams));
  if (!inputFileParsedCorrectly) {
    printf("Error parsing input file\n");
    return -1;
  }

  simLogParams.log_filename = (simLogParams.log_filename);
  RNGType rng(hmcParams.seed);
  std::mt19937 mt(hmcParams.seed);
  std::uniform_real_distribution<real_t> dist(0.0, 1.0);

  assert(hmcParams.Ndims == 4 && hmcParams.Nc == 2);

  using DGaugeFieldType = DeviceGaugeFieldType<4, 2>;
  using DAdjFieldType = DeviceAdjFieldType<4, 2>;
  using DSpinorFieldType = DeviceSpinorFieldType<4, 2, 4>;
  typename DGaugeFieldType::type g_4_SU2(hmcParams.L0, hmcParams.L1,
                                         hmcParams.L2, hmcParams.L3, rng,
                                         hmcParams.rngDelta);
  // typename DGaugeFieldType::type g_4_SU2(
  //     hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3,
  //     identitySUN<2>());
  typename DAdjFieldType::type a_4_SU2(hmcParams.L0, hmcParams.L1, hmcParams.L2,
                                       hmcParams.L3, traceT(identitySUN<2>()));
  typename DSpinorFieldType::type s_4_SU2(hmcParams.L0, hmcParams.L1,
                                          hmcParams.L2, hmcParams.L3, 0);
  auto integrator =
      createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
          g_4_SU2, a_4_SU2, s_4_SU2, integratorParams, gaugeMonomialParams,
          fermionParams, resParsef);
  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  HField hamiltonian_field = HField(g_4_SU2, a_4_SU2);

  using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
  HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng, dist,
          mt);
  hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
  hmc.add_kinetic_monomial(0);
  if (resParsef > 0) {
    auto diracParams = getDiracParams(fermionParams);
    hmc.add_fermion_monomial<CGSolver, HWilsonDiracOperator, DSpinorFieldType>(
        s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
  }

  for (size_t step = 0; step < integratorParams.nsteps; ++step) {
    // perform hmc_step
    bool accept = hmc.hmc_step(true);
  }

  return 0;
}
// for now, the test only allows for Nd=4
int main(int argc, char* argv[]) {
  printf(HLINE);
  printf("HMC for SU(N) gauge fields\n");
  printf(HLINE);

  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);

  Kokkos::initialize(argc, argv);
  int rc;
  std::string input_file;
  rc = parse_args(argc, argv, input_file);
  if (rc == 2) {
    Kokkos::finalize();
    return 0;
  }
  if (rc == 0) {
    rc = test_reversability(input_file, "./");
  }
  Kokkos::finalize();

  return rc;
}
