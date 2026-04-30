
#include "wilsonflow_improv_testing.hpp"

#include <getopt.h>
#include <filesystem>
#include "InputParser.hpp"
#include "PTBC.hpp"

using namespace klft;

// we are hard coding the RNG now to use Kokkos::Random_XorShift64_Pool
// we might want to use our own RNG or allow the user to choose from
// different RNGs in the future
#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

#define HLINE \
  "====================================================================\n"

int parse_args(int argc,
               char** argv,
               std::string& input_file,
               std::string& output_directory) {
  // Defaults
  input_file = "../../../input.yaml";
  output_directory = "./";
  const std::string help_string =
      "  -f <file_name> --filename <file_name>\n"
      "     Name of the input file.\n"
      "     Default: input.yaml\n"
      "  -o <file_name> --output <file_name>\n"
      "     Path to the output folder.\n"
      "     Hint: if the folder does not exist, it will be created.\n"
      "     Default: .\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"filename", required_argument, NULL, 'f'},
      {"output", required_argument, NULL, 'o'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "f:o:h", long_options, &option_index)) !=
         -1)
    switch (c) {
      case 'f':
        input_file = optarg;
        break;
      case 'o':
        output_directory = optarg;
        if (output_directory.back() != '/') {
          output_directory += '/';
        }
        if (!std::filesystem::exists(output_directory)) {
          std::filesystem::create_directories(output_directory);
        }
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

int test_wflow_improvement(const std::string& input_file,
                           const std::string& output_directory) {
  PTBCParams ptbcParams;
  HMCParams hmcParams;
  GaugeObservableParams gaugeObsParams;
  SimulationLoggingParams simLogParams;
  PTBCSimulationLoggingParams ptbcSimLogParams;
  Integrator_Params integratorParams;
  FermionMonomial_Params fermionParams;
  Hasenbusch_Params hbparams;
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

  assert(hmcParams.Ndims == 4 && (hmcParams.Nc == 2 || hmcParams.Nc == 3));

  if (hmcParams.Nc == 2) {
    using DGaugeFieldType = DeviceGaugeFieldType<4, 2>;
    using DAdjFieldType = DeviceAdjFieldType<4, 2>;
    using DSpinorFieldType = DeviceSpinorFieldType<4, 2, 4, complex_t>;
    typename DGaugeFieldType::type g_4_SU2(hmcParams.L0, hmcParams.L1,
                                           hmcParams.L2, hmcParams.L3, rng,
                                           hmcParams.rngDelta);
    typename DAdjFieldType::type a_4_SU2(hmcParams.L0, hmcParams.L1,
                                         hmcParams.L2, hmcParams.L3,
                                         traceT(identitySUN<2>()));
    typename DSpinorFieldType::type s_4_SU2(hmcParams.L0, hmcParams.L1,
                                            hmcParams.L2, hmcParams.L3, 0);
    auto integrator =
        createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
            g_4_SU2, a_4_SU2, s_4_SU2, s_4_SU2, integratorParams,
            gaugeMonomialParams, fermionParams, hbparams, resParsef);
    using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
    HField hamiltonian_field = HField(g_4_SU2, a_4_SU2);

    using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
    HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
            dist, mt);
    hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
    hmc.add_kinetic_monomial(0);
    if (resParsef > 0) {
      auto diracParams = getDiracParams(fermionParams);
      hmc.add_fermion_monomial<
          CGSolver, WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
          s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
    }
    return do_wilsonflow_improv_test<DGaugeFieldType, HMC>(
        hmc, gaugeObsParams, simLogParams, output_directory);
  } else {
    using DGaugeFieldType = DeviceGaugeFieldType<4, 3>;
    using DAdjFieldType = DeviceAdjFieldType<4, 3>;
    using DSpinorFieldType = DeviceSpinorFieldType<4, 3, 4, complex_t>;
    typename DGaugeFieldType::type g_4_SU3(hmcParams.L0, hmcParams.L1,
                                           hmcParams.L2, hmcParams.L3, rng,
                                           hmcParams.rngDelta);
    // typename DGaugeFieldType::type g_4_SU2(
    //     hmcParams.L0, hmcParams.L1, hmcParams.L2, hmcParams.L3,
    //     identitySUN<2>());
    typename DAdjFieldType::type a_4_SU3(hmcParams.L0, hmcParams.L1,
                                         hmcParams.L2, hmcParams.L3,
                                         traceT(identitySUN<3>()));
    typename DSpinorFieldType::type s_4_SU3(hmcParams.L0, hmcParams.L1,
                                            hmcParams.L2, hmcParams.L3, 0);
    auto integrator =
        createIntegrator<DGaugeFieldType, DAdjFieldType, DSpinorFieldType>(
            g_4_SU3, a_4_SU3, s_4_SU3, s_4_SU3, integratorParams,
            gaugeMonomialParams, fermionParams, hbparams, resParsef);
    using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
    HField hamiltonian_field = HField(g_4_SU3, a_4_SU3);

    using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
    HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng,
            dist, mt);
    hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
    hmc.add_kinetic_monomial(0);
    if (resParsef > 0) {
      auto diracParams = getDiracParams(fermionParams);
      hmc.add_fermion_monomial<
          CGSolver, WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>(
          s_4_SU3, diracParams, fermionParams.tol_accept, rng, 0);
    }
    return do_wilsonflow_improv_test<DGaugeFieldType, HMC>(
        hmc, gaugeObsParams, simLogParams, output_directory);
  }
  return 0;
}

int main(int argc, char* argv[]) {
  printf(HLINE);
  printf("HMC for SU(N) gauge fields\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  KTune::initialize();

  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);
  int rc;
  std::string input_file;
  std::string output_directory;
  rc = parse_args(argc, argv, input_file, output_directory);
  if (rc == 0) {
    rc = test_wflow_improvement(input_file, output_directory);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}
