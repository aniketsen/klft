
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

int test_wilsonflow(const std::string& input_file,
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
  Hasenbusch_Params hbparams;
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
          g_4_SU2, a_4_SU2, s_4_SU2, s_4_SU2, integratorParams,
          gaugeMonomialParams, fermionParams, hbparams, resParsef);
  using HField = HamiltonianField<DGaugeFieldType, DAdjFieldType>;
  HField hamiltonian_field = HField(g_4_SU2, a_4_SU2);

  using HMC = HMC<DGaugeFieldType, DAdjFieldType, RNGType>;
  HMC hmc(integratorParams, ioParams, hamiltonian_field, integrator, rng, dist,
          mt);
  hmc.add_gauge_monomial(gaugeMonomialParams.beta, 0);
  hmc.add_kinetic_monomial(0);
  if (resParsef > 0) {
    auto diracParams = getDiracParams(fermionParams);
    hmc.add_fermion_monomial<CGSolver, WilsonDiracOperator, DSpinorFieldType>(
        s_4_SU2, diracParams, fermionParams.tol_accept, rng, 0);
  }

  // Construct the output filename. Each MPI rank will get its own file.
  std::string output_filename =
      output_directory + "topological_charge_cumulative.txt";
  std::ofstream output_file_topologicalcharge(output_filename);

  if (!output_file_topologicalcharge.is_open()) {
    fprintf(stderr, "Error: Could not open output file %s\n",
            output_filename.c_str());
    return -1;  // Or handle the error as appropriate
  }

  std::string output_filename_action_density =
      output_directory + "action_densities_cumulative.txt";
  std::ofstream output_file_acitondensity(output_filename_action_density);

  if (!output_file_topologicalcharge.is_open()) {
    fprintf(stderr, "Error: Could not open output file %s\n",
            output_filename.c_str());
    return -1;  // Or handle the error as appropriate
  }

  // Set precision for floating point numbers in the output file
  output_file_topologicalcharge << std::fixed << std::setprecision(8);
  bool header_written = false;

  Kokkos::Timer timer;
  bool accept;
  real_t acc_sum{0.0};
  real_t acc_rate{0.0};

  index_t flow_steps = gaugeObsParams.wilson_flow_params.n_steps;
  gaugeObsParams.wilson_flow_params.n_steps = 1;
  std::vector<real_t> flow_times;
  std::vector<real_t> topological_charges;
  std::vector<real_t> action_densities;

  WilsonFlow<DGaugeFieldType> wilson_flow(hamiltonian_field.gauge_field,
                                          gaugeObsParams.wilson_flow_params);
  flow_times.push_back(0.0);
  topological_charges.push_back(
      get_topological_charge<DGaugeFieldType>(hamiltonian_field.gauge_field));
  action_densities.push_back(
      getActionDensity<DGaugeFieldType>(hamiltonian_field.gauge_field));

  for (int flow_Step = 1; flow_Step <= flow_steps; ++flow_Step) {
    // perform wilson flow step
    wilson_flow.flow();

    flow_times.push_back(flow_Step * gaugeObsParams.wilson_flow_params.eps);
    topological_charges.push_back(
        get_topological_charge<DGaugeFieldType>(wilson_flow.field));
    action_densities.push_back(
        getActionDensity<DGaugeFieldType>(wilson_flow.field));

    // measure observables
  }
  // Write the header only once, before the first line of data
  if (!header_written) {
    output_file_topologicalcharge << "hmc_step";
    output_file_acitondensity << "hmc_step";
    for (const auto& t : flow_times) {
      output_file_topologicalcharge << "," << t;
      output_file_acitondensity << "," << t;
    }
    output_file_topologicalcharge << "\n";
    output_file_acitondensity << "\n";
    header_written = true;
  }

  // Write the data for the current step
  output_file_topologicalcharge << 0;
  for (const auto& charge : topological_charges) {
    output_file_topologicalcharge << "," << charge;
  }
  output_file_topologicalcharge << "\n";
  output_file_acitondensity << 0;
  for (const auto& density : action_densities) {
    output_file_acitondensity << "," << density;
  }
  output_file_acitondensity << "\n";

  // hmc loop
  for (size_t step = 0; step < integratorParams.nsteps; ++step) {
    timer.reset();
    flow_times.clear();
    topological_charges.clear();
    action_densities.clear();

    // perform hmc_step
    accept = hmc.hmc_step();

    const real_t time = timer.seconds();
    acc_sum += static_cast<real_t>(accept);
    acc_rate = acc_sum / static_cast<real_t>(step + 1);

    if (KLFT_VERBOSITY > 0) {
      printf("Step: %ld, accepted: %ld, Acceptance rate: %f, Time: %f\n", step,
             static_cast<size_t>(accept), acc_rate, time);
    }

    if (accept) {
      WilsonFlow<DGaugeFieldType> wilson_flow(
          hamiltonian_field.gauge_field, gaugeObsParams.wilson_flow_params);
      flow_times.push_back(0.0);
      topological_charges.push_back(get_topological_charge<DGaugeFieldType>(
          hamiltonian_field.gauge_field));
      action_densities.push_back(
          getActionDensity<DGaugeFieldType>(hamiltonian_field.gauge_field));

      for (int flow_Step = 1; flow_Step <= flow_steps; ++flow_Step) {
        // perform wilson flow step
        wilson_flow.flow();

        flow_times.push_back(flow_Step * gaugeObsParams.wilson_flow_params.eps);
        topological_charges.push_back(
            get_topological_charge<DGaugeFieldType>(wilson_flow.field));
        action_densities.push_back(
            getActionDensity<DGaugeFieldType>(wilson_flow.field));

        // measure observables
      }
      // Write the header only once, before the first line of data
      if (!header_written) {
        output_file_topologicalcharge << "hmc_step";
        output_file_acitondensity << "hmc_step";
        for (const auto& t : flow_times) {
          output_file_topologicalcharge << "," << t;
          output_file_acitondensity << "," << t;
        }
        output_file_topologicalcharge << "\n";
        output_file_acitondensity << "\n";
        header_written = true;
      }

      // Write the data for the current step
      output_file_topologicalcharge << step;
      for (const auto& charge : topological_charges) {
        output_file_topologicalcharge << "," << charge;
      }
      output_file_topologicalcharge << "\n";
      output_file_acitondensity << step;
      for (const auto& density : action_densities) {
        output_file_acitondensity << "," << density;
      }
      output_file_acitondensity << "\n";
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
  printf(HLINE);
  printf("HMC for SU(N) gauge fields\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  KTune::initialize();

  int rc;
  std::string input_file;
  std::string output_directory;
  rc = parse_args(argc, argv, input_file, output_directory);
  if (rc == 0) {
    rc = test_wilsonflow(input_file, output_directory);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}
