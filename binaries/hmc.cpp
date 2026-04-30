
#include <getopt.h>

#include <filesystem>

#include "klft.hpp"  // or wherever HMC_execute is declared
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
  std::filesystem::copy(input_file, output_directory + "used_input.yaml",
                        std::filesystem::copy_options::overwrite_existing);
  return 0;
}

int main(int argc, char* argv[]) {
  printf(HLINE);
  printf("HMC for SU(N) gauge fields\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  std::string input_file;
  std::string output_directory;
  rc = parse_args(argc, argv, input_file, output_directory);
  if (rc == 0) {
    rc = build_and_run_HMC(input_file, output_directory);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}
