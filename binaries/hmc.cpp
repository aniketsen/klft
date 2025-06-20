
#include "klft.hpp" // or wherever HMC_execute is declared
#include <getopt.h>

using namespace klft;

// we are hard coding the RNG now to use Kokkos::Random_XorShift64_Pool
// we might want to use our own RNG or allow the user to choose from
// different RNGs in the future
#include <Kokkos_Random.hpp>

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

#define HLINE                                                                  \
  "====================================================================\n"

int parse_args(int argc, char **argv, std::string &input_file) {
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

int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("HMC for SU(N) gauge fields\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  std::string input_file;
  rc = parse_args(argc, argv, input_file);
  if (rc == 0) {
    rc = HMC_execute(input_file);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}
