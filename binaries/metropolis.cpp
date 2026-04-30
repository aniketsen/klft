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

// this file performs metropolis for gauge fields
// for different dimensions and gauge groups

#include <getopt.h>

#include <filesystem>
#include "klft.hpp"
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
  input_file = "../../../new_test.yaml";
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
      {"output", optional_argument, NULL, 'o'},
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

int main(int argc, char* argv[]) {
  printf(HLINE);
  printf("Metropolis for SU(N) gauge fields\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);

  int rc;
  std::string input_file;
  std::string output_directory;
  rc = parse_args(argc, argv, input_file, output_directory);
  if (rc == 0) {
    rc = Metropolis(input_file, output_directory);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}
