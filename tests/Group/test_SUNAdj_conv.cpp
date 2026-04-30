
#include <getopt.h>
#include <cassert>
#include <cmath>
#include <iostream>

// Placeholder includes – replace with actual library headers
#include "AdjointSUN.hpp"
#include "GLOBAL.hpp"
using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

using namespace klft;

template <typename T>
bool approxEqual(const T& a, const T& b, double tol = 1e-10) {
  return std::abs(a - b) < tol;
}

template <size_t Nc>
std::string printSUN(const SUN<Nc>& e) {
  std::ostringstream out;
  out << "[";
  for (int i = 0; i < Nc; ++i) {
    out << "[";
    for (int j = 0; j < Nc; ++j) {
      out << e[i][j];
      if (j < Nc - 1) {
        out << ", ";
      }
    }
    out << "]";
    if (i < Nc - 1) {
      out << ", ";
    }
  }
  out << "]";
  return out.str();
}

template <size_t Nc>
std::string printSUNAdj(const SUNAdj<Nc>& e) {
  std::ostringstream out;
  out << "[";
  for (int i = 0; i < NcAdj<Nc>; ++i) {
    out << e[i];
    if (i < NcAdj<Nc> - 1) {
      out << ", ";
    }
  }
  out << "]";
  return out.str();
}

template <size_t Nc>
void testConversionAccuracy(size_t& seed) {
  std::cout << "Testing SUN<" << Nc << "> and SUNAdj<" << Nc << ">\n";

  // Create a SUNAdj<Nc> algebra element (e.g., random anti-Hermitian traceless)
  SUNAdj<Nc> adjA;
  RNGType rng(seed);
  auto generator = rng.get_state();
  randSUNAdj<Nc>(adjA, generator);
  rng.free_state(generator);
  std::cout << "Start Adjoint: ";
  std::cout << printSUNAdj<Nc>(adjA) << "\n";

  // Compute the exponential map to get a SUN group element
  SUN<Nc> U = expoSUN(adjA);
  std::cout << "Generated SUN: " << printSUN<Nc>(U) << "\n";

  SUNAdj<Nc> trU = traceT(U);
  std::cout << "recreated Adjoint: " << printSUNAdj<Nc>(trU) << "\n";

  SUN<Nc> nextU = expoSUN(adjA);
  std::cout << "re Generated SUN: " << printSUN<Nc>(nextU) << "\n";

  SUN<Nc> diffSUNNc = U - nextU;
  // Construct an approximate U' from exp(A) again to see stability
  SUNAdj<Nc> diffV;
  for (int i = 0; i < NcAdj<Nc>; ++i) {
    diffV[i] = adjA[i] - trU[i];
  }
  real_t delta = norm2<Nc>(diffV);
  complex_t deltaSUN;

  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      deltaSUN += diffSUNNc[i][j];
    }
  }

  std::cout << "delta (Adjoint): " << delta << "\n";
  std::cout << "delta (SUN): " << deltaSUN << "\n";
}

int parse_args(int argc, char** argv, size_t& seed) {
  // Defaults
  seed = 1232;

  const std::string help_string =
      "  -s <N>, --seed <N>\n"
      "    Seed for the test.\n"
      "     Default: 1232\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {{"seed", required_argument, NULL, 's'},
                                         {"help", no_argument, NULL, 'h'},
                                         {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "n:h", long_options, &option_index)) !=
         -1)
    switch (c) {
      case 's':
        seed = atoi(optarg);
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
  size_t seed;
  parse_args(argc, argv, seed);
  Kokkos::initialize(argc, argv);
  KTune::initialize();
  testConversionAccuracy<1>(seed);
  testConversionAccuracy<2>(seed);
  Kokkos::finalize();
  return 0;
}
