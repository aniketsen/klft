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

// this file tests and benchmarks the gauge Wilson loop kernel for different
// 2D, 3D and 4D gauge fields for U(1), SU(2) and SU(3) gauge groups

#include "WilsonLoop.hpp"
#include <getopt.h>

#define HLINE                                                                  \
  "====================================================================\n"

using namespace klft;

int run_benchmark(const size_t stream_size_array) {
  // get verbosity from environment
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);
  // get tuning from environment
  const int tuning =
      std::getenv("KLFT_TUNING") ? std::atoi(std::getenv("KLFT_TUNING")) : 0;
  setTuning(tuning);
  // 4D volume of the gauge fields
  const real_t volume4D = (real_t)stream_size_array *
                          (real_t)stream_size_array *
                          (real_t)stream_size_array * (real_t)stream_size_array;
  // 3D volume of the gauge fields
  const real_t volume3D = (real_t)stream_size_array *
                          (real_t)stream_size_array * (real_t)stream_size_array;
  // 2D volume of the gauge fields
  const real_t volume2D = (real_t)stream_size_array * (real_t)stream_size_array;

  // timer
  Kokkos::Timer timer;

  // wilson loop time
  real_t wlTime = std::numeric_limits<real_t>::max();

  printf("Starting benchmark...\n");
  printf("Benchmark kernels will be performed for %d iterations.\n",
         STREAM_NTIMES);
  printf("Reports fastest timing per kernel\n");
  std::vector<Kokkos::Array<index_t, 2>> LT_pairs;
  LT_pairs.push_back(
      {index_t(stream_size_array / 4), index_t(stream_size_array / 4)});
  LT_pairs.push_back(
      {index_t(stream_size_array / 2), index_t(stream_size_array / 4)});
  LT_pairs.push_back(
      {index_t(stream_size_array / 2), index_t(stream_size_array / 2)});
  printf("Lattice extent: %ld\n", stream_size_array);
  printf("Wilson Loop pairs: \n");
  for (const auto &pair : LT_pairs) {
    printf("L = %d, T = %d\n", pair[0], pair[1]);
  }
  printf(HLINE);
  // output vector
  std::vector<Kokkos::Array<real_t, 3>> Wtemporal_vals;

  printf("U(1) Gauge 4D: \n");
  const real_t U1_field_4D = volume4D * 1.0 * 1.0;
  const real_t U1_gauge_4D = 4.0 * U1_field_4D;
  printf("- Size of U(1) GaugeField:                  %12.2f MB\n",
         1.0e-6 * U1_gauge_4D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");

  {
    deviceGaugeField<4, 1> dev_g_U1_4D(stream_size_array, stream_size_array,
                                       stream_size_array, stream_size_array,
                                       identitySUN<1>());

    printf("Running benchmark...\n");

    // run the benchmark
    for (index_t k = 0; k < STREAM_NTIMES; ++k) {
      Wtemporal_vals.clear();
      timer.reset();
      WilsonLoop_temporal<4, 1>(dev_g_U1_4D, LT_pairs, Wtemporal_vals);
      Kokkos::fence();
      wlTime = std::min(wlTime, timer.seconds());
    }
  }
  printf(HLINE);
  printf("Wilson Loop values: \n");
  for (const auto &Wtemporal : Wtemporal_vals) {
    printf("L = %d, T = %d, W(L,T) = %11.4e            Expected: %11.4e\n",
           static_cast<index_t>(Wtemporal[0]),
           static_cast<index_t>(Wtemporal[1]), Wtemporal[2], 1.0);
  }
  printf("Wilson Loop Kernel Time:     %11.4e s\n", wlTime);
  printf("Wilson Loop BW:              %11.4f GB/s\n",
         1.0e-9 * (real_t)sizeof(complex_t) *
             (volume4D + 2.0 * U1_field_4D + U1_gauge_4D) / wlTime);
  printf(HLINE);

  // reset plaquette time
  wlTime = std::numeric_limits<real_t>::max();

  printf("SU(2) Gauge 4D: \n");
  const real_t SU2_field_4D = volume4D * 2.0 * 2.0;
  const real_t SU2_gauge_4D = 4.0 * SU2_field_4D;
  printf("- Size of SU(2) GaugeField:                  %12.2f MB\n",
         1.0e-6 * SU2_gauge_4D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");

  {
    deviceGaugeField<4, 2> dev_g_SU2_4D(stream_size_array, stream_size_array,
                                        stream_size_array, stream_size_array,
                                        identitySUN<2>());

    printf("Running benchmark...\n");

    // run the benchmark
    for (index_t k = 0; k < STREAM_NTIMES; ++k) {
      Wtemporal_vals.clear();
      timer.reset();
      WilsonLoop_temporal<4, 2>(dev_g_SU2_4D, LT_pairs, Wtemporal_vals);
      Kokkos::fence();
      wlTime = std::min(wlTime, timer.seconds());
    }
  }
  printf(HLINE);
  printf("Wilson Loop values: \n");
  for (const auto &Wtemporal : Wtemporal_vals) {
    printf("L = %d, T = %d, W(L,T) = %11.4e            Expected: %11.4e\n",
           static_cast<index_t>(Wtemporal[0]),
           static_cast<index_t>(Wtemporal[1]), Wtemporal[2], 1.0);
  }
  printf("Wilson Loop Kernel Time:     %11.4e s\n", wlTime);
  printf("Wilson Loop BW:              %11.4f GB/s\n",
         1.0e-9 * (real_t)sizeof(complex_t) *
             (volume4D + 2.0 * SU2_field_4D + SU2_gauge_4D) / wlTime);
  printf(HLINE);

  // reset plaquette time
  wlTime = std::numeric_limits<real_t>::max();

  printf("SU(3) Gauge 4D: \n");
  const real_t SU3_field_4D = volume4D * 3.0 * 3.0;
  const real_t SU3_gauge_4D = 4.0 * SU3_field_4D;
  printf("- Size of SU(3) GaugeField:                  %12.2f MB\n",
         1.0e-6 * SU3_gauge_4D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");

  {
    deviceGaugeField<4, 3> dev_g_SU3_4D(stream_size_array, stream_size_array,
                                        stream_size_array, stream_size_array,
                                        identitySUN<3>());

    printf("Running benchmark...\n");

    // run the benchmark
    for (index_t k = 0; k < STREAM_NTIMES; ++k) {
      Wtemporal_vals.clear();
      timer.reset();
      WilsonLoop_temporal<4, 3>(dev_g_SU3_4D, LT_pairs, Wtemporal_vals);
      Kokkos::fence();
      wlTime = std::min(wlTime, timer.seconds());
    }
  }
  printf(HLINE);
  printf("Wilson Loop values: \n");
  for (const auto &Wtemporal : Wtemporal_vals) {
    printf("L = %d, T = %d, W(L,T) = %11.4e            Expected: %11.4e\n",
           static_cast<index_t>(Wtemporal[0]),
           static_cast<index_t>(Wtemporal[1]), Wtemporal[2], 1.0);
  }
  printf("Wilson Loop Kernel Time:     %11.4e s\n", wlTime);
  printf("Wilson Loop BW:              %11.4f GB/s\n",
         1.0e-9 * (real_t)sizeof(complex_t) *
             (volume4D + 2.0 * SU3_field_4D + SU3_gauge_4D) / wlTime);
  printf(HLINE);

  // reset plaquette time
  wlTime = std::numeric_limits<real_t>::max();

  printf("U(1) Gauge 3D: \n");
  const real_t U1_field_3D = volume3D * 1.0 * 1.0;
  const real_t U1_gauge_3D = 3.0 * U1_field_3D;
  printf("- Size of U(1) GaugeField:                  %12.2f MB\n",
         1.0e-6 * U1_gauge_3D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");

  {
    deviceGaugeField3D<3, 1> dev_g_U1_3D(stream_size_array, stream_size_array,
                                         stream_size_array, identitySUN<1>());

    printf("Running benchmark...\n");

    // run the benchmark
    for (index_t k = 0; k < STREAM_NTIMES; ++k) {
      Wtemporal_vals.clear();
      timer.reset();
      WilsonLoop_temporal<3, 1>(dev_g_U1_3D, LT_pairs, Wtemporal_vals);
      Kokkos::fence();
      wlTime = std::min(wlTime, timer.seconds());
    }
  }
  printf(HLINE);
  printf("Wilson Loop values: \n");
  for (const auto &Wtemporal : Wtemporal_vals) {
    printf("L = %d, T = %d, W(L,T) = %11.4e            Expected: %11.4e\n",
           static_cast<index_t>(Wtemporal[0]),
           static_cast<index_t>(Wtemporal[1]), Wtemporal[2], 1.0);
  }
  printf("Wilson Loop Kernel Time:     %11.4e s\n", wlTime);
  printf("Wilson Loop BW:              %11.4f GB/s\n",
         1.0e-9 * (real_t)sizeof(complex_t) *
             (volume3D + 2.0 * U1_field_3D + U1_gauge_3D) / wlTime);
  printf(HLINE);

  // reset plaquette time
  wlTime = std::numeric_limits<real_t>::max();

  printf("SU(2) Gauge 3D: \n");
  const real_t SU2_field_3D = volume3D * 2.0 * 2.0;
  const real_t SU2_gauge_3D = 3.0 * SU2_field_3D;
  printf("- Size of SU(2) GaugeField:                  %12.2f MB\n",
         1.0e-6 * SU2_gauge_3D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");

  {
    deviceGaugeField3D<3, 2> dev_g_SU2_3D(stream_size_array, stream_size_array,
                                          stream_size_array, identitySUN<2>());

    printf("Running benchmark...\n");

    // run the benchmark
    for (index_t k = 0; k < STREAM_NTIMES; ++k) {
      Wtemporal_vals.clear();
      timer.reset();
      WilsonLoop_temporal<3, 2>(dev_g_SU2_3D, LT_pairs, Wtemporal_vals);
      Kokkos::fence();
      wlTime = std::min(wlTime, timer.seconds());
    }
  }
  printf(HLINE);
  printf("Wilson Loop values: \n");
  for (const auto &Wtemporal : Wtemporal_vals) {
    printf("L = %d, T = %d, W(L,T) = %11.4e            Expected: %11.4e\n",
           static_cast<index_t>(Wtemporal[0]),
           static_cast<index_t>(Wtemporal[1]), Wtemporal[2], 1.0);
  }
  printf("Wilson Loop Kernel Time:     %11.4e s\n", wlTime);
  printf("Wilson Loop BW:              %11.4f GB/s\n",
         1.0e-9 * (real_t)sizeof(complex_t) *
             (volume3D + 2.0 * SU2_field_3D + SU2_gauge_3D) / wlTime);
  printf(HLINE);

  // reset plaquette time
  wlTime = std::numeric_limits<real_t>::max();

  printf("SU(3) Gauge 3D: \n");
  const real_t SU3_field_3D = volume3D * 3.0 * 3.0;
  const real_t SU3_gauge_3D = 3.0 * SU3_field_3D;
  printf("- Size of SU(3) GaugeField:                  %12.2f MB\n",
         1.0e-6 * SU3_gauge_3D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");

  {
    deviceGaugeField3D<3, 3> dev_g_SU3_3D(stream_size_array, stream_size_array,
                                          stream_size_array, identitySUN<3>());

    printf("Running benchmark...\n");

    // run the benchmark
    for (index_t k = 0; k < STREAM_NTIMES; ++k) {
      Wtemporal_vals.clear();
      timer.reset();
      WilsonLoop_temporal<3, 3>(dev_g_SU3_3D, LT_pairs, Wtemporal_vals);
      Kokkos::fence();
      wlTime = std::min(wlTime, timer.seconds());
    }
  }
  printf(HLINE);
  printf("Wilson Loop values: \n");
  for (const auto &Wtemporal : Wtemporal_vals) {
    printf("L = %d, T = %d, W(L,T) = %11.4e            Expected: %11.4e\n",
           static_cast<index_t>(Wtemporal[0]),
           static_cast<index_t>(Wtemporal[1]), Wtemporal[2], 1.0);
  }
  printf("Wilson Loop Kernel Time:     %11.4e s\n", wlTime);
  printf("Wilson Loop BW:              %11.4f GB/s\n",
         1.0e-9 * (real_t)sizeof(complex_t) *
             (volume3D + 2.0 * SU3_field_3D + SU3_gauge_3D) / wlTime);
  printf(HLINE);

  // reset plaquette time
  wlTime = std::numeric_limits<real_t>::max();

  printf("U(1) Gauge 2D: \n");
  const real_t U1_field_2D = volume2D * 1.0 * 1.0;
  const real_t U1_gauge_2D = 2.0 * U1_field_2D;
  printf("- Size of U(1) GaugeField:                  %12.2f MB\n",
         1.0e-6 * U1_gauge_2D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");

  {
    deviceGaugeField2D<2, 1> dev_g_U1_2D(stream_size_array, stream_size_array,
                                         identitySUN<1>());

    printf("Running benchmark...\n");

    // run the benchmark
    for (index_t k = 0; k < STREAM_NTIMES; ++k) {
      Wtemporal_vals.clear();
      timer.reset();
      WilsonLoop_temporal<2, 1>(dev_g_U1_2D, LT_pairs, Wtemporal_vals);
      Kokkos::fence();
      wlTime = std::min(wlTime, timer.seconds());
    }
  }
  printf(HLINE);
  printf("Wilson Loop values: \n");
  for (const auto &Wtemporal : Wtemporal_vals) {
    printf("L = %d, T = %d, W(L,T) = %11.4e            Expected: %11.4e\n",
           static_cast<index_t>(Wtemporal[0]),
           static_cast<index_t>(Wtemporal[1]), Wtemporal[2], 1.0);
  }
  printf("Wilson Loop Kernel Time:     %11.4e s\n", wlTime);
  printf("Wilson Loop BW:              %11.4f GB/s\n",
         1.0e-9 * (real_t)sizeof(complex_t) *
             (volume2D + 2.0 * U1_field_2D + U1_gauge_2D) / wlTime);
  printf(HLINE);

  // reset plaquette time
  wlTime = std::numeric_limits<real_t>::max();

  printf("SU(2) Gauge 2D: \n");
  const real_t SU2_field_2D = volume2D * 2.0 * 2.0;
  const real_t SU2_gauge_2D = 2.0 * SU2_field_2D;
  printf("- Size of SU(2) GaugeField:                  %12.2f MB\n",
         1.0e-6 * SU2_gauge_2D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");

  {
    deviceGaugeField2D<2, 2> dev_g_SU2_2D(stream_size_array, stream_size_array,
                                          identitySUN<2>());

    printf("Running benchmark...\n");

    // run the benchmark
    for (index_t k = 0; k < STREAM_NTIMES; ++k) {
      Wtemporal_vals.clear();
      timer.reset();
      WilsonLoop_temporal<2, 2>(dev_g_SU2_2D, LT_pairs, Wtemporal_vals);
      Kokkos::fence();
      wlTime = std::min(wlTime, timer.seconds());
    }
  }
  printf(HLINE);
  printf("Wilson Loop values: \n");
  for (const auto &Wtemporal : Wtemporal_vals) {
    printf("L = %d, T = %d, W(L,T) = %11.4e            Expected: %11.4e\n",
           static_cast<index_t>(Wtemporal[0]),
           static_cast<index_t>(Wtemporal[1]), Wtemporal[2], 1.0);
  }
  printf("Wilson Loop Kernel Time:     %11.4e s\n", wlTime);
  printf("Wilson Loop BW:              %11.4f GB/s\n",
         1.0e-9 * (real_t)sizeof(complex_t) *
             (volume2D + 2.0 * SU2_field_2D + SU2_gauge_2D) / wlTime);
  printf(HLINE);

  // reset plaquette time
  wlTime = std::numeric_limits<real_t>::max();
  printf("SU(3) Gauge 2D: \n");
  const real_t SU3_field_2D = volume2D * 3.0 * 3.0;
  const real_t SU3_gauge_2D = 2.0 * SU3_field_2D;
  printf("- Size of SU(3) GaugeField:                  %12.2f MB\n",
         1.0e-6 * SU3_gauge_2D * (real_t)sizeof(complex_t));
  printf(HLINE);
  printf("Initializing Gauge...\n");

  {
    deviceGaugeField2D<2, 3> dev_g_SU3_2D(stream_size_array, stream_size_array,
                                          identitySUN<3>());

    printf("Running benchmark...\n");

    // run the benchmark
    for (index_t k = 0; k < STREAM_NTIMES; ++k) {
      Wtemporal_vals.clear();
      timer.reset();
      WilsonLoop_temporal<2, 3>(dev_g_SU3_2D, LT_pairs, Wtemporal_vals);
      Kokkos::fence();
      wlTime = std::min(wlTime, timer.seconds());
    }
  }
  printf(HLINE);
  printf("Wilson Loop values: \n");
  for (const auto &Wtemporal : Wtemporal_vals) {
    printf("L = %d, T = %d, W(L,T) = %11.4e            Expected: %11.4e\n",
           static_cast<index_t>(Wtemporal[0]),
           static_cast<index_t>(Wtemporal[1]), Wtemporal[2], 1.0);
  }
  printf("Wilson Loop Kernel Time:     %11.4e s\n", wlTime);
  printf("Wilson Loop BW:              %11.4f GB/s\n",
         1.0e-9 * (real_t)sizeof(complex_t) *
             (volume2D + 2.0 * SU3_field_2D + SU3_gauge_2D) / wlTime);
  printf(HLINE);

  return 0;
}

int parse_args(int argc, char **argv, size_t &stream_array_size) {
  // Defaults
  stream_array_size = 32;

  const std::string help_string =
      "  -n <N>, --nelements <N>\n"
      "     Length of the gauge field dimensions.\n"
      "     Default: 32\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"nelements", required_argument, NULL, 'n'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "n:h", long_options, &option_index)) !=
         -1)
    switch (c) {
    case 'n':
      stream_array_size = atoi(optarg);
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
  printf(
      "SU(N) GaugeField 2D, 3D and 4D Wilson Loop kernel test and benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  size_t stream_array_size;
  rc = parse_args(argc, argv, stream_array_size);
  if (rc == 0) {
    rc = run_benchmark(stream_array_size);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}
