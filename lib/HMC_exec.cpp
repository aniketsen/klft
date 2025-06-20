#include "HMC_exec.hpp"
#include "../include/InputParser.hpp"
#include "../include/klft.hpp"
#include "AdjointSUN.hpp"
#include "FieldTypeHelper.hpp"
#include "GaugeObservable.hpp"
#include "HMC_Params.hpp"

using RNGType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;

namespace klft {

int HMC_execute(const std::string &input_file) {
  // get verbosity from environment
  const int verbosity = std::getenv("KLFT_VERBOSITY")
                            ? std::atoi(std::getenv("KLFT_VERBOSITY"))
                            : 0;
  setVerbosity(verbosity);
  // get tuning from environment
  const int tuning =
      std::getenv("KLFT_TUNING") ? std::atoi(std::getenv("KLFT_TUNING")) : 0;
  setTuning(tuning);
  // if tuning is enbled, check if the user has set the
  // KLFT_CACHE_FILE environment variable
  if (tuning) {
    const char *cache_file = std::getenv("KLFT_CACHE_FILE");
    // if it exists, read the cache
    if (cache_file) {
      if (KLFT_VERBOSITY > 0) {
        printf("Reading cache file: %s\n", cache_file);
      }
      readTuneCache(cache_file);
    }
  }
  HMCParams hmcParams;
  GaugeObservableParams gaugeObsParams;
  if (!parseInputFile(input_file, gaugeObsParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  if (!parseInputFile(input_file, hmcParams)) {
    printf("Error parsing input file\n");
    return -1;
  }
  // print the parameters
  hmcParams.print();
  // initialize RNG
  RNGType rng(hmcParams.seed);

  // initialize gauge field and run hmc
  // based on the system parameters
  // case 4D
  if (hmcParams.coldStart) {
    if (hmcParams.Ndims == 4) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType = DeviceGaugeFieldType<4, 1>;
        using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        typename DGaugeFieldType::type dev_g_U1_4D(hmcParams.L0, hmcParams.L1,
                                                   hmcParams.L2, hmcParams.L3,
                                                   identitySUN<1>());
        typename DAdjFieldType::type dev_a_U1_4D(hmcParams.L0, hmcParams.L1,
                                                 hmcParams.L2, hmcParams.L3,
                                                 traceT(identitySUN<1>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_U1_4D, dev_a_U1_4D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType = DeviceGaugeFieldType<4, 2>;
        using DAdjFieldType = DeviceAdjFieldType<4, 2>;
        typename DGaugeFieldType::type dev_g_SU2_4D(hmcParams.L0, hmcParams.L1,
                                                    hmcParams.L2, hmcParams.L3,
                                                    identitySUN<2>());
        typename DAdjFieldType::type dev_a_SU2_4D(hmcParams.L0, hmcParams.L1,
                                                  hmcParams.L2, hmcParams.L3,
                                                  traceT(identitySUN<2>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_SU2_4D, dev_a_SU2_4D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<4, 3>;
      //   using DAdjFieldType = DeviceAdjFieldType<4, 3>;
      //   typename DGaugeFieldType::type dev_g_SU3_4D(hmcParams.L0,
      //   hmcParams.L1,
      //                                              hmcParams.L2,
      //                                              hmcParams.L3,
      //                                              identitySUN<3>());
      //   typename DAdjFieldType::type dev_a_SU3_4D(hmcParams.L0, hmcParams.L1,
      //                                            hmcParams.L2, hmcParams.L3,
      //                                            traceT(identitySUN<3>()));
      //   run_HMC<DGaugeFieldType, DAdjFieldType>(
      //       dev_g_SU3_4D, dev_a_SU3_4D, hmcParams, gaugeObsParams, rng);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
    // case 3D
    else if (hmcParams.Ndims == 3) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType = DeviceGaugeFieldType<3, 1>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        typename DGaugeFieldType::type dev_g_U1_3D(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, identitySUN<1>());
        typename DAdjFieldType::type dev_a_U1_3D(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<1>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_U1_3D, dev_a_U1_3D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType = DeviceGaugeFieldType<3, 2>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        typename DGaugeFieldType::type dev_g_SU2_3D(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, identitySUN<2>());
        typename DAdjFieldType::type dev_a_SU2_3D(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<2>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_SU2_3D, dev_a_SU2_3D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<3, 3>;
      //   using DAdjFieldType = DeviceAdjFieldType<3, 3>;
      //   typename DGaugeFieldType::type dev_g_SU3_3D(hmcParams.L0,
      //   hmcParams.L1,
      //                                              hmcParams.L2,
      //                                              identitySUN<3>());
      //   typename DAdjFieldType::type dev_a_SU3_3D(hmcParams.L0, hmcParams.L1,
      //                                            hmcParams.L2,
      //                                            traceT(identitySUN<3>()));
      //   run_HMC<DGaugeFieldType, DAdjFieldType>(
      //       dev_g_SU3_3D, dev_a_SU3_3D, hmcParams, gaugeObsParams, rng);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
    // case 2D
    else if (hmcParams.Ndims == 2) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType = DeviceGaugeFieldType<2, 1>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        typename DGaugeFieldType::type dev_g_U1_2D(hmcParams.L0, hmcParams.L1,
                                                   identitySUN<1>());
        typename DAdjFieldType::type dev_a_U1_2D(hmcParams.L0, hmcParams.L1,
                                                 traceT(identitySUN<1>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_U1_2D, dev_a_U1_2D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType = DeviceGaugeFieldType<2, 2>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        typename DGaugeFieldType::type dev_g_SU2_2D(hmcParams.L0, hmcParams.L1,
                                                    identitySUN<2>());
        typename DAdjFieldType::type dev_a_SU2_2D(hmcParams.L0, hmcParams.L1,
                                                  traceT(identitySUN<2>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_SU2_2D, dev_a_SU2_2D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<2, 3>;
      //   using DAdjFieldType = DeviceAdjFieldType<2, 3>;
      //   typename DGaugeFieldType::type dev_g_SU3_2D(hmcParams.L0,
      //   hmcParams.L1,
      //                                              identitySUN<3>());
      //   typename DAdjFieldType::type dev_a_SU3_2D(hmcParams.L0, hmcParams.L1,
      //                                            traceT(identitySUN<3>()));
      //   run_HMC<DGaugeFieldType, DAdjFieldType>(
      //       dev_g_SU3_2D, dev_a_SU3_2D, hmcParams, gaugeObsParams, rng);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
  } else {
    if (hmcParams.Ndims == 4) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType = DeviceGaugeFieldType<4, 1>;
        using DAdjFieldType = DeviceAdjFieldType<4, 1>;
        typename DGaugeFieldType::type dev_g_U1_4D(hmcParams.L0, hmcParams.L1,
                                                   hmcParams.L2, hmcParams.L3,
                                                   rng, hmcParams.rngDelta);
        typename DAdjFieldType::type dev_a_U1_4D(hmcParams.L0, hmcParams.L1,
                                                 hmcParams.L2, hmcParams.L3,
                                                 traceT(identitySUN<1>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_U1_4D, dev_a_U1_4D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType = DeviceGaugeFieldType<4, 2>;
        using DAdjFieldType = DeviceAdjFieldType<4, 2>;
        typename DGaugeFieldType::type dev_g_SU2_4D(hmcParams.L0, hmcParams.L1,
                                                    hmcParams.L2, hmcParams.L3,
                                                    rng, hmcParams.rngDelta);
        typename DAdjFieldType::type dev_a_SU2_4D(hmcParams.L0, hmcParams.L1,
                                                  hmcParams.L2, hmcParams.L3,
                                                  traceT(identitySUN<2>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_SU2_4D, dev_a_SU2_4D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<4, 3>;
      //   using DAdjFieldType = DeviceAdjFieldType<4, 3>;
      //   typename DGaugeFieldType::type dev_g_SU3_4D(hmcParams.L0,
      //   hmcParams.L1,
      //                                              hmcParams.L2,
      //                                              hmcParams.L3, rng,
      //                                              hmcParams.rngDelta);
      //   typename DAdjFieldType::type dev_a_SU3_4D(hmcParams.L0, hmcParams.L1,
      //                                            hmcParams.L2, hmcParams.L3,
      //                                            traceT(identitySUN<3>()));
      //   run_HMC<DGaugeFieldType, DAdjFieldType>(
      //       dev_g_SU3_4D, dev_a_SU3_4D, hmcParams, gaugeObsParams, rng);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
    // case 3D
    else if (hmcParams.Ndims == 3) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType = DeviceGaugeFieldType<3, 1>;
        using DAdjFieldType = DeviceAdjFieldType<3, 1>;
        typename DGaugeFieldType::type dev_g_U1_3D(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, rng, hmcParams.rngDelta);
        typename DAdjFieldType::type dev_a_U1_3D(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<1>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_U1_3D, dev_a_U1_3D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType = DeviceGaugeFieldType<3, 2>;
        using DAdjFieldType = DeviceAdjFieldType<3, 2>;
        typename DGaugeFieldType::type dev_g_SU2_3D(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, rng, hmcParams.rngDelta);
        typename DAdjFieldType::type dev_a_SU2_3D(
            hmcParams.L0, hmcParams.L1, hmcParams.L2, traceT(identitySUN<2>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_SU2_3D, dev_a_SU2_3D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<3, 3>;
      //   using DAdjFieldType = DeviceAdjFieldType<3, 3>;
      //   typename DGaugeFieldType::type dev_g_SU3_3D(hmcParams.L0,
      //   hmcParams.L1,
      //                                              hmcParams.L2,
      //                                              rng, hmcParams.rngDelta);
      //   typename DAdjFieldType::type dev_a_SU3_3D(hmcParams.L0, hmcParams.L1,
      //                                            hmcParams.L2,
      //                                            traceT(identitySUN<3>()));
      //   run_HMC<DGaugeFieldType, DAdjFieldType>(
      //       dev_g_SU3_3D, dev_a_SU3_3D, hmcParams, gaugeObsParams, rng);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
    // case 2D
    else if (hmcParams.Ndims == 2) {
      // case U(1)
      if (hmcParams.Nc == 1) {
        using DGaugeFieldType = DeviceGaugeFieldType<2, 1>;
        using DAdjFieldType = DeviceAdjFieldType<2, 1>;
        typename DGaugeFieldType::type dev_g_U1_2D(hmcParams.L0, hmcParams.L1,
                                                   rng, hmcParams.rngDelta);
        typename DAdjFieldType::type dev_a_U1_2D(hmcParams.L0, hmcParams.L1,
                                                 traceT(identitySUN<1>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_U1_2D, dev_a_U1_2D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(2)
      else if (hmcParams.Nc == 2) {
        using DGaugeFieldType = DeviceGaugeFieldType<2, 2>;
        using DAdjFieldType = DeviceAdjFieldType<2, 2>;
        typename DGaugeFieldType::type dev_g_SU2_2D(hmcParams.L0, hmcParams.L1,
                                                    rng, hmcParams.rngDelta);
        typename DAdjFieldType::type dev_a_SU2_2D(hmcParams.L0, hmcParams.L1,
                                                  traceT(identitySUN<2>()));
        run_HMC<DGaugeFieldType, DAdjFieldType>(dev_g_SU2_2D, dev_a_SU2_2D,
                                                hmcParams, gaugeObsParams, rng);
      }
      // case SU(3)
      // else if (hmcParams.Nc == 3) {
      //   using DGaugeFieldType = DeviceGaugeFieldType<2, 3>;
      //   using DAdjFieldType = DeviceAdjFieldType<2, 3>;
      //   typename DGaugeFieldType::type dev_g_SU3_2D(hmcParams.L0,
      //   hmcParams.L1,
      //                                              rng, hmcParams.rngDelta);
      //   typename DAdjFieldType::type dev_a_SU3_2D(hmcParams.L0, hmcParams.L1,
      //                                            traceT(identitySUN<3>()));
      //   run_HMC<DGaugeFieldType, DAdjFieldType>(
      //       dev_g_SU3_2D, dev_a_SU3_2D, hmcParams, gaugeObsParams, rng);
      // }
      // case SU(N)
      else {
        printf("Error: Unsupported gauge group\n");
        return -1;
      }
    }
  }
  // if tuning is enabled, write the cache file
  if (KLFT_TUNING) {
    const char *cache_file = std::getenv("KLFT_CACHE_FILE");
    if (cache_file) {
      writeTuneCache(cache_file);
    } else {
      printf("KLFT_CACHE_FILE not set\n");
    }
  }

  return 0;
}

} // namespace klft
