#include "../include/klft.hpp"
#include "../include/Metropolis.hpp"
#include <iostream>
#include <fstream>

namespace klft {

  template <typename T>
  void Metropolis_SU3_4D(const int &LX, const int &LY, const int &LZ, const int &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename) {
    std::cout << "Running Metropolis_SU3_4D" << std::endl;
    std::cout << "Gauge Field Dimensions:" << std::endl;
    std::cout << "LX = " << LX << std::endl;
    std::cout << "LY = " << LY << std::endl;
    std::cout << "LZ = " << LZ << std::endl;
    std::cout << "LT = " << LT << std::endl;
    std::cout << "Metropolis Parameters:" << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "delta = " << delta << std::endl;
    std::cout << "n_hit = " << n_hit << std::endl;
    std::cout << "n_sweep = " << n_sweep << std::endl;
    std::cout << "seed = " << seed << std::endl;
    std::cout << "start condition = " << (cold_start ? "cold" : "hot") << std::endl;
    std::cout << "output file = " << outfilename << std::endl;
    std::ofstream outfile;
    if(outfilename != "") {
      outfile.open(outfilename);
      outfile << "step, plaquette, acceptance_rate, time" << std::endl;
    }
    Kokkos::initialize();
    {
      using Group = SU3<T>;
      using GaugeFieldType = GaugeField<T,Group,4,3>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      RNG rng = RNG(seed);
      GaugeFieldType gauge_field = GaugeFieldType({LX,LY,LZ,LT});
      Metropolis<T,Group,GaugeFieldType,RNG> metropolis = Metropolis<T,Group,GaugeFieldType,RNG>(gauge_field,rng,n_hit,beta,delta);
      metropolis.initGauge(cold_start);
      std::cout << "Starting Plaquette: " << gauge_field.get_plaquette() << std::endl;
      std::cout << "Starting Metropolis: " << std::endl;
      auto metropolis_start_time = std::chrono::high_resolution_clock::now();
      for(size_t i = 0; i < n_sweep; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        T acceptance_rate = metropolis.sweep();
        T plaquette = gauge_field.get_plaquette();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sweep_time = end_time - start_time;
        std::cout << "Step: " << i << " Plaquette: " << plaquette << " Acceptance Rate: " << acceptance_rate << " Time: " << sweep_time.count() << std::endl;
        if(outfilename != "") {
          outfile << i << ", " << plaquette << ", " << acceptance_rate << ", " << sweep_time.count() << std::endl;
        }
      }
    auto metropolis_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metropolis_time = metropolis_end_time - metropolis_start_time;
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  template <typename T>
  void Metropolis_SU3_3D(const int &LX, const int &LY, const int &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename) {
    std::cout << "Running Metropolis_SU3_3D" << std::endl;
    std::cout << "Gauge Field Dimensions:" << std::endl;
    std::cout << "LX = " << LX << std::endl;
    std::cout << "LY = " << LY << std::endl;
    std::cout << "LT = " << LT << std::endl;
    std::cout << "Metropolis Parameters:" << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "delta = " << delta << std::endl;
    std::cout << "n_hit = " << n_hit << std::endl;
    std::cout << "n_sweep = " << n_sweep << std::endl;
    std::cout << "seed = " << seed << std::endl;
    std::cout << "start condition = " << (cold_start ? "cold" : "hot") << std::endl;
    std::cout << "output file = " << outfilename << std::endl;
    std::ofstream outfile;
    if(outfilename != "") {
      outfile.open(outfilename);
      outfile << "step, plaquette, acceptance_rate, time" << std::endl;
    }
    Kokkos::initialize();
    {
      using Group = SU3<T>;
      using GaugeFieldType = GaugeField<T,Group,3,3>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      RNG rng = RNG(seed);
      GaugeFieldType gauge_field = GaugeFieldType({LX,LY,LT});
      Metropolis<T,Group,GaugeFieldType,RNG> metropolis = Metropolis<T,Group,GaugeFieldType,RNG>(gauge_field,rng,n_hit,beta,delta);
      metropolis.initGauge(cold_start);
      std::cout << "Starting Plaquette: " << gauge_field.get_plaquette() << std::endl;
      std::cout << "Starting Metropolis: " << std::endl;
      auto metropolis_start_time = std::chrono::high_resolution_clock::now();
      for(size_t i = 0; i < n_sweep; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        T acceptance_rate = metropolis.sweep();
        T plaquette = gauge_field.get_plaquette();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sweep_time = end_time - start_time;
        std::cout << "Step: " << i << " Plaquette: " << plaquette << " Acceptance Rate: " << acceptance_rate << " Time: " << sweep_time.count() << std::endl;
        if(outfilename != "") {
          outfile << i << ", " << plaquette << ", " << acceptance_rate << ", " << sweep_time.count() << std::endl;
        }
      }
    auto metropolis_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metropolis_time = metropolis_end_time - metropolis_start_time;
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  template <typename T>
  void Metropolis_SU3_2D(const int &LX, const int &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename) {
    std::cout << "Running Metropolis_SU3_2D" << std::endl;
    std::cout << "Gauge Field Dimensions:" << std::endl;
    std::cout << "LX = " << LX << std::endl;
    std::cout << "LT = " << LT << std::endl;
    std::cout << "Metropolis Parameters:" << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "delta = " << delta << std::endl;
    std::cout << "n_hit = " << n_hit << std::endl;
    std::cout << "n_sweep = " << n_sweep << std::endl;
    std::cout << "seed = " << seed << std::endl;
    std::cout << "start condition = " << (cold_start ? "cold" : "hot") << std::endl;
    std::cout << "output file = " << outfilename << std::endl;
    std::ofstream outfile;
    if(outfilename != "") {
      outfile.open(outfilename);
      outfile << "step, plaquette, acceptance_rate, time" << std::endl;
    }
    Kokkos::initialize();
    {
      using Group = SU3<T>;
      using GaugeFieldType = GaugeField<T,Group,2,3>;
      using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
      RNG rng = RNG(seed);
      GaugeFieldType gauge_field = GaugeFieldType({LX,LT});
      Metropolis<T,Group,GaugeFieldType,RNG> metropolis = Metropolis<T,Group,GaugeFieldType,RNG>(gauge_field,rng,n_hit,beta,delta);
      metropolis.initGauge(cold_start);
      std::cout << "Starting Plaquette: " << gauge_field.get_plaquette() << std::endl;
      std::cout << "Starting Metropolis: " << std::endl;
      auto metropolis_start_time = std::chrono::high_resolution_clock::now();
      for(size_t i = 0; i < n_sweep; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        T acceptance_rate = metropolis.sweep();
        T plaquette = gauge_field.get_plaquette();
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sweep_time = end_time - start_time;
        std::cout << "Step: " << i << " Plaquette: " << plaquette << " Acceptance Rate: " << acceptance_rate << " Time: " << sweep_time.count() << std::endl;
        if(outfilename != "") {
          outfile << i << ", " << plaquette << ", " << acceptance_rate << ", " << sweep_time.count() << std::endl;
        }
      }
    auto metropolis_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> metropolis_time = metropolis_end_time - metropolis_start_time;
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
    }
    Kokkos::finalize();
    outfile.close();
  }

  template void Metropolis_SU3_4D<float>(const int &LX, const int &LY, const int &LZ, const int &LT, 
                                   const size_t &n_hit, const float &beta, const float &delta,
                                   const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                   const std::string &outfilename);

  template void Metropolis_SU3_4D<double>(const int &LX, const int &LY, const int &LZ, const int &LT, 
                                   const size_t &n_hit, const double &beta, const double &delta,
                                   const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                   const std::string &outfilename);

  template void Metropolis_SU3_3D<float>(const int &LX, const int &LY, const int &LT, 
                                   const size_t &n_hit, const float &beta, const float &delta,
                                   const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                   const std::string &outfilename);

  template void Metropolis_SU3_3D<double>(const int &LX, const int &LY, const int &LT,
                                   const size_t &n_hit, const double &beta, const double &delta,
                                   const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                   const std::string &outfilename);    

  template void Metropolis_SU3_2D<float>(const int &LX, const int &LT,
                                   const size_t &n_hit, const float &beta, const float &delta,
                                   const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                   const std::string &outfilename);

  template void Metropolis_SU3_2D<double>(const int &LX, const int &LT,
                                   const size_t &n_hit, const double &beta, const double &delta,
                                   const size_t &seed, const size_t &n_sweep, const bool cold_start,
                                   const std::string &outfilename);                                                                                                                                                                    
}