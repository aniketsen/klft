#include "include/klft.hpp"
#include "include/Metropolis.hpp"
#include <mpi.h>
#include <iostream>
#include <fstream>

using real_t = double;
#define Ndim 4
#define Nc 1

int main(int argc, char **argv) {
  Kokkos::Array<int,Ndim> dims = {4,4,4,4};
  Kokkos::Array<int,Ndim> mpi_dims = {1,1,1,1};
  Kokkos::Array<int,Ndim> comm_dims = {0,0,0,0};
  Kokkos::Array<int,Ndim> local_dims = {4,4,4,4};
  size_t n_hit = 100;
  real_t beta = 2.0;
  real_t delta = 0.05;
  size_t seed = 1234;
  size_t n_sweep = 1000;
  bool cold_start = true;
  std::string outfilename = "";
  std::ofstream outfile;
  for(int i = 1; i < argc; i++) {
    if(std::string(argv[i]) == "--dims") {
      for(int j = 0; j < Ndim; j++) {
        dims[j] = std::stoi(argv[i+1+j]);
      }
    }
    if(std::string(argv[i]) == "--mpi-dims") {
      for(int j = 0; j < Ndim; j++) {
        mpi_dims[j] = std::stoi(argv[i+1+j]);
      }
    }
    if(std::string(argv[i]) == "--n-hit") {
      n_hit = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--beta") {
      beta = std::stod(argv[i+1]);
    }
    if(std::string(argv[i]) == "--delta") {
      delta = std::stod(argv[i+1]);
    }
    if(std::string(argv[i]) == "--seed") {
      seed = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--n-sweep") {
      n_sweep = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--cold-start") {
      cold_start = std::string(argv[i+1]) == "true";
    }
    if(std::string(argv[i]) == "--outfilename") {
      outfilename = argv[i+1];
    }
    if(std::string(argv[i]) == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "--dims dimensions of the lattice" << std::endl;
      std::cout << "--mpi-dims dimensions of the MPI grid" << std::endl;
      std::cout << "--n-hit number of hits per sweep" << std::endl;
      std::cout << "--beta inverse coupling constant" << std::endl;
      std::cout << "--delta step size" << std::endl;
      std::cout << "--seed random number generator seed" << std::endl;
      std::cout << "--n-sweep number of sweeps" << std::endl;
      std::cout << "--cold-start true or false" << std::endl;
      std::cout << "--outfilename output filename" << std::endl;
      return 0;
    }
  }
#ifndef KLFT_USE_MPI
  std::cout << "KLFT_USE_MPI not defined" << std::endl;
  return 1;
#else
  for(int i = 0; i < Ndim; i++) {
    if(mpi_dims[i] > 1) {
      comm_dims[i] = 1;
    }
  }
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int check_size = 1;
  for(int i = 0; i < Ndim; i++) {
    check_size *= mpi_dims[i];
  }
  if(size != check_size) {
    if(rank == 0) {
      std::cout << "Error: size != check_size" << std::endl;
    }
    MPI_Finalize();
    return 1;
  }
  const char* envThreads = std::getenv("OMP_NUM_THREADS");
  int numThreads = envThreads ? std::stoi(envThreads) : 1;
  Kokkos::InitializationSettings settings;
  settings.set_device_id(rank);
  settings.set_num_threads(numThreads);
  Kokkos::initialize(settings);
  {
  klft::Comm<Ndim> comm(mpi_dims);
  if(rank == 0) {
    std::cout << "Running Metropolis for U(1)" << std::endl;
    std::cout << "Lattice Dimensions: " << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << std::endl;
    std::cout << "MPI Dimensions: " << mpi_dims[0] << " " << mpi_dims[1] << " " << mpi_dims[2] << " " << mpi_dims[3] << std::endl;
    std::cout << "Number of Hits per Sweep: " << n_hit << std::endl;
    std::cout << "Inverse Coupling Constant: " << beta << std::endl;
    std::cout << "Step Size: " << delta << std::endl;
    std::cout << "Random Number Generator Seed: " << seed << std::endl;
    std::cout << "Number of Sweeps: " << n_sweep << std::endl;
    std::cout << "Cold Start: " << (cold_start ? "true" : "false") << std::endl;
    std::cout << "Output Filename: " << outfilename << std::endl;
    if(outfilename != "") {
      outfile.open(outfilename);
      outfile << "step, plaquette, acceptance_rate, time" << std::endl;
    }
  }
  for(int i = 0; i < Ndim; i++) {
    local_dims[i] = (int)(dims[i]/mpi_dims[i]);
  }
  using Group = klft::U1<real_t>;
  using GaugeFieldType = klft::GaugeField<real_t,Group,Ndim,Nc>;
  using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
  RNG rng = RNG(2412*seed+rank*112);
  GaugeFieldType gauge_field = GaugeFieldType(dims,local_dims,comm_dims,comm.prev_rank,comm.next_rank);
  klft::Metropolis<real_t,Group,GaugeFieldType,RNG> metropolis = klft::Metropolis<real_t,Group,GaugeFieldType,RNG>(gauge_field,rng,n_hit,beta,delta);
  metropolis.initGauge(cold_start);
  metropolis.gauge_field.update_halo_plus(comm);
  Kokkos::fence();
  metropolis.gauge_field.update_halo_minus(comm);
  Kokkos::fence();
  real_t lplaq = gauge_field.get_plaquette();
  real_t gplaq;
  MPI_Allreduce(&lplaq, &gplaq, 1, MPI_DOUBLE, MPI_SUM, comm);
  if(rank == 0) {
    std::cout << "Starting Plaquette: " << gplaq << std::endl;
    std::cout << "Starting Metropolis: " << std::endl;
  }
  auto metropolis_start_time = std::chrono::high_resolution_clock::now();
  for(size_t i = 0; i < n_sweep; i++) {
    auto start_time = std::chrono::high_resolution_clock::now();
    real_t acceptance_rate_local = metropolis.sweep();
    Kokkos::fence();
    real_t acceptance_rate;
    MPI_Allreduce(&acceptance_rate_local, &acceptance_rate, 1, MPI_DOUBLE, MPI_SUM, comm);
    metropolis.gauge_field.update_halo_plus(comm);
    Kokkos::fence();
    metropolis.gauge_field.update_halo_minus(comm);
    Kokkos::fence();
    real_t plaquette_local = gauge_field.get_plaquette();
    Kokkos::fence();
    real_t plaquette;
    MPI_Allreduce(&plaquette_local, &plaquette, 1, MPI_DOUBLE, MPI_SUM, comm);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sweep_time = end_time - start_time;
    if(rank == 0) {
      std::cout << "Step: " << i << " Plaquette: " << plaquette << " Acceptance Rate: " << acceptance_rate << " Time: " << sweep_time.count() << std::endl;
      if(outfilename != "") {
        std::ofstream outfile;
        outfile.open(outfilename);
        outfile << i << ", " << plaquette << ", " << acceptance_rate << ", " << sweep_time.count() << std::endl;
        outfile.close();
      }
    }
  }
  auto metropolis_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> metropolis_time = metropolis_end_time - metropolis_start_time;
  if(rank == 0) {
    std::cout << "Metropolis Time: " << metropolis_time.count() << std::endl;
  }
  }
  Kokkos::finalize();
  if(rank == 0) {
    outfile.close();
  }
  MPI_Finalize();
  return 0;
#endif
}