#include "include/klft.hpp"
#include "include/HMC.hpp"
#include <mpi.h>
#include <iostream>
#include <fstream>

using real_t = double;
#define Ndim 4
#define Nc 2

int main(int argc, char **argv) {
  Kokkos::Array<int,Ndim> dims = {4,4,4,4};
  Kokkos::Array<int,Ndim> mpi_dims = {1,1,1,1};
  Kokkos::Array<int,Ndim> comm_dims = {0,0,0,0};
  Kokkos::Array<int,Ndim> local_dims = {4,4,4,4};
  size_t n_traj = 1000;
  size_t n_steps = 50;
  real_t tau = 1.0;
  real_t beta = 2.0;
  size_t seed = 1234;
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
    if(std::string(argv[i]) == "--n-traj") {
      n_traj = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--n-steps") {
      n_steps = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--tau") {
      tau = std::stod(argv[i+1]);
    }
    if(std::string(argv[i]) == "--beta") {
      beta = std::stod(argv[i+1]);
    }
    if(std::string(argv[i]) == "--seed") {
      seed = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--outfilename") {
      outfilename = argv[i+1];
    }
    if(std::string(argv[i]) == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "--dims dimensions of the lattice" << std::endl;
      std::cout << "--mpi-dims dimensions of the MPI grid" << std::endl;
      std::cout << "--n-traj number of trajectories" << std::endl;
      std::cout << "--n-steps number of steps per trajectory" << std::endl;
      std::cout << "--tau step size" << std::endl;
      std::cout << "--beta coupling" << std::endl;
      std::cout << "--seed random seed" << std::endl;
      std::cout << "--outfilename output file name" << std::endl;
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
    std::cout << "Running HMC for U(1)" << std::endl;
    std::cout << "Lattice Dimensions: " << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << std::endl;
    std::cout << "MPI Dimensions: " << mpi_dims[0] << " " << mpi_dims[1] << " " << mpi_dims[2] << " " << mpi_dims[3] << std::endl;
    std::cout << "Number of Trajectories: " << n_traj << std::endl;
    std::cout << "Number of Steps per Trajectory: " << n_steps << std::endl;
    std::cout << "Step Size: " << tau << std::endl;
    std::cout << "Inverse Coupling Constant: " << beta << std::endl;
    std::cout << "Random Number Generator Seed: " << seed << std::endl;
    std::cout << "Output Filename: " << outfilename << std::endl;
    if(outfilename != "") {
      outfile.open(outfilename);
      outfile << "traj, accept, plaquette, time, acceptance rate" << std::endl;
    }
  }
  for(int i = 0; i < Ndim; i++) {
    local_dims[i] = (int)(dims[i]/mpi_dims[i]);
  }
  using Group = klft::SU2<real_t>;
  using Adjoint = klft::AdjointSU2<real_t>;
  using GaugeFieldType = klft::GaugeField<real_t,Group,Ndim,Nc>;
  using AdjointFieldType = klft::AdjointField<real_t,Adjoint,Ndim,Nc>;
  using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;
  using HamiltonianFieldType = klft::HamiltonianField<real_t,Group,Adjoint,Ndim,Nc>;
  RNG rng = RNG(2412*seed+rank*112);
  std::mt19937 mt(2412*seed+rank*112);
  std::uniform_real_distribution<real_t> dist(0.0,1.0);
  GaugeFieldType gauge_field = GaugeFieldType(dims,local_dims,comm_dims,comm.prev_rank,comm.next_rank);
  gauge_field.set_random(0.5,RNG(seed*2+rank*214));
  AdjointFieldType adjoint_field = AdjointFieldType(local_dims);
  HamiltonianFieldType hamiltonian_field = HamiltonianFieldType(gauge_field,adjoint_field);
  klft::HMC_Params params(n_steps,tau);
  klft::HMC<real_t,Group,Adjoint,RNG,Ndim,Nc> hmc(params,rng,dist,mt);
  hmc.add_kinetic_monomial(0);
  hmc.add_gauge_monomial(beta,0);
  hmc.add_hamiltonian_field(hamiltonian_field);
  hmc.set_integrator(klft::LEAPFROG);
  hamiltonian_field.gauge_field.update_halo_plus(comm);
  Kokkos::fence();
  hamiltonian_field.gauge_field.update_halo_minus(comm);
  Kokkos::fence();
  real_t plaq = hamiltonian_field.gauge_field.get_plaquette();
  // real_t plaq_g = 0.0;
  // MPI_Allreduce(&plaq_l,&plaq_g,1,MPI_DOUBLE,MPI_SUM,comm);
  if(rank == 0) {
    std::cout << "Starting Plaquette: " << plaq << std::endl;
    std::cout << "Starting HMC: " << std::endl;
  }
  bool accept;
  size_t n_accept = 0;
  auto hmc_start_time = std::chrono::high_resolution_clock::now();
  for(size_t i = 0; i < n_traj; i++) {
    auto start_time = std::chrono::high_resolution_clock::now();
    accept = hmc.hmc_step(comm);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> traj_time = end_time - start_time;
    if(accept) n_accept++;
    plaq = hamiltonian_field.gauge_field.get_plaquette();
    // MPI_Allreduce(&plaq_l,&plaq_g,1,MPI_DOUBLE,MPI_SUM,comm);
    if(rank == 0) {
      std::cout << "Traj: " << i << " Accept: " << accept << " Plaquette: " << plaq << " Time: " << traj_time.count() << " Acceptance Rate: " << real_t(n_accept)/real_t(i+1) << std::endl;
      if(outfilename != "") {
        outfile << i << ", " << accept << ", " << plaq << ", " << traj_time.count() << ", " << real_t(n_accept)/real_t(i+1) << std::endl;
      }
    }    
  }
  auto hmc_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> hmc_time = hmc_end_time - hmc_start_time;
  if(rank == 0) {
    std::cout << "HMC Time: " << hmc_time.count() << std::endl;
    outfile.close();
  }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
#endif
}