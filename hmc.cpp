#include "include/klft.hpp"
#include <iostream>

using real_t = double;

int main(int argc, char **argv) {
  std::string gauge_group = "SU2";
  int ndim = 4;
  int LX = 8;
  int LY = 8;
  int LZ = 8;
  int LT = 16;
  size_t n_traj = 1000;
  size_t n_steps = 50;
  real_t tau = 1.0;
  real_t beta = 2.0;
  size_t seed = 1234;
  std::string outfilename = "";
  for(int i = 1; i < argc; i++) {
    if(std::string(argv[i]) == "--gauge-group") {
      gauge_group = argv[i+1];
    }
    if(std::string(argv[i]) == "--ndim") {
      ndim = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--LX") {
      LX = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--LY") {
      LY = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--LZ") {
      LZ = std::stoi(argv[i+1]);
    }
    if(std::string(argv[i]) == "--LT") {
      LT = std::stoi(argv[i+1]);
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
      std::cout << "--gauge-group SU2 or U1" << std::endl;
      std::cout << "--ndim 2, 3, or 4" << std::endl;
      std::cout << "--LX lattice size in x direction" << std::endl;
      std::cout << "--LY lattice size in y direction" << std::endl;
      std::cout << "--LZ lattice size in z direction" << std::endl;
      std::cout << "--LT lattice size in t direction" << std::endl;
      std::cout << "--n-traj number of trajectories" << std::endl;
      std::cout << "--n-steps number of steps in leapfrog" << std::endl;
      std::cout << "--tau trajectory length" << std::endl;
      std::cout << "--beta inverse coupling constant" << std::endl;
      std::cout << "--seed random number generator seed" << std::endl;
      std::cout << "--outfilename output filename" << std::endl;
      return 0;
    }
  }
#ifdef KLFT_USE_MPI
  std::cerr << "This is not built for MPI" << std::endl;
  return 1;
#else
  if(gauge_group == "SU2" && ndim == 4) klft::HMC_SU2_4D<real_t>(LX,LY,LZ,LT,n_traj,n_steps,tau,beta,seed,outfilename);
  if(gauge_group == "SU2" && ndim == 3) klft::HMC_SU2_3D<real_t>(LX,LY,LT,n_traj,n_steps,tau,beta,seed,outfilename);
  if(gauge_group == "SU2" && ndim == 2) klft::HMC_SU2_2D<real_t>(LX,LT,n_traj,n_steps,tau,beta,seed,outfilename);
  if(gauge_group == "U1" && ndim == 4) klft::HMC_U1_4D<real_t>(LX,LY,LZ,LT,n_traj,n_steps,tau,beta,seed,outfilename);
  if(gauge_group == "U1" && ndim == 3) klft::HMC_U1_3D<real_t>(LX,LY,LT,n_traj,n_steps,tau,beta,seed,outfilename);
  if(gauge_group == "U1" && ndim == 2) klft::HMC_U1_2D<real_t>(LX,LT,n_traj,n_steps,tau,beta,seed,outfilename);
  return 0;
#endif
}