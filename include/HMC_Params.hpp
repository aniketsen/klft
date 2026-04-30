
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

// this file defines all Paramets structs used in the HMC
#pragma once
#include "GLOBAL.hpp"

namespace klft {

struct HMCParams {
  // general parameters
  index_t Ndims;  // number of dimensions of the simulated system
  index_t L0;     // length of the first dimension
  index_t L1;     // length of the second dimension
  index_t L2;     // length of the third dimension
  index_t L3;     // length of the fourth dimension
  // hard cutoff at 4 for now, since we have not
  // implemented any 5D cases yet
  // more dimensions can and will be added when necessary
  real_t rngDelta;  // integration "time" for the hmc
  index_t seed;     // seed for the random number generator
  bool coldStart;

  // parameters specific to the GaugeField
  size_t Nd;  // number of mu degrees of freedom
  size_t Nc;  // number of color degrees of freedom
  std::string loadfile;
  // add more parameters above this line as needed
  // ...

  // default parameters
  // in practice, user should set these
  // parameters after constructing the object
  HMCParams() {
    Ndims = 4;
    L0 = 4;
    L1 = 4;
    L2 = 4;
    L3 = 4;
    seed = 1234;
    rngDelta = 1.0;
    coldStart = false;

    Nd = 4;
    Nc = 2;

    // set default values to newly added parameters
    // above this line
    // ...
  }

  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("HMC Parameters:\n");
      printf("General Parameters:\n");
      printf("Ndims: %d\n", Ndims);
      printf("L0: %d\n", L0);
      printf("L1: %d\n", L1);
      printf("L2: %d\n", L2);
      printf("L3: %d\n", L3);

      printf("coldStart: %d", coldStart);
      printf("rngDelta: %.3f\n", rngDelta);
      printf("seed: %d\n", seed);

      printf("GaugeField Parameters:\n");
      printf("Nd: %zu\n", Nd);
      printf("Nc: %zu\n", Nc);
    }
  }
  void fprint(std::FILE* stream) {
    std::fprintf(stream, "HMC Parameters:\n");
    std::fprintf(stream, "General Parameters:\n");
    std::fprintf(stream, "Ndims: %d\n", Ndims);
    std::fprintf(stream, "L0: %d\n", L0);
    std::fprintf(stream, "L1: %d\n", L1);
    std::fprintf(stream, "L2: %d\n", L2);
    std::fprintf(stream, "L3: %d\n", L3);

    std::fprintf(stream, "coldStart: %d", coldStart);
    std::fprintf(stream, "rngDelta: %.3f\n", rngDelta);
    std::fprintf(stream, "seed: %d\n", seed);

    std::fprintf(stream, "GaugeField Parameters:\n");
    std::fprintf(stream, "Nd: %zu\n", Nd);
    std::fprintf(stream, "Nc: %zu\n", Nc);
  }
};

struct GaugeMonomial_Params {
  real_t beta;
  index_t level;  // level of integration
  // GaugeMonomial_Params(real_t _beta = 1.0) : beta(_beta) {}
  GaugeMonomial_Params() = default;
  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("Gauge Monomial Parameters:\n");
      printf("  level: %d\n", level);
      printf("  beta: %.10f\n", beta);
    }
  }
  std::string to_string() const {
    return "GaugeMonomial_Params{beta: " + std::to_string(beta) +
           ", level: " + std::to_string(level) + "}";
  }
};

struct FermionMonomial_Params {
  index_t level;             // level of integration
  std::string fermion_type;  // type of fermion, e.g. Wilson, Staggered
  std::string Solver;
  size_t RepDim;
  real_t kappa;
  real_t tol_accept;
  real_t tol_MD;
  bool preconditioning;
  // FermionMonomial_Params(const std::string& _fermion_type = "HWilson",
  //                        const std::string& _Solver = "CG", size_t _RepDim =
  //                        4, real_t _kappa = 0.1, real_t _tol = 1e-6)
  //     : fermion_type(_fermion_type),
  //       Solver(_Solver),
  //       RepDim(_RepDim),
  //       kappa(_kappa),
  //       tol(_tol) {}
  FermionMonomial_Params() = default;
  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("Fermion Parameters:\n");
      printf("  Level: %d\n", level);
      printf("  Fermion Type: %s\n", fermion_type.c_str());
      printf("  Solver: %s\n", Solver.c_str());
      printf("  RepDim: %zu\n", RepDim);
      printf("  Kappa: %.20f\n", kappa);
      printf("  Tolerance (MD/Accept): %.20f/%.20f\n", tol_MD, tol_accept);
      printf("  Preconditioning: %i\n", preconditioning);
    }
  }
};

struct Hasenbusch_Params {
  index_t level;             // level of integration
  std::string fermion_type;  // type of fermion, e.g. Wilson, Staggered
  std::string Solver;
  size_t RepDim;
  real_t kappa;
  real_t massShift;
  real_t tol_accept;
  real_t tol_MD;
  bool preconditioning;
  // FermionMonomial_Params(const std::string& _fermion_type = "HWilson",
  //                        const std::string& _Solver = "CG", size_t _RepDim =
  //                        4, real_t _kappa = 0.1, real_t _tol = 1e-6)
  //     : fermion_type(_fermion_type),
  //       Solver(_Solver),
  //       RepDim(_RepDim),
  //       kappa(_kappa),
  //       tol(_tol) {}
  Hasenbusch_Params() = default;
  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("Hasenbusch Parameters HB:\n");
      printf("  Level: %d\n", level);
      printf("  Fermion Type: %s\n", fermion_type.c_str());
      printf("  Solver: %s\n", Solver.c_str());
      printf("  RepDim: %zu\n", RepDim);
      printf("  Kappa: %.20f\n", kappa);
      printf("  Mass Shift: %.20f\n", massShift);
      printf("  Tolerance (MD/Accept): %.20f/%.20f\n", tol_MD, tol_accept);

      printf("  Preconditioning: %i\n", preconditioning);
    }
  }
};
struct Integrator_Monomial_Params {
  // defines Kind of monomial, i.e. gauge, fermions
  std::string
      type;       // defines the type of integrator to use for now only Leapfrog
  index_t level;  // level of integration
  index_t steps;  // num of steps
  real_t lambda;
  // Integrator_Monomial_Params(const std::string& _Kind,
  //                            const std::string& _type = "Leapfrog",
  //                            index_t _level = 0, index_t _steps = 20)
  //     : Kind(_Kind), type(_type), level(_level), steps(_steps) {}
  Integrator_Monomial_Params() = default;
};

struct Integrator_Params {
  real_t tau;      // integration "time" for the hmc
  index_t nsteps;  // number of hmc steps
  std::vector<Integrator_Monomial_Params> monomials;

  // Integrator_Params(real_t _tau = 1.0, index_t _nsteps = 10,
  //                   std::vector<Integrator_Monomial_Params> _monomials,
  //                   Integrator_Monomial_Params _fermionMonomial)
  //     : tau(_tau), nsteps(_nsteps), monomials(_monomials) {}
  Integrator_Params() = default;
  void print() const {
    if (KLFT_VERBOSITY > 0) {
      printf("Integrator Parameters:\n");
      printf("  tau: %.3f\n", tau);
      printf("  nsteps: %d\n", nsteps);
      for (auto& monomial : monomials) {
        printf("  Monomial:\n");
        printf("    Type: %s\n", monomial.type.c_str());
        printf("    Level: %d\n", monomial.level);
        printf("    Steps: %d\n", monomial.steps);
      }
    }
  }
};
}  // namespace klft
