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

// this file defines input parser for running different simulations
// we are going to overload the the fuction parseInputFile for all
// different Param structs

#pragma once
#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "GaugeObservable.hpp"
#include "HMC_Params.hpp"
#include "IOParams.hpp"
#include "Metropolis_Params.hpp"
#include "PTBC.hpp"
#include "SimulationLogging.hpp"

namespace klft {

// get MetropolisParams from input file
inline bool parseInputFile(const std::string& filename,
                           const std::string& output_directory,
                           MetropolisParams& metropolisParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse MetropolisParams
    if (config["MetropolisParams"]) {
      const auto& mp = config["MetropolisParams"];
      // general parameters
      metropolisParams.Ndims = mp["Ndims"].as<index_t>(4);
      metropolisParams.L0 = mp["L0"].as<index_t>(32);
      metropolisParams.L1 = mp["L1"].as<index_t>(32);
      metropolisParams.L2 = mp["L2"].as<index_t>(32);
      metropolisParams.L3 = mp["L3"].as<index_t>(32);
      metropolisParams.nHits = mp["nHits"].as<index_t>(10);
      metropolisParams.nSweep = mp["nSweep"].as<index_t>(1000);
      metropolisParams.seed = mp["seed"].as<index_t>(1234);
      // parameters specific to the GaugeField
      metropolisParams.Nd = mp["Nd"].as<size_t>(4);
      metropolisParams.Nc = mp["Nc"].as<size_t>(2);
      // parameters specific to the Wilson action
      metropolisParams.beta = mp["beta"].as<double>(1.0);
      metropolisParams.delta = mp["delta"].as<double>(0.1);
      // ...
      // add more parameters above this as needed
    } else {
      printf("Error: MetropolisParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception& e) {
    printf("(MetropolisParams) Error parsing input file: %s\n", e.what());
    return false;
  }
}

// get GaugeObservableParams from input file
inline int parseInputFile(const std::string& filename,
                          const std::string& output_directory,
                          GaugeObservableParams& gaugeObservableParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse GaugeObservableParams
    if (config["GaugeObservableParams"]) {
      const auto& gp = config["GaugeObservableParams"];
      gaugeObservableParams.thermalization_steps =
          gp["thermalization_steps"].as<size_t>(0);
      // interval between measurements
      gaugeObservableParams.measurement_interval =
          gp["measurement_interval"].as<size_t>(1);
      // whether to measure the topological charge
      gaugeObservableParams.measure_topological_charge =
          gp["measure_topological_charge"].as<bool>(false);
      // whether to measure the plaquette
      gaugeObservableParams.measure_plaquette =
          gp["measure_plaquette"].as<bool>(false);
      // whether to measure the temporal Wilson loop
      gaugeObservableParams.measure_wilson_loop_temporal =
          gp["measure_wilson_loop_temporal"].as<bool>(false);
      // whether to measure the mu-nu Wilson loop
      gaugeObservableParams.measure_wilson_loop_mu_nu =
          gp["measure_wilson_loop_mu_nu"].as<bool>(false);
      gaugeObservableParams.measure_action_density =
          gp["measure_action_density"].as<bool>(false);
      gaugeObservableParams.measure_sp_max =
          gp["measure_sp_max"].as<bool>(false);

      // pairs of (L,T) for the temporal Wilson loop
      if (gp["W_temp_L_T_pairs"]) {
        for (const auto& pair : gp["W_temp_L_T_pairs"]) {
          gaugeObservableParams.W_temp_L_T_pairs.push_back(
              IndexArray<2>({pair[0].as<index_t>(), pair[1].as<index_t>()}));
        }
      }

      // pairs of (mu,nu) for the mu-nu Wilson loop
      if (gp["W_mu_nu_pairs"]) {
        for (const auto& pair : gp["W_mu_nu_pairs"]) {
          gaugeObservableParams.W_mu_nu_pairs.push_back(
              IndexArray<2>({pair[0].as<index_t>(), pair[1].as<index_t>()}));
        }
      }

      // pairs of (Lmu,Lnu) for the Wilson loop
      if (gp["W_Lmu_Lnu_pairs"]) {
        for (const auto& pair : gp["W_Lmu_Lnu_pairs"]) {
          gaugeObservableParams.W_Lmu_Lnu_pairs.push_back(
              IndexArray<2>({pair[0].as<index_t>(), pair[1].as<index_t>()}));
        }
      }

      // Check whether to perform Wilson Flow
      gaugeObservableParams.do_wilson_flow =
          gp["do_wilson_flow"].as<bool>(false);

      // Parse the WilsonFlowParams sub-node if it exists
      if (gp["WilsonFlowParams"]) {
        const auto& wfp_node = gp["WilsonFlowParams"];

        // Populate the single wilson_flow_params object directly
        gaugeObservableParams.wilson_flow_params.tau =
            wfp_node["tau"].as<real_t>();
        gaugeObservableParams.wilson_flow_params.eps =
            wfp_node["eps"].as<real_t>();
        if (wfp_node["style"]) {
          std::string style_str = wfp_node["style"].as<std::string>();
          if (style_str == "RK3") {
            gaugeObservableParams.wilson_flow_params.style =
                WilsonFlowStyle::RK3;
          } else if (style_str == "RK4") {
            gaugeObservableParams.wilson_flow_params.style =
                WilsonFlowStyle::RK4;
          } else if (style_str == "Adaptive") {
            gaugeObservableParams.wilson_flow_params.style =
                WilsonFlowStyle::Adaptive;
          } else if (style_str == "Dynamic") {
            gaugeObservableParams.wilson_flow_params.style =
                WilsonFlowStyle::Dynamic;

          } else {
            printf(
                "Warning: Unknown Wilson flow style '%s', defaulting to RK3\n",
                style_str.c_str());
            gaugeObservableParams.wilson_flow_params.style =
                WilsonFlowStyle::RK3;
          }
        } else {
          gaugeObservableParams.wilson_flow_params.style =
              WilsonFlowStyle::RK3;  // default
        }
        if (wfp_node["Adaptive"]) {
          const auto& adapt_node = wfp_node["Adaptive"];
          gaugeObservableParams.wilson_flow_params.adaptiveParams.rho =
              adapt_node["rho"].as<real_t>(0.95);
          gaugeObservableParams.wilson_flow_params.adaptiveParams.abs_tol =
              adapt_node["abs_tol"].as<real_t>(1e-3);
          gaugeObservableParams.wilson_flow_params.adaptiveParams.rel_tol =
              adapt_node["rel_tol"].as<real_t>(1e-1);
          gaugeObservableParams.wilson_flow_params.adaptiveParams.max_increase =
              adapt_node["max_increase"].as<real_t>(1.1);
          gaugeObservableParams.wilson_flow_params.adaptiveParams.max_decrease =
              adapt_node["max_decrease"].as<real_t>(0.6);
        }
        if (wfp_node["Dynamic"]) {
          const auto& dyn_node = wfp_node["Dynamic"];
          gaugeObservableParams.wilson_flow_params.dynamicParams.min_flow_time =
              dyn_node["min_flow_time"].as<real_t>(-1.0);
          gaugeObservableParams.wilson_flow_params.dynamicParams.max_flow_time =
              dyn_node["max_flow_time"].as<real_t>(-1.0);
          gaugeObservableParams.wilson_flow_params.dynamicParams.sp_max_target =
              dyn_node["sp_max_target"].as<real_t>(0.067);
          gaugeObservableParams.wilson_flow_params.dynamicParams
              .t_sqrd_E_target = dyn_node["t_sqrd_E_target"].as<real_t>(0.01);
          gaugeObservableParams.wilson_flow_params.dynamicParams
              .first_tE_measure_step =
              dyn_node["first_tE_measure_step"].as<size_t>(10);
          gaugeObservableParams.wilson_flow_params.dynamicParams.log_details =
              dyn_node["log_details"].as<bool>(false);
          gaugeObservableParams.wilson_flow_params.dynamicParams
              .wilson_flow_filename =
              dyn_node["wilson_flow_filename"].as<std::string>("");
        }

        // Recalculate eps based on parsed values
        if (gaugeObservableParams.wilson_flow_params.eps > 0) {
          gaugeObservableParams.wilson_flow_params.n_steps =
              static_cast<int>(gaugeObservableParams.wilson_flow_params.tau /
                               gaugeObservableParams.wilson_flow_params.eps);
        } else {
          gaugeObservableParams.wilson_flow_params.n_steps =
              0;  // Avoid division by zero
        }
      }
      // filenames for the measurements
      gaugeObservableParams.topological_charge_filename =
          output_directory +
          gp["topological_charge_filename"].as<std::string>("");
      gaugeObservableParams.plaquette_filename =
          output_directory + gp["plaquette_filename"].as<std::string>("");
      gaugeObservableParams.W_temp_filename =
          output_directory + gp["W_temp_filename"].as<std::string>("");
      gaugeObservableParams.W_mu_nu_filename =
          output_directory + gp["W_mu_nu_filename"].as<std::string>("");
      gaugeObservableParams.action_density_filename =
          output_directory + gp["action_density_filename"].as<std::string>("");
      gaugeObservableParams.sp_max_filename =
          output_directory + gp["sp_max_filename"].as<std::string>("");

      // whether to write to file
      gaugeObservableParams.write_to_file = gp["write_to_file"].as<bool>(false);

      // flush interval
      gaugeObservableParams.flush = gp["flush"].as<size_t>(25);
      // ...
      // add more parameters above this line as needed
    } else {
      printf("Error: GaugeObservableParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception& e) {
    printf("(GaugeObservableParams) Error parsing input file: %s\n", e.what());
    return false;
  }
}

inline int parseInputFile(const std::string& filename,
                          const std::string& output_directory,
                          FermionObservableParams& fobs) {
  try {
    YAML::Node config = YAML::LoadFile(filename);
    if (config["FermionObservableParams"]) {
      const auto& mp = config["FermionObservableParams"];
      fobs.measurement_interval = mp["measurement_interval"].as<size_t>(0);

      fobs.measure_pion_correlator =
          mp["measure_pion_correlator"].as<bool>(false);
      fobs.pion_correlator_filename =
          output_directory + mp["pion_correlator_filename"].as<std::string>("");
      fobs.tol = mp["tol"].as<real_t>(10e-8);
      fobs.flush = mp["flush"].as<size_t>(25);
      fobs.write_to_file = mp["write_to_file"].as<bool>(false);
      fobs.kappa = mp["kappa"].as<real_t>(0.15);
      fobs.RepDim = mp["RepDim"].as<size_t>(4);
      fobs.write_to_file = mp["write_to_file"].as<bool>(false);
      fobs.preconditioning = mp["preconditioning"].as<bool>(false);
      fobs.n_sources = mp["n_sources"].as<index_t>(12);
    } else {
      printf("Info: No Fermionic Measurments are done!\n");
      return false;
    }

  } catch (const YAML::Exception& e) {
    printf("(FermionObservableParams) Error parsing input file: %s\n",
           e.what());
    return false;
  }
  return true;
}

// get PTBCParams from input file
inline bool parseInputFile(const std::string& filename,
                           const std::string& output_directory,
                           PTBCParams& ptbcParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse PTBCParams
    if (config["PTBCParams"]) {
      const auto& pp = config["PTBCParams"];
      ptbcParams.defect_length = pp["defect_length"].as<index_t>(1);
      if (pp["defect_values"]) {
        auto defect_node = pp["defect_values"];
        // Set n_sims based on the number of values
        ptbcParams.n_sims = defect_node.size();
        // Resize and read values into std::vector
        ptbcParams.defects.resize(ptbcParams.n_sims);
        for (size_t i = 0; i < defect_node.size(); ++i) {
          ptbcParams.defects[i] = defect_node[i].as<real_t>();
        }
      } else {
        // Default: single defect value
        ptbcParams.n_sims = 1;
        ptbcParams.defects = {1.0};  // will this error without resizing?
      }  // ...
    } else {
      printf("Error: PTBCParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception& e) {
    printf("(PTBCParams) Error parsing input file: %s\n", e.what());
    return false;
  }
}

// get HMCParams from input file
inline bool parseInputFile(const std::string& filename,
                           const std::string& output_directory,
                           HMCParams& hmcParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse HMCParams
    if (config["HMCParams"]) {
      const auto& mp = config["HMCParams"];
      // general parameters
      hmcParams.Ndims = mp["Ndims"].as<index_t>(4);
      hmcParams.L0 = mp["L0"].as<index_t>(32);
      hmcParams.L1 = mp["L1"].as<index_t>(32);
      hmcParams.L2 = mp["L2"].as<index_t>(32);
      hmcParams.L3 = mp["L3"].as<index_t>(32);
      hmcParams.seed = mp["seed"].as<long>(1234);
      hmcParams.rngDelta = mp["rngDelta"].as<double>(1.0);
      hmcParams.coldStart = mp["coldStart"].as<bool>(false);
      // parameters specific to the GaugeField
      hmcParams.Nd = mp["Nd"].as<size_t>(4);
      hmcParams.Nc = mp["Nc"].as<size_t>(2);
      hmcParams.loadfile = mp["loadfile"].as<std::string>("");
      // add more parameters above this as needed
    } else {
      printf("Error: HMCParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception& e) {
    printf("(HMC Params) Error parsing input file: %s\n", e.what());
    return false;
  }
}

inline int parseInputFile(const std::string& filename,
                          const std::string& output_directory,
                          GaugeMonomial_Params& gmparams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse HMCParams
    if (config["Gauge Monomial"]) {
      const auto& mp = config["Gauge Monomial"];
      gmparams.level = mp["level"].as<index_t>(1);
      gmparams.beta = mp["beta"].as<real_t>(1.0);
    } else {
      printf(
          "Error: You have to provide a  Gauge Monomial  in the input file\n");
      // TODO: Change back
      // return false;
    }
  } catch (const YAML::Exception& e) {
    printf("(Gauge Monomial) Error parsing input file: %s\n", e.what());
    return false;
  }
  return true;
}

inline int parseInputFile(const std::string& filename,
                          const std::string& output_directory,
                          FermionMonomial_Params& fermionParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse FermionParams
    if (config["Fermion Monomial"]) {
      const auto& fp = config["Fermion Monomial"];
      fermionParams.level = fp["level"].as<index_t>(0);
      fermionParams.fermion_type = fp["fermion"].as<std::string>("HWilson");
      fermionParams.Solver = fp["solver"].as<std::string>("CG");
      fermionParams.RepDim = fp["RepDim"].as<size_t>(4);
      fermionParams.kappa = fp["kappa"].as<real_t>(0.1);
      fermionParams.preconditioning = fp["preconditioning"].as<bool>(false);
      if (fp["tol"]) {
        fermionParams.tol_accept = fp["tol"].as<real_t>(1e-8);
        fermionParams.tol_MD = fp["tol"].as<real_t>(1e-8);

      } else {
        fermionParams.tol_accept = fp["tol_accept"].as<real_t>(1e-8);
        fermionParams.tol_MD = fp["tol_MD"].as<real_t>(1e-10);
      }
    } else {
      // No Fermions
      return -1;
    }
    return true;
  } catch (const YAML::Exception& e) {
    printf("(Fermion Params) Error parsing input file: %s\n", e.what());
    return false;
  }
}

inline int parseInputFile(const std::string& filename,
                          const std::string& output_directory,
                          Hasenbusch_Params& fermionParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);
    if (config["Fermion Monomial"]) {
      const auto& fp = config["Fermion Monomial"];
      fermionParams.fermion_type = fp["fermion"].as<std::string>("HWilson");
      fermionParams.RepDim = fp["RepDim"].as<size_t>(4);
      fermionParams.kappa = fp["kappa"].as<real_t>(0.1);
      fermionParams.preconditioning = fp["preconditioning"].as<bool>(true);
    }
    // Parse FermionParams
    if (config["Hasenbusch Monomial"]) {
      const auto& fp = config["Hasenbusch Monomial"];
      fermionParams.Solver = fp["solver"].as<std::string>("CG");
      fermionParams.level = fp["level"].as<index_t>(-1);
      fermionParams.massShift = fp["massShift"].as<real_t>(0.1);
      if (fp["tol"]) {
        fermionParams.tol_accept = fp["tol"].as<real_t>(1e-8);
        fermionParams.tol_MD = fp["tol"].as<real_t>(1e-8);

      } else {
        fermionParams.tol_accept = fp["tol_accept"].as<real_t>(1e-8);
        fermionParams.tol_MD = fp["tol_MD"].as<real_t>(1e-10);
      }
    } else {
      fermionParams.level = -1;
    }
    return true;
  } catch (const YAML::Exception& e) {
    printf("(Hasenbusch Params) Error parsing input file: %s\n", e.what());
    return false;
  }
}

// get SimulationLoggingParams from input file
inline int parseInputFile(const std::string& filename,
                          const std::string& output_directory,
                          SimulationLoggingParams& simParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse MetropolisParams
    if (config["SimulationLoggingParams"]) {
      const auto& mp = config["SimulationLoggingParams"];
      // general parameters
      simParams.log_interval = mp["log_interval"].as<size_t>(0);
      simParams.log_filename =
          output_directory + mp["log_filename"].as<std::string>("");
      simParams.write_to_file = mp["write_to_file"].as<bool>(false);
      simParams.flush = mp["flush"].as<size_t>(25);

      simParams.log_delta_H = mp["log_delta_H"].as<bool>(false);
      simParams.log_acceptance = mp["log_acceptance"].as<bool>(false);
      simParams.log_accept = mp["log_accept"].as<bool>(false);
      simParams.log_time = mp["log_time"].as<bool>(false);
      simParams.log_observable_time = mp["log_observable_time"].as<bool>(false);
    } else {
      printf("Error: SimulationLoggingParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception& e) {
    printf("(SimulationLoggingParams) Error parsing input file: %s\n",
           e.what());
    return false;
  }
}

inline int parseInputFile(const std::string& filename,
                          const std::string& output_directory,
                          Integrator_Params& intParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse Integrator_Params
    if (config["Integrator"]) {
      const auto& ip = config["Integrator"];

      intParams.tau = ip["tau"] ? ip["tau"].as<real_t>() : 0.01;
      intParams.nsteps = ip["nSteps"] ? ip["nSteps"].as<index_t>() : 10;

      if (ip["Monomials"]) {
        const auto& monomials = ip["Monomials"];
        if (!monomials.IsSequence()) {
          printf("Error: 'Monomials' must be a YAML sequence\n");
          return false;
        }

        for (const auto& mon : monomials) {
          auto mono = mon.as<Integrator_Monomial_Params>();
          intParams.monomials.push_back(mono);
        }
        // Sort monomials by level (descending)
        std::sort(intParams.monomials.begin(), intParams.monomials.end(),
                  [](const Integrator_Monomial_Params& a,
                     const Integrator_Monomial_Params& b) {
                    return a.level < b.level;
                  });
      } else {
        printf("Warning: No monomials found\n");
      }

    } else {
      printf("Error: Integrator parameters not found in input file\n");
      return false;
    }

    return true;
  } catch (const YAML::Exception& e) {
    printf("YAML Error:  %s\n", e.what());
    return false;
  }
}

// get SimulationLoggingParams from input file
inline int parseInputFile(const std::string& filename,
                          const std::string& output_directory,
                          PTBCSimulationLoggingParams& simParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse MetropolisParams
    if (config["PTBCSimulationLoggingParams"]) {
      const auto& mp = config["PTBCSimulationLoggingParams"];
      // general parameters
      simParams.log_interval = mp["log_interval"].as<size_t>(0);
      simParams.log_filename =
          output_directory + mp["log_filename"].as<std::string>("");
      simParams.write_to_file = mp["write_to_file"].as<bool>(false);
      simParams.flush = mp["flush"].as<size_t>(25);

      simParams.log_swap_accepts = mp["log_swap_accepts"].as<bool>(false);
      simParams.log_delta_H_swap = mp["log_delta_H_swap"].as<bool>(false);
      simParams.log_defects = mp["log_defects"].as<bool>(false);
    } else {
      printf("Error: PTBCSimulationLoggingParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception& e) {
    printf("(PTBCSimulationLoggingParams) Error parsing input file: %s\n",
           e.what());
    return false;
  }
}

inline int parseSanityChecks(const Integrator_Params& iparams,
                             const GaugeMonomial_Params& gmparams,
                             const FermionMonomial_Params& fparams,
                             const int resParsef) {
  // Check if the integrator has at least one monomial
  if (iparams.monomials.empty()) {
    printf("Error: Integrator must have at least one monomial\n");
    return false;
  }

  // Check if the gauge monomial parameters are set
  if (gmparams.beta <= 0) {
    printf("Error: Gauge Monomial beta must be positive\n");

    return false;
  }
  printf("resParsef: %d\n", resParsef);
  if (!(resParsef <= 0)) {
    if (fparams.fermion_type.empty()) {
      printf("Error: Fermion Monomial type must be specified\n");
      return false;
    }
    if (!(fparams.fermion_type == "HWilson" ||
          fparams.fermion_type == "Wilson")) {
      printf("Error: Unsupported Fermion Monomial type: %s\n",
             fparams.fermion_type.c_str());
      return false;
    }
    // Check for correct solver
    if (!((fparams.Solver == "CG" || fparams.Solver == "CGMultiP" ||
           fparams.Solver == "BiCGStab" ||
           fparams.Solver == "BiCGStabMultiP") &&
          (fparams.fermion_type == "HWilson" ||
           fparams.fermion_type == "Wilson"))) {
      printf(
          "Error: Unsupported Fermion Monomial solver: %s for Fermion Type: "
          "%s\n",
          fparams.Solver.c_str(), fparams.fermion_type.c_str());
      return false;
    }

    // Check if the fermion monomial parameters are set
    if (fparams.RepDim != 4 && fparams.RepDim != 2) {
      printf("Error: Fermion Monomial RepDim must be 2 or 4\n");
      return false;
    }
    // Check if the fermion monomial parameters are set
    if (fparams.kappa < 0) {
      printf("Error: Fermion Monomial kappa must be positive\n");
      return false;
    }
  }
  return true;
  //
}
inline int parseInputFile(const std::string& filename,
                          const std::string& output_directory,
                          IOParams& ioParams) {
  try {
    YAML::Node config = YAML::LoadFile(filename);

    // Parse MetropolisParams
    if (config["IOParams"]) {
      const auto& mp = config["IOParams"];
      // general parameters
      ioParams.save_gauge_field = mp["save"].as<bool>(0);
      ioParams.overwrite_gauge_field_file = mp["overwrite"].as<bool>(true);
      ioParams.save_gauge_field_interval = mp["interval"].as<size_t>(0);

      ioParams.save_after_trajectory =
          mp["save_after_trajectory"].as<bool>(true);
      size_t pos =
          mp["filename"].as<std::string>("config.dat").find_last_of("/");

      if (pos != std::string::npos) {
        ioParams.output_dir =
            output_directory +
            mp["filename"].as<std::string>("config.dat").substr(0, pos);
        ioParams.gauge_field_filename =
            mp["filename"].as<std::string>("config.dat").substr(pos + 1);
      } else {
        ioParams.output_dir = output_directory;
        ioParams.gauge_field_filename =
            mp["filename"].as<std::string>("config.dat");
      }
      if (!std::filesystem::exists(ioParams.output_dir) &&
          ioParams.save_gauge_field) {
        std::filesystem::create_directories(ioParams.output_dir);
      }
    } else {
      printf("Error: IOParams not found in input file\n");
      return false;
    }
    return true;
  } catch (const YAML::Exception& e) {
    printf("(SimulationLoggingParams) Error parsing input file: %s\n",
           e.what());
    return false;
  }
}

}  // namespace klft

namespace YAML {
template <>
struct convert<klft::Integrator_Monomial_Params> {
  static int decode(const Node& node, klft::Integrator_Monomial_Params& rhs) {
    if (!node["level"]) {
      printf("Monomial missing required field 'level'\n");
      return false;
    }
    rhs.type = node["Type"] ? node["Type"].as<std::string>() : "Leapfrog";
    rhs.level = node["level"] ? node["level"].as<klft::index_t>() : 0;
    rhs.steps = node["steps"] ? node["steps"].as<klft::index_t>() : 20;
    rhs.lambda = node["lambda"].as<klft::real_t>(0.1931833275037836);
    return true;
  }
};
}  // namespace YAML
