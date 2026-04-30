# klft

A library for lattice field theory simulation accelerated using Kokkos

# Installation

use git to clone the repository

```bash
git clone https://github.com/aniketsen/klft.git /path/to/klft
cd /path/to/klft
```

setup `kokkos`, `KTune` and `yaml-cpp` 

```bash
git submodule update --init --recursive
```

build the library

```bash
mkdir /path/to/build
cd /path/to/build

cmake [Kokkos options] /path/to/klft

make -j<number of threads>
```

### Kokkos options

The most important Kokkos options are:

`-DKokkos_ENABLE_CUDA=ON` to enable CUDA support

`-DKokkos_ENABLE_OPENMP=ON` to enable OpenMP support

`-DKokkos_ARCH_<arch>=ON` to enable a specific architecture (e.g. `-DKokkos_ARCH_AMPERE80=ON` for NVIDIA A100 gpus)

see the [Kokkos documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#cmake-keywords) for more options



# Usage

## Metropolis

```bash
binaries/metropolis
  -f <file_name> --filename <file_name>
    Name of the input file.
    Default: input.yaml
  -o <file_name> --output <file_name>
    Path to the output folder.
    Hint: if the folder does not exist, it will be created.
    Default: .   
  -h, --help
     Prints this message.
     Hint: use --kokkos-help to see command line options provided by Kokkos.
```

### Example input.yaml for Metropolis

```yaml
# input.yaml
MetropolisParams:    # parameters for the Metropolis algorithm
  Ndims: 4    # number of dimensions [2,3,4]
  Nd: 4       # number of link dimensions (this must be strictly same as Ndims)
  Nc: 1       # number of colors (defines SU(Nc)) [1,2,3]
  L0: 8       # lattice extent in 0 direction
  L1: 8       # lattice extent in 1 direction
  L2: 8       # lattice extent in 2 direction
  L3: 8       # lattice extent in 3 direction
  nHits: 10       # number of hits per sweep
  nSweep: 1000      # number of sweeps
  seed: 32091     # random seed
  beta: 2.0       # inverse coupling constant
  delta: 0.1      # step size for the Metropolis algorithm

GaugeObservableParams:
  measurement_interval: 10               # interval for measurements
  measure_plaquette: true                # measure the plaquette
  measure_wilson_loop_temporal: true    # measure the temporal Wilson loop
  measure_wilson_loop_mu_nu: true       # measure the spatial Wilson loop
  W_temp_L_T_pairs:      # pairs of (L, T) values for the temporal Wilson loop
    - [2, 2]
    - [3, 4]             # keep a non-decreasing order (as much as possible)
    - [4, 3]             # for maximum efficiency
    - [4, 4]
  W_mu_nu_pairs:      # pairs of (mu, nu) values for the planar Wilson loop
    - [0, 1]
    - [1, 2]
    - [3, 2]
  W_Lmu_Lnu_pairs:      # pairs of (Lmu, Lnu) values for the lengths of the 
    - [2, 2]            # planar Wilson loop in the mu and nu directions
    - [3, 3]            # again, keep a non-decreasing order (as much as possible)
    - [4, 3]
  plaquette_filename: "plaquette.out"  # filename to output the plaquette
  W_temp_filename: "W_temp.out"        # filename to output the temporal Wilson loop
  W_mu_nu_filename: "W_mu_nu.out"      # filename to output the planar Wilson loop
  write_to_file: true                  # write the measurements to file
```
## HMC
Currently supported:
- Pure Gauge
    - 2D / 3D / 4D with U(1) / SU(2) / SU(3)
    - Leapfrog and OMF2 integrator
    - Wilson Flow:
        - RK3, RK4, Adaptive RK4
    - Measurements:
        - Topological Charge
        - Action Density
        - planar / temporal Wilson Loops
- 2 Mass degenerate Wilson Fermions
    - currently only in 4D, with U(1) / SU(2) / SU(3)
    - even-odd preconditioning
    - Mass / Hasenbusch preconditioning ( only together with even-odd prec.)
    - CG Solver and Mixed precision CG via [reliable updates](https://arxiv.org/abs/0911.3191) (currently only FP32 supported for Mixed precision)
    - BiCGStab and mixed precision BiCGStab via [reliable updates](https://arxiv.org/abs/0911.3191) (currently only FP32 supported for Mixed precision), (only together with even-odd prec.)
    - Additional Measurements:
        - Pion Corrector using point sources 
    
    

```bash
binaries/hmc
  -f <file_name> --filename <file_name>
    Name of the input file.
    Default: input.yaml
  -o <file_name> --output <file_name>
    Path to the output folder.
    Hint: if the folder does not exist, it will be created.
    Default: .   
  -h, --help
     Prints this message.
     Hint: use --kokkos-help to see command line options provided by Kokkos.
```
### Example input.yaml for HMC

```yaml 
HMCParams:
  Ndims: 4    # number of dimensions [2,3,4]
  Nd: 4       # number of link dimensions (this must be strictly same as Ndims)
  Nc: 2       # number of colors (defines SU(Nc)) [1,2] (SU(3) is still WiP)
  L0: 8       # lattice extent in 0 direction
  L1: 8       # lattice extent in 1 direction
  L2: 8       # lattice extent in 2 direction
  L3: 8       # lattice extent in 3 direction
  seed: 1234  # random seed
  coldStart: false # start with GaugeFIeld set to 1
  rngDelta: 1 # step size for the Metropolis algorithm
  loadfile: "gaugeconfig/gaugeconfig.txt"

IOParams:
  save: true
  interval: 10
  filename: "gaugeconfig/gaugeconfig.txt" 
  overwrite: false
  save_after_trajectory: true

# .
Integrator: # parameters to configure the Integrator, Level 0 is the innermost level of the Integrator, i.e that is executed most frequently
  tau: 1    # time for md trajectory
  nSteps: 1000 # Number of md trajectory 
  Monomials: # Monomial types 
    - Type: "OMF2"  # Integrator to be used for this Level [Leapfrog,OMF2]
      level: 0          # Level for this Monomial used for matching the specific Monomial (see below)
      steps: 1  # Integration steps for specific Monomial
      lambda: 0.194   # OMF2 specific setting  
    - Type: "OMF2"
      level: 1
      steps: 1
      lambda: 0.2
    - Type: "OMF2"
      level: 2
      steps: 6
      lambda: 0.22


Gauge Monomial: # Monomial for Pure Gauge [Must be used] 
  level: 0      # Level to identifiy it with the Integrator 
  beta: 2.12    # inverse coupling constant

Fermion Monomial: # Monomial for Fermions (2 mass degenerate Flavours) [For now only in 4D]
  level: 1  # Level to identifiy it with the Integrator  
  fermion: "Wilson" # Typ of Fermion(operator) [Wilson]
  solver: "CG" # "Solver for Matrix Inversion" [CG,CGMultiP (, BiCGStab, BiCGStabMultiP)]
  RepDim: 4 # Spinor Representation [currently only 4 supported]
  kappa: 0.15  # hopping parameter
  tol: 1e-10 # use tol for overall Solver precision
  tol_accept: 1e-10 # or specify separate tolerance for accept step and MD
  tol_MD: 1e-10 

Hasenbusch Monomial: # Monomial for Mass preconditioning 
  level: 2
  massShift: 0.19 
  tol: 1e-10 # see Fermion Monomial

GaugeObservableParams:
  thermalization_steps: 5
  measurement_interval: 1
  measure_plaquette: true
  measure_wilson_loop_temporal: false
  measure_wilson_loop_mu_nu: false
  measure_topological_charge: true
  measure_action_density: true
  measure_sp_max: true
  flush: 2
  do_wilson_flow: false
  WilsonFlowParams:
    style: "Adaptive"
    tau: 10.0
    eps: 0.02
    Adaptive:
      rho: 0.95
      abs_tol: 1e-3
      rel_tol: 1e-1
      max_increase: 1.1
      max_decrease: 0.6

  W_temp_L_T_pairs:
    - [2, 3]
    - [3, 4]

  W_mu_nu_pairs:
    - [0, 1]
    - [1, 2]

  W_Lmu_Lnu_pairs:
    - [2, 2]
    - [3, 3]

  plaquette_filename: "plaquette_output.txt"
  W_temp_filename: "wilson_temp_output.txt"
  W_mu_nu_filename: "wilson_mu_nu_output.txt"
  topological_charge_filename: "topological_charge.txt"
  action_density_filename: "action_density.txt"
  sp_max_filename: "sp_max.txt"
  write_to_file: true

FermionObservableParams:
  pion_correlator_filename: "pion_correlator_output.txt"
  measurement_interval: 2
  measure_pion_correlator: false
  tol: 1e-10
  kappa: 0.15
  RepDim: 4
  write_to_file: true
  flush: 5
  preconditioning: true
  n_sources: 12 # number of sources to average over


SimulationLoggingParams:
  log_interval: 1
  log_delta_H: true
  log_acceptance: true
  log_accept: true
  log_time: true
  log_observable_time: true
  flush: 2
  log_filename: "simulation_log.txt"
  write_to_file: true
```

## Parallel Tampering (PTBC)
Additional to normal HMC we also provide an implementation of [Parallel Tampering algorithm](https://arxiv.org/abs/2404.14151).

```bash
binaries/ptbc
  -f <file_name> --filename <file_name>
    Name of the input file.
    Default: input.yaml
  -o <file_name> --output <file_name>
    Path to the output folder.
    Hint: if the folder does not exist, it will be created.
    Default: .   
  -h, --help
     Prints this message.
     Hint: use --kokkos-help to see command line options provided by Kokkos.
```
For OpenMPI use `mpirun`:
```bash
mpirun -n $Number_of_replicas binaries/ptbc $options
``` 
`$Number_of_replicas` is the number of replicas used in the simualtions (see `defect_values` below)
where 
### Example input.yaml for HMC
 for PTBC
 See the  [HMC section](#hmc) for HMC specific setup. Additionally specify the following in the `input.yaml`:

```yaml 
PTBCParams:
  defect_length: 2 # length of the defect in the 3 spacial directions
  defect_values: [0.0, 0.25,0.5,0.75,1] # define the number of defect values implicitly sets the number of needed MPI ranks and replicas.

PTBCSimulationLoggingParams:
  log_interval: 1
  log_delta_H_swap: true
  log_swap_start: true
  log_swap_accepts: true
  log_defects: true
  flush: 2
  log_filename: "ptbcsimulation_log.txt"
  write_to_file: true 
```
> [!NOTE]
> Logging of `swap_Accept` and `delta_H_swap` is relative to `swap_start`. 
 
### Tuning
Tuning is done via [KTune](link_to_ktune). 
To deactivate Tuning, set the enviroment variable  `KTUNE_DISABLE_TUNING` to `1`.
Seek the documentation for more in-depth information.


# Environment variables

### KLFT_VERBOSITY
Set the verbosity level of the library.
- 0: silent
- 1: summarize
- 2: verbose
- &gt;=3: debug
- Default: 0
