#pragma once
#include <string>
#include "ActionDensity.hpp"
#include "AdjointSUN.hpp"
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Gauge_Util.hpp"
#include "Kokkos_Complex.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_Macros.hpp"
#include "Kokkos_MathematicalFunctions.hpp"
#include "Kokkos_Printf.hpp"
#include "decl/Kokkos_Declare_OPENMP.hpp"
#include "impl/Kokkos_HostThreadTeam.hpp"

namespace klft {

enum class WilsonFlowStyle { RK3, RK4, Adaptive, Dynamic };

struct WilsonFlowParams_Dynamic {
  real_t min_flow_time;
  real_t max_flow_time;
  real_t sp_max_target;
  real_t t_sqrd_E_target;
  size_t first_tE_measure_step;
  bool log_details;
  std::string wilson_flow_filename;

  std::vector<std::string> log_strings;

  WilsonFlowParams_Dynamic() {
    min_flow_time = -1.0;
    max_flow_time = -1.0;
    sp_max_target = real_t(0.067);
    t_sqrd_E_target = real_t(0.1);  // this is Nc dependent
    first_tE_measure_step = 10;
    log_details =
        false;  // for now this will only do something for the dynamical wflow
    wilson_flow_filename = "";
  }
};

struct WilsonFlowParams_Adaptive {
  real_t rho;
  real_t abs_tol;
  real_t rel_tol;
  real_t max_increase;
  real_t max_decrease;

  WilsonFlowParams_Adaptive() {
    rho = 0.95;
    abs_tol = 1e-3;
    rel_tol = 1e-1;
    max_increase = 1.1;
    max_decrease = 0.6;
  }
};

struct WilsonFlowParams {
  // define parameters for a given Wilson Flow calculation, the wilson flow
  // follows arxiv.org/pdf/1006.4518
  // flow time
  real_t tau;
  // number of Wilson flow steps that should be generated
  index_t n_steps;
  // flow time step size
  real_t eps;
  WilsonFlowStyle style;
  WilsonFlowParams_Adaptive adaptiveParams;
  WilsonFlowParams_Dynamic dynamicParams;

  WilsonFlowParams() {
    // default parameters (eps = 0.01)
    n_steps = 10;
    tau = 1.0;
    eps = real_t(tau / n_steps);
    style = WilsonFlowStyle::RK3;
  }
};

struct WilsonFlowData {
  size_t step;
  real_t flow_time;
  real_t sp_max_init;
  real_t sp_max;
  real_t sp_max_old;
  real_t sp_max_deriv;
  real_t t_sqrd_E;
  real_t t_sqrd_E_old;
  size_t measure_step;
  size_t measure_step_old;

  std::string to_string() {
    return std::to_string(step) + ", " + std::to_string(flow_time) + ", " +
           std::to_string(sp_max_init) + ", " + std::to_string(sp_max) + ", " +
           std::to_string(sp_max_deriv) + ", " + std::to_string(t_sqrd_E_old) +
           ", " + std::to_string(measure_step) + ", " +
           std::to_string(t_sqrd_E);
  }

  std::string header_string() {
    return "# step, flow_step, flow_time, sp_max_init, sp_max, sp_max_deriv, "
           "tsquaredxaction_density(old), next_measure_t^E_step, "
           "tsquaredxaction_density\n";
  }
};

template <typename DGaugeFieldType>
struct WilsonFlow {
  // implement the Wilson flow, for now the field will not be copied, but it
  // will be flown in place -> copying needs to be done before
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind Kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  static_assert(rank == 4);  // The wilson flow is only defined for 4D Fields
  WilsonFlowParams& params;
  WilsonFlowData wfdata;

  struct RK3 {};
  struct RK4 {};

  // get the correct deviceGaugeFieldType
  using GaugeFieldT = typename DGaugeFieldType::type;
  GaugeFieldT field;
  GaugeFieldT field_bak;
  GaugeFieldT tmp_staple;
  SUNAdjField<rank, Nc> tmp_Z;
  SUNAdjField<rank, Nc> tmp_Z_err;
  index_t current_step{0};
  real_t eps{0.01};

  WilsonFlow() = delete;

  WilsonFlow(const GaugeFieldT& _field, WilsonFlowParams& _params)
      : params(_params),
        field(_field.field),
        field_bak(_field.field),
        tmp_staple(_field.field),
        wfdata() {
    const IndexArray<rank> dims = _field.dimensions;
    eps = params.eps;
    Kokkos::realloc(Kokkos::WithoutInitializing, tmp_Z, dims[0], dims[1],
                    dims[2], dims[3]);
    Kokkos::realloc(Kokkos::WithoutInitializing, tmp_Z_err, dims[0], dims[1],
                    dims[2], dims[3]);
    Kokkos::fence();
  }

  void flow_step() {
#pragma unroll
    for (index_t fstep = 0; fstep < 3; ++fstep) {
      this->current_step = fstep;
      stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
      Kokkos::fence();

      KTune::parallel_for(
          "Wilsonflow-flow",
          Policy<rank>(IndexArray<rank>{0, 0, 0, 0}, field.dimensions), *this);
      Kokkos::fence();
    }
  }

  void flow_step_RK4() {
#pragma unroll
    for (index_t fstep = 0; fstep < 6; ++fstep) {
      this->current_step = fstep;
      stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
      Kokkos::fence();

      KTune::parallel_for(
          "Wilsonflow-flow",
          Policy<rank, RK4>(IndexArray<rank>{0, 0, 0, 0}, field.dimensions),
          *this);
      Kokkos::fence();
    }
  }

  // execute the wilson flow
  void flow() {  // todo: check this once by saving a staple field and once by
                 // locally calculating the staple
    switch (params.style) {
      case WilsonFlowStyle::RK3:
        if (KLFT_VERBOSITY > 2) {
          printf("Using RK3 Wilson flow...\n");
        }
        for (int step = 0; step < params.n_steps; ++step) {
          flow_step();
        }
        break;
      case WilsonFlowStyle::RK4:
        if (KLFT_VERBOSITY > 2) {
          printf("Using RK4 Wilson flow...\n");
        }
        for (int step = 0; step < params.n_steps; ++step) {
          flow_step_RK4();
        }
        break;
      case WilsonFlowStyle::Adaptive:
        if (KLFT_VERBOSITY > 2) {
          printf("Using adaptive Wilson flow...\n");
        }
        flow_adaptive();
        break;
      case WilsonFlowStyle::Dynamic:
        if (KLFT_VERBOSITY > 2) {
          printf("Using dynamical Wilson flow...\n");
        }
        flow_dynamical();
        break;
    }
  }

  void flow_adaptive() {
    real_t eps_init = eps;
    real_t flow_time{0.0};

    if (KLFT_VERBOSITY > 2) {
      Kokkos::printf("Starting adaptive Wilson flow\n");
    }

    while (flow_time < params.tau) {
      // Save current field state
      Kokkos::deep_copy(field_bak.field, field.field);
      Kokkos::fence();

      flow_step();
      // Perform RK3 step - result will be in tmp_Z_err
      Kokkos::fence();

      // Restore field and perform RK4 step - result will be in tmp_Z
      Kokkos::deep_copy(field.field, field_bak.field);
      Kokkos::fence();

      flow_step_RK4();
      Kokkos::fence();

      // Now tmp_Z_err contains RK4 result and tmp_Z contains RK3 result
      // Calculate error between the two methods
      real_t eps_opt = params.eps;
      auto aparams = params.adaptiveParams;
      real_t err = 0.0;

      const auto rp = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(
          IndexArray<rank>{0}, field.dimensions);

      // Capture the views explicitly to avoid issues with 'this' in CUDA lambda
      auto Z_rk3 = this->tmp_Z_err;
      auto Z_rk4 = this->tmp_Z;

      KTune::parallel_reduce(
          "WilsonFlow_Error", rp,
          KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                        const index_t i3, real_t& local_err) {
            for (index_t mu = 0; mu < 4; ++mu) {
              SUNAdj<Nc> diff =
                  Z_rk4(i0, i1, i2, i3, mu) - Z_rk3(i0, i1, i2, i3, mu);
              real_t rk3_norm =
                  Kokkos::sqrt(norm2<Nc>(Z_rk3(i0, i1, i2, i3, mu)));
              local_err += Kokkos::pow(
                  Kokkos::sqrt(norm2<Nc>(diff)) /
                      (aparams.abs_tol + aparams.rel_tol * rk3_norm),
                  2);
            }
          },
          err);
      Kokkos::fence();

      // Normalize error
      err /= (this->field.dimensions[0] * this->field.dimensions[1] *
              this->field.dimensions[2] * this->field.dimensions[3] * 4.0);
      err = Kokkos::sqrt(err);

      // Calculate optimal step size (RK4 is 4th order, RK3 is 3rd order)
      eps_opt = eps * Kokkos::pow(1.0 / (err + 1e-10), 1.0 / 4.0) * aparams.rho;

      eps_opt = Kokkos::min(aparams.max_increase * eps,
                            Kokkos::max(aparams.max_decrease * eps, eps_opt));

      if (KLFT_VERBOSITY > 2) {
        Kokkos::printf(
            "Adaptive Flow: err=%e, eps=%e, eps_opt=%e, flow_time=%e\n", err,
            eps, eps_opt, flow_time);
      }

      // Check if error is acceptable
      if (err < 1.0) {
        // Accept step
        flow_time += eps;
      } else {
        // Reject step and restore field
        Kokkos::deep_copy(field.field, field_bak.field);
        Kokkos::fence();
      }

      // Update step size for next iteration
      eps = eps_opt;
      // Ensure we don't overshoot the target flow time
      if ((flow_time + eps) > params.tau) {
        eps = params.tau - flow_time + REAL_T_EPSILON;
      }
    }

    eps = eps_init;
  }

  void flow_dynamical() {
    // dynamically does the wilson flow either until sp_max is below
    // sp_max_target or until t^2E is above t_sqd_E_target.
    bool continue_flow = true;
    auto dparams = params.dynamicParams;
    wfdata.sp_max = 0;
    wfdata.sp_max_old = 0;
    wfdata.step = 0;
    wfdata.measure_step = dparams.first_tE_measure_step;
    wfdata.measure_step_old = 0;
    wfdata.t_sqrd_E = 0;
    wfdata.t_sqrd_E_old = 0;

    wfdata.sp_max_deriv = 0;

    wfdata.sp_max_init = get_spmax<DGaugeFieldType>(this->field);

    while (continue_flow) {
      flow_step();
      wfdata.step++;
      this->params.tau = wfdata.step * params.eps;
      wfdata.flow_time = this->params.tau;

      if (this->params.tau > dparams.min_flow_time) {
        // quick napkin math:
        // each flow_step() call calculates the staple field 3 times and
        // multiplies this to the field(...) each time. A staple_field * field
        // should be a bit less than 2 plaquette calculations in terms of
        // instruction count. The get_spmax call does one sweep over the GPlaq
        // Functor within a parallel reduction.
        // -> a get_spmax call (- the overhead from the reduction vs. the
        // parallel for) should be about(slightly larger than) 1/6'th of a
        // flow_step() call
        wfdata.sp_max = get_spmax<DGaugeFieldType>(this->field);
        if (wfdata.step > 1) {
          wfdata.sp_max_deriv += (wfdata.sp_max - wfdata.sp_max_old);
        } else {
          wfdata.sp_max_deriv += (wfdata.sp_max - wfdata.sp_max_init);
        }
        wfdata.sp_max_old = wfdata.sp_max;

        if (wfdata.step >= wfdata.measure_step) {
          // linearlise the measured wfdata.t_sqrd_E to get a prediction for the
          // end step
          wfdata.measure_step = wfdata.step;
          wfdata.t_sqrd_E =
              getActionDensity_clover<DGaugeFieldType>(this->field) *
              params.tau * params.tau;
          real_t slope = (wfdata.t_sqrd_E - wfdata.t_sqrd_E_old) /
                         (wfdata.step - wfdata.measure_step_old);
          real_t intercept =
              wfdata.t_sqrd_E - slope * static_cast<real_t>(wfdata.step);
          wfdata.measure_step_old = wfdata.measure_step;
          wfdata.measure_step = static_cast<index_t>(
              ((dparams.t_sqrd_E_target - intercept) / slope) + 1);
          wfdata.measure_step =
              wfdata.measure_step > wfdata.measure_step_old + 10
                  ? wfdata.measure_step
                  : wfdata.measure_step_old + 10;
          wfdata.measure_step =
              wfdata.measure_step < wfdata.step + 100
                  ? wfdata.measure_step
                  : wfdata.measure_step_old + 100;  // avoid too large steps
          if (KLFT_VERBOSITY > 1) {
            printf("Wilson Flow prediction: next measurement at step %zu\n",
                   wfdata.measure_step);
            printf("  slope: %1.6f, intercept: %1.6f\n", slope, intercept);
            printf("  current t^2E: %1.6f\n", wfdata.t_sqrd_E);
          }
        }
        if (KLFT_VERBOSITY > 4) {
          printf("Wilson Flow step %zu: wfdata.sp_max = %1.6f, t^2E = %1.6f\n",
                 wfdata.step, wfdata.sp_max, wfdata.t_sqrd_E);
        }
        if ((wfdata.sp_max <= dparams.sp_max_target ||
             wfdata.t_sqrd_E >= dparams.t_sqrd_E_target)) {
          continue_flow = false;
        }
      }

      if (this->params.tau >= dparams.max_flow_time &&
          dparams.max_flow_time > 0.0) {
        continue_flow = false;
      }
    }
    wfdata.sp_max_deriv =
        wfdata.sp_max_deriv / static_cast<real_t>(wfdata.step - 1);
    if (KLFT_VERBOSITY > 1) {
      printf("Wilson Flow completed in %zu steps, total flow time %1.6f\n",
             wfdata.step, wfdata.step * params.eps);
    }
    if (dparams.log_details) {
      wfdata.t_sqrd_E_old = wfdata.t_sqrd_E;
      wfdata.t_sqrd_E = getActionDensity_clover<DGaugeFieldType>(this->field) *
                        params.tau * params.tau;
      dparams.log_strings.push_back(wfdata.to_string());
    }
  }  // namespace klft

  void flow_impr(real_t b1) {
    for (int step = 0; step < params.n_steps; ++step) {
#pragma unroll
      for (index_t fstep = 0; fstep < 3; ++fstep) {
        this->current_step = fstep;
        stapleField<DGaugeFieldType>(this->field, this->tmp_staple, b1);
        Kokkos::fence();

        KTune::parallel_for(
            "Wilsonflow-flow-impr",
            Policy<rank>(IndexArray<rank>{0, 0, 0, 0}, field.dimensions),
            *this);
        Kokkos::fence();
      }
    }
  }

  void flow_DBW2() { flow_impr(-1.4011); }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW1(indexType i0,
                                     indexType i1,
                                     indexType i2,
                                     indexType i3,
                                     index_t mu) const {
    // SUN<Nc> Z0_SUN = (field.field(i0, i1, i2, i3, mu)) *
    //                  (tmp_staple.field(i0, i1, i2, i3, mu));
    SUN<Nc> Z0_SUN = (field.field(i0, i1, i2, i3, mu) *
                      tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z0 = 2.0 * traceT(Z0_SUN);  //* (Nc / params.beta);
    tmp_Z(i0, i1, i2, i3, mu) = Z0;        // does this need to be deep copied?
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z0 * -0.25 * this->eps) * field.field(i0, i1, i2, i3, mu);
    // restoreSUN(field.field(i0, i1, i2, i3, mu));
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW2(indexType i0,
                                     indexType i1,
                                     indexType i2,
                                     indexType i3,
                                     index_t mu) const {
    // SUN<Nc> Z1_SUN = (field.field(i0, i1, i2, i3, mu)) *
    //                  (tmp_staple.field(i0, i1, i2, i3, mu));
    SUN<Nc> Z1_SUN = (field.field(i0, i1, i2, i3, mu) *
                      tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z1 = 2.0 * traceT(Z1_SUN);  // * (Nc / params.beta);
    SUNAdj<Nc> Z0 = tmp_Z(i0, i1, i2, i3, mu);
    Z1 = Z1 * static_cast<real_t>(8.0 / 9.0) -
         Z0 * static_cast<real_t>(17.0 / 36.0);
    tmp_Z(i0, i1, i2, i3, mu) = Z1;
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z1 * -this->eps) * field.field(i0, i1, i2, i3, mu);
    // restoreSUN(field.field(i0, i1, i2, i3, mu));
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepV(indexType i0,
                                    indexType i1,
                                    indexType i2,
                                    indexType i3,
                                    index_t mu) const {
    SUN<Nc> Z2_SUN = (field.field(i0, i1, i2, i3, mu) *
                      tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z2 = 2.0 * traceT(Z2_SUN);  // * (Nc / params.beta);
    SUNAdj<Nc> Z_old = tmp_Z(i0, i1, i2, i3, mu);
    Z2 = (Z2 - Z_old);
    tmp_Z_err(i0, i1, i2, i3, mu) = Z2;
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z2 * -this->eps * 0.75) * field.field(i0, i1, i2, i3, mu);
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void stepW(indexType i0,
                                    indexType i1,
                                    indexType i2,
                                    indexType i3,
                                    index_t mu,
                                    const real_t a,
                                    const real_t b,
                                    const bool first_step,
                                    const bool last_step) const {
    SUN<Nc> Z_SUN = (field.field(i0, i1, i2, i3, mu) *
                     tmp_staple.field(i0, i1, i2, i3, mu));
    SUNAdj<Nc> Z = 2.0 * traceT(Z_SUN);  // * (Nc / params.beta);
    if (!first_step) {
      SUNAdj<Nc> Z_old = tmp_Z(i0, i1, i2, i3, mu);
      Z_old *= a;
      Z = (Z - Z_old);
    }
    if (!last_step) {
      tmp_Z(i0, i1, i2, i3, mu) = Z;
    } else {
      tmp_Z(i0, i1, i2, i3, mu) = Z;
    }
    Z *= -b * this->eps;
    field.field(i0, i1, i2, i3, mu) =
        expoSUN(Z) * field.field(i0, i1, i2, i3, mu);
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void operator()(const indexType i0,
                                         const indexType i1,
                                         const indexType i2,
                                         const indexType i3) const {
#pragma unroll
    for (index_t mu = 0; mu < 4; ++mu) {
      switch (this->current_step) {
        case 0:
          stepW1(i0, i1, i2, i3, mu);
          break;
        case 1:
          stepW2(i0, i1, i2, i3, mu);
          break;
        case 2:
          stepV(i0, i1, i2, i3, mu);
          break;
      }
    }
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void operator()(RK4,
                                         const indexType i0,
                                         const indexType i1,
                                         const indexType i2,
                                         const indexType i3) const {
    // params taken from quda codebase
    real_t a[] = {0.0,
                  0.737101392796,
                  1.634740794341,
                  0.744739003780,
                  1.469897351522,
                  2.813971388035};
    real_t b[] = {0.032918605146, 0.823256998200, 0.381530948900,
                  0.200092213184, 1.718581042715, 0.27};
#pragma unroll
    for (index_t mu = 0; mu < 4; ++mu) {
      switch (this->current_step) {
        case 0:
          stepW(i0, i1, i2, i3, mu, a[0], b[0], true, false);
          break;
        case 1:
          stepW(i0, i1, i2, i3, mu, a[1], b[1], false, false);
          break;
        case 2:
          stepW(i0, i1, i2, i3, mu, a[2], b[2], false, false);
          break;
        case 3:
          stepW(i0, i1, i2, i3, mu, a[3], b[3], false, false);
          break;
        case 4:
          stepW(i0, i1, i2, i3, mu, a[4], b[4], false, false);
          break;
        case 5:
          stepW(i0, i1, i2, i3, mu, a[5], b[5], false, true);
          break;
      }
    }
  }
};

}  // namespace klft
