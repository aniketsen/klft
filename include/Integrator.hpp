#pragma once
#include "HMC_Params.hpp"
#include "Solver.hpp"
#include "UpdateMomentum.hpp"
#include "UpdatePosition.hpp"
#include "updateMomentumFermion.hpp"
#include "updateMomentumFermionEO.hpp"
#include "updateMomentumFermionEOHasenbusch.hpp"
namespace klft {

typedef enum IntegratorType_s { LEAPFROG = 0, LP_LEAPFROG } IntegratorType;

// template <class UpdatePosition, class UpdateMomentum>
class Integrator : public std::enable_shared_from_this<Integrator> {
 public:
  Integrator() = delete;
  virtual ~Integrator() = default;

  Integrator(const size_t n_steps_,
             const bool outermost_,
             std::shared_ptr<Integrator> nested_,
             std::shared_ptr<UpdatePosition> update_q_,
             std::shared_ptr<UpdateMomentum> update_p_)
      : n_steps(n_steps_),
        outermost(outermost_),
        nested(nested_),
        update_q(update_q_),
        update_p(update_p_) {};

  virtual void integrate(const real_t tau, const bool last_step) const = 0;
  virtual void halfstep(const real_t tau) const = 0;

  std::shared_ptr<Integrator> nested;
  std::shared_ptr<UpdatePosition> update_q;
  std::shared_ptr<UpdateMomentum> update_p;
  const size_t n_steps;
  const bool outermost;
};

// template <class UpdatePosition, class UpdateMomentum>
class LeapFrog : public Integrator {  // <UpdatePosition, UpdateMomentum> {
 public:
  LeapFrog() = delete;

  LeapFrog(const size_t n_steps_,
           const bool outermost_,
           std::shared_ptr<Integrator> nested_,
           std::shared_ptr<UpdatePosition> update_q_,
           std::shared_ptr<UpdateMomentum> update_p_)
      : Integrator(n_steps_, outermost_, nested_, update_q_, update_p_) {};

  ~LeapFrog() override = default;

  void halfstep(const real_t tau) const override {
    const real_t eps = tau / n_steps;
    update_p->update(eps * 0.5);
    if (nested)
      nested->halfstep(eps);
  }

  void integrate(const real_t tau, const bool last_step) const override {
    if (outermost)
      halfstep(tau);
    const real_t eps = tau / n_steps;
    for (size_t i = 0; i < n_steps - 1; ++i) {
      if (nested) {
        Kokkos::Profiling::pushRegion("Nested Integrator Hot Loop");
        nested->integrate(eps, false);
        Kokkos::Profiling::popRegion();
      } else {
        update_q->update(eps);
      }
      update_p->update(eps);
    }
    if (nested) {
      Kokkos::Profiling::pushRegion("Nested Integrator");
      if (outermost) {
        nested->integrate(eps, true);
      } else {
        nested->integrate(eps, last_step);
      }
      Kokkos::Profiling::popRegion();
    } else {
      update_q->update(eps);
    }
    if (!last_step && !outermost)
      update_p->update(eps);
    if (outermost)
      halfstep(tau);
  }

};  // class LeapFrog
//

class OMF2 : public Integrator {
 public:
  OMF2() = delete;

  OMF2(const unsigned n_steps,
       const bool outermost,
       std::shared_ptr<Integrator> nested,
       std::shared_ptr<UpdatePosition> update_q,
       std::shared_ptr<UpdateMomentum> update_p,
       const double lambda = 0.1931833275037836)
      : Integrator(n_steps, outermost, nested, update_q, update_p),
        m_lambda(lambda) {};

  ~OMF2() override = default;

  void halfstep(const double tau) const override {
    const double eps = tau / n_steps;
    update_p->update(m_lambda * eps);
    if (nested)
      nested->halfstep(0.5 * eps);
  }

  void integrate(const double tau, const bool last_step) const override {
    if (outermost)
      halfstep(tau);

    const double eps = tau / n_steps;
    const double one_m_2lambda = (1.0 - 2 * m_lambda);

    for (unsigned int i = 0; i < n_steps - 1; ++i) {
      if (nested) {
        nested->integrate(0.5 * eps, false);
        update_p->update(one_m_2lambda * eps);
        nested->integrate(0.5 * eps, false);
        update_p->update(2 * m_lambda * eps);
      } else {
        update_q->update(0.5 * eps);
        update_p->update(one_m_2lambda * eps);
        update_q->update(0.5 * eps);
        update_p->update(2 * m_lambda * eps);
      }
    }

    if (nested) {
      nested->integrate(0.5 * eps, false);
      update_p->update(one_m_2lambda * eps);
      if (outermost) {
        nested->integrate(0.5 * eps, true);
      } else {
        nested->integrate(0.5 * eps, last_step);
      }
    } else {
      update_q->update(0.5 * eps);
      update_p->update(one_m_2lambda * eps);
      update_q->update(0.5 * eps);
    }

    // combine two half-steps on the inner time scales
    // if we're not at the "last" step
    if (!last_step && !outermost)
      update_p->update(2 * m_lambda * eps);

    if (outermost)
      halfstep(tau);
  }

 private:
  double m_lambda;
};

// Still need to add check for different Dirac Operators
template <typename DGaugeFieldType,
          typename DAdjFieldType,
          typename DSpinorFieldType>
std::shared_ptr<Integrator> createIntegrator(
    typename DGaugeFieldType::type& g_in,
    typename DAdjFieldType::type& a_in,
    typename DSpinorFieldType::type& s_in,
    typename DSpinorFieldType::type& s_in_HB,
    const Integrator_Params& integratorParams,
    const GaugeMonomial_Params& gaugeMonomialParams,
    const FermionMonomial_Params& fermionParams,
    const Hasenbusch_Params& fermionHBparams,
    const int& resParsef) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));
  static constexpr SpinorFieldLayout Layout =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Layout;
  constexpr const size_t Nd = rank;
  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;
  // Create the integrator based on the type
  if (integratorParams.monomials.empty()) {
    printf("Error: Integrator must have at least one monomial\n");
    return nullptr;
  }
  // startingpoit of integrator chain
  std::shared_ptr<Integrator> nested_integrator = nullptr;
  for (const auto& monomial : integratorParams.monomials) {
    std::shared_ptr<Integrator> integrator = nullptr;
    if (monomial.level == 0) {
      // if the level is 0, we create a new integrator with nullptr as inner
      // integrator
      if (gaugeMonomialParams.level == 0) {
        UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
        UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType> update_p(
            g_in, a_in, gaugeMonomialParams.beta);
        if (monomial.type == "Leapfrog") {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nullptr, std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              std::make_shared<
                  UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(
                  update_p));
        } else if (monomial.type == "OMF2") {
          integrator = std::make_shared<OMF2>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nullptr, std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              std::make_shared<
                  UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(
                  update_p),
              monomial.lambda);

        } else {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nullptr, std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              std::make_shared<
                  UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(
                  update_p));
        }

      } else if (fermionParams.level == 0 && resParsef > 0) {
        // if the level is 0, we create a new integrator with nullptr as inner
        // integrator
        if (fermionParams.RepDim == 4) {
          auto diracParams = getDiracParams(fermionHBparams);
          UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
          std::shared_ptr<UpdateMomentum> momentum_ptr;
          if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
            // UpdateMomentumWilsonEO<DSpinorFieldType, DGaugeFieldType,
            //                        DAdjFieldType,

            //                        BiCGStab,
            //                        WilsonDiracOperator<DSpinorFieldType,
            //                        DGaugeFieldType>>
            //     update_p(s_in, g_in, a_in, diracParams,
            //     fermionParams.tol_MD);
            if (fermionHBparams.level < 0) {
              if (fermionParams.Solver == "BiCGStab") {
                momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                    DAdjFieldType, BiCGStab,
                    EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                    s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
                /* code */
              } else if (fermionParams.Solver == "CGMultiP") {
                momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                    DAdjFieldType, CGMultiP,
                    EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                    s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
              } else if (fermionParams.Solver == "BiCGStabMultiP") {
                momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                    DAdjFieldType, BiCGStabMultiP,
                    EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                    s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
              } else {
                momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                    DAdjFieldType, CGSolver,
                    EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                    s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
              }

            } else {
              if (fermionParams.Solver == "BiCGStab") {
                momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                    DAdjFieldType, BiCGStab,
                    EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                          true>>>(
                    s_in, g_in, a_in, diracParams,
                    fermionParams.tol_MD);  // in this case there is HB accl.
                                            // used, such that the normal F.mon.
                                            // uses the shifted operator
              } else if (fermionParams.Solver == "CGMultiP") {
                momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                    DAdjFieldType, CGMultiP,
                    EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                          true>>>(s_in, g_in, a_in, diracParams,
                                                  fermionParams.tol_MD);
              } else if (fermionParams.Solver == "BiCGStabMultiP") {
                momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                    DAdjFieldType, BiCGStabMultiP,
                    EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                          true>>>(s_in, g_in, a_in, diracParams,
                                                  fermionParams.tol_MD);
              } else {
                momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                    DAdjFieldType, CGSolver,
                    EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                          true>>>(s_in, g_in, a_in, diracParams,
                                                  fermionParams.tol_MD);
              }
            }
          } else {
            if (fermionParams.Solver == "CGSSolver") {
              momentum_ptr = std::make_shared<UpdateMomentumWilson<
                  DAdjFieldType, CGSolver,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
            } else {
              momentum_ptr = std::make_shared<UpdateMomentumWilson<
                  DAdjFieldType, CGMultiP,
                  WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
            }
          }

          if (monomial.type == "Leapfrog") {
            integrator = std::make_shared<LeapFrog>(
                monomial.steps,
                monomial.level == integratorParams.monomials.back().level,
                nullptr,
                std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
                momentum_ptr);
          } else if (monomial.type == "OMF2") {
            integrator = std::make_shared<OMF2>(
                monomial.steps,
                monomial.level == integratorParams.monomials.back().level,
                nullptr,
                std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
                momentum_ptr, monomial.lambda);

          } else {
            printf(
                "Warning: Integrator %s isn't available, falllback Leapfrog ",
                monomial.type.c_str());
            integrator = std::make_shared<LeapFrog>(
                monomial.steps,
                monomial.level == integratorParams.monomials.back().level,
                nullptr,
                std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
                momentum_ptr);
          }
        } else {
          printf("Error: Fermion RepDim must be 4\n");
          return nullptr;
        }
      } else if (fermionHBparams.level == 0 && resParsef > 0) {
        if (fermionHBparams.RepDim == 4) {
          auto diracParams_light =
              getDiracParams(fermionHBparams);  // these are the light pion ones
          auto diracParams_heavy =
              getDiracParams(fermionParams);  // these are the heavy pion ones

          UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
          std::shared_ptr<UpdateMomentum> momentum_ptr;
          if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
            if (fermionHBparams.Solver == "BiCGStab") {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEOHasenbusch<
                  DAdjFieldType, BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in_HB, g_in, a_in, diracParams_light, fermionParams.tol_MD);
            } else if (fermionHBparams.Solver == "CGMultiP") {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEOHasenbusch<
                  DAdjFieldType, CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in_HB, g_in, a_in, diracParams_light, fermionParams.tol_MD);
            } else if (fermionHBparams.Solver == "BiCGStabMultiP") {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEOHasenbusch<
                  DAdjFieldType, BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in_HB, g_in, a_in, diracParams_light, fermionParams.tol_MD);
            } else {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEOHasenbusch<
                  DAdjFieldType, CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in_HB, g_in, a_in, diracParams_light, fermionParams.tol_MD);
            }

            // UpdateMomentumWilsonEO<DSpinorFieldType, DGaugeFieldType,
            //                        DAdjFieldType,

            //                        BiCGStab, WilsonDiracOperator>
            //     update_p(s_in, g_in, a_in, diracParams,
            //     fermionParams.tol_MD);
          }

          if (monomial.type == "Leapfrog") {
            integrator = std::make_shared<LeapFrog>(
                monomial.steps,
                monomial.level == integratorParams.monomials.back().level,
                nullptr,
                std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
                momentum_ptr);
          } else if (monomial.type == "OMF2") {
            integrator = std::make_shared<OMF2>(
                monomial.steps,
                monomial.level == integratorParams.monomials.back().level,
                nullptr,
                std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
                momentum_ptr, monomial.lambda);
          }
        }
      }
      nested_integrator = integrator;
    } else if (gaugeMonomialParams.level == monomial.level) {
      // if the level is the same, we create a new integrator with the
      // previous one as inner integrator
      UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
      UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType> update_p(
          g_in, a_in, gaugeMonomialParams.beta);
      if (monomial.type == "Leapfrog") {
        integrator = std::make_shared<LeapFrog>(
            monomial.steps,
            monomial.level == integratorParams.monomials.back().level,
            nested_integrator,
            std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
            std::make_shared<
                UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(update_p));

      } else if (monomial.type == "OMF2") {
        integrator = std::make_shared<OMF2>(
            monomial.steps,
            monomial.level == integratorParams.monomials.back().level,
            nested_integrator,
            std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
            std::make_shared<
                UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(update_p),
            monomial.lambda);

      } else {
        integrator = std::make_shared<LeapFrog>(
            monomial.steps,
            monomial.level == integratorParams.monomials.back().level,
            nested_integrator,
            std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
            std::make_shared<
                UpdateMomentumGauge<DGaugeFieldType, DAdjFieldType>>(update_p));
      }
    } else if (fermionParams.level == monomial.level && resParsef > 0) {
      // if the level is 0, we create a new integrator with nullptr as inner
      // integrator
      if (fermionParams.RepDim == 4) {
        auto diracParams = getDiracParams(fermionHBparams);

        UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
        std::shared_ptr<UpdateMomentum> momentum_ptr;
        if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
          // UpdateMomentumWilsonEO<DSpinorFieldType, DGaugeFieldType,
          //                        DAdjFieldType, BiCGStab,
          //                        WilsonDiracOperator>
          //     update_p(s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
          if (fermionHBparams.level < 0) {
            if (fermionParams.Solver == "BiCGStab") {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                  DAdjFieldType, BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
            } else if (fermionParams.Solver == "CGMultiP") {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                  DAdjFieldType, CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                  DAdjFieldType, BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
            } else {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                  DAdjFieldType, CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                  s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
            }

          } else {
            if (fermionParams.Solver == "BiCGStab") {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                  DAdjFieldType, BiCGStab,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>>>(
                  s_in, g_in, a_in, diracParams,
                  fermionParams.tol_MD);  // in this case there is HB accl.
                                          // used, such that the normal F.mon.
                                          // uses the shifted operator
            } else if (fermionParams.Solver == "CGMultiP") {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                  DAdjFieldType, CGMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>>>(
                  s_in, g_in, a_in, diracParams,
                  fermionParams.tol_MD);  // in this case there is HB accl.
                                          // used, such that the normal F.mon.
                                          // uses the shifted operator

            } else if (fermionParams.Solver == "BiCGStabMultiP") {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                  DAdjFieldType, BiCGStabMultiP,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>>>(
                  s_in, g_in, a_in, diracParams,
                  fermionParams.tol_MD);  // in this case there is HB accl.
                                          // used, such that the normal F.mon.
                                          // uses the shifted operator
            } else {
              momentum_ptr = std::make_shared<UpdateMomentumWilsonEO<
                  DAdjFieldType, CGSolver,
                  EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType,
                                        true>>>(
                  s_in, g_in, a_in, diracParams,
                  fermionParams.tol_MD);  // in this case there is HB accl.
                                          // used, such that the normal F.mon.
                                          // uses the shifted operator
            }
          }
        } else {
          if (fermionParams.Solver == "CGSolver") {
            momentum_ptr = std::make_shared<UpdateMomentumWilson<
                DAdjFieldType, CGSolver,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
          } else {
            momentum_ptr = std::make_shared<UpdateMomentumWilson<
                DAdjFieldType, CGMultiP,
                WilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                s_in, g_in, a_in, diracParams, fermionParams.tol_MD);
          }
        }

        if (monomial.type == "Leapfrog") {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nested_integrator,
              std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              momentum_ptr);
        } else if (monomial.type == "OMF2") {
          integrator = std::make_shared<OMF2>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nested_integrator,
              std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              momentum_ptr, monomial.lambda);
        } else {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nested_integrator,
              std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              momentum_ptr);
        }
      } else {
        printf("Error: Fermion RepDim must be 4\n");
        return nullptr;
      }
    } else if (fermionHBparams.level == monomial.level && resParsef > 0) {
      if (fermionParams.RepDim == 4) {
        auto diracParams_light =
            getDiracParams(fermionHBparams);  // these are the light pion ones
        auto diracParams_heavy =
            getDiracParams(fermionParams);  // these are the heavy pion ones

        UpdatePositionGauge<Nd, Nc> update_q(g_in, a_in);
        std::shared_ptr<UpdateMomentum> momentum_ptr;
        if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
          // UpdateMomentumWilsonEO<DSpinorFieldType, DGaugeFieldType,
          //                        DAdjFieldType,

          //                        BiCGStab,
          //                        WilsonDiracOperator<DSpinorFieldType,
          //                        DGaugeFieldType>>
          //     update_p(s_in, g_in, a_in, diracParams,
          //     fermionParams.tol_MD);
          if (fermionHBparams.Solver == "BiCGStab") {
            momentum_ptr = std::make_shared<UpdateMomentumWilsonEOHasenbusch<
                DAdjFieldType, BiCGStab,
                EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType, true>,
                EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                s_in_HB, g_in, a_in, diracParams_light, fermionParams.tol_MD);
          } else if (fermionHBparams.Solver == "CGMultiP") {
            momentum_ptr = std::make_shared<UpdateMomentumWilsonEOHasenbusch<
                DAdjFieldType, CGMultiP,
                EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType, true>,
                EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                s_in_HB, g_in, a_in, diracParams_light, fermionParams.tol_MD);
          } else if (fermionHBparams.Solver == "BiCGStabMultiP") {
            momentum_ptr = std::make_shared<UpdateMomentumWilsonEOHasenbusch<
                DAdjFieldType, BiCGStabMultiP,
                EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType, true>,
                EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                s_in_HB, g_in, a_in, diracParams_light, fermionParams.tol_MD);
          } else {
            momentum_ptr = std::make_shared<UpdateMomentumWilsonEOHasenbusch<
                DAdjFieldType, CGSolver,
                EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType, true>,
                EOWilsonDiracOperator<DSpinorFieldType, DGaugeFieldType>>>(
                s_in_HB, g_in, a_in, diracParams_light, fermionParams.tol_MD);
          }
        }
        if (monomial.type == "Leapfrog") {
          integrator = std::make_shared<LeapFrog>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nested_integrator,
              std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              momentum_ptr);
        } else if (monomial.type == "OMF2") {
          integrator = std::make_shared<OMF2>(
              monomial.steps,
              monomial.level == integratorParams.monomials.back().level,
              nested_integrator,
              std::make_shared<UpdatePositionGauge<Nd, Nc>>(update_q),
              momentum_ptr, monomial.lambda);
        }
      }
    }

    nested_integrator = integrator;
  }

  return nested_integrator;
}

}  // namespace klft