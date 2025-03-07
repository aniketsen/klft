#pragma once
#include "GaugeMonomial.hpp"
#include "HamiltonianField.hpp"
#include "GaugeField.hpp"
#include "AdjointField.hpp"
#include "HMC_Params.hpp"

namespace klft {
  
  typedef enum IntegratorType_s {
    LEAPFROG = 0,
    LP_LEAPFROG
  } IntegratorType;

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class Integrator {
  public:
    Integrator() {};
    virtual void integrate(std::vector<std::unique_ptr<Monomial<T,Group,Adjoint,Ndim,Nc>>> &monomials, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h, HMC_Params params
#ifdef KLFT_USE_MPI
      , Comm<Ndim> &comm
#endif
    ) = 0;
  };

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class Leapfrog : public Integrator<T,Group,Adjoint,Ndim,Nc> {
  public:
    Leapfrog() : Integrator<T,Group,Adjoint,Ndim,Nc>::Integrator() {}
    void integrate(std::vector<std::unique_ptr<Monomial<T,Group,Adjoint,Ndim,Nc>>> &monomials, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h, HMC_Params params
#ifdef KLFT_USE_MPI
      , Comm<Ndim> &comm
#endif
    ) override {
      AdjointField<T,Adjoint,Ndim,Nc> deriv(h.gauge_field.local_dims);
      T dtau = params.get_tau()/T(params.get_n_steps());
      // initial half step
      deriv.set_zero();
      for(int i = 0; i < monomials.size(); ++i) {
        if(monomials[i]->get_monomial_type() != KLFT_MONOMIAL_KINETIC) monomials[i]->derivative(deriv,h);
      }
      h.update_momentum(deriv,dtau/2.0);
      // full step for gauge
      h.update_gauge(dtau);
#ifdef KLFT_USE_MPI
      h.gauge_field.update_halo_plus(comm);
      Kokkos::fence();
      h.gauge_field.update_halo_minus(comm);
      Kokkos::fence();
#endif
      // leapfrog steps
      for(size_t i = 0; i < params.get_n_steps(); ++i) {
        deriv.set_zero();
        for(int i = 0; i < monomials.size(); ++i) {
          if(monomials[i]->get_monomial_type() != KLFT_MONOMIAL_KINETIC) monomials[i]->derivative(deriv,h);
        }
        h.update_momentum(deriv,dtau);
        h.update_gauge(dtau);
#ifdef KLFT_USE_MPI
      h.gauge_field.update_halo_plus(comm);
      Kokkos::fence();
      h.gauge_field.update_halo_minus(comm);
      Kokkos::fence();
#endif
      }
      // final half step
      deriv.set_zero();
      for(int i = 0; i < monomials.size(); ++i) {
        if(monomials[i]->get_monomial_type() != KLFT_MONOMIAL_KINETIC) monomials[i]->derivative(deriv,h);
      }
      h.update_momentum(deriv,dtau/2.0);
      h.gauge_field.restoreGauge();
#ifdef KLFT_USE_MPI
      h.gauge_field.update_halo_plus(comm);
      Kokkos::fence();
      h.gauge_field.update_halo_minus(comm);
      Kokkos::fence();
#endif
    }
  };

} // namespace klft

