#pragma once
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "Monomial.hpp"
#include "HamiltonianField.hpp"

namespace klft {

  template <typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class GaugeMonomial : public Monomial<T,Group,Adjoint,Ndim,Nc> {
    public:
      T beta;
    
    GaugeMonomial(T _beta, unsigned int _time_scale) : Monomial<T,Group,Adjoint,Ndim,Nc>::Monomial(_time_scale) {
      beta = _beta;
      Monomial<T,Group,Adjoint,Ndim,Nc>::monomial_type = KLFT_MONOMIAL_GAUGE;
    }

    void heatbath(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      Monomial<T,Group,Adjoint,Ndim,Nc>::H_old = -(beta/T(h.gauge_field.get_Nc()))*h.gauge_field.get_plaquette(false);
    }

    void accept(HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      Monomial<T,Group,Adjoint,Ndim,Nc>::H_new = -(beta/T(h.gauge_field.get_Nc()))*h.gauge_field.get_plaquette(false);
    }

    void derivative(AdjointField<T,Adjoint,Ndim,Nc> deriv, HamiltonianField<T,Group,Adjoint,Ndim,Nc> h) override {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({h.gauge_field.get_start_dim(0),h.gauge_field.get_start_dim(1),h.gauge_field.get_start_dim(2),h.gauge_field.get_start_dim(3),0},{h.gauge_field.get_end_dim(0),h.gauge_field.get_end_dim(1),h.gauge_field.get_end_dim(2),h.gauge_field.get_end_dim(3),h.gauge_field.get_Ndim()});
      Kokkos::parallel_for("derivative", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &mu) {
        Kokkos::Array<int,4> site = {x,y,z,t};
#ifdef KLFT_USE_MPI
        for(int i = 0; i < Ndim; i++) {
          if(h.gauge_field.comm_dims[i]) {
            site[i] -= 1;
          }
        }
#endif
        Group S = h.gauge_field.get_staple(x,y,z,t,mu);
        S = h.gauge_field.get_link(x,y,z,t,mu)*S;
        Adjoint dS(S);
        dS = (beta/h.gauge_field.get_Nc())*dS;
        dS += deriv.get_adjoint(site[0],site[1],site[2],site[3],mu);
        deriv.set_adjoint(site[0],site[1],site[2],site[3],mu,dS);
      });
    }

  };

} // namespace klft