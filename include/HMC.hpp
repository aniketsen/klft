#pragma once
#include "GLOBAL.hpp"
#include "HMC_Params.hpp"
#include "HamiltonianField.hpp"
#include "Integrator.hpp"
#include <random>
#ifdef KLFT_USE_MPI
#include "Comm.hpp"
#endif

namespace klft {

  template<typename T, class Group, class Adjoint, class RNG, int Ndim = 4, int Nc = 3>
  class HMC {
  public:
    struct randomize_momentum_s {};
    HMC_Params params;
    HamiltonianField<T,Group,Adjoint,Ndim,Nc> hamiltonian_field;
    std::vector<std::unique_ptr<Monomial<T,Group,Adjoint,Ndim,Nc>>> monomials;
    Integrator<T,Group,Adjoint,Ndim,Nc> *integrator;
    RNG rng;
    std::mt19937 mt;
    std::uniform_real_distribution<T> dist;

    HMC() = default;

    HMC(const HMC_Params _params, RNG _rng, std::uniform_real_distribution<T> _dist, std::mt19937 _mt) : params(_params), rng(_rng), dist(_dist), mt(_mt) {}

    void add_gauge_monomial(const T _beta, const unsigned int _time_scale) {
      monomials.emplace_back(std::make_unique<GaugeMonomial<T,Group,Adjoint,Ndim,Nc>>(_beta,_time_scale));
    }

    void add_kinetic_monomial(const unsigned int _time_scale) {
      monomials.emplace_back(std::make_unique<KineticMonomial<T,Group,Adjoint,Ndim,Nc>>(_time_scale));
    }

    void add_hamiltonian_field(const HamiltonianField<T,Group,Adjoint,Ndim,Nc> _hamiltonian_field) {
      hamiltonian_field = _hamiltonian_field;
    }

    void set_integrator(const IntegratorType _integrator_type) {
      switch(_integrator_type) {
        case LEAPFROG:
          integrator = new Leapfrog<T,Group,Adjoint,Ndim,Nc>();
          break;
        default:
          break;
      }
    }

    bool hmc_step(
#ifdef KLFT_USE_MPI
      Comm<Ndim> &comm
#endif
    ) {
      hamiltonian_field.randomize_momentum(rng);
#ifdef KLFT_USE_MPI
      GaugeField<T,Group,Ndim,Nc> gauge_old(hamiltonian_field.gauge_field.global_dims,
                                            hamiltonian_field.gauge_field.local_dims,
                                            hamiltonian_field.gauge_field.comm_dims,
                                            hamiltonian_field.gauge_field.prev_rank,
                                            hamiltonian_field.gauge_field.next_rank);
#else
      GaugeField<T,Group,Ndim,Nc> gauge_old(hamiltonian_field.gauge_field.local_dims);
#endif
      gauge_old.copy(hamiltonian_field.gauge_field);
      for(int i = 0; i < monomials.size(); ++i) {
        monomials[i]->heatbath(hamiltonian_field);
      }
      integrator->integrate(monomials, hamiltonian_field, params
#ifdef KLFT_USE_MPI
        , comm
#endif
      );
// #ifdef KLFT_USE_MPI
//       hamiltonian_field.gauge_field.update_halo_plus(comm);
//       Kokkos::fence();
//       hamiltonian_field.gauge_field.update_halo_minus(comm);
//       Kokkos::fence();
// #endif
      double delta_H = 0.0;
      for(int i = 0; i < monomials.size(); ++i) {
        monomials[i]->accept(hamiltonian_field);
        delta_H += monomials[i]->get_delta_H();
      }
// #ifdef KLFT_USE_MPI
//       double global_delta_H;
//       MPI_Allreduce(&delta_H,&global_delta_H,1,MPI_DOUBLE,MPI_SUM,comm);
//       delta_H = global_delta_H;
// #endif
      bool accept = true;
#ifdef KLFT_USE_MPI
      if(comm.rank == 0) {
#endif 
      if(delta_H > 0.0) {
        if(dist(mt) > Kokkos::exp(-delta_H)) {
          accept = false;
        }
      }
#ifdef KLFT_USE_MPI
      }
      MPI_Bcast(&accept,1,MPI_CXX_BOOL,0,comm);
#endif
      if(!accept) {
        hamiltonian_field.gauge_field.copy(gauge_old);
      }
      return accept;
    }

  };
} // namespace klft