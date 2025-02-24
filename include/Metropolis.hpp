#pragma once
#include "GLOBAL.hpp"

namespace klft {

  template <class FieldType, class RNG>
  class Metropolis {
    public:
      template <int odd_even> struct sweep_s {};
      FieldType field;
      RNG rng;
      size_t n_hit;
      using site_t = typename FieldType::site_t;
      using real_t = typename FieldType::real_t;
      real_t delta;

      Metropolis() = default;

      Metropolis(FieldType _field, const RNG &_rng, const size_t &_n_hit, const real_t &_delta) {
        this->field = _field;
        this->rng = _rng;
        this->n_hit = _n_hit;
        this->delta = _delta;
      }

      KOKKOS_FUNCTION real_t get_delta() const { return delta; }
      
      template <int odd_even>
      KOKKOS_INLINE_FUNCTION void operator()(sweep_s<odd_even>, const int &x, const int &y, const int &z, const int &t, const int &mu, real_t &update) const {
        auto generator = rng.get_state();
        real_t num_accepted = 0.0;
        real_t delS = 0.0;
        site_t R;
        const int tt = 2*t + 1*odd_even;
        for(size_t i = 0; i < n_hit; i++) {
          R.get_random(generator, delta);
          delS = field.get_delta_S(x,y,z,tt,mu,R);
          bool accept = delS < 0.0;
          if(!accept) {
            real_t r = generator.drand(0.0,1.0);
            accept = r < Kokkos::exp(-delS);
          }
          if(accept) {
            field.update(x,y,z,tt,mu,R);
            num_accepted += 1.0;
          }
        }
        rng.free_state(generator);
        update += num_accepted;
      }

      void init(const bool cold_start) {
        if(cold_start) {
          field.set_one();
        } else {
          field.set_random(delta,rng);
        }
      }

      real_t sweep() {
        auto BulkPolicy_odd = Kokkos::MDRangePolicy<sweep_s<1>,Kokkos::Rank<5>>({0,0,0,0,0},{field.get_max_dim(0),field.get_max_dim(1),field.get_max_dim(2),(int)(field.get_max_dim(3)/2),field.get_Ndim()});
        auto BulkPolicy_even = Kokkos::MDRangePolicy<sweep_s<0>,Kokkos::Rank<5>>({0,0,0,0,0},{field.get_max_dim(0),field.get_max_dim(1),field.get_max_dim(2),(int)(field.get_max_dim(3)/2),field.get_Ndim()});
        real_t accept = 0.0;
        real_t accept_rate = 0.0;
        Kokkos::parallel_reduce("sweep_even", BulkPolicy_even, *this, accept);
        Kokkos::fence();
        accept_rate += accept;
        accept = 0.0;
        Kokkos::parallel_reduce("sweep_odd", BulkPolicy_odd, *this, accept);
        Kokkos::fence();
        accept_rate += accept;
        return accept_rate/(field.get_volume()*field.get_Ndim()*n_hit);
      }

  };
}