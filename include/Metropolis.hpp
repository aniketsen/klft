#pragma once
#include "GaugeField.hpp"

namespace klft {

  template <typename T, class Group, class GaugeFieldType, class RNG>
  class Metropolis {
    public:
      struct initGauge_cold_s {};
      struct initGauge_hot_s {};
      template <int odd_even> struct sweep_s {};
      GaugeFieldType gauge_field;
      RNG rng;
      int n_hit;
      T beta;
      T delta;

      Metropolis() = default;

      Metropolis(GaugeFieldType gauge_field, RNG &rng, const int &n_hit, const T &beta, const T &delta) {
        this->gauge_field = gauge_field;
        this->rng = rng;
        this->n_hit = n_hit;
        this->beta = beta;
        this->delta = delta;
      }

      KOKKOS_FUNCTION T get_beta() const { return beta; }

      KOKKOS_FUNCTION T get_delta() const { return delta; }

      KOKKOS_FUNCTION T get_plaq() const { return gauge_field.get_plaquette(); }

      KOKKOS_INLINE_FUNCTION void operator()(initGauge_cold_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        Group U;
        U.set_identity();
        this->gauge_field.set_link(x,y,z,t,mu,U);
      }

      KOKKOS_INLINE_FUNCTION void operator()(initGauge_hot_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        auto generator = rng.get_state();
        Group U;
        U.get_random(generator,delta);
        this->gauge_field.set_link(x,y,z,t,mu,U);
        rng.free_state(generator);
      }
      
      template <int odd_even>
      KOKKOS_INLINE_FUNCTION void operator()(sweep_s<odd_even>, const int &x, const int &y, const int &z, const int &t, const int &mu, T &update) const {
        auto generator = rng.get_state();
        T num_accepted = 0.0;
        T delS = 0.0;
        const int tt = 2*t + 1*odd_even;
        Group U = gauge_field.get_link(x,y,z,tt,mu);
        Group staple = gauge_field.get_staple(x,y,z,tt,mu);
        Group tmp1 = U*staple;
        Group R;
        for(int i = 0; i < n_hit; ++i) {
          R.get_random(generator, delta);
          Group U_new = U*R;
          Group tmp2 = U_new*staple;
          delS = (beta/static_cast<T>(gauge_field.get_Nc()))*(tmp1.retrace() - tmp2.retrace());
          bool accept = delS < 0.0;
          if(!accept) {
            T r = generator.drand(0.0,1.0);
            accept = r < Kokkos::exp(-delS);
          }
          if(accept) {
            U_new.restoreGauge();
            gauge_field.set_link(x,y,z,tt,mu,U_new);
            num_accepted += 1.0;
          }
        }
        rng.free_state(generator);
        update += num_accepted;
      }

      void initGauge(const bool cold_start) {
        if(cold_start) {
          auto BulkPolicy = Kokkos::MDRangePolicy<initGauge_cold_s,Kokkos::Rank<5>>({gauge_field.get_start_dim(0),gauge_field.get_start_dim(1),gauge_field.get_start_dim(2),gauge_field.get_start_dim(3),0},{gauge_field.get_end_dim(0),gauge_field.get_end_dim(1),gauge_field.get_end_dim(2),gauge_field.get_end_dim(3),gauge_field.get_Ndim()});
          Kokkos::parallel_for("initGauge_cold", BulkPolicy, *this);
        } else {
          auto BulkPolicy = Kokkos::MDRangePolicy<initGauge_hot_s,Kokkos::Rank<5>>({gauge_field.get_start_dim(0),gauge_field.get_start_dim(1),gauge_field.get_start_dim(2),gauge_field.get_start_dim(3),0},{gauge_field.get_end_dim(0),gauge_field.get_end_dim(1),gauge_field.get_end_dim(2),gauge_field.get_end_dim(3),gauge_field.get_Ndim()});
          Kokkos::parallel_for("initGauge_hot", BulkPolicy, *this);
        }
      }

      T sweep() {
        int t_start_odd, t_start_even, t_end_odd, t_end_even;
        t_start_odd = gauge_field.get_start_dim(3);
        t_start_even = gauge_field.get_start_dim(3);
        t_end_odd = (int)(gauge_field.get_end_dim(3)/2);
        t_end_even = (int)(gauge_field.get_end_dim(3)/2);
#ifdef KLFT_USE_MPI
        if(gauge_field.comm_dims[3]) {
          t_start_odd = 0;
          t_start_even = 1;
          t_end_odd = (int)(gauge_field.get_end_dim(3)/2);
          t_end_even = (int)(gauge_field.get_end_dim(3)/2) + 1;
        }
#endif
        auto BulkPolicy_odd = Kokkos::MDRangePolicy<sweep_s<1>,Kokkos::Rank<5>>({gauge_field.get_start_dim(0),gauge_field.get_start_dim(1),gauge_field.get_start_dim(2),t_start_odd,0},{gauge_field.get_end_dim(0),gauge_field.get_end_dim(1),gauge_field.get_end_dim(2),t_end_odd,gauge_field.get_Ndim()});
        auto BulkPolicy_even = Kokkos::MDRangePolicy<sweep_s<0>,Kokkos::Rank<5>>({gauge_field.get_start_dim(0),gauge_field.get_start_dim(1),gauge_field.get_start_dim(2),t_start_even,0},{gauge_field.get_end_dim(0),gauge_field.get_end_dim(1),gauge_field.get_end_dim(2),t_end_even,gauge_field.get_Ndim()});
        T accept = 0.0;
        T accept_rate = 0.0;
        Kokkos::parallel_reduce("sweep_even", BulkPolicy_even, *this, accept);
        Kokkos::fence();
        accept_rate += accept;
        accept = 0.0;
        Kokkos::parallel_reduce("sweep_odd", BulkPolicy_odd, *this, accept);
        Kokkos::fence();
        accept_rate += accept;
        return accept_rate/(gauge_field.get_volume()*gauge_field.get_Ndim()*n_hit);
      }

  };
}