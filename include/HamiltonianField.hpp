#pragma once
#include "GLOBAL.hpp"
#include "GaugeField.hpp"
#include "AdjointField.hpp"

namespace klft {

  template<typename T, class Group, class Adjoint, int Ndim = 4, int Nc = 3>
  class HamiltonianField {
  public:
    struct update_momentum_s {};
    struct update_gauge_s {};
    GaugeField<T,Group,Ndim,Nc> gauge_field;
    AdjointField<T,Adjoint,Ndim,Nc> adjoint_field;

    typedef T value_type;
    
    HamiltonianField() = default;

    HamiltonianField(const GaugeField<T,Group,Ndim,Nc> _gauge_field, const AdjointField<T,Adjoint,Ndim,Nc> _adjoint_field) :
      gauge_field(_gauge_field), adjoint_field(_adjoint_field) {}
    
    void update_momentum(AdjointField<T,Adjoint,Ndim,Nc> deriv, const T dtau) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({adjoint_field.get_start_dim(0),adjoint_field.get_start_dim(1),adjoint_field.get_start_dim(2),adjoint_field.get_start_dim(3),0},{adjoint_field.get_end_dim(0),adjoint_field.get_end_dim(1),adjoint_field.get_end_dim(2),adjoint_field.get_end_dim(3),adjoint_field.get_Ndim()});
      Kokkos::parallel_for("update_momentum", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t, const int mu) {
        Adjoint tmp = adjoint_field.get_adjoint(x,y,z,t,mu);
        tmp -= dtau*deriv.get_adjoint(x,y,z,t,mu);
        adjoint_field.set_adjoint(x,y,z,t,mu,tmp);
      });
    }

    void update_gauge(const T dtau) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({gauge_field.get_start_dim(0),gauge_field.get_start_dim(1),gauge_field.get_start_dim(2),gauge_field.get_start_dim(3),0},{gauge_field.get_end_dim(0),gauge_field.get_end_dim(1),gauge_field.get_end_dim(2),gauge_field.get_end_dim(3),gauge_field.get_Ndim()});
      Kokkos::parallel_for("update_gauge", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t, const int mu) {
        Group U = exp(dtau*adjoint_field.get_adjoint(x,y,z,t,mu))*gauge_field.get_link(x,y,z,t,mu);
        gauge_field.set_link(x,y,z,t,mu,U);
      });
    }

    T kinetic_energy() {
      T kinetic_energy = 0.0;
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({adjoint_field.get_start_dim(0),adjoint_field.get_start_dim(1),adjoint_field.get_start_dim(2),adjoint_field.get_start_dim(3),0},{adjoint_field.get_end_dim(0),adjoint_field.get_end_dim(1),adjoint_field.get_end_dim(2),adjoint_field.get_end_dim(3),adjoint_field.get_Ndim()});
      Kokkos::parallel_reduce("kinetic_energy", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t, const int mu, T &local_sum) {
        Adjoint tmp = adjoint_field.get_adjoint(x,y,z,t,mu);
        local_sum += 0.5*tmp.norm2();
      }, kinetic_energy);
      return kinetic_energy;
    }

    template <class RNG>
    void randomize_momentum(RNG rng) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({adjoint_field.get_start_dim(0),adjoint_field.get_start_dim(1),adjoint_field.get_start_dim(2),adjoint_field.get_start_dim(3),0},{adjoint_field.get_end_dim(0),adjoint_field.get_end_dim(1),adjoint_field.get_end_dim(2),adjoint_field.get_end_dim(3),adjoint_field.get_Ndim()});
      Kokkos::parallel_for("randomize_momentum", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int x, const int y, const int z, const int t, const int mu) {
        auto generator = rng.get_state();
        Adjoint U;
        U.get_random(generator);
        adjoint_field.set_adjoint(x,y,z,t,mu,U);
        rng.free_state(generator);
      });
    }

  };

} // namespace klft