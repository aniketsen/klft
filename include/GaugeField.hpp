#pragma once
#include "GaugeGroup.hpp"

namespace klft {

  template <class Group, size_t Ndim>
  class GaugeField {
  public:
    struct set_one_s {};
    struct plaq_s {};
    struct restoreGauge_s {};
    using real_t = typename Group::real_type;
    using complex_t = Kokkos::complex<real_t>;
    using DeviceView = Kokkos::View<Group****>;
    // using HostView = typename DeviceView::HostMirror;

    DeviceView gauge[Ndim];
    // HostView gauge_host;
    
    size_t LT,LX,LY,LZ;
    Kokkos::Array<size_t,Ndim> dims;
    Kokkos::Array<size_t,4> max_dims;
    Kokkos::Array<size_t,4> array_dims;

    typedef Group gauge_group_t;
    static constexpr size_t Nc = Group::Nc;

    GaugeField() = default;

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    GaugeField(const Kokkos::Array<size_t,4> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LZ = _dims[2];
      this->LT = _dims[3];
      for(int mu = 0; mu < Ndim; ++mu) {
        this->gauge[mu] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      }
      this->dims = {LX,LY,LZ,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    GaugeField(const Kokkos::Array<size_t,3> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LT = _dims[2];
      this->LZ = 1;
      for(int mu = 0; mu < Ndim; ++mu) {
        this->gauge[mu] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      }
      this->dims = {LX,LY,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,999};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    GaugeField(const Kokkos::Array<size_t,2> &_dims) {
      this->LX = _dims[0];
      this->LT = _dims[1];
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,3,999,999};
    }

    KOKKOS_FUNCTION int get_Ndim() const { return Ndim; }

    KOKKOS_FUNCTION int get_Nc() const { return Nc; }

    KOKKOS_FUNCTION size_t get_volume() const { return this->LX*this->LY*this->LZ*this->LT; }

    KOKKOS_FUNCTION size_t get_size() const { return this->LX*this->LY*this->LZ*this->LT*Ndim*Nc*Nc; }

    KOKKOS_FUNCTION int get_dim(const int &mu) const {
      return this->dims[mu];
    }

    KOKKOS_FUNCTION int get_max_dim(const int &mu) const {
      return this->max_dims[mu];
    }

    KOKKOS_FUNCTION int get_array_dim(const int &mu) const {
      return this->array_dims[mu];
    }

    KOKKOS_INLINE_FUNCTION void operator()(set_one_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      this->gauge[mu](x,y,z,t).set_identity();
    }

    void set_one() {
      auto BulkPolicy = Kokkos::MDRangePolicy<set_one_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      Kokkos::parallel_for("set_one", BulkPolicy, *this);
    }

    KOKKOS_INLINE_FUNCTION void operator()(plaq_s, const int &x, const int &y, const int &z, const int &t, const int &mu, real_t &plaq) const {
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
      site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      #pragma unroll
      for(int nu = 0; nu < mu; ++nu){
        Kokkos::Array<int,4> site_plus_nu = {x,y,z,t};
        site_plus_nu[this->array_dims[nu]] = (site_plus_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        plaq += (gauge[mu](site[0],site[1],site[2],site[3])
                * gauge[nu](site_plus_mu[0],site_plus_mu[1],site_plus_mu[2],site_plus_mu[3])
                * dagger(gauge[mu](site_plus_nu[0],site_plus_nu[1],site_plus_nu[2],site_plus_nu[3]))
                * dagger(gauge[nu](site[0],site[1],site[2],site[3]))).retrace();
      }
    }

    real_t get_plaquette(bool Normalize = true) {
      auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      real_t plaq = 0.0;
      Kokkos::parallel_reduce("plaquette", BulkPolicy, *this, plaq);
      if(Normalize) plaq /= this->get_volume()*((Ndim-1)*Ndim/2)*Nc;
      return plaq;
    }

    KOKKOS_INLINE_FUNCTION Group get_staple(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group staple(0.0);
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
      site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      Kokkos::Array<int,4> site_pm_nu = {x,y,z,t};
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        staple += gauge[nu](site_plus_mu[0],site_plus_mu[1],site_plus_mu[2],site_plus_mu[3])
                * dagger(gauge[mu](site_pm_nu[0],site_pm_nu[1],site_pm_nu[2],site_pm_nu[3]))
                * dagger(gauge[nu](site[0],site[1],site[2],site[3]));
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
      }
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_plus_mu[this->array_dims[nu]] = (site_plus_mu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        staple += dagger(gauge[nu](site_plus_mu[0],site_plus_mu[1],site_plus_mu[2],site_plus_mu[3]))
                * dagger(gauge[mu](site_pm_nu[0],site_pm_nu[1],site_pm_nu[2],site_pm_nu[3]))
                * gauge[nu](site_pm_nu[0],site_pm_nu[1],site_pm_nu[2],site_pm_nu[3]);
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        site_plus_mu[this->array_dims[nu]] = (site_plus_mu[this->array_dims[nu]] + 1) % this->dims[nu];
      }
      return staple;
    }

    void copy(const GaugeField<Group,Ndim> &in) {
      for(int mu = 0; mu < Ndim; ++mu) {
        Kokkos::deep_copy(this->gauge[mu], in.gauge[mu]);
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(restoreGauge_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      this->gauge[mu](x,y,z,t).restoreGauge();
    }

    void restoreGauge() {
      auto BulkPolicy = Kokkos::MDRangePolicy<restoreGauge_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      Kokkos::parallel_for("restoreGauge", BulkPolicy, *this);
    }

    template <typename T, class RNG>
    void set_random(T delta, RNG rng) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      Kokkos::parallel_for("set_random", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &mu) {
        this->gauge[mu](x,y,z,t).get_random(rng, delta);
      });
    }
    
  };

}
