#pragma once
#include "GaugeGroup.hpp"

namespace klft {

  template <typename T, class GaugeGroup, int Ndim = 4, int Nc = 3>
  class GaugeField {
    public:
      struct initGauge_cold_s {};
      template <class RNG, RNG &rng> struct initGauge_hot_s {};
      struct plaq_s {};
      using DeviceView = Kokkos::View<T****[Ndim][Nc*Nc]>;
      using HostView = typename DeviceView::HostMirror;

      DeviceView gauge;
      HostView gauge_host;
      
      int LT,LX,LY,LZ;
      Kokkos::Array<int,4> dims;

      GaugeField() = default;

      template <std::enable_if<Ndim == 4, int>::type = 0>
      GaugeField(const int &_LX, const int &_LY, const int &_LZ, const int &_LT) {
        this->LT = _LT;
        this->LX = _LX;
        this->LY = _LY;
        this->LZ = _LZ;
        this->gauge = DeviceView("gauge", LX, LY, LZ, LT);
        this->dims = {LX,LY,LZ,LT};
        this->gauge_host = Kokkos::create_mirror_view(gauge);
      }

      template <std::enable_if<Ndim == 3, int>::type = 0>
      GaugeField(const int &_LX, const int &_LY, const int &_LT) {
        this->LT = _LT;
        this->LX = _LX;
        this->LY = _LY;
        this->LZ = 1;
        this->gauge = DeviceView("gauge", LX, LY, LZ, LT);
        this->dims = {LX,LY,LZ,LT};
        this->gauge_host = Kokkos::create_mirror_view(gauge);
      }

      template <std::enable_if<Ndim == 2, int>::type = 0>
      GaugeField(const int &_LX, const int &_LT) {
        this->LT = _LT;
        this->LX = _LX;
        this->LY = 1;
        this->LZ = 1;
        this->gauge = DeviceView("gauge", LX, LY, LZ, LT);
        this->dims = {LX,LY,LZ,LT};
        this->gauge_host = Kokkos::create_mirror_view(gauge);
      }

      KOKKOS_FUNCTION int get_Ndim() const { return Ndim; }

      KOKKOS_FUNCTION int get_Nc() const { return Nc; }

      KOKKOS_FUNCTION size_t get_volume() const { return this->LX*this->LY*this->LZ*this->LT; }

      KOKKOS_FUNCTION size_t get_size() const { return this->LX*this->LY*this->LZ*this->LT*Ndim*Nc*Nc; }

      KOKKOS_FUNCTION int get_dim(const int &mu) const {
        return this->gauge.extent_int(mu);
      }

      KOKKOS_INLINE_FUNCTION GaugeGroup get_link(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        T link[Nc*Nc];
        for(int i = 0; i < Nc*Nc; i++) {
          link[i] = this->gauge(x,y,z,t,mu,i);
        }
        return GaugeGroup(link);
      }

      KOKKOS_INLINE_FUNCTION GaugeGroup get_link(const Kokkos::Array<int,4> &site, const int &mu) const {
        T link[Nc*Nc];
        for(int i = 0; i < Nc*Nc; i++) {
          link[i] = this->gauge(site[0],site[1],site[2],site[3],mu,i);
        }
        return GaugeGroup(link);
      }

      KOKKOS_INLINE_FUNCTION void set_link(const int &x, const int &y, const int &z, const int &t, const int &mu, const GaugeGroup &U) const {
        for(int i = 0; i < Nc*Nc; i++) {
          this->gauge(x,y,z,t,mu,i) = U(i);
        }
      }

      KOKKOS_INLINE_FUNCTION void set_link(const Kokkos::Array<int,4> &site, const int &mu, const GaugeGroup &U) {
        for(int i = 0; i < Nc*Nc; i++) {
          this->gauge(site[0],site[1],site[2],site[3],mu,i) = U(i);
        }
      }

      KOKKOS_INLINE_FUNCTION void operator()(initGauge_cold_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        this->gauge(x,y,z,t,mu,0) = (Nc == 1) ? 0.0 : 1.0;
        for(int i = 1; i < Nc*Nc; i++) {
          this->gauge(x,y,z,t,mu,i) = 0.0;
        }
      }

      template <class RNG, RNG &rng>
      KOKKOS_INLINE_FUNCTION void operator()(initGauge_hot_s<RNG,rng>, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        auto generator = rng.get_state();
        this->set_link(x,y,z,t,mu,get_random<T,GaugeGroup,rng>(generator,0.01));
        rng.free_state(generator);
      }

      void initGauge() {
        auto BulkPolicy = Kokkos::MDRangePolicy<initGauge_cold_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->LX,this->LY,this->LZ,this->LT,Ndim});
        Kokkos::parallel_for("initGauge_cold", BulkPolicy, *this);
      }

      template <class RNG>
      void initGauge(RNG &rng) {
        auto BulkPolicy = Kokkos::MDRangePolicy<initGauge_hot_s<RNG,rng>,Kokkos::Rank<5>>({0,0,0,0,0},{this->LX,this->LY,this->LZ,this->LT,Ndim});
        Kokkos::parallel_for("initGauge_hot", BulkPolicy, *this);
      }

      KOKKOS_INLINE_FUNCTION void operator()(plaq_s, const int &x, const int &y, const int &z, const int &t, const int &mu, T &plaq) const {
        GaugeGroup U1, U2, U3, U4, tmp;
        Kokkos::Array<int,4> site = {x,y,z,t};
        Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
        site_plus_mu[mu] = (site_plus_mu[mu] + 1) % this->dims[mu];
        for(int nu = 0; nu < mu; ++nu){
          Kokkos::Array<int,4> site_plus_nu = {x,y,z,t};
          site_plus_nu[nu] = (site_plus_nu[nu] + 1) % this->dims[nu];
          U1 = this->get_link(site,mu);
          U2 = this->get_link(site_plus_mu,nu);
          U3 = this->get_link(site_plus_nu,mu);
          U4 = this->get_link(site,nu);
          tmp = U1*U2*dagger(U3)*dagger(U4);
          plaq += tmp.retrace();
        }
      }

      T get_plaquette() {
        auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->LX,this->LY,this->LZ,this->LT,Ndim});
        T plaq = 0.0;
        Kokkos::parallel_reduce("plaquette", BulkPolicy, *this, plaq);
        return plaq/(this->get_volume()*((Ndim-1)*Ndim/2)*Nc);
      }

      KOKKOS_INLINE_FUNCTION GaugeGroup get_staple(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
        GaugeGroup staple(0.0);
        GaugeGroup U1, U2, U3;
        Kokkos::Array<int,4> site = {x,y,z,t};
        Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
        site_plus_mu[mu] = (site_plus_mu[mu] + 1) % this->dims[mu];
        Kokkos::Array<int,4> site_pm_nu = {x,y,z,t};
        for(int nu = 0; nu < Ndim; ++nu) {
          if(nu == mu) continue;
          site_pm_nu[nu] = (site_pm_nu[nu] + 1) % this->dims[nu];
          U1 = get_link(site_plus_mu,nu);
          U2 = get_link(site_pm_nu,mu);
          U3 = get_link(site,nu);
          staple += U1*dagger(U2)*dagger(U3);
          site_pm_nu[nu] = (site_pm_nu[nu] - 1 + this->dims[nu]) % this->dims[nu];
        }
        for(int nu = 0; nu < Ndim; ++nu) {
          if(nu == mu) continue;
          site_plus_mu[nu] = (site_plus_mu[nu] - 1 + this->dims[nu]) % this->dims[nu];
          site_pm_nu[nu] = (site_pm_nu[nu] - 1 + this->dims[nu]) % this->dims[nu];
          U1 = get_link(site_plus_mu,nu);
          U2 = get_link(site_pm_nu,mu);
          U3 = get_link(site_pm_nu,nu);
          staple += dagger(U1)*dagger(U2)*U3;
          site_pm_nu[nu] = (site_pm_nu[nu] + 1) % this->dims[nu];
          site_plus_mu[nu] = (site_plus_mu[nu] + 1) % this->dims[nu];
        }
        return staple;
      }
  };

}