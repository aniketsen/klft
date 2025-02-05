#pragma once
#include "GaugeGroup.hpp"

namespace klft {

  template <typename T, class Group, int Ndim = 4, int Nc = 3>
  class GaugeField {
  public:
    struct set_one_s {};
    struct plaq_s {};
    struct wloop_temporal_s{};
    // obc related operators
    struct plaq_obc_s{};
    struct wloop_temporal_obc_s{};
    using complex_t = Kokkos::complex<T>;
    using DeviceView = Kokkos::View<complex_t****>;
    // using HostView = typename DeviceView::HostMirror;

    DeviceView gauge[Ndim][Nc*Nc];
    // HostView gauge_host;
    
    int LT,LX,LY,LZ;
    Kokkos::Array<int,Ndim> dims;
    Kokkos::Array<int,4> max_dims;
    Kokkos::Array<int,4> array_dims;


    typedef Group gauge_group_t;

    GaugeField() = default;

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    GaugeField(const int &_LX, const int &_LY, const int &_LZ, const int &_LT) {
      this->LT = _LT;
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = _LZ;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LY,LZ,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    GaugeField(const int &_LX, const int &_LY, const int &_LT) {
      this->LT = _LT;
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LY,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,-100};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    GaugeField(const int &_LX, const int &_LT) {
      this->LT = _LT;
      this->LX = _LX;
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->dims = {LX,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,3,-100,-100};
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

/*       KOKKOS_FUNCTION int get_array_dim(const int &mu) const {
        return this->array_dims[mu];
      } */

      void set_open_bc_x() {
        auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3)});
        Kokkos::parallel_for("set_open_bc_x", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &y, const int &z, const int &t) {
          #pragma unroll
          for(int i = 0; i < Nc*Nc; i++) {
            this->gauge[0][i](this->max_dims[0]-1,y,z,t) = Kokkos::complex<T>(0.0,0.0);
          }
        });
      }

      void set_open_bc_y() {
        if(Ndim < 3) return;
        auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{this->get_max_dim(0),this->get_max_dim(2),this->get_max_dim(3)});
        Kokkos::parallel_for("set_open_bc_y", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &z, const int &t) {
          #pragma unroll
          for(int i = 0; i < Nc*Nc; i++) {
            this->gauge[1][i](x,this->max_dims[1]-1,z,t) = Kokkos::complex<T>(0.0,0.0);
          }
        });
      }

      void set_open_bc_z() {
        if(Ndim < 4) return;
        auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(3)});
        Kokkos::parallel_for("set_open_bc_z", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &t) {
          #pragma unroll
          for(int i = 0; i < Nc*Nc; i++) {
            this->gauge[2][i](x,y,this->max_dims[2]-1,t) = Kokkos::complex<T>(0.0,0.0);
          }
        });
      }
    KOKKOS_FUNCTION int get_array_dim(const int &mu) const {
      return this->array_dims[mu];
    }

    KOKKOS_INLINE_FUNCTION void operator()(set_one_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        this->gauge[mu][i](x,y,z,t) = Kokkos::complex<T>(0.0,0.0);
      }
      #pragma unroll
      for(int i = 0; i < Nc; i++) {
        this->gauge[mu][i*Nc+i](x,y,z,t) = Kokkos::complex<T>(1.0,0.0);
      }
    }

    void set_one() {
      auto BulkPolicy = Kokkos::MDRangePolicy<set_one_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      Kokkos::parallel_for("set_one", BulkPolicy, *this);
    }

    KOKKOS_INLINE_FUNCTION Group get_link(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Kokkos::Array<Kokkos::complex<T>,Nc*Nc> link;
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        link[i] = this->gauge[mu][i](x,y,z,t);
      }
      return Group(link);
    }

    KOKKOS_INLINE_FUNCTION Group get_link(const Kokkos::Array<int,4> &site, const int &mu) const {
      Kokkos::Array<Kokkos::complex<T>,Nc*Nc> link;
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        link[i] = this->gauge[mu][i](site[0],site[1],site[2],site[3]);
      }
      return Group(link);
    }

    KOKKOS_INLINE_FUNCTION void set_link(const int &x, const int &y, const int &z, const int &t, const int &mu, const Group &U) const {
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        this->gauge[mu][i](x,y,z,t) = U(i);
      }
    }

    KOKKOS_INLINE_FUNCTION void set_link(const Kokkos::Array<int,4> &site, const int &mu, const Group &U) {
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        this->gauge[mu][i](site[0],site[1],site[2],site[3]) = U(i);
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(plaq_s, const int &x, const int &y, const int &z, const int &t, const int &mu, T &plaq) const {
      Group U1, U2, U3, U4;
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
      site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      #pragma unroll
      for(int nu = 0; nu < mu; ++nu){
        Kokkos::Array<int,4> site_plus_nu = {x,y,z,t};
        site_plus_nu[this->array_dims[nu]] = (site_plus_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        U1 = this->get_link(site,mu);
        U2 = this->get_link(site_plus_mu,nu);
        U3 = this->get_link(site_plus_nu,mu);
        U4 = this->get_link(site,nu);
        plaq += (U1*U2*dagger(U3)*dagger(U4)).retrace();
      }
    }

    T get_plaquette(bool Normalize = true) {
      auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({0,0,0,0,0},{this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Ndim});
      T plaq = 0.0;
      Kokkos::parallel_reduce("plaquette", BulkPolicy, *this, plaq);
      if(Normalize) plaq /= this->get_volume()*((Ndim-1)*Ndim/2)*Nc;
      return plaq;
    }

    KOKKOS_INLINE_FUNCTION void operator()(wloop_temporal_s, const int &x, const int &y, const int &z, const int &t, T &wloop_temporal) const
    {
      Group U1, U2, U3, U4;
      int R_side = 1;
      int T_side = 1;
      const int mu = Ndim - 1;

      #pragma unroll
      for (int nu = 0; nu < mu; ++nu)
      {
        Kokkos::Array<int, 4> site = {x, y, z, t};

        U1.set_identity();
        #pragma unroll
        for (int t_it = 0; t_it < T_side; t_it++)
        {
          U1 *= this->get_link(site, mu);
          site[this->array_dims[mu]] = (site[this->array_dims[mu]] + 1) % this->dims[mu];
        }

        U2.set_identity();
        #pragma unroll
        for (int r_it = 0; r_it < R_side; r_it++)
        {
          U2 *= this->get_link(site, nu);
          site[this->array_dims[nu]] = (site[this->array_dims[nu]] + 1) % this->dims[nu];
        }

        U4.set_identity();
        Kokkos::Array<int,4> site2 = {x, y, z, t};
        #pragma unroll
        for (int r_it = 0; r_it < R_side; r_it++)
        {
          U4 *= this->get_link(site2, nu);
          site2[this->array_dims[nu]] = (site2[this->array_dims[nu]] + 1) % this->dims[nu];
        }

        U3.set_identity();
        #pragma unroll
        for (int t_it = 0; t_it < T_side; t_it++)
        {
          U3 *= this->get_link(site2, mu);
          site2[this->array_dims[mu]] = (site2[this->array_dims[mu]] + 1) % this->dims[mu];
        }

        wloop_temporal += (U1 * U2 * dagger(U3) * dagger(U4)).retrace();
      }
    }

    T get_wloop_temporal(bool Normalize = true)
    {
      auto BulkPolicy = Kokkos::MDRangePolicy<wloop_temporal_s, Kokkos::Rank<4>>({0, 0, 0, 0}, {this->get_max_dim(0), this->get_max_dim(1), this->get_max_dim(2), this->get_max_dim(3)});
      T wloop_temporal = 0.0;
      Kokkos::parallel_reduce("wloop_temporal", BulkPolicy, *this, wloop_temporal);
      if (Normalize)
        wloop_temporal /= this->get_volume() * (Ndim - 1) * Nc;
      return wloop_temporal;
    }

    KOKKOS_INLINE_FUNCTION Group get_staple(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group staple(0.0);
      Group U1, U2, U3;
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
      site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1) % this->dims[mu];
      Kokkos::Array<int,4> site_pm_nu = {x,y,z,t};
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        U1 = get_link(site_plus_mu,nu);
        U2 = get_link(site_pm_nu,mu);
        U3 = get_link(site,nu);
        staple += U1*dagger(U2)*dagger(U3);
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
      }
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_plus_mu[this->array_dims[nu]] = (site_plus_mu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] - 1 + this->dims[nu]) % this->dims[nu];
        U1 = get_link(site_plus_mu,nu);
        U2 = get_link(site_pm_nu,mu);
        U3 = get_link(site_pm_nu,nu);
        staple += dagger(U1)*dagger(U2)*U3;
        site_pm_nu[this->array_dims[nu]] = (site_pm_nu[this->array_dims[nu]] + 1) % this->dims[nu];
        site_plus_mu[this->array_dims[nu]] = (site_plus_mu[this->array_dims[nu]] + 1) % this->dims[nu];
      }
      return staple;
    }

    // obc related stuff
    KOKKOS_INLINE_FUNCTION void operator()(plaq_obc_s, const int &t, const int &mu, T &plaq) const
    {
      Group U1, U2, U3, U4;
      Kokkos::Array<int, 4> site = {(int)(LX / 2), (int)(LY / 2), (int)(LZ / 2), t};
      Kokkos::Array<int, 4> site_plus_mu = {(int)(LX / 2), (int)(LY / 2), (int)(LZ / 2), t};
      site_plus_mu[this->array_dims[mu]] = (site_plus_mu[this->array_dims[mu]] + 1);
      #pragma unroll
      for (int nu = 0; nu < mu; ++nu)
      {
        Kokkos::Array<int, 4> site_plus_nu = {(int)(LX / 2), (int)(LY / 2), (int)(LZ / 2), t};
        site_plus_nu[this->array_dims[nu]] = (site_plus_nu[this->array_dims[nu]] + 1);
        U1 = this->get_link(site, mu);
        U2 = this->get_link(site_plus_mu, nu);
        U3 = this->get_link(site_plus_nu, mu);
        U4 = this->get_link(site, nu);
        plaq += (U1 * U2 * dagger(U3) * dagger(U4)).retrace();
      }
    }

    T get_plaquette_obc(bool Normalize = true)
    {
      auto BulkPolicy = Kokkos::MDRangePolicy<plaq_obc_s, Kokkos::Rank<2>>({0, 0}, {this->get_max_dim(3), Ndim - 1});
      T plaq = 0.0;
      Kokkos::parallel_reduce("plaquette", BulkPolicy, *this, plaq);
      if (Normalize)
        plaq /= this->get_max_dim(3) * ((Ndim - 2) * (Ndim - 1) / 2) * Nc;
      return plaq;
    }

    KOKKOS_INLINE_FUNCTION void operator()(wloop_temporal_obc_s, const int &t, T &wloop_temporal_obc) const
    {
      Group U1, U2, U3, U4;
      const int x0 = 0;//(int)(LX / 2);
      const int y0 = 0;//(int)(LY / 2);
      const int z0 = 0;//(int)(LZ / 2);
      int R_side = 1;
      int T_side = 1;
      const int mu = Ndim - 1;

      #pragma unroll
      for (int nu = 0; nu < mu; ++nu)
      {
        Kokkos::Array<int, 4> site = {x0, y0, z0, t};

        U1.set_identity();
        #pragma unroll
        for (int t_it = 0; t_it < T_side; t_it++)
        {
          U1 *= this->get_link(site, mu);
          site[this->array_dims[mu]] = (site[this->array_dims[mu]] + 1);
        }

        U2.set_identity();
        #pragma unroll
        for (int r_it = 0; r_it < R_side; r_it++)
        {
          U2 *= this->get_link(site, nu);
          site[this->array_dims[nu]] = (site[this->array_dims[nu]] + 1);
        }

        U4.set_identity();
        Kokkos::Array<int,4> site2 = {x0, y0, z0, t};
        #pragma unroll
        for (int r_it = 0; r_it < R_side; r_it++)
        {
          U4 *= this->get_link(site2, nu);
          site2[this->array_dims[nu]] = (site2[this->array_dims[nu]] + 1);
        }

        U3.set_identity();
        #pragma unroll
        for (int t_it = 0; t_it < T_side; t_it++)
        {
          U3 *= this->get_link(site2, mu);
          site2[this->array_dims[mu]] = (site2[this->array_dims[mu]] + 1);
        }

        wloop_temporal_obc += (U1 * U2 * dagger(U3) * dagger(U4)).retrace();
      }
    }

    T get_wloop_temporal_obc(bool Normalize = true)
    {
      auto BulkPolicy = Kokkos::RangePolicy<wloop_temporal_obc_s>(0, this->get_max_dim(3));
      T wloop_temporal_obc = 0.0;
      Kokkos::parallel_reduce("wloop_temporal_obc", BulkPolicy, *this, wloop_temporal_obc);
      if (Normalize)
        wloop_temporal_obc /= this->get_max_dim(3) * (Ndim - 1) * Nc;
      return wloop_temporal_obc;
    }

    void copy(const GaugeField<T,Group,Ndim,Nc> &in) {
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          Kokkos::deep_copy(this->gauge[mu][i], in.gauge[mu][i]);
        }
      }
    }
  };

}
