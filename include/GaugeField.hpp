#pragma once
#include "GaugeGroup.hpp"

#ifdef KLFT_USE_MPI
#include "Comm.hpp"
#endif

namespace klft {

  template <typename T, class Group, int Ndim = 4, int Nc = 3>
  class GaugeField {
  public:
    struct set_one_s {};
    struct plaq_s {};
    struct restoreGauge_s {};
    using complex_t = Kokkos::complex<T>;
    using DeviceView = Kokkos::View<complex_t****>;
    // using HostView = typename DeviceView::HostMirror;

    DeviceView gauge[Ndim][Nc*Nc];
    // HostView gauge_host;
    
    int LT,LX,LY,LZ;
    Kokkos::Array<int,Ndim> local_dims;
    Kokkos::Array<int,4> global_dims;
    Kokkos::Array<int,4> start_dims;
    Kokkos::Array<int,4> end_dims;
    Kokkos::Array<int,4> array_dims;
    
#ifdef KLFT_USE_MPI
    Kokkos::Array<int,4> comm_dims;
    Kokkos::Array<int,4> prev_rank;
    Kokkos::Array<int,4> next_rank;
    Kokkos::Array<int,4> field_length;
#endif

    typedef Group gauge_group_t;

    GaugeField() = default;

#ifdef KLFT_USE_MPI
    GaugeField(const Kokkos::Array<int,4> &_global_dims, const Kokkos::Array<int,4> &_local_dims,
               const Kokkos::Array<int,4> &_comm_dims, const Kokkos::Array<int,4> &_prev_rank,
               const Kokkos::Array<int,4> &_next_rank) {
      this->global_dims = _global_dims;
      this->local_dims = _local_dims;
      this->comm_dims = _comm_dims;
      this->prev_rank = _prev_rank;
      this->next_rank = _next_rank;
      for(int i = 0; i < Ndim; i++) {
        if(comm_dims[i]) {
          start_dims[i] = 1;
          end_dims[i] = local_dims[i]+1;
          field_length[i] = local_dims[i]+2;
        } else {
          start_dims[i] = 0;
          end_dims[i] = local_dims[i];
          field_length[i] = local_dims[i];
        }
      }
      this->LX = global_dims[0];
      this->LY = global_dims[1];
      this->LZ = global_dims[2];
      this->LT = global_dims[3];
      this->array_dims = {0,1,2,3};
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), field_length[0], field_length[1], field_length[2], field_length[3]);
        }
      }
    }
#endif                

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    GaugeField(const int &_LX, const int &_LY, const int &_LZ, const int &_LT) {
      this->LX = _LX;
      this->LY = _LY;
      this->LZ = _LZ;
      this->LT = _LT;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->local_dims = {LX,LY,LZ,LT};
      this->global_dims = {LX,LY,LZ,LT};
      this->start_dims = {0,0,0,0};
      this->end_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
#ifdef KLFT_USE_MPI
      this->comm_dims = {0,0,0,0};
      this->prev_rank = {-100,-100,-100,-100};
      this->next_rank = {-100,-100,-100,-100};
      this->field_length = {LX,LY,LZ,LT};
#endif      
    }

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    GaugeField(const Kokkos::Array<int,4> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LZ = _dims[2];
      this->LT = _dims[3];
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->local_dims = {LX,LY,LZ,LT};
      this->global_dims = {LX,LY,LZ,LT};
      this->start_dims = {0,0,0,0};
      this->end_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
#ifdef KLFT_USE_MPI
      this->comm_dims = {0,0,0,0};
      this->prev_rank = {-100,-100,-100,-100};
      this->next_rank = {-100,-100,-100,-100};
      this->field_length = {LX,LY,LZ,LT};
#endif 
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    GaugeField(const int &_LX, const int &_LY, const int &_LT) {
      this->LX = _LX;
      this->LY = _LY;
      this->LT = _LT;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->local_dims = {LX,LY,LT};
      this->global_dims = {LX,LY,LT};
      this->start_dims = {0,0,0,0};
      this->end_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,-100};
#ifdef KLFT_USE_MPI
      this->comm_dims = {0,0,0,0};
      this->prev_rank = {-100,-100,-100,-100};
      this->next_rank = {-100,-100,-100,-100};
      this->field_length = {LX,LY,LZ,LT};
#endif 
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    GaugeField(const Kokkos::Array<int,3> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LT = _dims[2];
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->local_dims = {LX,LY,LT};
      this->global_dims = {LX,LY,LT};
      this->start_dims = {0,0,0,0};
      this->end_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,-100};
#ifdef KLFT_USE_MPI
      this->comm_dims = {0,0,0,0};
      this->prev_rank = {-100,-100,-100,-100};
      this->next_rank = {-100,-100,-100,-100};
      this->field_length = {LX,LY,LZ,LT};
#endif 
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    GaugeField(const int &_LX, const int &_LT) {
      this->LX = _LX;
      this->LT = _LT;
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->local_dims = {LX,LT};
      this->global_dims = {LX,LT};
      this->start_dims = {0,0,0,0};
      this->end_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,3,-100,-100};
#ifdef KLFT_USE_MPI
      this->comm_dims = {0,0,0,0};
      this->prev_rank = {-100,-100,-100,-100};
      this->next_rank = {-100,-100,-100,-100};
      this->field_length = {LX,LY,LZ,LT};
#endif 
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    GaugeField(const Kokkos::Array<int,2> &_dims) {
      this->LX = _dims[0];
      this->LT = _dims[1];
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->gauge[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
        }
      }
      // this->gauge = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "gauge"), LX, LY, LZ, LT);
      this->local_dims = {LX,LT};
      this->global_dims = {LX,LT};
      this->start_dims = {0,0,0,0};
      this->end_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,3,-100,-100};
#ifdef KLFT_USE_MPI
      this->comm_dims = {0,0,0,0};
      this->prev_rank = {-100,-100,-100,-100};
      this->next_rank = {-100,-100,-100,-100};
      this->field_length = {LX,LY,LZ,LT};
#endif 
    }

    KOKKOS_FUNCTION int get_Ndim() const { return Ndim; }

    KOKKOS_FUNCTION int get_Nc() const { return Nc; }

    KOKKOS_FUNCTION size_t get_volume() const { return this->LX*this->LY*this->LZ*this->LT; }

    KOKKOS_FUNCTION size_t get_size() const { return this->LX*this->LY*this->LZ*this->LT*Ndim*Nc*Nc; }

    KOKKOS_FUNCTION int get_local_dim(const int &mu) const {
      return this->local_dims[mu];
    }

    KOKKOS_FUNCTION int get_global_dim(const int &mu) const {
      return this->global_dims[mu];
    }

    KOKKOS_INLINE_FUNCTION int get_start_dim(const int &mu) const {
      return this->start_dims[mu];
    }

    KOKKOS_FUNCTION int get_end_dim(const int &mu) const {
      return this->end_dims[mu];
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
      auto BulkPolicy = Kokkos::MDRangePolicy<set_one_s,Kokkos::Rank<5>>({start_dims[0],start_dims[1],start_dims[2],start_dims[3],0},{end_dims[0],end_dims[1],end_dims[2],end_dims[3],Ndim});
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
      site_plus_mu[array_dims[mu]] = (site_plus_mu[array_dims[mu]] + 1);
#ifdef KLFT_USE_MPI
      if(!comm_dims[mu]) {
#endif
      site_plus_mu[array_dims[mu]] = site_plus_mu[array_dims[mu]] % global_dims[mu];
#ifdef KLFT_USE_MPI
      }
#endif
      #pragma unroll
      for(int nu = 0; nu < mu; ++nu){
        Kokkos::Array<int,4> site_plus_nu = {x,y,z,t};
        site_plus_nu[array_dims[nu]] = (site_plus_nu[array_dims[nu]] + 1);
#ifdef KLFT_USE_MPI
        if(!comm_dims[nu]) {
#endif
        site_plus_nu[array_dims[nu]] = site_plus_nu[array_dims[nu]] % global_dims[nu];
#ifdef KLFT_USE_MPI
        }
#endif  
        U1 = get_link(site,mu);
        U2 = get_link(site_plus_mu,nu);
        U3 = get_link(site_plus_nu,mu);
        U4 = get_link(site,nu);
        plaq += (U1*U2*dagger(U3)*dagger(U4)).retrace();
      }
    }

    T get_plaquette(bool Normalize = true) {
      auto BulkPolicy = Kokkos::MDRangePolicy<plaq_s,Kokkos::Rank<5>>({start_dims[0],start_dims[1],start_dims[2],start_dims[3],0},{end_dims[0],end_dims[1],end_dims[2],end_dims[3],Ndim});
      T plaq = 0.0;
      Kokkos::parallel_reduce("plaquette", BulkPolicy, *this, plaq);
      if(Normalize) plaq /= get_volume()*((Ndim-1)*Ndim/2)*Nc;
      return plaq;
    }

    KOKKOS_INLINE_FUNCTION Group get_staple(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group staple(0.0);
      Group U1, U2, U3;
      Kokkos::Array<int,4> site = {x,y,z,t};
      Kokkos::Array<int,4> site_plus_mu = {x,y,z,t};
      site_plus_mu[array_dims[mu]] = (site_plus_mu[array_dims[mu]] + 1);
#ifdef KLFT_USE_MPI
      if(!comm_dims[mu]) {
#endif
      site_plus_mu[array_dims[mu]] = site_plus_mu[array_dims[mu]] % local_dims[mu];
#ifdef KLFT_USE_MPI
      }
#endif        
      Kokkos::Array<int,4> site_pm_nu = {x,y,z,t};
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_pm_nu[array_dims[nu]] = (site_pm_nu[array_dims[nu]] + 1);
#ifdef KLFT_USE_MPI
        if(!comm_dims[nu]) {
#endif
        site_pm_nu[array_dims[nu]] = site_pm_nu[array_dims[nu]] % local_dims[nu];
#ifdef KLFT_USE_MPI
        }
#endif
        U1 = get_link(site_plus_mu,nu);
        U2 = get_link(site_pm_nu,mu);
        U3 = get_link(site,nu);
        staple += U1*dagger(U2)*dagger(U3);
        site_pm_nu[array_dims[nu]] = (site_pm_nu[array_dims[nu]] - 1);
#ifdef KLFT_USE_MPI
        if(!comm_dims[nu]) {
#endif
        site_pm_nu[array_dims[nu]] = (site_pm_nu[array_dims[nu]] + local_dims[nu]) % local_dims[nu];
#ifdef KLFT_USE_MPI
        }
#endif
      }
      #pragma unroll
      for(int nu = 0; nu < Ndim; ++nu) {
        if(nu == mu) continue;
        site_plus_mu[array_dims[nu]] = (site_plus_mu[array_dims[nu]] - 1);
#ifdef KLFT_USE_MPI
        if(!comm_dims[nu]) {
#endif
        site_plus_mu[array_dims[nu]] = (site_plus_mu[array_dims[nu]] + local_dims[nu]) % local_dims[nu];
#ifdef KLFT_USE_MPI
        }
#endif
        site_pm_nu[array_dims[nu]] = (site_pm_nu[array_dims[nu]] - 1);
#ifdef KLFT_USE_MPI
        if(!comm_dims[nu]) {
#endif
        site_pm_nu[array_dims[nu]] = (site_pm_nu[array_dims[nu]] + local_dims[nu]) % local_dims[nu];
#ifdef KLFT_USE_MPI
        }
#endif
        U1 = get_link(site_plus_mu,nu);
        U2 = get_link(site_pm_nu,mu);
        U3 = get_link(site_pm_nu,nu);
        staple += dagger(U1)*dagger(U2)*U3;
        site_pm_nu[array_dims[nu]] = (site_pm_nu[array_dims[nu]] + 1);
#ifdef KLFT_USE_MPI
        if(!comm_dims[nu]) {
#endif
        site_pm_nu[array_dims[nu]] = site_pm_nu[array_dims[nu]] % local_dims[nu];
#ifdef KLFT_USE_MPI
        }
#endif
        site_plus_mu[array_dims[nu]] = (site_plus_mu[array_dims[nu]] + 1);
#ifdef KLFT_USE_MPI
        if(!comm_dims[nu]) {
#endif
        site_plus_mu[array_dims[nu]] = site_plus_mu[array_dims[nu]] % local_dims[nu];
#ifdef KLFT_USE_MPI
        }
#endif
      }
      return staple;
    }

    void copy(const GaugeField<T,Group,Ndim,Nc> &in) {
      for(int i = 0; i < Nc*Nc; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          Kokkos::deep_copy(this->gauge[mu][i], in.gauge[mu][i]);
        }
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(restoreGauge_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Group tmp = this->get_link(x,y,z,t,mu);
      tmp.restoreGauge();
      this->set_link(x,y,z,t,mu,tmp);
    }

    void restoreGauge() {
      auto BulkPolicy = Kokkos::MDRangePolicy<restoreGauge_s,Kokkos::Rank<5>>({start_dims[0],start_dims[1],start_dims[2],start_dims[3],0},{end_dims[0],end_dims[1],end_dims[2],end_dims[3],Ndim});
      Kokkos::parallel_for("restoreGauge", BulkPolicy, *this);
    }

    template <class RNG>
    void set_random(T delta, RNG rng) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({start_dims[0],start_dims[1],start_dims[2],start_dims[3],0},{end_dims[0],end_dims[1],end_dims[2],end_dims[3],Ndim});
      Kokkos::parallel_for("set_random", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &mu) {
        auto generator = rng.get_state();
        Group U;
        U.get_random(generator,delta);
        this->set_link(x,y,z,t,mu,U);
        rng.free_state(generator);
      });
    }

#ifdef KLFT_USE_MPI
    void update_halo_minus(Comm<Ndim> &comm) {
      #pragma unroll
      for(int mu = 0; mu < Ndim; mu++) {
        #pragma unroll
        for(int i = 0; i < Nc*Nc; i++) {
          comm.update_halo_minus(this->gauge[mu][i], this->start_dims, this->end_dims, this->local_dims);
        }
      }
    }

    void update_halo_plus(Comm<Ndim> &comm) {
      #pragma unroll
      for(int mu = 0; mu < Ndim; mu++) {
        #pragma unroll
        for(int i = 0; i < Nc*Nc; i++) {
          comm.update_halo_plus(this->gauge[mu][i], this->start_dims, this->end_dims, this->local_dims);
        }
      }
    }
#endif
    
  };

}
