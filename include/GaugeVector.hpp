#pragma once
#include "GaugeGroup.hpp"
#include "GaugeField.hpp"

namespace klft {

  template <typename T, class Group, int Ndim = 4, int Nc = 3>
  class GaugeVector {
  public:
    struct wilson_line_s {};

    using complex_t = Kokkos::complex<T>;
    using DeviceView = Kokkos::View<complex_t****>;

    DeviceView vector[Nc*Nc];
      
    int LT,LX,LY,LZ;
    Kokkos::Array<int,Ndim> dims;
    Kokkos::Array<int,4> max_dims;
    Kokkos::Array<int,4> array_dims;

    GaugeVector() = default;

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    GaugeVector(const Kokkos::Array<int,4> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LZ = _dims[2];
      this->LT = _dims[3];
      for(int i = 0; i < Nc*Nc; ++i) {
        this->vector[i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vector"), LX, LY, LZ, LT);
      }
      this->dims = {LX,LY,LZ,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    GaugeVector(const Kokkos::Array<int,3> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LT = _dims[2];
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        this->vector[i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vector"), LX, LY, LZ, LT);
      }
      this->dims = {LX,LY,LT};
      this->max_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,-100};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    GaugeVector(const Kokkos::Array<int,2> &_dims) {
      this->LX = _dims[0];
      this->LT = _dims[1];
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < Nc*Nc; ++i) {
        this->vector[i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "vector"), LX, LY, LZ, LT);
      }
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

    KOKKOS_FUNCTION int get_array_dim(const int &mu) const {
      return this->array_dims[mu];
    }

    void absorb_direction(const GaugeField<T,Group,Ndim,Nc> &gauge, const int &mu) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Nc*Nc});
      Kokkos::parallel_for("absorb_direction", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &i) {
        this->vector[i](x,y,z,t) = gauge.gauge[mu][i](x,y,z,t);
      });
    }

    void set_one() {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3)});
      Kokkos::parallel_for("set_one", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t) {
        for(int i = 0; i < Nc*Nc; i++) {
          this->vector[i](x,y,z,t) = Kokkos::complex<T>(0.0,0.0);
        }
        for(int i = 0; i < Nc; i++) {
          this->vector[i*Nc+i](x,y,z,t) = Kokkos::complex<T>(1.0,0.0);
        }
      });
    }

    KOKKOS_INLINE_FUNCTION Group get_link(const int &x, const int &y, const int &z, const int &t) const {
      Kokkos::Array<Kokkos::complex<T>,Nc*Nc> link;
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        link[i] = this->vector[i](x,y,z,t);
      }
      return Group(link);
    }

    KOKKOS_INLINE_FUNCTION Group get_link(const Kokkos::Array<int,4> &site) const {
      Kokkos::Array<Kokkos::complex<T>,Nc*Nc> link;
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        link[i] = this->vector[i](site[0],site[1],site[2],site[3]);
      }
      return Group(link);
    }

    KOKKOS_INLINE_FUNCTION void set_link(const int &x, const int &y, const int &z, const int &t, const Group &U) const {
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        this->vector[i](x,y,z,t) = U(i);
      }
    }

    KOKKOS_INLINE_FUNCTION void set_link(const Kokkos::Array<int,4> &site, const Group &U) {
      #pragma unroll
      for(int i = 0; i < Nc*Nc; i++) {
        this->vector[i](site[0],site[1],site[2],site[3]) = U(i);
      }
    }

    void UxU(const GaugeVector<T,Group,Ndim,Nc> &in) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3)});
      Kokkos::parallel_for("UxU", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t) {
        Group tmp = this->get_link(x,y,z,t);
        tmp *= in.get_link(x,y,z,t);
        this->set_link(x,y,z,t,tmp);
      });
    }

    void UxU(const GaugeVector<T,Group,Ndim,Nc> &in1, const GaugeVector<T,Group,Ndim,Nc> &in2) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3)});
      Kokkos::parallel_for("UxU", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t) {
        Group tmp = in1.get_link(x,y,z,t);
        tmp *= in2.get_link(x,y,z,t);
        this->set_link(x,y,z,t,tmp);
      });
    }

    void UxUdag(const GaugeVector<T,Group,Ndim,Nc> &in) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3)});
      Kokkos::parallel_for("UxUdag", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t) {
        Group tmp = this->get_link(x,y,z,t);
        tmp *= dagger(in.get_link(x,y,z,t));
        this->set_link(x,y,z,t,tmp);
      });
    }

    void UxUdag(const GaugeVector<T,Group,Ndim,Nc> &in1, const GaugeVector<T,Group,Ndim,Nc> &in2) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3)});
      Kokkos::parallel_for("UxUdag", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t) {
        Group tmp = in1.get_link(x,y,z,t);
        tmp *= dagger(in2.get_link(x,y,z,t));
        this->set_link(x,y,z,t,tmp);
      });
    }

    void Udag() {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3)});
      Kokkos::parallel_for("Udag", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t) {
        Group tmp = this->get_link(x,y,z,t);
        tmp.dagger();
        this->set_link(x,y,z,t,tmp);
      });
    }

    void copy(const GaugeVector<T,Group,Ndim,Nc> &in) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Nc*Nc});
      Kokkos::parallel_for("copy", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &i) {
        this->vector[i](x,y,z,t) = in.vector[i](x,y,z,t);
      });
    }

    void shift_plus(const GaugeVector<T,Group,Ndim,Nc> &tmp, const int &mu, const int &dist) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Nc*Nc});
      Kokkos::parallel_for("shift_plus", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &i) {
        Kokkos::Array<int,4> site = {x,y,z,t};
        site[this->array_dims[mu]] = (site[this->array_dims[mu]] + dist) % this->dims[mu];
        tmp.vector[i](x,y,z,t) = this->vector[i](site[0],site[1],site[2],site[3]);
      });
      this->copy(tmp);
    }

    void shift_minus(const GaugeVector<T,Group,Ndim,Nc> &tmp, const int &mu, const int &dist) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>({0,0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3),Nc*Nc});
      Kokkos::parallel_for("shift_minus", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, const int &i) {
        Kokkos::Array<int,4> site = {x,y,z,t};
        site[this->array_dims[mu]] = (site[this->array_dims[mu]] - dist + this->dims[mu]) % this->dims[mu];
        tmp.vector[i](x,y,z,t) = this->vector[i](site[0],site[1],site[2],site[3]);
      });
      this->copy(tmp);
    }

    T sum_retrace() {
      T sum = 0.0;
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3)});
      Kokkos::parallel_reduce("sum_retrace", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t, T &local_sum) {
        Group tmp = this->get_link(x,y,z,t);
        local_sum += tmp.retrace();
      }, sum);
      return sum;
    }

    void wilson_line(GaugeVector<T,Group,Ndim,Nc> &Umu, const int &mu, const int &Lmu, const bool &reverse = false) {
      auto BulkPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {this->get_max_dim(0),this->get_max_dim(1),this->get_max_dim(2),this->get_max_dim(3)});
      Kokkos::parallel_for("wilson_line", BulkPolicy, KOKKOS_CLASS_LAMBDA(const int &x, const int &y, const int &z, const int &t) {
        Kokkos::Array<int,4> site = {x,y,z,t};
        Group tmp1 = this->get_link(x,y,z,t);
        #pragma unroll
        for(int ii = 0; ii < Lmu; ++ii) {
          Group tmp2 = Umu.get_link(site[0],site[1],site[2],site[3]);
          tmp1 *= tmp2;
          site[this->array_dims[mu]] = (site[this->array_dims[mu]] + 1) % this->dims[mu];
        }
        this->set_link(x,y,z,t,tmp1);
      });
      // Umu.copy(tmp);
    }

  };

  template <typename T, class Group, int Ndim, int Nc>
  T get_W_loop_mu_nu(GaugeField<T,Group,Ndim,Nc> &gauge, const int &mu, const int &nu, const int &Lmu, const int &Lnu, const bool &Normalize = true) {
    GaugeVector<T,Group,Ndim,Nc> U1(gauge.dims);
    GaugeVector<T,Group,Ndim,Nc> U2(gauge.dims);
    GaugeVector<T,Group,Ndim,Nc> U3(gauge.dims);
    GaugeVector<T,Group,Ndim,Nc> U4(gauge.dims);
    GaugeVector<T,Group,Ndim,Nc> tmp(gauge.dims);
    GaugeVector<T,Group,Ndim,Nc> Umu(gauge.dims);
    GaugeVector<T,Group,Ndim,Nc> Unu(gauge.dims);
    Umu.absorb_direction(gauge,mu);
    Unu.absorb_direction(gauge,nu);
    U1.set_one();
    U1.wilson_line(Umu,mu,Lmu);
    U2.set_one();
    U2.wilson_line(Unu,nu,Lnu);
    U3.copy(U1);
    U4.copy(U2);
    U2.shift_plus(tmp,mu,Lmu);
    U1.UxU(U2);
    U3.shift_plus(tmp,nu,Lnu);
    U1.UxUdag(U3);
    U1.UxUdag(U4);
    T W = U1.sum_retrace();
    if(Normalize) W  /= (gauge.get_volume()*gauge.get_Nc());
    return W;
  }

} // namespace klft
