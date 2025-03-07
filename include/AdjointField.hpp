#pragma once
#include "AdjointGroup.hpp"

#ifdef KLFT_USE_MPI
#include "Comm.hpp"
#endif

namespace klft {

  template <typename T, class Adjoint, int Ndim = 4, int Nc = 3>
  class AdjointField {
  public:
    struct set_zero_s {};
    using DeviceView = Kokkos::View<T****>;

    DeviceView adjoint[Ndim][2*Nc-1];

    int LT,LX,LY,LZ;
    Kokkos::Array<int,Ndim> local_dims;
    Kokkos::Array<int,4> global_dims;
    Kokkos::Array<int,4> start_dims;
    Kokkos::Array<int,4> end_dims;
    Kokkos::Array<int,4> array_dims;

    typedef Adjoint adjoint_group_t;

    AdjointField() = default;

    template <int N = Ndim, typename std::enable_if<N == 4, int>::type = 0>
    AdjointField(const Kokkos::Array<int,4> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LZ = _dims[2];
      this->LT = _dims[3];
      for(int i = 0; i < 2*Nc-1; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->adjoint[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "adjoint"), LX, LY, LZ, LT);
        }
      }
      this->local_dims = {LX,LY,LZ,LT};
      this->global_dims = {LX,LY,LZ,LT};
      this->start_dims = {0,0,0,0};
      this->end_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,2,3};
    }

    template <int N = Ndim, typename std::enable_if<N == 3, int>::type = 0>
    AdjointField(const Kokkos::Array<int,3> &_dims) {
      this->LX = _dims[0];
      this->LY = _dims[1];
      this->LT = _dims[2];
      this->LZ = 1;
      for(int i = 0; i < 2*Nc-1; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->adjoint[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "adjoint"), LX, LY, LZ, LT);
        }
      }
      this->local_dims = {LX,LY,LT};
      this->global_dims = {LX,LY,LT};
      this->start_dims = {0,0,0,0};
      this->end_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,1,3,-100};
    }

    template <int N = Ndim, typename std::enable_if<N == 2, int>::type = 0>
    AdjointField(const Kokkos::Array<int,2> &_dims) {
      this->LX = _dims[0];
      this->LT = _dims[1];
      this->LY = 1;
      this->LZ = 1;
      for(int i = 0; i < 2*Nc-1; ++i) {
        for(int mu = 0; mu < Ndim; ++mu) {
          this->adjoint[mu][i] = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "adjoint"), LX, LY, LZ, LT);
        }
      }
      this->local_dims = {LX,LT};
      this->global_dims = {LX,LT};
      this->start_dims = {0,0,0,0};
      this->end_dims = {LX,LY,LZ,LT};
      this->array_dims = {0,3,-100,-100};
    }

    KOKKOS_INLINE_FUNCTION int get_Ndim() const { return Ndim; }

    KOKKOS_INLINE_FUNCTION int get_Nc() const { return Nc; }

    KOKKOS_INLINE_FUNCTION int get_volume() const { return this->LX*this->LY*this->LZ*this->LT; }

    KOKKOS_INLINE_FUNCTION int get_size() const { return this->LX*this->LY*this->LZ*this->LT*Ndim*(2*Nc-1); }

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

    KOKKOS_INLINE_FUNCTION Adjoint get_adjoint(const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      Kokkos::Array<T,2*Nc-1> adj;
      #pragma unroll
      for(int i = 0; i < 2*Nc-1; i++) {
        adj[i] = this->adjoint[mu][i](x,y,z,t);
      }
      return Adjoint(adj);
    }

    KOKKOS_INLINE_FUNCTION Adjoint get_adjoint(const Kokkos::Array<int,4> &site, const int &mu) const {
      Kokkos::Array<T,2*Nc-1> adj;
      #pragma unroll
      for(int i = 0; i < 2*Nc-1; i++) {
        adj[i] = this->adjoint[mu][i](site[0],site[1],site[2],site[3]);
      }
      return Adjoint(adj);
    }

    KOKKOS_INLINE_FUNCTION void set_adjoint(const int &x, const int &y, const int &z, const int &t, const int &mu, const Adjoint &U) const {
      #pragma unroll
      for(int i = 0; i < 2*Nc-1; i++) {
        this->adjoint[mu][i](x,y,z,t) = U(i);
      }
    }

    KOKKOS_INLINE_FUNCTION void set_adjoint(const Kokkos::Array<int,4> &site, const int &mu, const Adjoint &U) {
      #pragma unroll
      for(int i = 0; i < 2*Nc-1; i++) {
        this->adjoint[mu][i](site[0],site[1],site[2],site[3]) = U(i);
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(set_zero_s, const int &x, const int &y, const int &z, const int &t, const int &mu) const {
      #pragma unroll
      for(int i = 0; i < 2*Nc-1; i++) {
        this->adjoint[mu][i](x,y,z,t) = 0.0;
      }
    }

    void set_zero() {
      auto BulkPolicy = Kokkos::MDRangePolicy<set_zero_s,Kokkos::Rank<5>>({start_dims[0],start_dims[1],start_dims[2],start_dims[3],0},{end_dims[0],end_dims[1],end_dims[2],end_dims[3],Ndim});
      Kokkos::parallel_for("set_zero", BulkPolicy, *this);
    }

  };

} // namespace klft