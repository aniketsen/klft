#pragma once
#include "GLOBAL.hpp"
#include <mpi.h>
#include <mpi-ext.h>

namespace klft {

  template <class ViewType,
            std::enable_if_t<std::is_same<typename ViewType::value_type,
            typename Kokkos::complex<double>>::value, bool> = true>
  MPI_Datatype get_mpi_datatype(ViewType) {
    return MPI_DOUBLE_COMPLEX;
  }

  template <class ViewType,
            std::enable_if_t<std::is_same<typename ViewType::value_type,
            typename Kokkos::complex<float>>::value, bool> = true>
  MPI_Datatype get_mpi_datatype(ViewType) {
    return MPI_COMPLEX;
  }

  template <int Ndim = 4>
  class Comm {
  public:
    MPI_Comm comm;
    int rank, size;
    Kokkos::Array<int,Ndim> dims;
    Kokkos::Array<int,Ndim> prev_rank;
    Kokkos::Array<int,Ndim> next_rank;
    Kokkos::Array<int,Ndim> periodic;

    Comm(const Kokkos::Array<int,Ndim> _dims) : dims(_dims) {
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      MPI_Dims_create(size, Ndim, dims.data());
      for(int i = 0; i < Ndim; i++) {
        periodic[i] = 1;
      }
      MPI_Cart_create(MPI_COMM_WORLD, Ndim, dims.data(), periodic.data(), 1, &comm);
      MPI_Comm_rank(comm, &rank);
      for(int i = 0; i < Ndim; i++) {
        MPI_Cart_shift(comm, i, 1, &prev_rank[i], &next_rank[i]);
      }
    }

    operator const MPI_Comm &() const { return comm; }

    template <class ViewType>
    void update_halo_minus(ViewType &view, const Kokkos::Array<int,Ndim> start_dims, const Kokkos::Array<int,Ndim> end_dims, const Kokkos::Array<int,Ndim> local_dims) {
      MPI_Request req[2*Ndim];
      int count = 0;
      MPI_Datatype mpi_datatype = get_mpi_datatype(view);
      typedef typename ViewType::value_type value_type;
      auto view_host = Kokkos::create_mirror_view(view);
      Kokkos::deep_copy(view_host, view);
      auto halo_send_x_next = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_send_x_next", local_dims[1], local_dims[2], local_dims[3]);
      auto halo_recv_x_prev = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_recv_x_next", local_dims[1], local_dims[2], local_dims[3]);
      auto halo_send_y_next = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_send_y_next", local_dims[0], local_dims[2], local_dims[3]);
      auto halo_recv_y_prev = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_recv_y_next", local_dims[0], local_dims[2], local_dims[3]);
      auto halo_send_z_next = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_send_z_next", local_dims[0], local_dims[1], local_dims[3]);
      auto halo_recv_z_prev = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_recv_z_next", local_dims[0], local_dims[1], local_dims[3]);
      auto halo_send_t_next = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_send_t_next", local_dims[0], local_dims[1], local_dims[2]);
      auto halo_recv_t_prev = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_recv_t_next", local_dims[0], local_dims[1], local_dims[2]);
      if(dims[0] > 1) {
        Kokkos::deep_copy(halo_send_x_next, Kokkos::subview(view_host,local_dims[0],Kokkos::make_pair(start_dims[1],end_dims[1]),Kokkos::make_pair(start_dims[2],end_dims[2]),Kokkos::make_pair(start_dims[3],end_dims[3])));
        MPI_Isend(halo_send_x_next.data(), size_t(halo_send_x_next.size()), mpi_datatype, next_rank[0], 0, comm, &req[count]);
        MPI_Irecv(halo_recv_x_prev.data(), size_t(halo_recv_x_prev.size()), mpi_datatype, prev_rank[0], 0, comm, &req[count]);
        count++;
      }
      if(dims[1] > 1) {
        Kokkos::deep_copy(halo_send_y_next, Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),local_dims[1],Kokkos::make_pair(start_dims[2],end_dims[2]),Kokkos::make_pair(start_dims[3],end_dims[3])));
        MPI_Isend(halo_send_y_next.data(), size_t(halo_send_y_next.size()), mpi_datatype, next_rank[1], 1, comm, &req[count]);
        MPI_Irecv(halo_recv_y_prev.data(), size_t(halo_recv_y_prev.size()), mpi_datatype, prev_rank[1], 1, comm, &req[count]);
        count++;
      }
      if(dims[2] > 1) {
        Kokkos::deep_copy(halo_send_z_next, Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),Kokkos::make_pair(start_dims[1],end_dims[1]),local_dims[2],Kokkos::make_pair(start_dims[3],end_dims[3])));
        MPI_Isend(halo_send_z_next.data(), size_t(halo_send_z_next.size()), mpi_datatype, next_rank[2], 2, comm, &req[count]);
        MPI_Irecv(halo_recv_z_prev.data(), size_t(halo_recv_z_prev.size()), mpi_datatype, prev_rank[2], 2, comm, &req[count]);
        count++;
      }
      if(dims[3] > 1) {
        Kokkos::deep_copy(halo_send_t_next, Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),Kokkos::make_pair(start_dims[1],end_dims[1]),Kokkos::make_pair(start_dims[2],end_dims[2]),local_dims[3]));
        MPI_Isend(halo_send_t_next.data(), size_t(halo_send_t_next.size()), mpi_datatype, next_rank[3], 3, comm, &req[count]);
        MPI_Irecv(halo_recv_t_prev.data(), size_t(halo_recv_t_prev.size()), mpi_datatype, prev_rank[3], 3, comm, &req[count]);
        count++;
      }
      MPI_Waitall(count, req, MPI_STATUSES_IGNORE);
      if(dims[0] > 1) {
        Kokkos::deep_copy(Kokkos::subview(view_host,0,Kokkos::make_pair(start_dims[1],end_dims[1]),Kokkos::make_pair(start_dims[2],end_dims[2]),Kokkos::make_pair(start_dims[3],end_dims[3])), halo_recv_x_prev);
      }
      if(dims[1] > 1) {
        Kokkos::deep_copy(Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),0,Kokkos::make_pair(start_dims[2],end_dims[2]),Kokkos::make_pair(start_dims[3],end_dims[3])), halo_recv_y_prev);
      }
      if(dims[2] > 1) {
        Kokkos::deep_copy(Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),Kokkos::make_pair(start_dims[1],end_dims[1]),0,Kokkos::make_pair(start_dims[3],end_dims[3])), halo_recv_z_prev);
      }
      if(dims[3] > 1) {
        Kokkos::deep_copy(Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),Kokkos::make_pair(start_dims[1],end_dims[1]),Kokkos::make_pair(start_dims[2],end_dims[2]),0), halo_recv_t_prev);
      }
      Kokkos::deep_copy(view, view_host);
    }

    template <class ViewType>
    void update_halo_plus(ViewType &view, const Kokkos::Array<int,Ndim> start_dims, const Kokkos::Array<int,Ndim> end_dims, const Kokkos::Array<int,Ndim> local_dims) {
      MPI_Request req[20];
      int count = 0;
      MPI_Datatype mpi_datatype = get_mpi_datatype(view);
      typedef typename ViewType::value_type value_type;
      auto view_host = Kokkos::create_mirror_view(view);
      Kokkos::deep_copy(view_host, view);
      auto halo_send_x_prev = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_send_x_next", local_dims[1], local_dims[2], local_dims[3]);
      auto halo_recv_x_next = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_recv_x_next", local_dims[1], local_dims[2], local_dims[3]);
      auto halo_send_y_prev = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_send_y_next", local_dims[0], local_dims[2], local_dims[3]);
      auto halo_recv_y_next = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_recv_y_next", local_dims[0], local_dims[2], local_dims[3]);
      auto halo_send_z_prev = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_send_z_next", local_dims[0], local_dims[1], local_dims[3]);
      auto halo_recv_z_next = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_recv_z_next", local_dims[0], local_dims[1], local_dims[3]);
      auto halo_send_t_prev = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_send_t_next", local_dims[0], local_dims[1], local_dims[2]);
      auto halo_recv_t_next = Kokkos::View<value_type***, Kokkos::HostSpace>("halo_recv_t_next", local_dims[0], local_dims[1], local_dims[2]);
      if(dims[0] > 1) {
        Kokkos::deep_copy(halo_send_x_prev, Kokkos::subview(view_host,1,Kokkos::make_pair(start_dims[1],end_dims[1]),Kokkos::make_pair(start_dims[2],end_dims[2]),Kokkos::make_pair(start_dims[3],end_dims[3])));
        MPI_Isend(halo_send_x_prev.data(), size_t(halo_send_x_prev.size()), mpi_datatype, prev_rank[0], 0, comm, &req[count++]);
        MPI_Irecv(halo_recv_x_next.data(), size_t(halo_recv_x_next.size()), mpi_datatype, next_rank[0], 0, comm, &req[count++]);
      }
      if(dims[1] > 1) {
        Kokkos::deep_copy(halo_send_y_prev, Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),1,Kokkos::make_pair(start_dims[2],end_dims[2]),Kokkos::make_pair(start_dims[3],end_dims[3])));
        MPI_Isend(halo_send_y_prev.data(), size_t(halo_send_y_prev.size()), mpi_datatype, prev_rank[1], 1, comm, &req[count++]);
        MPI_Irecv(halo_recv_y_next.data(), size_t(halo_recv_y_next.size()), mpi_datatype, next_rank[1], 1, comm, &req[count++]);
      }
      if(dims[2] > 1) {
        Kokkos::deep_copy(halo_send_z_prev, Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),Kokkos::make_pair(start_dims[1],end_dims[1]),1,Kokkos::make_pair(start_dims[3],end_dims[3])));
        MPI_Isend(halo_send_z_prev.data(), size_t(halo_send_z_prev.size()), mpi_datatype, prev_rank[2], 2, comm, &req[count++]);
        MPI_Irecv(halo_recv_z_next.data(), size_t(halo_recv_z_next.size()), mpi_datatype, next_rank[2], 2, comm, &req[count++]);
      }
      if(dims[3] > 1) {
        Kokkos::deep_copy(halo_send_t_prev, Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),Kokkos::make_pair(start_dims[1],end_dims[1]),Kokkos::make_pair(start_dims[2],end_dims[2]),1));
        MPI_Isend(halo_send_t_prev.data(), size_t(halo_send_t_prev.size()), mpi_datatype, prev_rank[3], 3, comm, &req[count++]);
        MPI_Irecv(halo_recv_t_next.data(), size_t(halo_recv_t_next.size()), mpi_datatype, next_rank[3], 3, comm, &req[count++]);
      }
      MPI_Waitall(count, req, MPI_STATUSES_IGNORE);
      if(dims[0] > 1) {
        Kokkos::deep_copy(Kokkos::subview(view_host,local_dims[0]+1,Kokkos::make_pair(start_dims[1],end_dims[1]),Kokkos::make_pair(start_dims[2],end_dims[2]),Kokkos::make_pair(start_dims[3],end_dims[3])), halo_recv_x_next);
      }
      if(dims[1] > 1) {
        Kokkos::deep_copy(Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),local_dims[1]+1,Kokkos::make_pair(start_dims[2],end_dims[2]),Kokkos::make_pair(start_dims[3],end_dims[3])), halo_recv_y_next);
      }
      if(dims[2] > 1) {
        Kokkos::deep_copy(Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),Kokkos::make_pair(start_dims[1],end_dims[1]),local_dims[2]+1,Kokkos::make_pair(start_dims[3],end_dims[3])), halo_recv_z_next);
      }
      if(dims[3] > 1) {
        Kokkos::deep_copy(Kokkos::subview(view_host,Kokkos::make_pair(start_dims[0],end_dims[0]),Kokkos::make_pair(start_dims[1],end_dims[1]),Kokkos::make_pair(start_dims[2],end_dims[2]),local_dims[3]+1), halo_recv_t_next);
      }
      Kokkos::deep_copy(view, view_host);
    }
  };

} // namespace klft