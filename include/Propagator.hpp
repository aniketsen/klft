#include "GLOBAL.hpp"
namespace klft {

template <size_t Nc, size_t RepDim>
struct devicePropagator {
  static constexpr size_t rank = 4;
  devicePropagator(const index_t L0,
                   const index_t L1,
                   const index_t L2,
                   const index_t L3,
                   const complex_t init)
      : dimensions({L0, L1, L2, L3}) {
    do_init(L0, L1, L2, L3, field, init);
  }
  devicePropagator(const IndexArray<rank>& dimensions, const complex_t& init)
      : dimensions(dimensions) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            init);
  }
  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               const index_t L3,
               Propagator<Nc, RepDim>& V,
               complex_t init) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    KTune::parallel_for(
        "init_devicePropagator",
        Policy<4>(IndexArray<rank>{0, 0, 0, 0},
                  IndexArray<rank>{L0, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {

#pragma unroll
          for (index_t c1 = 0; c1 < Nc * RepDim; ++c1) {
#pragma unroll
            for (index_t c2 = 0; c2 < Nc * RepDim; ++c2) {
              V(i0, i1, i2, i3)[c1][c2] = init;
            }
          }
        });
    Kokkos::fence();
  }
  Propagator<Nc, RepDim> field;
  const IndexArray<4> dimensions;

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION PropagatorMatrix<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1,
      const indexType i2,
      const indexType i3) const {
    return field(i0, i1, i2, i3);
  }
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION PropagatorMatrix<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1,
      const indexType i2,
      const indexType i3) {
    return field(i0, i1, i2, i3);
  }
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION PropagatorMatrix<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, 4> site) const {
    return field(site[0], site[1], site[2], site[3]);
  }
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION PropagatorMatrix<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, 4> site) {
    return field(site[0], site[1], site[2], site[3]);
  }
};
}  // namespace klft
