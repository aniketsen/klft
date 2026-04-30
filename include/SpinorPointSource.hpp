#pragma once

#include "GLOBAL.hpp"
#include "Spinor.hpp"
namespace klft {
template <size_t _Nc, size_t _RepDim>
struct deviceSpinorPointSource {
  static const size_t rank = 4;  // SpinorField is always 4D
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorPointSource() = default;
  deviceSpinorPointSource(const index_t L0,
                          const index_t L1,
                          const index_t L2,
                          const index_t L3,
                          const IndexArray<rank>& sourcePoint,
                          const index_t& deltaIndex)
      : dimensions({L0, L1, L2, L3}),
        sourcePoint(sourcePoint),
        deltaIndex(deltaIndex) {
    do_init(L0, L1, L2, L3, field, sourcePoint, deltaIndex);
  };
  deviceSpinorPointSource(const IndexArray<rank>& dimensions,
                          const IndexArray<rank>& sourcePoint,
                          const index_t& deltaIndex)
      : dimensions(dimensions),
        sourcePoint(sourcePoint),
        deltaIndex(deltaIndex) {
    do_init(dimensions[0], dimensions[1], dimensions[2], dimensions[3], field,
            sourcePoint, deltaIndex);
  };

  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               const index_t L3,
               SpinorField<Nc, RepDim>& V,
               const IndexArray<RepDim>& sourcePoint,
               const index_t& deltaIndex) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2, L3);
    KTune::parallel_for(
        "init_deviceSpinorField",
        Policy<rank>(IndexArray<RepDim>{0, 0, 0, 0},
                     IndexArray<rank>{L0, L1, L2, L3}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          IndexArray<RepDim>{i0, i1, i2, i3} == sourcePoint
              ? V(i0, i1, i2, i3) = deltaSpinor<Nc, RepDim>(deltaIndex)
              : V(i0, i1, i2, i3) = zeroSpinor<Nc, RepDim>();
        });
    Kokkos::fence();
  }
  SpinorField<Nc, RepDim> field;
  IndexArray<rank> dimensions;
  IndexArray<rank> sourcePoint;
  index_t deltaIndex;
  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1,
      const indexType i2,
      const indexType i3) const {
    return field(i0, i1, i2, i3);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1,
      const indexType i2,
      const indexType i3) {
    return field(i0, i1, i2, i3);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, 4> site) const {
    return field(site[0], site[1], site[2], site[3]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, 4> site) {
    return field(site[0], site[1], site[2], site[3]);
  }
};
template <size_t _Nc, size_t _RepDim>
struct deviceSpinorPointSource3D {
  static const size_t rank = 3;
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorPointSource3D() = default;
  deviceSpinorPointSource3D(const index_t L0,
                            const index_t L1,
                            const index_t L2,
                            const IndexArray<rank>& sourcePoint,
                            const index_t& deltaIndex)
      : dimensions({L0, L1, L2}),
        sourcePoint(sourcePoint),
        deltaIndex(deltaIndex) {
    do_init(L0, L1, L2, field, sourcePoint, deltaIndex);
  };
  deviceSpinorPointSource3D(const IndexArray<rank>& dimensions,
                            const IndexArray<rank>& sourcePoint,
                            const index_t& deltaIndex)
      : dimensions(dimensions),
        sourcePoint(sourcePoint),
        deltaIndex(deltaIndex) {
    do_init(dimensions[0], dimensions[1], dimensions[2], field, sourcePoint,
            deltaIndex);
  };

  void do_init(const index_t L0,
               const index_t L1,
               const index_t L2,
               SpinorField3D<Nc, RepDim>& V,
               const IndexArray<rank>& sourcePoint,
               const index_t& deltaIndex) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1, L2);
    KTune::parallel_for(
        "init_deviceSpinorField",
        Policy<rank>(IndexArray<rank>{0, 0, 0}, IndexArray<rank>{L0, L1, L2}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          IndexArray<rank>{i0, i1, i2} == sourcePoint
              ? V(i0, i1, i2) = deltaSpinor<Nc, RepDim>(deltaIndex)
              : V(i0, i1, i2) = zeroSpinor<Nc, RepDim>();
        });
    Kokkos::fence();
  }
  SpinorField3D<Nc, RepDim> field;
  IndexArray<rank> dimensions;
  IndexArray<rank> sourcePoint;
  index_t deltaIndex;
  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>&
  operator()(const indexType i0, const indexType i1, const indexType i2) const {
    return field(i0, i1, i2);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>&
  operator()(const indexType i0, const indexType i1, const indexType i2) {
    return field(i0, i1, i2);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) const {
    return field(site[0], site[1], site[2]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) {
    return field(site[0], site[1], site[2]);
  }
};
template <size_t _Nc, size_t _RepDim>
struct deviceSpinorPointSource2D {
  static const size_t rank = 2;  // SpinorField is always 4D
  static const size_t Nc = _Nc;
  static const size_t RepDim =
      _RepDim;  // RepDim is the dimension of the Gamma matrices
  deviceSpinorPointSource2D() = default;
  deviceSpinorPointSource2D(const index_t L0,
                            const index_t L1,
                            const IndexArray<rank>& sourcePoint,
                            const index_t& deltaIndex)
      : dimensions({L0, L1}), sourcePoint(sourcePoint), deltaIndex(deltaIndex) {
    do_init(L0, L1, field, sourcePoint, deltaIndex);
  };
  deviceSpinorPointSource2D(const IndexArray<rank>& dimensions,
                            const IndexArray<rank>& sourcePoint,
                            const index_t& deltaIndex)
      : dimensions(dimensions),
        sourcePoint(sourcePoint),
        deltaIndex(deltaIndex) {
    do_init(dimensions[0], dimensions[1], field, sourcePoint, deltaIndex);
  };

  void do_init(const index_t L0,
               const index_t L1,
               SpinorField2D<Nc, RepDim>& V,
               const IndexArray<rank>& sourcePoint,
               const index_t& deltaIndex) {
    Kokkos::realloc(Kokkos::WithoutInitializing, V, L0, L1);
    KTune::parallel_for(
        "init_deviceSpinorField",
        Policy<rank>(IndexArray<rank>{0, 0}, IndexArray<rank>{L0, L1}),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          IndexArray<rank>{i0, i1} == sourcePoint
              ? V(i0, i1) = deltaSpinor<Nc, RepDim>(deltaIndex)
              : V(i0, i1) = zeroSpinor<Nc, RepDim>();
        });
    Kokkos::fence();
  }
  SpinorField2D<Nc, RepDim> field;
  IndexArray<rank> dimensions;
  IndexArray<rank> sourcePoint;
  const index_t deltaIndex;
  // define accessors
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1) const {
    return field(i0, i1);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const indexType i0,
      const indexType i1) {
    return field(i0, i1);
  }

  // define accessors with 4D Kokkos array
  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) const {
    return field(site[0], site[1]);
  }

  template <typename indexType>
  KOKKOS_FORCEINLINE_FUNCTION Spinor<Nc, RepDim>& operator()(
      const Kokkos::Array<indexType, rank> site) {
    return field(site[0], site[1]);
  }
};

}  // namespace klft
