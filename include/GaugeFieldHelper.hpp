#pragma once
#include "FieldTypeHelper.hpp"
#include "SUN.hpp"

namespace klft {
template <typename DGaugeFieldType>
struct UnitarityCheckFunctor {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;

  using Field = typename DGaugeFieldType::type;
  Field field;

  UnitarityCheckFunctor() = delete;

  UnitarityCheckFunctor(const Field& f_in) : field(f_in) {}

  // ——— rank-2 lattice  (x , t) ————————————————————————————————
  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(const int x, const int t, real_t& local_max) const {
    for (int mu = 0; mu < 2; ++mu) {
      const real_t d = unitary_defect(field(x, t, mu));
      local_max = fmax(local_max, d);
    }
  }

  // ——— rank-3 lattice  (x , y , t) ————————————————————————————
  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(const int x,
                  const int y,
                  const int t,
                  real_t& local_max) const {
    for (int mu = 0; mu < 3; ++mu) {
      const real_t d = unitary_defect(field(x, y, t, mu));
      local_max = fmax(local_max, d);
    }
  }

  // ——— rank-4 lattice  (x , y , z , t) ————————————————————————
  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(const int x,
                  const int y,
                  const int z,
                  const int t,
                  real_t& local_max) const {
    for (int mu = 0; mu < 4; ++mu) {
      const real_t d = unitary_defect(field(x, y, z, t, mu));
      local_max = fmax(local_max, d);
    }
  }
};

template <typename DGaugeFieldType>
real_t unitarity_check(const typename DGaugeFieldType::type& field) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;

  // initialize the local sum
  real_t defect_max = 0.0;
  // create the functor
  UnitarityCheckFunctor<DGaugeFieldType> functor(field);
  // launch the kernel

  const auto rp = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(IndexArray<rank>{0},
                                                            field.dimensions);
  KTune::parallel_reduce("UnitarityCheck", rp, functor,
                         Kokkos::Max<real_t>(defect_max));
  // return the local sum
  return defect_max;
}

template <typename DGaugeFieldType>
void unitarity_restore(const typename DGaugeFieldType::type& field) {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;

  if (KLFT_VERBOSITY > 5) {
    Kokkos::printf("Restoring Unitarity of GaugeField.\n");
    Kokkos::printf("Unitarity defect before restoration: %f\n",
                   unitarity_check<DGaugeFieldType>(field));
  }

  const auto rp = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>(IndexArray<rank>{0},
                                                            field.dimensions);
  if constexpr (rank == 4) {
    KTune::parallel_for(
        "UnitaryRestore", Policy<rank>(IndexArray<rank>{0}, field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2,
                      const index_t i3) {
          for (index_t mu = 0; mu < rank; ++mu) {
            restoreSUN(field(i0, i1, i2, i3, mu));
          }
        });
  } else if constexpr (rank == 3) {
    KTune::parallel_for(
        "UnitaryRestore", Policy<rank>(IndexArray<rank>{0}, field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1, const index_t i2) {
          for (index_t mu = 0; mu < rank; ++mu) {
            restoreSUN(field(i0, i1, i2, mu));
          }
        });
  } else if constexpr (rank == 2) {
    KTune::parallel_for(
        "UnitaryRestore", Policy<rank>(IndexArray<rank>{0}, field.dimensions),
        KOKKOS_LAMBDA(const index_t i0, const index_t i1) {
          for (index_t mu = 0; mu < rank; ++mu) {
            restoreSUN(field(i0, i1, mu));
          }
        });
  } else {
    static_assert(rank == 2 || rank == 3 || rank == 4, "Unsupported rank");
  }

  if (KLFT_VERBOSITY > 5) {
    Kokkos::printf("Unitarity defect after restoration: %f\n",
                   unitarity_check<DGaugeFieldType>(field));
  }
}

}  // namespace klft
