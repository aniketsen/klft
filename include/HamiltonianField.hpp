#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Kokkos_Macros.hpp"
#include "decl/Kokkos_Declare_OPENMP.hpp"
#include "impl/Kokkos_Profiling.hpp"

namespace klft {

template <typename DGaugeFieldType, typename DAdjFieldType>
struct HamiltonianField {
  // template argument deduction and safety
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceAdjFieldType<DAdjFieldType>::value);
  constexpr static size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static size_t Nc = DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  static_assert((rank == DeviceAdjFieldTypeTraits<DAdjFieldType>::Rank) &&
                (Nc == DeviceAdjFieldTypeTraits<DAdjFieldType>::Nc));

  using GaugeField = typename DGaugeFieldType::type;
  using AdjointField = typename DAdjFieldType::type;

  GaugeField &gauge_field;
  AdjointField &adjoint_field;
  struct EKin {};

  HamiltonianField() = delete;

  HamiltonianField(GaugeField &_gauge_field, AdjointField &_adjoint_field)
      : gauge_field(_gauge_field), adjoint_field(_adjoint_field) {}

  // Rank-4 version
  KOKKOS_INLINE_FUNCTION void operator()(EKin, index_t i0, index_t i1,
                                         index_t i2, index_t i3,
                                         real_t &rtn) const {
    real_t tmp = 0.0;
    for (index_t mu = 0; mu < 4; ++mu) {
      tmp += 0.5 * norm2<Nc>(adjoint_field(i0, i1, i2, i3, mu));
    }
    rtn += tmp;
  }

  // Rank-3 version
  KOKKOS_INLINE_FUNCTION void operator()(EKin, index_t i0, index_t i1,
                                         index_t i2, real_t &rtn) const {
    real_t tmp = 0.0;
    for (index_t mu = 0; mu < 3; ++mu) {
      tmp += 0.5 * norm2<Nc>(adjoint_field(i0, i1, i2, mu));
    }
    rtn += tmp;
  }

  // Rank-2 version
  KOKKOS_INLINE_FUNCTION void operator()(EKin, index_t i0, index_t i1,
                                         real_t &rtn) const {
    real_t tmp = 0.0;
    for (index_t mu = 0; mu < 2; ++mu) {
      tmp += 0.5 * norm2<Nc>(adjoint_field(i0, i1, mu));
    }
    rtn += tmp;
  }

  real_t kinetic_energy() {
    real_t kinetic_energy = 0.0;
    auto rp = Kokkos::MDRangePolicy<EKin, Kokkos::Rank<this->rank>>(
        IndexArray<this->rank>{0}, this->adjoint_field.dimensions);
    Kokkos::parallel_reduce("kinetic_energy", rp, *this, kinetic_energy);
    return kinetic_energy;
  }

  template <class RNG> void randomize_momentum(RNG &rng) {
    adjoint_field.template randomize_field<RNG>(rng);
  }
};

} // namespace klft
