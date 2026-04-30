#pragma once
#include "FieldTypeHelper.hpp"
#include "GLOBAL.hpp"
#include "Gauge_Util.hpp"

namespace klft {
struct CoolingParams {
  // define parameters for a given Wilson Flow calculation, the wilson flow
  // follows arxiv.org/pdf/1006.4518
  // flow time
  index_t n_steps;
  // flow time step size

  CoolingParams() {
    // default parameters (eps = 0.01)
    n_steps = 10;
  }
};

template <typename DGaugeFieldType, class RNG>
struct CoolingFunctors {
  // implement the Wilson flow, for now the field will not be copied, but it
  // will be flown in place -> copying needs to be done before
  constexpr static const size_t rank =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank;
  constexpr static const size_t Nc =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc;
  constexpr static const GaugeFieldKind Kind =
      DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Kind;
  static_assert(rank == 4);  // The wilson flow is only defined for 4D Fields
  CoolingParams params;

  // get the correct deviceGaugeFieldType
  using GaugeFieldT = typename DGaugeFieldType::type;
  GaugeFieldT field;
  GaugeFieldT tmp_staple;
  SUNAdjField<rank, Nc> tmp_Z;
  RNG& rng;

  CoolingFunctors() = delete;

  CoolingFunctors(const GaugeFieldT& _field, CoolingParams& _params, RNG& rng)
      : params(_params), field(_field.field), tmp_staple(_field.field) {
    const IndexArray<rank> dims = _field.dimensions;
    Kokkos::realloc(Kokkos::WithoutInitializing, tmp_Z, dims[0], dims[1],
                    dims[2], dims[3]);
    // Kokkos::realloc(Kokkos::WithoutInitializing, tmp_staple, dims[0],
    // dims[1],
    //                 dims[2], dims[3]);

    //            Kokkos::deep_copy(_field.field, tmp_stap);
    //            Kokkos::deep_copy(_field.field, tmp_Z);
    Kokkos::fence();
  }

  // execute the wilson flow
  void cool() {  // todo: check this once by saving a staple field and once by
                 // locally calculating the staple
    for (int step = 0; step < params.n_steps; ++step) {
      stapleField<DGaugeFieldType>(this->field, this->tmp_staple);
      Kokkos::fence();

      KTune::parallel_for(
          "Wilsonflow-flow",
          Policy<rank>(IndexArray<rank>{0, 0, 0, 0}, field.dimensions), *this);
      Kokkos::fence();
    }
  }

  template <typename indexType>
  KOKKOS_INLINE_FUNCTION void operator()(const indexType i0,
                                         const indexType i1,
                                         const indexType i2,
                                         const indexType i3) const {
#pragma unroll
    for (index_t mu = 0; mu < 4; ++mu) {
      bool accepted = false;

      this->field(i0, i1, i2, i3, mu) = (this->tmp_staple(i0, i1, i2, i3, mu));
      restoreSUN(this->field(i0, i1, i2, i3, mu));
    }
  }
};

}  // namespace klft
