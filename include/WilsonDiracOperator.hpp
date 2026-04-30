#pragma once
#include "DiracOperator.hpp"
namespace klft {

template <typename DSpinorFieldType,
          typename DGaugeFieldType,
          bool HasMassShift = false>
class WilsonDiracOperator : public DiracOperator<WilsonDiracOperator,
                                                 DSpinorFieldType,
                                                 DGaugeFieldType,
                                                 HasMassShift> {
 public:
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;

  template <typename NewSpinor,
            typename NewGauge = DGaugeFieldType,
            bool NewHasMassShift = HasMassShift>
  using rebind = WilsonDiracOperator<NewSpinor, NewGauge, NewHasMassShift>;
  ~WilsonDiracOperator() = default;
  using Base = DiracOperator<WilsonDiracOperator,
                             DSpinorFieldType,
                             DGaugeFieldType,
                             HasMassShift>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp;
    Kokkos::Array<size_t, rank> idx{Idcs...};
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                   this->s_in.dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                  this->s_in.dimensions);

      auto temp1 =
          this->g_in(Idcs..., mu) * project(mu, -1, this->s_in(xp.first));

      auto temp2 =
          conj(this->g_in(xm.first, mu)) * project(mu, 1, this->s_in(xm.first));
      temp += reconstruct(mu, -1, (this->params.kappa * xp.second) * temp1) +
              reconstruct(mu, 1, (this->params.kappa * xm.second) * temp2);
    }

    this->s_out(Idcs...) = this->s_in(Idcs...) - temp;
    // this->s_out(Idcs...) = 1 * temp;
  }

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagDdagger,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp;
    Kokkos::Array<size_t, rank> idx{Idcs...};

#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                   this->s_in.dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                  this->s_in.dimensions);

      auto temp1 =
          this->g_in(Idcs..., mu) * project(mu, 1, this->s_in(xp.first));

      //
      auto temp2 = conj(this->g_in(xm.first, mu)) *
                   project(mu, -1, this->s_in(xm.first));
      temp += reconstruct(mu, 1, (this->params.kappa * xp.second) * temp1) +
              reconstruct(mu, -1, (this->params.kappa * xm.second) * temp2);
    }
    this->s_out(Idcs...) = this->s_in(Idcs...) - temp;
  }
};

template <typename DSpinorFieldType,
          typename DGaugeFieldType,
          bool HasMassShift = false>
class HWilsonDiracOperator : public DiracOperator<HWilsonDiracOperator,
                                                  DSpinorFieldType,
                                                  DGaugeFieldType,
                                                  HasMassShift> {
 public:
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  template <typename NewSpinor,
            typename NewGauge = DGaugeFieldType,
            bool NewHasMassShift = HasMassShift>
  using rebind = HWilsonDiracOperator<NewSpinor, NewGauge, NewHasMassShift>;
  ~HWilsonDiracOperator() = default;
  using Base = DiracOperator<HWilsonDiracOperator,
                             DSpinorFieldType,
                             DGaugeFieldType,
                             HasMassShift>;
  using Base::Base;
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagD,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim> temp;
    Kokkos::Array<size_t, rank> idx{Idcs...};
#pragma unroll
    for (size_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                   this->s_in.dimensions);
      auto xp = shift_index_plus_bc<rank, size_t>(idx, mu, 1, 3, -1,
                                                  this->s_in.dimensions);

      auto temp1 =
          this->g_in(Idcs..., mu) * project(mu, 1, this->s_in(xp.first));

      //
      auto temp2 = conj(this->g_in(xm.first, mu)) *
                   project(mu, -1, this->s_in(xm.first));
      temp += reconstruct(mu, 1, (this->params.kappa * xp.second) * temp1) +
              reconstruct(mu, -1, (this->params.kappa * xm.second) * temp2);
    }

    this->s_out(Idcs...) = gamma5(this->s_in(Idcs...) - temp);
  }

  // only for testing porpose, not the real Ddagger operator
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagDdagger,
                                              const Indices... Idcs) const {
    operator()(typename Tags::TagD(), Idcs...);
  }
};

template <typename DSpinorFieldType,
          typename DGaugeFieldType,
          bool HasMassShift = false>
class EOWilsonDiracOperator : public EODiracOperator<EOWilsonDiracOperator,
                                                     DSpinorFieldType,
                                                     DGaugeFieldType,
                                                     HasMassShift> {
 public:
  using Base = EODiracOperator<EOWilsonDiracOperator,
                               DSpinorFieldType,
                               DGaugeFieldType,
                               HasMassShift>;
  using Base::Base;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  template <typename NewSpinor,
            typename NewGauge = DGaugeFieldType,
            bool NewHasMassShift = HasMassShift>
  using rebind = EOWilsonDiracOperator<NewSpinor, NewGauge, NewHasMassShift>;
  // odd to even so H_eo
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagHeo,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp{};
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 0);
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                    this->g_in.dimensions);
      auto xp = shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                   this->g_in.dimensions);

      auto temp1 =
          this->g_even(Idcs..., mu) *
          project(mu, -1, this->s_in(index_full_to_half(xp.first).first));

      auto temp2 =
          conj(this->g_odd(index_full_to_half(xm.first).first, mu)) *
          project(mu, 1, this->s_in(index_full_to_half(xm.first).first));
      temp += reconstruct(mu, -1, (xp.second) * temp1) +
              reconstruct(mu, 1, (xm.second) * temp2);
    }

    this->s_out(Idcs...) = temp;
  }
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagHeodagger,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp{};
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 0);
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                    this->g_in.dimensions);
      auto xp = shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                   this->g_in.dimensions);

      auto temp1 =
          this->g_even(Idcs..., mu) *
          project(mu, 1, this->s_in(index_full_to_half(xp.first).first));

      auto temp2 =
          conj(this->g_odd(index_full_to_half(xm.first).first, mu)) *
          project(mu, -1, this->s_in(index_full_to_half(xm.first).first));
      temp += reconstruct(mu, +1, (xp.second) * temp1) +
              reconstruct(mu, -1, (xm.second) * temp2);
    }

    this->s_out(Idcs...) = temp;
  }

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::Tag1minusHeo,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp{};
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 0);
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                    this->g_in.dimensions);
      auto xp = shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                   this->g_in.dimensions);

      auto temp1 =
          this->g_even(Idcs..., mu) *
          project(mu, -1, this->s_in(index_full_to_half(xp.first).first));

      auto temp2 =
          conj(this->g_odd(index_full_to_half(xm.first).first, mu)) *
          project(mu, 1, this->s_in(index_full_to_half(xm.first).first));
      temp += reconstruct(mu, -1, (xp.second) * temp1) +
              reconstruct(mu, 1, (xm.second) * temp2);
    }

    if constexpr (HasMassShift == false) {
      this->s_out(Idcs...) = this->temp(Idcs...) -
                             (this->params.kappa * this->params.kappa) * temp;
    } else {
      this->s_out(Idcs...) =
          (1 + this->params.massShift) * this->temp(Idcs...) -
          (this->params.kappa * this->params.kappa) * temp;
    }
  }
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::Tag1minusHeodagger,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp{};
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 0);
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                    this->g_in.dimensions);
      auto xp = shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                   this->g_in.dimensions);

      auto temp1 =
          this->g_even(Idcs..., mu) *
          project(mu, 1, this->s_in(index_full_to_half(xp.first).first));

      auto temp2 =
          conj(this->g_odd(index_full_to_half(xm.first).first, mu)) *
          project(mu, -1, this->s_in(index_full_to_half(xm.first).first));
      temp += reconstruct(mu, 1, (xp.second) * temp1) +
              reconstruct(mu, -1, (xm.second) * temp2);
    }

    if constexpr (HasMassShift == false) {
      this->s_out(Idcs...) = this->temp(Idcs...) -
                             (this->params.kappa * this->params.kappa) * temp;
    } else {
      this->s_out(Idcs...) =
          (1 + this->params.massShift) * this->temp(Idcs...) -
          (this->params.kappa * this->params.kappa) * temp;
    }
  }
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::Tagg51minusHeo,
                                              const Indices... Idcs) const {
    operator()(typename Base::Tag1minusHeo(), Idcs...);
    this->s_out(Idcs...) = gamma5(this->s_out(Idcs...));
  }

  // even to odd = Hoe

  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagHoe,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp{};
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 1);
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                    this->g_in.dimensions);
      auto xp = shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                   this->g_in.dimensions);

      auto temp1 =
          this->g_odd(Idcs..., mu) *
          project(mu, -1, this->s_in(index_full_to_half(xp.first).first));

      auto temp2 =
          conj(this->g_even(index_full_to_half(xm.first).first, mu)) *
          project(mu, 1, this->s_in(index_full_to_half(xm.first).first));
      temp += reconstruct(mu, -1, (xp.second) * temp1) +
              reconstruct(mu, 1, (xm.second) * temp2);
    }

    this->s_out(Idcs...) = temp;
  }
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagHoedagger,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp{};
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 1);
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                    this->g_in.dimensions);
      auto xp = shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                   this->g_in.dimensions);

      auto temp1 =
          this->g_odd(Idcs..., mu) *
          project(mu, 1, this->s_in(index_full_to_half(xp.first).first));

      auto temp2 =
          conj(this->g_even(index_full_to_half(xm.first).first, mu)) *
          project(mu, -1, this->s_in(index_full_to_half(xm.first).first));
      temp += reconstruct(mu, 1, (xp.second) * temp1) +
              reconstruct(mu, -1, (xm.second) * temp2);
    }

    this->s_out(Idcs...) = temp;
  }
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::Tag1minusHoe,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp{};
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 1);
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                    this->g_in.dimensions);
      auto xp = shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                   this->g_in.dimensions);

      auto temp1 =
          this->g_odd(Idcs..., mu) *
          project(mu, -1, this->s_in(index_full_to_half(xp.first).first));

      auto temp2 =
          conj(this->g_even(index_full_to_half(xm.first).first, mu)) *
          project(mu, 1, this->s_in(index_full_to_half(xm.first).first));
      temp += reconstruct(mu, -1, (xp.second) * temp1) +
              reconstruct(mu, 1, (xm.second) * temp2);
    }
    if constexpr (HasMassShift == false) {
      this->s_out(Idcs...) = this->temp(Idcs...) -
                             (this->params.kappa * this->params.kappa) * temp;
    } else {
      this->s_out(Idcs...) =
          (1 + this->params.massShift) * this->temp(Idcs...) -
          (this->params.kappa * this->params.kappa) * temp;
    }
  }
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::Tag1minusHoedagger,
                                              const Indices... Idcs) const {
    Spinor<Nc, RepDim, typename Base::precision> temp{};
    Kokkos::Array<size_t, rank> idx{Idcs...};
    auto full_idx = index_half_to_full(idx, 1);
#pragma unroll
    for (index_t mu = 0; mu < rank; ++mu) {
      auto xm = shift_index_minus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                    this->g_in.dimensions);
      auto xp = shift_index_plus_bc<rank, index_t>(full_idx, mu, 1, 3, -1,
                                                   this->g_in.dimensions);

      auto temp1 =
          this->g_odd(Idcs..., mu) *
          project(mu, 1, this->s_in(index_full_to_half(xp.first).first));

      auto temp2 =
          conj(this->g_even(index_full_to_half(xm.first).first, mu)) *
          project(mu, -1, this->s_in(index_full_to_half(xm.first).first));
      temp += reconstruct(mu, 1, (xp.second) * temp1) +
              reconstruct(mu, -1, (xm.second) * temp2);
    }
    if constexpr (HasMassShift == false) {
      this->s_out(Idcs...) = this->temp(Idcs...) -
                             (this->params.kappa * this->params.kappa) * temp;
    } else {
      this->s_out(Idcs...) =
          (1 + this->params.massShift) * this->temp(Idcs...) -
          (this->params.kappa * this->params.kappa) * temp;
    }
  }
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Base::Tagg51minusHoe,
                                              const Indices... Idcs) const {
    operator()(typename Base::Tag1minusHoe(), Idcs...);
    this->s_out(Idcs...) = gamma5(this->s_out(Idcs...));
  }
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagD,
                                              const Indices... Idcs) const {
    operator()(typename Tags::TagHeo(), Idcs...);
    this->s_out(Idcs...) *= this->params.kappa;
    this->s_out(Idcs...) -= (1 + HasMassShift * this->params.massShift) *
                            this->s_in_same_parity(Idcs...);
    this->s_out(Idcs...) *= -1;
  }
  template <typename... Indices>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(typename Tags::TagDdagger,
                                              const Indices... Idcs) const {
    operator()(typename Tags::TagHoe(), Idcs...);
    this->s_out(Idcs...) *= this->params.kappa;
    this->s_out(Idcs...) -= (1 + HasMassShift * this->params.massShift) *
                            this->s_in_same_parity(Idcs...);
    this->s_out(Idcs...) *= -1;
  }
};

}  // namespace klft
