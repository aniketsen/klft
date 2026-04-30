#pragma once
#include "FermionParams.hpp"
#include "FieldTypeHelper.hpp"
#include "GammaMatrix.hpp"
#include "Gauge_Util.hpp"
#include "IndexHelper.hpp"
#include "Spinor.hpp"
#include "SpinorFieldLinAlg.hpp"

namespace klft {
// Base Class for Single Dirac Operator
namespace Tags {
struct TagD {};
struct TagDdagger {};
// applys Composid Operator M= DDdagger*s_in
struct TagDDdagger {};
// applys Composid Operator Mdagger= DdaggerD*s_in
// If spinorfield are Checkerboard the operator S_e is used
struct TagDdaggerD {};
struct TagHoe {};
struct TagHeo {};
struct TagHoedagger {};
struct TagHeodagger {};
struct TagSe {};
struct TagSo {};
struct TagSedagger {};
struct TagSodagger {};
struct TagG5Se {};
struct TagG5So {};
}  // namespace Tags
template <class _Derived,
          typename _DSpinorFieldType,
          typename _DGaugeFieldType,
          bool _HasMassShift = false>
class BaseDiracOperator {
  static_assert(isDeviceGaugeFieldType<_DGaugeFieldType>::value);
  static_assert(isDeviceFermionFieldType<_DSpinorFieldType>::value);
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<_DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<_DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<_DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<_DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<_DGaugeFieldType>::Nc));
  constexpr static SpinorFieldLayout Layout =
      DeviceFermionFieldTypeTraits<_DSpinorFieldType>::Layout;

  // using Derived = _Derived<_Derived, _DSpinorFieldType, DGaugeFieldType>;
  // Define Tags for template dispatch:
 public:
  constexpr static bool HasMassShift = _HasMassShift;
  using precision = DeviceFermionFieldTypeTraits<_DSpinorFieldType>::value_type;
  using Derived = _Derived;
  using DSpinorFieldType = _DSpinorFieldType;
  using DGaugeFieldType = _DGaugeFieldType;
  using SpinorFieldType = typename _DSpinorFieldType::type;
  using strippedGaugeField = DeviceGaugeFieldType<rank, Nc, precision>;
  using GaugeFieldType = strippedGaugeField::type;

  BaseDiracOperator(const GaugeFieldType& g_in, const diracParams& params)
      : g_in(g_in), params(params) {}
  BaseDiracOperator(const diracParams& params) : params(params) {}
  ~BaseDiracOperator() = default;

  void init(const IndexArray<rank>& dim_gauge) {
    static_cast<_Derived&>(*this).init_gaugefield(dim_gauge);
  }
  void set_gauge(const GaugeFieldType& g_in) { this->g_in = g_in; }
  void init_gaugefield(const IndexArray<rank> dims) {};
  std::string constexpr getPrecision() {
    if constexpr (std::is_same_v<precision, complex_t>) {
      return "double";
    } else if (std::is_same_v<precision, complexsingle_t>) {
      return "single";
    }
    return "";
  }
  // Define callabale apply functions
  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION SpinorFieldType
  apply(const SpinorFieldType& s_in) {
    this->s_in = s_in;
    this->s_out = SpinorFieldType(s_in.dimensions, complex_t(0.0, 0.0));
    // Apply the operator

    return static_cast<_Derived&>(*this).apply_(Tag{});
  }

  /// @brief applys the DiracOperator to the field s_in and stores the result in
  /// s_out
  /// @tparam Tag Can either be TagD or TagDdagger or TagDDdagger or TagDdaggerD
  /// @param s_in
  /// @param s_out
  /// @return Nothing, result is stored in s_out
  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION void apply(const SpinorFieldType& s_in,
                                         const SpinorFieldType& s_out) {
    this->s_in = s_in;

    this->s_out = s_out;

    // forward to Derived class only for TagHeo
    static_cast<_Derived&>(*this).apply_(Tag{});
  }

  // Special overload t reduce allocations in solver further, only works with
  // composed operators

  template <typename Tag>
  KOKKOS_FORCEINLINE_FUNCTION void apply(const SpinorFieldType& s_in,
                                         const SpinorFieldType& s_temp,
                                         const SpinorFieldType& s_out) {
    this->s_in = s_in;
    this->s_out = s_temp;

    // Apply the operator
    static_cast<_Derived&>(*this).apply_(Tag{}, s_out);
  }

 public:
  SpinorFieldType s_in;
  SpinorFieldType s_out;
  GaugeFieldType g_in;
  diracParams params;

  BaseDiracOperator() = default;
};
template <template <typename, typename, bool> class _Derived,
          typename DSpinorFieldType,
          typename DGaugeFieldType,
          bool HasMassShift = false>
class DiracOperator : public BaseDiracOperator<DiracOperator<_Derived,
                                                             DSpinorFieldType,
                                                             DGaugeFieldType,
                                                             HasMassShift>,
                                               DSpinorFieldType,
                                               DGaugeFieldType,
                                               HasMassShift> {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));
  constexpr static SpinorFieldLayout Layout =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Layout;

  using BaseDiracOperator<
      DiracOperator<_Derived, DSpinorFieldType, DGaugeFieldType, HasMassShift>,
      DSpinorFieldType,
      DGaugeFieldType,
      HasMassShift>::BaseDiracOperator;
  using Derived = _Derived<DSpinorFieldType, DGaugeFieldType, HasMassShift>;
  using SpinorFieldType = typename DSpinorFieldType::type;

 public:
  SpinorFieldType apply_(Tags::TagD) {
    // Apply the operator
    KTune::parallel_for(
        typeid(Derived).name(),
        Policy<rank, Tags::TagD>(IndexArray<rank>{}, this->s_in.dimensions),
        static_cast<Derived&>(*this));
    Kokkos::fence();
    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagDdagger) {
    // Apply the operator
    KTune::parallel_for(typeid(Derived).name(),
                        Policy<rank, Tags::TagDdagger>(IndexArray<rank>{},
                                                       this->s_in.dimensions),
                        static_cast<Derived&>(*this));
    Kokkos::fence();
    return this->s_out;
  }
  // applys Composid Operator M= DDdagger*s_in

  SpinorFieldType apply_(Tags::TagDDdagger) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDDdagger{}, cached_out);
  }

  // applys Composid Operator Mdagger= DdaggerD*s_in
  SpinorFieldType apply_(Tags::TagDdaggerD) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDdaggerD{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagDDdagger, const SpinorFieldType& s_out) {
    if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
      // printf("Iam a CheckerboardLayout\n");
      static_cast<Derived&>(*this).apply_(Tags::TagDDdagger{}, s_out);
      // auto temp_field = this->s_in;
      // apply<Tags::TagHoe>(this->s_in, this->s_out);
      // this->s_in = this->s_out;
      // this->s_out = s_out;
      // apply<Tags::TagHeo>(this->s_in, this->s_out);
      // axpy<DSpinorFieldType>(-1, this->s_out, temp_field, this->s_out);
      return this->s_out;
    }

    this->apply_(Tags::TagDdagger{});
    this->s_in = this->s_out;
    this->s_out = s_out;
    return this->apply_(Tags::TagD{});
  }

  SpinorFieldType apply_(Tags::TagDdaggerD, const SpinorFieldType& s_out) {
    if constexpr (Layout == SpinorFieldLayout::Checkerboard) {
      static_cast<Derived&>(*this).apply_(Tags::TagDdaggerD{}, s_out);
      // printf("Iam a CheckerboardLayout\n");
      // auto temp_field = this->s_in;
      // this->apply_(Tags::TagD{});
      // this->s_in = this->s_out;
      // this->s_out = s_out;
      // this->apply_(Tags::TagDdagger{});
      // axpy<DSpinorFieldType>(-1, this->s_out, temp_field, this->s_out);
      return this->s_out;
    }
    this->apply_(Tags::TagD{});
    this->s_in = this->s_out;
    this->s_out = s_out;
    return this->apply_(Tags::TagDdagger{});
  }
};

template <template <typename, typename, bool> class _Derived,
          typename DSpinorFieldType,
          typename DGaugeFieldType,
          bool HasMassShift = false>
class EODiracOperator
    : public BaseDiracOperator<EODiracOperator<_Derived,
                                               DSpinorFieldType,
                                               DGaugeFieldType,
                                               HasMassShift>,
                               DSpinorFieldType,
                               DGaugeFieldType,
                               HasMassShift> {
  static_assert(isDeviceGaugeFieldType<DGaugeFieldType>::value);
  static_assert(isDeviceFermionFieldType<DSpinorFieldType>::value);
  constexpr static size_t rank =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Rank;
  constexpr static size_t Nc =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Nc;
  constexpr static size_t RepDim =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::RepDim;
  static_assert((rank == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Rank) &&
                (Nc == DeviceGaugeFieldTypeTraits<DGaugeFieldType>::Nc));
  constexpr static SpinorFieldLayout Layout =
      DeviceFermionFieldTypeTraits<DSpinorFieldType>::Layout;

 public:
  using BaseDiracOperator<EODiracOperator<_Derived,
                                          DSpinorFieldType,
                                          DGaugeFieldType,
                                          HasMassShift>,
                          DSpinorFieldType,
                          DGaugeFieldType,
                          HasMassShift>::BaseDiracOperator;
  using Derived = _Derived<DSpinorFieldType, DGaugeFieldType, HasMassShift>;
  using SpinorFieldType = typename DSpinorFieldType::type;
  SpinorFieldType s_in_same_parity;
  SpinorFieldType temp;
  BaseDiracOperator<EODiracOperator<_Derived,
                                    DSpinorFieldType,
                                    DGaugeFieldType,
                                    HasMassShift>,
                    DSpinorFieldType,
                    DGaugeFieldType,
                    HasMassShift>::GaugeFieldType g_even;
  BaseDiracOperator<EODiracOperator<_Derived,
                                    DSpinorFieldType,
                                    DGaugeFieldType,
                                    HasMassShift>,
                    DSpinorFieldType,
                    DGaugeFieldType,
                    HasMassShift>::GaugeFieldType g_odd;
  struct Tag1minusHeo {};
  struct Tag1minusHoe {};
  struct Tag1minusHeodagger {};
  struct Tag1minusHoedagger {};
  struct Tagg51minusHeo {};
  struct Tagg51minusHoe {};
  SpinorFieldType apply_(Tags::TagDDdagger) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDDdagger{}, cached_out);
  }

  // applys Composid Operator Mdagger= DdaggerD*s_in
  SpinorFieldType apply_(Tags::TagDdaggerD) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));

    return this->apply_(Tags::TagDdaggerD{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagHeo) {
    // this->s_out = SpinorFieldType(this->this->s_in.dimensions, complex_t(0.0,
    // 0.0));
    KTune::parallel_for(
        "Tags::TagHeo" + this->getPrecision(),
        Policy<rank, Tags::TagHeo>(IndexArray<rank>{}, this->s_in.dimensions),
        static_cast<Derived&>(*this));
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagHoe) {
    // this->s_out = SpinorFieldType(this->this->s_in.dimensions, complex_t(0.0,
    // 0.0));

    KTune::parallel_for(
        "Tags::TagHoe" + this->getPrecision(),
        Policy<rank, Tags::TagHoe>(IndexArray<rank>{}, this->s_in.dimensions),
        static_cast<Derived&>(*this));
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagHeodagger) {
    // this->s_out = SpinorFieldType(this->this->s_in.dimensions, complex_t(0.0,
    // 0.0));
    KTune::parallel_for("Tags::TagHeodagger" + this->getPrecision(),
                        Policy<rank, Tags::TagHeodagger>(IndexArray<rank>{},
                                                         this->s_in.dimensions),
                        static_cast<Derived&>(*this));
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagHoedagger) {
    // this->s_out = SpinorFieldType(this->this->s_in.dimensions, complex_t(0.0,
    // 0.0));

    KTune::parallel_for("Tags::TagHoedagger" + this->getPrecision(),
                        Policy<rank, Tags::TagHoedagger>(IndexArray<rank>{},
                                                         this->s_in.dimensions),
                        static_cast<Derived&>(*this));
    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagSe) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagSe{}, cached_out);
  }
  SpinorFieldType apply_(Tags::TagSedagger) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagSedagger{}, cached_out);
  }
  SpinorFieldType apply_(Tags::TagSo) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagSo{}, cached_out);
  }
  SpinorFieldType apply_(Tags::TagSodagger) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagSodagger{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagSe, const SpinorFieldType& s_out) {
    // printf("iam a EO Se\n");
    // printf("%i", this->test);
    this->apply_(Tags::TagHoe{});

    this->temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    KTune::parallel_for(
        "Tag1minusHeo" + this->getPrecision(),
        Policy<rank, Tag1minusHeo>(IndexArray<rank>{}, this->s_in.dimensions),
        static_cast<Derived&>(*this));

    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagSedagger, const SpinorFieldType& s_out) {
    // printf("iam a EO Se\n");
    // printf("%i", this->test);
    this->apply_(Tags::TagHoedagger{});

    this->temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    KTune::parallel_for("Tag1minusHeo" + this->getPrecision(),
                        Policy<rank, Tag1minusHeodagger>(IndexArray<rank>{},
                                                         this->s_in.dimensions),
                        static_cast<Derived&>(*this));

    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagSo, const SpinorFieldType& s_out) {
    this->apply_(Tags::TagHeo{});

    auto temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    KTune::parallel_for(
        "Tag1minusHoe" + this->getPrecision(),
        Policy<rank, Tag1minusHoe>(IndexArray<rank>{}, this->s_in.dimensions),
        static_cast<Derived&>(*this));

    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagSodagger, const SpinorFieldType& s_out) {
    this->apply_(Tags::TagHeodagger{});

    auto temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    KTune::parallel_for("Tag1minusHoedagger" + this->getPrecision(),
                        Policy<rank, Tag1minusHoedagger>(IndexArray<rank>{},
                                                         this->s_in.dimensions),
                        static_cast<Derived&>(*this));

    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagG5Se) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagG5Se{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagG5So) {
    auto cached_out = this->s_out;
    this->s_out = SpinorFieldType(this->s_in.dimensions, complex_t(0.0, 0.0));
    return this->apply_(Tags::TagG5So{}, cached_out);
  }

  SpinorFieldType apply_(Tags::TagG5Se, const SpinorFieldType& s_out) {
    // printf("iam a EO Se\n");
    // printf("%i", this->test);
    this->apply_(Tags::TagHoe{});

    this->temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    KTune::parallel_for(
        "Tagg51minusHeo" + this->getPrecision(),
        Policy<rank, Tagg51minusHeo>(IndexArray<rank>{}, this->s_in.dimensions),
        static_cast<Derived&>(*this));

    return this->s_out;
  }
  SpinorFieldType apply_(Tags::TagG5So, const SpinorFieldType& s_out) {
    this->apply_(Tags::TagHeo{});

    this->temp = this->s_in;
    this->s_in = this->s_out;
    this->s_out = s_out;

    KTune::parallel_for(
        "Tagg51minusHoe" + this->getPrecision(),
        Policy<rank, Tagg51minusHoe>(IndexArray<rank>{}, this->s_in.dimensions),
        static_cast<Derived&>(*this));
    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagDDdagger, const SpinorFieldType& s_out) {
    if (!s_in_same_parity.field.is_allocated()) {
      this->s_in_same_parity = SpinorFieldType(this->s_in.dimensions, 0);
    }

    auto cached_s_out = this->s_out;
    this->s_out = s_out;
    apply_(Tags::TagSedagger{}, s_in_same_parity);  // ergebnis in s_out
    this->s_in = s_in_same_parity;
    this->s_out = cached_s_out;

    apply_(Tags::TagSe{}, s_out);
    return s_out;
  }
  SpinorFieldType apply_(Tags::TagDdaggerD, const SpinorFieldType& s_out) {
    if (!s_in_same_parity.field.is_allocated()) {
      this->s_in_same_parity = SpinorFieldType(this->s_in.dimensions, 0);
    }

    auto cached_s_out = this->s_out;
    this->s_out = s_out;
    apply_(Tags::TagSe{}, s_in_same_parity);  // ergebnis in s_out
    this->s_in = s_in_same_parity;
    this->s_out = cached_s_out;

    apply_(Tags::TagSedagger{}, s_out);
    return s_out;
  }
  SpinorFieldType apply_(Tags::TagD) {
    // Apply the operator
    KTune::parallel_for(
        "Tags::TagD" + this->getPrecision(),
        Policy<rank, Tags::TagD>(IndexArray<rank>{}, this->s_in.dimensions),
        static_cast<Derived&>(*this));
    if constexpr (HasMassShift) {
      printf(
          "Warning: Mass shift in EO Dirac operator apply D is not "
          "implemented\n");
    }
    Kokkos::fence();
    return this->s_out;
  }

  SpinorFieldType apply_(Tags::TagDdagger) {
    // Apply the operator
    KTune::parallel_for("Tags::TagDdagger" + this->getPrecision(),
                        Policy<rank, Tags::TagDdagger>(IndexArray<rank>{},
                                                       this->s_in.dimensions),
                        static_cast<Derived&>(*this));
    if constexpr (HasMassShift) {
      printf(
          "Warning: Mass shift in EO Dirac operator apply Ddagger is not "
          "implemented\n");
    }
    Kokkos::fence();
    return this->s_out;
  }

  void init_gaugefield(const IndexArray<rank> dims) {
    if (!g_even.field.is_allocated() || !g_odd.field.is_allocated()) {
      this->g_even = typename BaseDiracOperator<
          EODiracOperator<_Derived, DSpinorFieldType, DGaugeFieldType,
                          HasMassShift>,
          DSpinorFieldType, DGaugeFieldType,
          HasMassShift>::GaugeFieldType(dims, complex_t(0.0));
      this->g_odd = typename BaseDiracOperator<
          EODiracOperator<_Derived, DSpinorFieldType, DGaugeFieldType,
                          HasMassShift>,
          DSpinorFieldType, DGaugeFieldType,
          HasMassShift>::GaugeFieldType(dims, complex_t(0.0));
    }

    alignGaugeFieldEvenOddFunctor<typename BaseDiracOperator<
        EODiracOperator<_Derived, DSpinorFieldType, DGaugeFieldType,
                        HasMassShift>,
        DSpinorFieldType, DGaugeFieldType, HasMassShift>::DGaugeFieldType>
        even(this->g_even, this->g_in, 0);
    alignGaugeFieldEvenOddFunctor<typename BaseDiracOperator<
        EODiracOperator<_Derived, DSpinorFieldType, DGaugeFieldType,
                        HasMassShift>,
        DSpinorFieldType, DGaugeFieldType, HasMassShift>::DGaugeFieldType>
        odd(this->g_odd, this->g_in, 1);
    KTune::parallel_for(
        "init_evengaugefield",
        Policy<rank>(IndexArray<rank>{}, this->g_even.dimensions), even);
    KTune::parallel_for(
        "init_oddgaugefield",
        Policy<rank>(IndexArray<rank>{}, this->g_odd.dimensions), odd);
  }
};
}  // namespace klft
