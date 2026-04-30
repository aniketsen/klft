#include "GLOBAL.hpp"
namespace klft {
// Correct datalayout of spinor?

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION index_t get_linear_index(const index_t& color,
                                                     const index_t& dirac) {
  return dirac * Nc + color;
}
template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::pair<index_t, index_t>
get_color_dirac_index(const index_t& idx) {
  index_t dirac = idx / Nc;
  index_t color = idx % Nc;
  return {dirac, color};
}
template <size_t Nc, size_t RepDim>
KOKKOS_FORCEINLINE_FUNCTION void add_inplace(PropagatorMatrix<Nc, RepDim>& mat,
                                             const Spinor<Nc, RepDim>& spinor,
                                             const index_t& alpha0) {
#pragma unroll
  for (size_t beta = 0; beta < Nc * RepDim; beta++) {
    auto idx = get_color_dirac_index<Nc>(beta);
    mat[beta][alpha0] +=
        spinor[idx.first][idx.second];  // index with alpha or beta here?
  }
}
}  // namespace klft
