//******************************************************************************/
//
// This file is part of the Kokkos Lattice Field Theory (KLFT) library.
//
// KLFT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KLFT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with KLFT.  If not, see <http://www.gnu.org/licenses/>.
//
//******************************************************************************/

// this file defines helper functions for index manipulation

#pragma once
#include "GLOBAL.hpp"

namespace klft {

// return x + shift  mu
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
shift_index_plus(const Kokkos::Array<indexType, rank>& idx,
                 const index_t mu,
                 const index_t shift,
                 const IndexArray<rank>& dimensions) {
  // make sure mu makes sense
  assert(mu < rank && mu >= 0);
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = static_cast<index_t>(idx[i]);
  }
  new_idx[mu] = (idx[mu] + shift) % dimensions[mu];
  return new_idx;
}

// return x - shift mu
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
shift_index_minus(const Kokkos::Array<indexType, rank>& idx,
                  const index_t mu,
                  const index_t shift,
                  const IndexArray<rank>& dimensions) {
  // make sure mu makes sense
  assert(mu < rank && mu >= 0);
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = static_cast<index_t>(idx[i]);
  }
  new_idx[mu] = (idx[mu] - shift + dimensions[mu]) % dimensions[mu];
  return new_idx;
}

// return index based on odd/even sublattice
// this does not check if the index is valid
// it is assumed that all of idx is
// less than half of the dimensional extents
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
index_odd_even(const Kokkos::Array<indexType, rank>& idx,
               const Kokkos::Array<bool, rank>& oddeven) {
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = oddeven[i] ? static_cast<index_t>(2 * idx[i] + 1)
                            : static_cast<index_t>(2 * idx[i]);
  }
  return new_idx;
}

// return an array of boolean values
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<bool, rank> oddeven_array(
    const indexType& val) {
  Kokkos::Array<bool, rank> oddeven;
  for (index_t i = 0; i < rank; ++i) {
    oddeven[rank - 1 - i] = (val & (1 << i)) != 0;
  }
  return oddeven;
}

// shifts index by shift*mu and gives scaler for bc
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION
    Kokkos::pair<Kokkos::Array<index_t, rank>, real_t>
    shift_index_plus_bc(const Kokkos::Array<indexType, rank>& idx,
                        const index_t mu,
                        const index_t shift,
                        const index_t bc_dim,
                        const real_t bc_value,
                        const IndexArray<rank>& dimensions) {
  // make sure mu makes sense
  assert(mu < rank && mu >= 0);
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = static_cast<index_t>(idx[i]);
  }
  new_idx[mu] = (idx[mu] + shift) % dimensions[mu];
  real_t sign = (mu == bc_dim && idx[mu] > new_idx[mu]) ? bc_value : 1.0;
  return {new_idx, sign};
}

// shifts index by shift*mu and gives scaler for bc
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION
    Kokkos::pair<Kokkos::Array<index_t, rank>, real_t>
    shift_index_minus_bc(const Kokkos::Array<indexType, rank>& idx,
                         const index_t mu,
                         const index_t shift,
                         const index_t bc_dim,
                         const real_t bc_value,
                         const IndexArray<rank>& dimensions) {
  // make sure mu makes sense
  assert(mu < rank && mu >= 0);
  Kokkos::Array<index_t, rank> new_idx;
#pragma unroll
  for (index_t i = 0; i < rank; ++i) {
    new_idx[i] = static_cast<index_t>(idx[i]);
  }
  new_idx[mu] = (idx[mu] - shift + dimensions[mu]) % dimensions[mu];
  real_t sign = (mu == bc_dim && idx[mu] < new_idx[mu]) ? bc_value : 1.0;
  return {new_idx, sign};
}

// Index helper for even odd spinor field

// returns index of half field and parity (dont no if i need that) based on full
// field index, assuming that e/o around x axis x axis is 0 index
//
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION
    Kokkos::pair<Kokkos::Array<index_t, rank>, int>
    index_full_to_half(const Kokkos::Array<indexType, rank>& idx) {
  Kokkos::Array<index_t, rank> new_idx{};
  // starting at index 0 is on purpose
#pragma unroll
  for (int i = 1; i < rank; ++i) {
    auto temp = static_cast<index_t>(idx[i]);
    new_idx[0] += temp;
    new_idx[i] = temp;
  }
  auto parity = (new_idx[0] + idx[0]) & 1;
  new_idx[0] = (idx[0] - ((new_idx[0] + parity) & 1)) >> 1;
  return {new_idx, static_cast<int>(parity)};
}

/// @brief Index Helper function with calculates the Full index based on the
/// half field index and parity
/// @param idx the input index to be converted
/// @param parity native parity of idx
/// @return Index array of the full index
template <size_t rank, typename indexType>
constexpr KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<index_t, rank>
index_half_to_full(const Kokkos::Array<indexType, rank>& idx,
                   const int& parity) {
  Kokkos::Array<index_t, rank> new_idx{};
  // using 0 entry as temp storage for the sum of all other dims
#pragma unroll
  for (index_t i = 1; i < rank; i++) {
    auto temp = static_cast<index_t>(idx[i]);
    new_idx[0] += temp;
    new_idx[i] = temp;
  }
  new_idx[0] = 2 * idx[0] + ((new_idx[0] + parity) & 1);

  return new_idx;
}

}  // namespace klft
