#pragma once
#include "GLOBAL.hpp"
#include "SUN.hpp"

#define SQRT3 1.7320508075688772936
#define SQRT3INV 0.5773502691896257645

namespace klft {

// template <typename T> struct SUNAdjTraits;
// template <size_t _Nc> struct SUNAdjTraits<SUNAdj<_Nc>> {
//   static constexpr size_t Nc = _Nc;
// };

template <size_t Nc, typename Tin>
KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc> operator*(const SUNAdj<Nc> &a,
                                                 const Tin &b) {
  SUNAdj<Nc> c;
#pragma unroll
  for (size_t i = 0; i < NcAdj<Nc>; ++i) {
    c[i] = a[i] * b;
  }
  return c;
}

template <size_t Nc, typename Tin>
KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc> operator*(const Tin &b,
                                                 const SUNAdj<Nc> &a) {
  return a * b;
}

template <size_t Nc, typename Tin>
KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc> multSUNAdj(const SUNAdj<Nc> &a,
                                                  const Tin &b) {
  SUNAdj<Nc> c;
#pragma unroll
  for (size_t i = 0; i < NcAdj<Nc>; ++i) {
    c[i] = a[i] * b;
  }
  return c;
}

template <size_t Nc, typename Tin>
KOKKOS_FORCEINLINE_FUNCTION void operator*=(SUNAdj<Nc> &a, const Tin &b) {
#pragma unroll
  for (size_t i = 0; i < NcAdj<Nc>; ++i) {
    a[i] *= b;
  }
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc> operator+(const SUNAdj<Nc> &a,
                                                 const SUNAdj<Nc> &b) {
  SUNAdj<Nc> c;
#pragma unroll
  for (size_t i = 0; i < NcAdj<Nc>; ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION void operator+=(SUNAdj<Nc> &a,
                                            const SUNAdj<Nc> &b) {
#pragma unroll
  for (size_t i = 0; i < NcAdj<Nc>; ++i) {
    a[i] += b[i];
  }
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION SUNAdj<Nc> operator-(const SUNAdj<Nc> &a,
                                                 const SUNAdj<Nc> &b) {
  SUNAdj<Nc> c;
#pragma unroll
  for (size_t i = 0; i < NcAdj<Nc>; ++i) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION void operator-=(SUNAdj<Nc> &a,
                                            const SUNAdj<Nc> &b) {
#pragma unroll
  for (size_t i = 0; i < NcAdj<Nc>; ++i) {
    a[i] -= b[i];
  }
}

template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION real_t norm2(const SUNAdj<Nc> &a) {
  real_t c = 0.0;
#pragma unroll
  for (size_t i = 0; i < NcAdj<Nc>; ++i) {
    c += a[i] * a[i];
  }
  return c;
}

// random SUNAdj matrix generator
template <size_t Nc, class RNG>
KOKKOS_FORCEINLINE_FUNCTION void randSUNAdj(SUNAdj<Nc> &r, RNG &generator) {
#pragma unroll
  for (size_t i = 0; i < NcAdj<Nc>; ++i) {
    r[i] = generator.normal(0.0, 1.0);
  }
}

// get the adjoint from an SU(N) matrix
// nneds to be defined for each Nc
KOKKOS_FORCEINLINE_FUNCTION
SUNAdj<1> traceT(const SUN<1> &a) {
  SUNAdj<1> c;
  c[0] = a[0][0].imag();
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION
SUNAdj<2> traceT(const SUN<2> &a) {
  SUNAdj<2> c;
  c[0] = 2.0 * a[0][1].imag();
  c[1] = 2.0 * a[0][1].real();
  c[2] = 2.0 * a[0][0].imag();
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION
SUNAdj<3> traceT(const SUN<3> &a) {
  SUNAdj<3> c;
  c[0] = 0.5 * (-a[0][1].imag() - a[1][0].imag());
  c[1] = 0.5 * (a[1][0].real() - a[0][1].real());
  c[2] = 0.5 * (a[1][1].imag() - a[0][0].imag());
  c[3] = 0.5 * (-a[0][2].imag() - a[2][0].imag());
  c[4] = 0.5 * (a[2][0].real() - a[0][2].real());
  c[5] = 0.5 * (-a[2][1].imag() - a[1][2].imag());
  c[6] = 0.5 * (a[2][1].real() - a[1][2].real());
  c[7] = 0.5 *
         ((-a[0][0].imag() - a[1][1].imag() + 2.0 * a[2][2].imag()) * SQRT3INV);
  return c;
}

// exponential of an adjoint matrix
// needs to be defined for each Nc
KOKKOS_FORCEINLINE_FUNCTION
SUN<1> expoSUN(const SUNAdj<1> &a) {
  SUN<1> c;
  c[0][0] = complex_t(Kokkos::cos(a[0]), Kokkos::sin(a[0]));
  return c;
}

KOKKOS_FORCEINLINE_FUNCTION
SUN<2> expoSUN(const SUNAdj<2> &a) {
  const real_t alpha = Kokkos::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const Kokkos::Array<real_t, 3> u = {a[0] / alpha, a[1] / alpha, a[2] / alpha};
  const real_t sin_alpha = Kokkos::sin(alpha);
  SUN<2> c;
  c[0][0] = complex_t(Kokkos::cos(alpha), u[2] * sin_alpha);
  c[0][1] = complex_t(u[1] * sin_alpha, u[0] * sin_alpha);
  c[1][0] = complex_t(-c[0][1].real(), c[0][1].imag());
  c[1][1] = complex_t(c[0][0].real(), -c[0][0].imag());
  return c;
}

// need some global variables for the SU(3) exponential
// check section 5 of https://luscher.web.cern.ch/luscher/notes/su3fcts.pdf
// number of iterations in Cayley-Hamilton expansion
// based on precision of real_t
// constexpr size_t n_iter_expoSU3 = []() {
//   real_t fact = 1.0;
//   size_t n_iter = 0;
//   while (fact > std::numeric_limits<real_t>::epsilon()) {
//     ++n_iter;
//     fact *= 1.0 / static_cast<real_t>(n_iter);
//   }
//   // add 7 (following tmLQCD) to be safe
//   n_iter += 7;
//   // round up to the next even number
//   n_iter += n_iter % 2;
//   return n_iter;
// }();
// // we also need to setup the expansion coefficients
// constexpr Kokkos::Array<real_t, n_iter_expoSU3 + 1> coeffs_expoSU3 = []() {
//   Kokkos::Array<real_t, n_iter_expoSU3 + 1> coeffs;
//   coeffs[0] = 1.0;
//   for (size_t i = 0; i < n_iter_expoSU3; ++i) {
//     coeffs[i + 1] = coeffs[i] / static_cast<real_t>(i);
//   }
//   return coeffs;
// }();
//
// // a function to generate SU(3) matrix from adjoint
// constexpr KOKKOS_FORCEINLINE_FUNCTION SUN<3>
// get_SU3_from_adj(const SUNAdj<3> &a) {
//   SUN<3> c;
//   c[0][0] = complex_t(0.0, 0.5 * (SQRT3INV * a[7] + a[2]));
//   c[0][1] = 0.5 * complex_t(a[1], a[0]);
//   c[0][2] = 0.5 * complex_t(a[4], a[3]);
//   c[1][0] = 0.5 * complex_t(-a[1], a[0]);
//   c[1][1] = complex_t(0.0, 0.5 * (SQRT3INV * a[7] - a[2]));
//   c[1][2] = 0.5 * complex_t(a[6], a[5]);
//   c[2][0] = 0.5 * complex_t(-a[4], a[3]);
//   c[2][1] = 0.5 * complex_t(-a[6], a[5]);
//   c[2][2] = complex_t(0.0, -SQRT3INV * a[7]);
//   return c;
// }
//
// // we also need the determinant of the SU(3) matrix
// // returns i det(a)
// constexpr KOKKOS_FORCEINLINE_FUNCTION real_t imag_det_SU3(const SUNAdj<3> &a)
// {
//   real_t d = -2.0 * SQRT3INV * a[7] * (a[7] * a[7] / 3.0 - a[2] * a[2]) -
//              2.0 * (a[1] * a[3] * a[6] - a[0] * a[3] * a[5] -
//                     a[1] * a[4] * a[5] - a[0] * a[4] * a[6]);
//   d -= (SQRT3INV * a[7] - a[2]) * (a[3] * a[3] + a[4] * a[4]) -
//        (SQRT3INV * a[7] + a[2]) * (a[5] * a[5] + a[6] * a[6]) +
//        2.0 * SQRT3INV * a[7] * (a[0] * a[0] + a[1] * a[1]);
//   return d;
// }
//
// KOKKOS_FORCEINLINE_FUNCTION
// SUN<3> expoSUN(const SUNAdj<3> &a) {
//   // Cayley-Hamilton expansion
//   // exp(X) = p0 + p1 X + p2 X^2
//   // first we need to ensure numerical stability
//   // expansion is well-behaved for ||X||_2 <= 1
//   // so we evaluate exp(X/2^m) and obtain the final
//   // result by squaring m times
//   // X = a_i * T_i
//   SUN<3> X = get_SU3_from_adj(a);
//   // X2 = X * X
//   SUN<3> X2 = X * X;
//   // tc = -2 * trace(X2)
//   // ||X|| = sqrt{(X,X)}
//   // (X,X) = -2 * trace(X^2)
//   real_t tc = -2.0 * trace(X2).real();
//   // store a in a temporary variable
//   // to perform the numerical stability step
//   // a_tmp = a
//   SUNadj<3> a_tmp = a;
//   // mm stores the number of times we need to
//   // multiply by 0.5
//   size_t mm = 0;
//   // do a_tmp *= 0.5 untill tc <= 1.0
//   while (tc > 1.0) {
//     a_tmp *= 0.5;
//     tc *= 0.5;
//     ++mm;
//   }
//   // get the new SU(3) matrix
//   X = get_SU3_from_adj(a_tmp);
//   // X2 = X * X
//   X2 = X * X;
//   // t = -tr(X^2)/2
//   complex_t t = -0.5 * trace(X2);
//   // d = i det(X)
//   real_t d = imag_det_SU3(a_tmp);
//   // now we can compute the exponential
//   // q_{N,0} = c_N
//   real_t p0 = coeffs_expoSU3[n_iter_expoSU3];
//   // q_{N,1} = q_{N,2} = 0
//   real_t p1 = 0.0;
//   real_t p2 = 0.0;
//   real_t q0, q1, q2;
//   // iteration going from N-1 to 0
//   // p_k = q_{0,k}
//   for (size_t i = (n_iter_expoSU3 - 1); i >= 0; --i) {
//     q0 = p0;
//     q1 = p1;
//     q2 = p2;
//     // q_{n,0} = c_n - i d q_{n+1,2}
//     p0 = coeffs_expoSU3[i] - complex_t(0.0, d) * q2;
//     // q_{n,1} = q_{n+1,0} - t q_{n+1,2}
//     p1 = q0 - t * q2;
//     // q_{n,2} = q_{n+1,1}
//     p2 = q1;
//   }
//   // final expansion
//   // p(X) = p0 + p1 X + p2 X^2
//   SUN<3> pt;
//   pt[0][0] = p0 + p1 * X[0][0] + p2 * X2[0][0];
//   pt[0][1] = p1 * X[0][1] + p2 * X2[0][1];
//   pt[0][2] = p1 * X[0][2] + p2 * X2[0][2];
//   pt[1][0] = p1 * X[1][0] + p2 * X2[1][0];
//   pt[1][1] = p0 + p1 * X[1][1] + p2 * X2[1][1];
//   pt[1][2] = p1 * X[1][2] + p2 * X2[1][2];
//   pt[2][0] = p1 * X[2][0] + p2 * X2[2][0];
//   pt[2][1] = p1 * X[2][1] + p2 * X2[2][1];
//   pt[2][2] = p0 + p1 * X[2][2] + p2 * X2[2][2];
//   // finally square m times to get the final result
//   for (size_t i = 0; i < mm; ++i) {
//     X2 = pt * pt;
//     pt = X2;
//   }
//   // return the result
//   return pt;
// }

} // namespace klft
