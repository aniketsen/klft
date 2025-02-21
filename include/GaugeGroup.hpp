#pragma once
#include "GLOBAL.hpp"

namespace klft {

  template <typename T>
  struct SU3 {

    Kokkos::Array<Kokkos::complex<T>,9> v;

    static constexpr size_t Nc = 3;
    static constexpr size_t Npar = 9;
    typedef T real_type;

    SU3() = default;

    KOKKOS_INLINE_FUNCTION SU3(const Kokkos::complex<T> &a, const Kokkos::complex<T> &b,
                               const Kokkos::complex<T> &c, const Kokkos::complex<T> &d,
                               const Kokkos::complex<T> &e, const Kokkos::complex<T> &f, 
                               const Kokkos::complex<T> &g, const Kokkos::complex<T> &h,
                               const Kokkos::complex<T> &i) {
      v[0] = a;
      v[1] = b;
      v[2] = c;
      v[3] = d;
      v[4] = e;
      v[5] = f;
      v[6] = g;
      v[7] = h;
      v[8] = i;
    }

    KOKKOS_INLINE_FUNCTION SU3(const Kokkos::complex<T> v_in[9]) {
      v[0] = v_in[0];
      v[1] = v_in[1];
      v[2] = v_in[2];
      v[3] = v_in[3];
      v[4] = v_in[4];
      v[5] = v_in[5];
      v[6] = v_in[6];
      v[7] = v_in[7];
      v[8] = v_in[8];
    }

    KOKKOS_INLINE_FUNCTION SU3(const Kokkos::Array<Kokkos::complex<T>,9> &v_in) {
      v[0] = v_in[0];
      v[1] = v_in[1];
      v[2] = v_in[2];
      v[3] = v_in[3];
      v[4] = v_in[4];
      v[5] = v_in[5];
      v[6] = v_in[6];
      v[7] = v_in[7];
      v[8] = v_in[8];
    }

    KOKKOS_INLINE_FUNCTION SU3(const Kokkos::complex<T> &v_in) {
      v[0] = v_in;
      v[1] = v_in;
      v[2] = v_in;
      v[3] = v_in;
      v[4] = v_in;
      v[5] = v_in;
      v[6] = v_in;
      v[7] = v_in;
      v[8] = v_in;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU3(const SU3<Tin> &in) {
      v[0] = in.v[0];
      v[1] = in.v[1];
      v[2] = in.v[2];
      v[3] = in.v[3];
      v[4] = in.v[4];
      v[5] = in.v[5];
      v[6] = in.v[6];
      v[7] = in.v[7];
      v[8] = in.v[8];
    }

    KOKKOS_INLINE_FUNCTION void set_identity() {
      v[0] = Kokkos::complex<T>(1.0,0.0);
      v[1] = Kokkos::complex<T>(0.0,0.0);
      v[2] = Kokkos::complex<T>(0.0,0.0);
      v[3] = Kokkos::complex<T>(0.0,0.0);
      v[4] = Kokkos::complex<T>(1.0,0.0);
      v[5] = Kokkos::complex<T>(0.0,0.0);
      v[6] = Kokkos::complex<T>(0.0,0.0);
      v[7] = Kokkos::complex<T>(0.0,0.0);
      v[8] = Kokkos::complex<T>(1.0,0.0);
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) {
      return v[i];
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) const {
      return v[i];
    }

    KOKKOS_INLINE_FUNCTION void dagger() {
      Kokkos::Array<Kokkos::complex<T>,3> tmp = {v[1],v[2],v[5]};
      v[0] = Kokkos::conj(v[0]);
      v[1] = Kokkos::conj(v[3]);
      v[2] = Kokkos::conj(v[6]);
      v[3] = Kokkos::conj(tmp[0]);
      v[4] = Kokkos::conj(v[4]);
      v[5] = Kokkos::conj(v[7]);
      v[6] = Kokkos::conj(tmp[1]);
      v[7] = Kokkos::conj(tmp[2]);
      v[8] = Kokkos::conj(v[8]);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator+=(const SU3<Tin> &in) {
      v[0] += in.v[0];
      v[1] += in.v[1];
      v[2] += in.v[2];
      v[3] += in.v[3];
      v[4] += in.v[4];
      v[5] += in.v[5];
      v[6] += in.v[6];
      v[7] += in.v[7];
      v[8] += in.v[8];
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator-=(const SU3<Tin> &in) {
      v[0] -= in.v[0];
      v[1] -= in.v[1];
      v[2] -= in.v[2];
      v[3] -= in.v[3];
      v[4] -= in.v[4];
      v[5] -= in.v[5];
      v[6] -= in.v[6];
      v[7] -= in.v[7];
      v[8] -= in.v[8];
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU3<T> operator+(const SU3<Tin> &in) const {
      SU3<T> tmp(v[0] + in.v[0], v[1] + in.v[1], v[2] + in.v[2], v[3] + in.v[3], v[4] + in.v[4], v[5] + in.v[5], v[6] + in.v[6], v[7] + in.v[7], v[8] + in.v[8]);
      return tmp;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU3<T> operator-(const SU3<Tin> &in) const {
      SU3<T> tmp(v[0] - in.v[0], v[1] - in.v[1], v[2] - in.v[2], v[3] - in.v[3], v[4] - in.v[4], v[5] - in.v[5], v[6] - in.v[6], v[7] - in.v[7], v[8] - in.v[8]);
      return tmp;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator*=(const SU3<Tin> &in) {
      Kokkos::Array<Kokkos::complex<T>,9> tmp{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      #pragma unroll
      for(int i = 0; i < 3; i++) {
        #pragma unroll
        for(int j = 0; j < 3; j++) {
          #pragma unroll
          for(int k = 0; k < 3; k++) {
            tmp[3*i+j] += v[3*i+k]*in.v[3*k+j];
          }
        }
      }
      for(int i = 0; i < 9; i++) {
        v[i] = tmp[i];
      }
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU3<T> operator*(const SU3<Tin> &in) const {
      Kokkos::Array<Kokkos::complex<T>,9> tmp{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      #pragma unroll
      for(int i = 0; i < 3; i++) {
        #pragma unroll
        for(int j = 0; j < 3; j++) {
          #pragma unroll
          for(int k = 0; k < 3; k++) {
            tmp[3*i+j] += v[3*i+k]*in.v[3*k+j];
          }
        }
      }
      SU3<T> out(tmp);
      return out;
    }

    KOKKOS_INLINE_FUNCTION T retrace() const {
      return (v[0] + v[4] + v[8]).real();
    }

    KOKKOS_INLINE_FUNCTION void restoreGauge() {
      T n0 = Kokkos::sqrt((Kokkos::conj(v[0])*v[0] + Kokkos::conj(v[1])*v[1] + Kokkos::conj(v[2])*v[2]).real());
      T n1 = Kokkos::sqrt((Kokkos::conj(v[3])*v[3] + Kokkos::conj(v[4])*v[4] + Kokkos::conj(v[5])*v[5]).real());
      v[0] /= n0;
      v[1] /= n0;
      v[2] /= n0;
      v[3] /= n1;
      v[4] /= n1;
      v[5] /= n1;
      v[6] = Kokkos::conj((v[1]*v[5]) - (v[2]*v[4]));
      v[7] = Kokkos::conj((v[2]*v[3]) - (v[0]*v[5]));
      v[8] = Kokkos::conj((v[0]*v[4]) - (v[1]*v[3]));
      v[3] = Kokkos::conj((v[7]*v[2]) - (v[8]*v[1]));
      v[4] = Kokkos::conj((v[8]*v[0]) - (v[6]*v[2]));
      v[5] = Kokkos::conj((v[6]*v[1]) - (v[7]*v[0]));
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator, T delta) {
      T r1[6],r2[6],norm, fact;
      Kokkos::complex<T> z1[3], z2[3], z3[3], z;
      while(1) {
        for(int i = 0; i < 6; i++) {
          r1[i] = generator.drand(0,delta);
        }
        norm = Kokkos::sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2] + r1[3]*r1[3] + r1[4]*r1[4] + r1[5]*r1[5]);
        if(1.0 != (1.0 + norm)) break;
      }
      fact = 1.0 / norm;
      z1[0] = fact * Kokkos::complex<T>(r1[0],r1[1]);
      z1[1] = fact * Kokkos::complex<T>(r1[2],r1[3]);
      z1[2] = fact * Kokkos::complex<T>(r1[4],r1[5]);
      while(1) {
        while(1) {
          for(int i = 0; i < 6; i++) {
            r2[i] = generator.drand(0,delta);
          }
          norm = Kokkos::sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2] + r2[3]*r2[3] + r2[4]*r2[4] + r2[5]*r2[5]);
          if(1.0 != (1.0 + norm)) break;
        }
        fact = 1.0 / norm;
        z2[0] = fact * Kokkos::complex<T>(r2[0],r2[1]);
        z2[1] = fact * Kokkos::complex<T>(r2[2],r2[3]);
        z2[2] = fact * Kokkos::complex<T>(r2[4],r2[5]);
        z = Kokkos::conj(z1[0])*z2[0] + Kokkos::conj(z1[1])*z2[1] + Kokkos::conj(z1[2])*z2[2];
        z2[0] -= z * z1[0];
        z2[1] -= z * z1[1];
        z2[2] -= z * z1[2];
        norm = Kokkos::sqrt(z2[0].real()*z2[0].real() + z2[0].imag()*z2[0].imag() + z2[1].real()*z2[1].real() + z2[1].imag()*z2[1].imag() + z2[2].real()*z2[2].real() + z2[2].imag()*z2[2].imag());
        if(1.0 != (1.0 + norm)) break;
      }
      fact = 1.0 / norm;
      z2[0] *= fact;
      z2[1] *= fact;
      z2[2] *= fact;
      z3[0] = Kokkos::conj((z1[1]*z2[2]) - (z1[2]*z2[1]));
      z3[1] = Kokkos::conj((z1[2]*z2[0]) - (z1[0]*z2[2]));
      z3[2] = Kokkos::conj((z1[0]*z2[1]) - (z1[1]*z2[0]));
      v[0] = z1[0];
      v[1] = z1[1];
      v[2] = z1[2];
      v[3] = z2[0];
      v[4] = z2[1];
      v[5] = z2[2];
      v[6] = z3[0];
      v[7] = z3[1];
      v[8] = z3[2];
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> det() const {
      return v[0]*(v[4]*v[8] - v[5]*v[7]) - v[1]*(v[3]*v[8] - v[5]*v[6]) + v[2]*(v[3]*v[7] - v[4]*v[6]);
    }

  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION SU3<T> dagger(const SU3<T> &in) {
    SU3<T> out(in);
    out.dagger();
    return out;
  }

  template <typename T>
  struct SU2 {

    Kokkos::complex<T> c00, c01;

    static constexpr size_t Nc = 2;
    static constexpr size_t Npar = 2;
    typedef T real_type;

    SU2() = default;

    KOKKOS_INLINE_FUNCTION SU2(const Kokkos::complex<T> &a, const Kokkos::complex<T> &b) :
      c00(a), c01(b) {}

    KOKKOS_INLINE_FUNCTION SU2(const Kokkos::Array<Kokkos::complex<T>,2> &v_in) :
      c00(v_in[0]), c01(v_in[1]) {}

    KOKKOS_INLINE_FUNCTION SU2(const Kokkos::complex<T> v_in[2]) :
      c00(v_in[0]), c01(v_in[1]) {}

    KOKKOS_INLINE_FUNCTION SU2(const Kokkos::complex<T> &v_in) :
      c00(v_in), c01(v_in) {}

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2(const SU2<Tin> &in) :
      c00(in.c00), c01(in.c01) {}

    KOKKOS_INLINE_FUNCTION void set_identity() {
      c00 = Kokkos::complex<T>(1.0,0.0);
      c01 = Kokkos::complex<T>(0.0,0.0);
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> get_c00() const {
      return c00;
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> get_c01() const {
      return c01;
    }

    KOKKOS_INLINE_FUNCTION void dagger() {
      c00 = Kokkos::conj(c00);
      c01 *= -1.0;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator+=(const SU2<Tin> &in) {
      c00 += in.c00;
      c01 += in.c01;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator-=(const SU2<Tin> &in) {
      c00 -= in.c00;
      c01 -= in.c01;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2<T> operator+(const SU2<Tin> &in) const {
      return SU2<T>(c00 + in.c00, c01 + in.c01);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2<T> operator-(const SU2<Tin> &in) const {
      return SU2<T>(c00 - in.c00, c01 - in.c01);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator*=(const SU2<Tin> &in) {
      T a = c00.real()*in.c00.real() - c00.imag()*in.c00.imag() - c01.real()*in.c01.real() - c01.imag()*in.c01.imag();
      T b = c00.real()*in.c00.imag() + c00.imag()*in.c00.real() + c01.real()*in.c01.imag() - c01.imag()*in.c01.real();
      T c = c00.real()*in.c01.real() - c00.imag()*in.c01.imag() + c01.real()*in.c00.real() + c01.imag()*in.c00.imag();
      T d = c00.real()*in.c01.imag() + c00.imag()*in.c01.real() - c01.real()*in.c00.imag() + c01.imag()*in.c00.real();
      c00 = Kokkos::complex<T>(a,b);
      c01 = Kokkos::complex<T>(c,d);
    }
  
    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2<T> operator*(const SU2<Tin> &in) const {
      T a = c00.real()*in.c00.real() - c00.imag()*in.c00.imag() - c01.real()*in.c01.real() - c01.imag()*in.c01.imag();
      T b = c00.real()*in.c00.imag() + c00.imag()*in.c00.real() + c01.real()*in.c01.imag() - c01.imag()*in.c01.real();
      T c = c00.real()*in.c01.real() - c00.imag()*in.c01.imag() + c01.real()*in.c00.real() + c01.imag()*in.c00.imag();
      T d = c00.real()*in.c01.imag() + c00.imag()*in.c01.real() - c01.real()*in.c00.imag() + c01.imag()*in.c00.real();
      SU2<T> tmp(Kokkos::complex<T>(a,b),Kokkos::complex<T>(c,d));
      return tmp;
    }

    KOKKOS_INLINE_FUNCTION T retrace() const {
      return c00.real()*2.0;
    }

    KOKKOS_INLINE_FUNCTION void restoreGauge() {
      T norm = Kokkos::sqrt(c00.real()*c00.real() + c00.imag()*c00.imag() + c01.real()*c01.real() + c01.imag()*c01.imag());
      c00 /= norm;
      c01 /= norm;
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator, T delta) {
      T alpha = generator.drand(0.0,delta*2*Kokkos::numbers::pi_v<T>);
      T u = generator.drand(-1.0,1.0);
      T theta = generator.drand(0.0,2.0*Kokkos::numbers::pi_v<T>);
      T salpha = Kokkos::sin(alpha);
      T n1 = Kokkos::sqrt(1.0 - u*u)*Kokkos::sin(theta);
      T n2 = Kokkos::sqrt(1.0 - u*u)*Kokkos::cos(theta);
      c00 = Kokkos::complex<T>(Kokkos::cos(alpha),u*salpha);
      c01 = Kokkos::complex<T>(n1*salpha,n2*salpha);
    }
  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION SU2<T> dagger(const SU2<T> &in) {
    return SU2<T>(Kokkos::conj(in.c00),-in.c01);
  }

  template <typename T>
  struct U1 {

    Kokkos::complex<T> v;

    static constexpr size_t Nc = 1;
    static constexpr size_t Npar = 1;
    typedef T real_type;

    U1() = default;

    KOKKOS_INLINE_FUNCTION U1(const Kokkos::complex<T> &a) : v(a) {}

    KOKKOS_INLINE_FUNCTION U1(const U1<T> &in) : v(in.v) {}

    KOKKOS_INLINE_FUNCTION U1(const Kokkos::complex<T> v_in[1]) : v(v_in[0]) {}

    KOKKOS_INLINE_FUNCTION U1(const Kokkos::Array<Kokkos::complex<T>,1> &v_in) : v(v_in[0]) {}

    KOKKOS_INLINE_FUNCTION void set_identity() {
      v = Kokkos::complex<T>(1.0,0.0);
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> get_v() const {
      return v;
    }

    KOKKOS_INLINE_FUNCTION void dagger() {
      v = Kokkos::conj(v);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator+=(const U1<Tin> &in) {
      v += in.v;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator-=(const U1<Tin> &in) {
      v -= in.v;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator+(const U1<Tin> &in) const {
      return U1<T>(v + in.v);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator-(const U1<Tin> &in) const {
      return U1<T>(v - in.v);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator*=(const U1<Tin> &in) {
      T a = v.real()*in.v.real() - v.imag()*in.v.imag();
      T b = v.real()*in.v.imag() + v.imag()*in.v.real();
      v = Kokkos::complex<T>(a,b);
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator*(const U1<Tin> &in) const {
      return U1<T>(Kokkos::complex<T>(v.real()*in.v.real() - v.imag()*in.v.imag(),v.real()*in.v.imag() + v.imag()*in.v.real()));
    }

    KOKKOS_INLINE_FUNCTION T retrace() const {
      return v.real();
    }

    KOKKOS_INLINE_FUNCTION void restoreGauge() {
      v /= Kokkos::sqrt(v.real()*v.real() + v.imag()*v.imag());
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator, T delta) {
      v = Kokkos::exp(Kokkos::complex(0.0,generator.drand(-delta*Kokkos::numbers::pi_v<T>,delta*Kokkos::numbers::pi_v<T>)));
    }
  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION U1<T> dagger(const U1<T> &in) {
    return U1<T>(Kokkos::complex(in.v.real(),-in.v.imag()));
  }

} // namespace klqcd
