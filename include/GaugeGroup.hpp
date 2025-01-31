#pragma once
#include "GLOBAL.hpp"

namespace klft {

  template <typename T>
  struct SU3 {

    Kokkos::Array<Kokkos::complex<T>,9> v;

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

    // KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) {
    //   return v[i];
    // }

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
      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
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
      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
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
      // T norm = Kokkos::sqrt((Kokkos::conj(v[0])*v[0] + Kokkos::conj(v[1])*v[1] + Kokkos::conj(v[2])*v[2]).real());
      // v[0] /= norm;
      // v[1] /= norm;
      // v[2] /= norm;
      // Kokkos::complex<T> z = Kokkos::conj(v[0])*v[3] + Kokkos::conj(v[1])*v[4] + Kokkos::conj(v[2])*v[5];
      // v[3] -= z * v[0];
      // v[4] -= z * v[1];
      // v[5] -= z * v[2];
      // norm = Kokkos::sqrt((Kokkos::conj(v[3])*v[3] + Kokkos::conj(v[4])*v[4] + Kokkos::conj(v[5])*v[5]).real());
      // v[3] /= norm;
      // v[4] /= norm;
      // v[5] /= norm;
      // v[6] = Kokkos::conj((v[1]*v[5]) - (v[5]*v[1]));
      // v[7] = Kokkos::conj((v[2]*v[3]) - (v[3]*v[2]));
      // v[8] = Kokkos::conj((v[0]*v[4]) - (v[4]*v[0]));
      return;
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

    Kokkos::Array<Kokkos::complex<T>,4> v;

    SU2() = default;

    KOKKOS_INLINE_FUNCTION SU2(const Kokkos::complex<T> &a, const Kokkos::complex<T> &b, const Kokkos::complex<T> &c, const Kokkos::complex<T> &d) {
      v[0] = a;
      v[1] = b;
      v[2] = c;
      v[3] = d;
    }

    KOKKOS_INLINE_FUNCTION SU2(const Kokkos::Array<Kokkos::complex<T>,4> &v_in) {
      v[0] = v_in[0];
      v[1] = v_in[1];
      v[2] = v_in[2];
      v[3] = v_in[3];
    }

    KOKKOS_INLINE_FUNCTION SU2(const Kokkos::complex<T> v_in[4]) {
      v[0] = v_in[0];
      v[1] = v_in[1];
      v[2] = v_in[2];
      v[3] = v_in[3];
    }

    KOKKOS_INLINE_FUNCTION SU2(const Kokkos::complex<T> &v_in) {
      v[0] = v_in;
      v[1] = v_in;
      v[2] = v_in;
      v[3] = v_in;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2(const SU2<Tin> &in) {
      v[0] = in.v[0];
      v[1] = in.v[1];
      v[2] = in.v[2];
      v[3] = in.v[3];
    }

    KOKKOS_INLINE_FUNCTION void set_identity() {
      v[0] = Kokkos::complex<T>(1.0,0.0);
      v[1] = Kokkos::complex<T>(0.0,0.0);
      v[2] = Kokkos::complex<T>(0.0,0.0);
      v[3] = Kokkos::complex<T>(1.0,0.0);
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) {
      return v[i];
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) const {
      return v[i];
    }

    KOKKOS_INLINE_FUNCTION void dagger() {
      v[0].imag(-v[0].imag());
      v[1] *= -1.0;
      v[2] *= -1.0;
      v[3].imag(-v[3].imag());
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator+=(const SU2<Tin> &in) {
      v[0] += in.v[0];
      v[1] += in.v[1];
      v[2] += in.v[2];
      v[3] += in.v[3];
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator-=(const SU2<Tin> &in) {
      v[0] -= in.v[0];
      v[1] -= in.v[1];
      v[2] -= in.v[2];
      v[3] -= in.v[3];
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2<T> operator+(const SU2<Tin> &in) const {
      SU2<T> tmp(v[0] + in.v[0], v[1] + in.v[1], v[2] + in.v[2], v[3] + in.v[3]);
      return tmp;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2<T> operator-(const SU2<Tin> &in) const {
      SU2<T> tmp(v[0] - in.v[0], v[1] - in.v[1], v[2] - in.v[2], v[3] - in.v[3]);
      return tmp;
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION void operator*=(const SU2<Tin> &in) {
      T a = v[0].real()*in.v[0].real() - v[0].imag()*in.v[0].imag() - v[1].real()*in.v[1].real() - v[1].imag()*in.v[1].imag();
      T b = v[0].real()*in.v[0].imag() + v[0].imag()*in.v[0].real() + v[1].real()*in.v[1].imag() - v[1].imag()*in.v[1].real();
      T c = v[0].real()*in.v[1].real() - v[0].imag()*in.v[1].imag() + v[1].real()*in.v[0].real() + v[1].imag()*in.v[0].imag();
      T d = v[0].real()*in.v[1].imag() + v[0].imag()*in.v[1].real() - v[1].real()*in.v[0].imag() + v[1].imag()*in.v[0].real();
      v[0] = Kokkos::complex<T>(a,b);
      v[1] = Kokkos::complex<T>(c,d);
      v[2] = Kokkos::complex<T>(-c,d);
      v[3] = Kokkos::complex<T>(a,-b);
    }
  
    template <typename Tin>
    KOKKOS_INLINE_FUNCTION SU2<T> operator*(const SU2<Tin> &in) const {
      T a = v[0].real()*in.v[0].real() - v[0].imag()*in.v[0].imag() - v[1].real()*in.v[1].real() - v[1].imag()*in.v[1].imag();
      T b = v[0].real()*in.v[0].imag() + v[0].imag()*in.v[0].real() + v[1].real()*in.v[1].imag() - v[1].imag()*in.v[1].real();
      T c = v[0].real()*in.v[1].real() - v[0].imag()*in.v[1].imag() + v[1].real()*in.v[0].real() + v[1].imag()*in.v[0].imag();
      T d = v[0].real()*in.v[1].imag() + v[0].imag()*in.v[1].real() - v[1].real()*in.v[0].imag() + v[1].imag()*in.v[0].real();
      SU2<T> tmp(Kokkos::complex<T>(a,b),Kokkos::complex<T>(c,d),Kokkos::complex<T>(-c,d),Kokkos::complex<T>(a,-b));
      return tmp;
    }

    KOKKOS_INLINE_FUNCTION T retrace() const {
      return v[0].real()*2.0;
    }

    KOKKOS_INLINE_FUNCTION void restoreGauge() {
      T norm = Kokkos::sqrt(v[0].real()*v[0].real() + v[0].imag()*v[0].imag() + v[1].real()*v[1].real() + v[1].imag()*v[1].imag());
      v[0] /= norm;
      v[1] /= norm;
      v[2] /= norm;
      v[3] /= norm;
    }

    template <class RNG>
    KOKKOS_INLINE_FUNCTION void get_random(RNG &generator, T delta) {
      T alpha = generator.drand(0.0,delta*2*Kokkos::numbers::pi_v<T>);
      T u = generator.drand(-1.0,1.0);
      T theta = generator.drand(0.0,2.0*Kokkos::numbers::pi_v<T>);
      T salpha = Kokkos::sin(alpha);
      T n1 = Kokkos::sqrt(1.0 - u*u)*Kokkos::sin(theta);
      T n2 = Kokkos::sqrt(1.0 - u*u)*Kokkos::cos(theta);
      v[0] = Kokkos::complex<T>(Kokkos::cos(alpha),u*salpha);
      v[1] = Kokkos::complex<T>(n1*salpha,n2*salpha);
      v[2] = Kokkos::complex<T>(-v[1].real(),v[1].imag());
      v[3] = Kokkos::complex<T>(v[0].real(),-v[0].imag());
    }
  };

  template <typename T>
  KOKKOS_INLINE_FUNCTION SU2<T> dagger(const SU2<T> &in) {
    SU2<T> tmp(in);
    tmp.dagger();
    return tmp;
  }

  // template <typename T, class GaugeGroup, class RNG>
  // KOKKOS_INLINE_FUNCTION GaugeGroup get_random(RNG &generator, T delta) {}

  // template <typename T, class RNG>
  // KOKKOS_INLINE_FUNCTION SU2<T> get_random(RNG &generator, T delta) {
  //   T alpha = generator.drand(0.0,delta*2*Kokkos::numbers::pi_v<T>);
  //   T u = generator.drand(-1.0,1.0);
  //   T theta = generator.drand(0.0,2.0*Kokkos::numbers::pi_v<T>);
  //   T salpha = Kokkos::sin(alpha);
  //   T n1 = Kokkos::sqrt(1.0 - u*u)*Kokkos::sin(theta);
  //   T n2 = Kokkos::sqrt(1.0 - u*u)*Kokkos::cos(theta);
  //   SU2<T> R(Kokkos::cos(alpha),u*salpha,n1*salpha,n2*salpha);
  //   return R;
  // }

  template <typename T>
  struct U1 {

    Kokkos::complex<T> v;

    U1() = default;

    KOKKOS_INLINE_FUNCTION U1(const Kokkos::complex<T> &a) {
      v = a;
    }

    KOKKOS_INLINE_FUNCTION U1(const U1<T> &in) {
      v = in.v;
    }

    KOKKOS_INLINE_FUNCTION U1(const Kokkos::complex<T> v_in[1]) {
      v = v_in[0];
    }

    KOKKOS_INLINE_FUNCTION U1(const Kokkos::Array<Kokkos::complex<T>,1> &v_in) {
      v = v_in[0];
    }

    KOKKOS_INLINE_FUNCTION void set_identity() {
      v = Kokkos::complex<T>(1.0,0.0);
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) {
      return v;
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<T> operator()(const int &i) const {
      return v;
    }

    KOKKOS_INLINE_FUNCTION void dagger() {
      v.imag(-v.imag());
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
      v.real(v.real()*in.v.real() - v.imag()*in.v.imag());
      v.imag(v.real()*in.v.imag() + v.imag()*in.v.real());
    }

    template <typename Tin>
    KOKKOS_INLINE_FUNCTION U1<T> operator*(const U1<Tin> &in) const {
      return U1<T>(Kokkos::complex<T>(v.real()*in.v.real() - v.imag()*in.v.imag(),v.real()*in.v.imag() + v.imag()*in.v.real()));
    }

    KOKKOS_INLINE_FUNCTION T retrace() const {
      return v.real();
    }

    KOKKOS_INLINE_FUNCTION void restoreGauge() {
      T norm = Kokkos::sqrt(v.real()*v.real() + v.imag()*v.imag());
      if(Kokkos::abs(norm-0.0) > 1e-12) v /= Kokkos::sqrt(v.real()*v.real() + v.imag()*v.imag());
      else v = Kokkos::complex<T>(0.0,0.0);
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

  // template <typename T, class G, class RNG, std::enable_if_t<std::is_same<G,U1<T>>::value,int> = 0>
  // KOKKOS_INLINE_FUNCTION U1<T> get_random(RNG &generator, T delta) {
  //   return U1<T>(generator.drand(-delta*Kokkos::numbers::pi_v<T>,delta*Kokkos::numbers::pi_v<T>));
  // }

} // namespace klqcd
