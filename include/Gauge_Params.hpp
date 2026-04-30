#pragma once

namespace klft {

template <class Group, class Adjoint>
class Gauge_Params {
 public:
  int Nc;
  int Ndim;
  int LX, LY, LZ, LT;
  typedef Group gauge_group_t;
  typedef Adjoint adjoint_group_t;
  Gauge_Params() : Nc(3), Ndim(4), LX(4), LY(4), LZ(4), LT(4) {}
  Gauge_Params(int _Nc, int _Ndim, int _LX, int _LY, int _LZ, int _LT)
      : Nc(_Nc), Ndim(_Ndim), LX(_LX), LY(_LY), LZ(_LZ), LT(_LT) {}
};

}  // namespace klft