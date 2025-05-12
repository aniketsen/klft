#include <assert.h>

#include <iomanip>

#include "../../include/GammaMatrix.hpp"
#include "GLOBAL.hpp"
#include "Spinor.hpp"

using namespace klft;

#define HLINE "=========================================================\n"

template <size_t Nc, size_t Nd>
void print_spinor(const Spinor<Nc, Nd> &s, const char *name = "Spinor")
{
  printf("%s:\n", name);
  for (size_t c = 0; c < Nc; ++c)
  {
    printf("  Color %zu:\n", c);
    for (size_t d = 0; d < Nd; ++d)
    {
      double re = s[c][d].real();
      double im = s[c][d].imag();
      printf("    [%zu] = (% .6f, % .6f i)\n", d, re, im);
    }
  }
}

int main(int argc, char const *argv[])
{
  printf(HLINE);
  printf("Testing Spinor GammaMatrix interaction\n");
  printf(HLINE);
  printf("Instantiate One spinor\n");
  auto ospinor = oneSpinor<3, 4>();
  print_spinor(ospinor);
  printf(HLINE);
  printf("Instantiate zero Spinor\n");
  auto zSpinor = zeroSpinor<3, 4>();
  printf(HLINE);
  printf("Testing Subtracting and Scalar multiplication:\n");
  assert(0 * ospinor == zSpinor);
  assert(zSpinor == ospinor - ospinor);
  printf(HLINE);
  printf("Testing scalar multiplication and addition\n");
  print_spinor(2 * ospinor);
  assert(2 * ospinor == ospinor + ospinor);
  printf(HLINE);
  printf(HLINE);
  printf("Testing Gamma*Spinor: \n");
  Spinor<3, 4> id = zeroSpinor<3, 4>();
#pragma unroll
  for (size_t i = 0; i < 3; ++i)
  {
#pragma unroll
    for (size_t j = 0; j < 4; ++j)
    {
      id[i][j] = complex_t(i * 4 + j, 0.0);
    }
  }
  get_gamma1();
  assert(get_gamma5() * id ==
         get_gamma0() * get_gamma1() * get_gamma2() * get_gamma3() * id);
  printf(HLINE);
  printf("Finished\n");
  return 0;
}