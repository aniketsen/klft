#include <iomanip>
#include <iostream>

#include "../../include/GammaMatrix.hpp"

using namespace klft;

#define HLINE "=========================================================\n"
template <size_t RepDim>
void print_matrix(const GammaMat<RepDim> &mat) {
  std::cout << std::fixed;
  for (int i = 0; i < 4; ++i) {
    std::cout << "[ ";
    for (int j = 0; j < 4; ++j) {
      const auto &val = mat(i, j);
      std::cout << "(" << std::setw(6) << std::setprecision(2) << val.real()
                << (val.imag() >= 0 ? "+" : "") << std::setw(6)
                << std::setprecision(2) << val.imag() << "i) ";
    }
    std::cout << "]\n";
  }
  std::cout << "\n";
}

int main(int argc, char const *argv[]) {
  std::cout << HLINE << "Testing Gamma matrices" << HLINE;
  GammaMat<4> ggamma0 = get_gamma0();
  GammaMat<4> ggamma1 = get_gamma1();
  GammaMat<4> ggamma2 = get_gamma2();
  GammaMat<4> ggamma3 = get_gamma3();
  GammaMat<4> ggamma5 = get_gamma5();
  const auto vec_gamma = get_gammas<4>();
  // print_matrix(ggamma0 * ggamma3 + ggamma3 * ggamma0);
  std::cout << "Gamma0:\n";
  print_matrix(ggamma0);
  std::cout << "Gamma1:\n";
  print_matrix(ggamma1);
  std::cout << "Gamma2:\n";
  print_matrix(ggamma2);
  std::cout << "Gamma3:\n";
  print_matrix(ggamma3);
  std::cout << "Gamma5 via multiplication:\n";
  print_matrix(ggamma0 * ggamma1 * ggamma2 * ggamma3);
  std::cout << "Gamma5:\n";
  print_matrix(ggamma5);
  std::cout << HLINE << "Testing Multiplication" << HLINE;
  std::cout << "gamma0*Gamma1*gamma2*gamma3\n";
  print_matrix(ggamma0 * ggamma1 * ggamma2 * ggamma3);
  std::cout << "gamma0*Gamma1*gamma2*gamma3 == gamma5:\n";

  std::cout << bool(ggamma0 * ggamma1 * ggamma2 * ggamma3 == ggamma5) << "\n";
  std::cout << HLINE << "testing get_gammas\n" << HLINE;
  std::cout << "Gamma0:\n";
  std::cout << (vec_gamma[0] == ggamma0) << "\n";
  std::cout << "Gamma1:\n";
  std::cout << (vec_gamma[1] == ggamma1) << "\n";
  std::cout << "Gamma2:\n";
  std::cout << (vec_gamma[2] == ggamma2) << "\n";
  std::cout << "Gamma3:\n";
  std::cout << (vec_gamma[3] == ggamma3) << "\n";
  /* code */
  return 0;
}