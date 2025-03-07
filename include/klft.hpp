#include "GLOBAL.hpp"

namespace klft {

  template <typename T>
  void Metropolis_SU2_4D(const int &LX, const int &LY, const int &LZ, const int &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);
  
  template <typename T>
  void Metropolis_SU2_3D(const int &LX, const int &LY, const int &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);

  template <typename T>
  void Metropolis_SU2_2D(const int &LX, const int &LT,
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);

  template <typename T>
  void Metropolis_U1_4D(const int &LX, const int &LY, const int &LZ, const int &LT, 
                        const size_t &n_hit, const T &beta, const T &delta,
                        const size_t &seed, const size_t &n_sweep, const bool cold_start,
                        const std::string &outfilename);

  template <typename T>
  void Metropolis_U1_3D(const int &LX, const int &LY, const int &LT, 
                        const size_t &n_hit, const T &beta, const T &delta,
                        const size_t &seed, const size_t &n_sweep, const bool cold_start,
                        const std::string &outfilename);

  template <typename T>
  void Metropolis_U1_2D(const int &LX, const int &LT,
                        const size_t &n_hit, const T &beta, const T &delta,
                        const size_t &seed, const size_t &n_sweep, const bool cold_start,
                        const std::string &outfilename);

  template <typename T>
  void Metropolis_SU3_4D(const int &LX, const int &LY, const int &LZ, const int &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);

  template <typename T>
  void Metropolis_SU3_3D(const int &LX, const int &LY, const int &LT, 
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);

  template <typename T>
  void Metropolis_SU3_2D(const int &LX, const int &LT,
                         const size_t &n_hit, const T &beta, const T &delta,
                         const size_t &seed, const size_t &n_sweep, const bool cold_start,
                         const std::string &outfilename);

  template <typename T>
  void HMC_SU2_4D(const int &LX, const int &LY, const int &LZ, const int &LT,
                  const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                  const size_t &seed, const std::string &outfilename);

  template <typename T>
  void HMC_SU2_3D(const int &LX, const int &LY, const int &LT,
                  const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                  const size_t &seed, const std::string &outfilename);

  template <typename T>
  void HMC_SU2_2D(const int &LX, const int &LT,
                  const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                  const size_t &seed, const std::string &outfilename);

  template <typename T>
  void HMC_U1_4D(const int &LX, const int &LY, const int &LZ, const int &LT,
                 const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                 const size_t &seed, const std::string &outfilename);

  template <typename T>
  void HMC_U1_3D(const int &LX, const int &LY, const int &LT,
                 const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                 const size_t &seed, const std::string &outfilename);

  template <typename T>
  void HMC_U1_2D(const int &LX, const int &LT,
                 const size_t &n_traj, const size_t &n_steps, const T &tau, const T &beta,
                 const size_t &seed, const std::string &outfilename);
                        
}