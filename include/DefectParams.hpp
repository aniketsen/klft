#pragma once
#include <filesystem>
#include <sstream>
#include <string>

#include "GLOBAL.hpp"
namespace klft {
template <size_t Nd>
struct defectParams {
  // A struct that is used to hold the defect information for a given
  // devicePTBCGaugeField
  index_t defect_length{1};
  real_t defect_value{1.0};
  IndexArray<Nd - 1> defect_position{
      0};  // origin of the defect in mu = 1,2,3 directions
  std::string format() const {
    std::ostringstream os;
    os << defect_value << "_" << defect_length;
    for (auto i : defect_position) {
      os << "_" << i;
    }
    return os.str();
  }
};
std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    result.push_back(item);
  }

  return result;
}
template <size_t Nd>
defectParams<Nd> parse(const std::string& str) {
  defectParams<Nd> params{};
  std::string filename = std::filesystem::path(str).filename().string();
  auto splitted = split(filename, '_');
  // first element is the defect defect_value

  params.defect_value = static_cast<real_t>(std::stod(splitted[1]));
  // second element is the defect_length
  params.defect_length = static_cast<index_t>(std::stoi(splitted[2]));
  IndexArray<Nd - 1> pos{};
  for (int i = 0; i < Nd - 1; i++) {
    pos[i] = static_cast<index_t>(std::stoi(splitted[i + 3]));
  }
  params.defect_position = pos;
  return params;
}
}  // namespace klft
