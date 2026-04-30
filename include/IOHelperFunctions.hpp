#pragma once
#include <filesystem>
#include <iomanip>
#include <regex>
// #include <sstream>

#include "DefectParams.hpp"
#include "GLOBAL.hpp"
namespace klft {

template <size_t Nd>
std::string encode_ptbc_info(const std::string& path,
                             const defectParams<Nd>& params) {
  // 1. Find directory / filename split
  std::size_t lastSlash = path.find_last_of('/');
  std::string dir;
  std::string filename;

  if (lastSlash != std::string::npos) {
    dir = path.substr(0, lastSlash + 1);
    filename = path.substr(lastSlash + 1);
  } else {
    filename = path;  // no directory
  }

  std::ostringstream prefix;
  prefix << params.defect_length;
  for (std::size_t i = 0; i < params.defect_position(); ++i) {
    prefix << "_" << params.defect_position[i];
  }
  prefix << "_" << std::fixed << std::setprecision(8) << params.defect_value
         << "_";

  // 3. Construct final path
  return dir + prefix.str() + filename;
}
template <size_t Nd>
void decode_ptbc_info(const std::string& filename, defectParams<Nd>& dParams) {
  // 1. Find directory / filename split
  std::size_t lastSlash = filename.find_last_of('/');
  std::string file;
  if (lastSlash != std::string::npos) {
    file = filename.substr(lastSlash + 1);
  } else {
    file = filename;  // no directory
  }

  // 2. Extract info from filename
  std::istringstream ss(file);
  std::string token;
  std::vector<std::string> tokens;
  while (std::getline(ss, token, '_')) {
    tokens.push_back(token);
  }
  dParams.defect_length = std::stoi(tokens[0]);
  for (size_t i = 1; i < Nd - 1; i++) {
    dParams.defect_position[i - 1] = std::stoi(tokens[i]);
    /* code */
  }

  dParams.defect_value = std::stod(tokens[Nd]);
}
std::string find_file_for_rank(const std::string& dir, int rank) {
  std::regex rankRegex(R"(.*\.rank([0-9]+)\.)");

  for (auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string filename = entry.path().filename().string();
    std::smatch match;
    if (std::regex_match(filename, match, rankRegex)) {
      int fileRank = std::stoi(match[1]);
      if (fileRank == rank) {
        return entry.path().string();
      }
    }
  }

  throw std::runtime_error("No file found for rank " + std::to_string(rank) +
                           " in " + dir);
}
}  // namespace klft