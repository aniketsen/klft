#pragma once

#include <filesystem>
#include "GLOBAL.hpp"
namespace klft {
struct IOParams {
  bool save_gauge_field = false;
  std::string gauge_field_filename = "gauge_field.dat";
  bool overwrite_gauge_field_file = true;
  size_t save_gauge_field_interval = 100;
  bool save_after_trajectory = true;
  std::string output_dir = "./";
  void print() {
    printf("IO Parameters:\n");
    printf("Save Gauge Field: %i\n", save_gauge_field);
    printf("Gauge Field Filename: %s\n", gauge_field_filename.c_str());
    printf("Overwrite Gauge Field File: %i\n", overwrite_gauge_field_file);
    printf("Save Gauge Field Interval: %zu\n", save_gauge_field_interval);
    printf("Save After Last Trajectory: %i\n", save_after_trajectory);
  }
};
template <typename DGaugeFieldType>
void flushIO(const IOParams& params,
             const size_t& step,
             const typename DGaugeFieldType::type& gauge_field,
             const bool last_step = false) {
  if (!params.save_gauge_field) {
    return;
  }
  if (step % params.save_gauge_field_interval == 0 ||
      (last_step && params.save_after_trajectory)) {
    if (params.overwrite_gauge_field_file) {
      gauge_field.save(params.output_dir + "/" + params.gauge_field_filename);
      return;
      /* code */
    }
    std::string output_dir =
        params.output_dir + "/step_" + std::to_string(step) + "/";
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }

    gauge_field.save(output_dir + params.gauge_field_filename);
  }
  return;
}

template <typename DGaugeFieldType>
void flushIOPTBC(const IOParams& params,
                 const int& rank,
                 const size_t& step,
                 const typename DGaugeFieldType::type& gauge_field,
                 const bool last_step = false) {
  if (!params.save_gauge_field) {
    return;
  }
  std::string output_dir =
      params.output_dir + "/step" + std::to_string(step) + "/";

  if (last_step && params.save_after_trajectory) {
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
    gauge_field.save((output_dir + "rank" + std::to_string(rank) + "_" +
                      gauge_field.dParams.format() + "_" +
                      params.gauge_field_filename));
    return;
  }

  if (params.save_gauge_field_interval > 0 &&
      step % params.save_gauge_field_interval == 0) {
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
    if (params.overwrite_gauge_field_file) {
      printf(
          "Warning: overwriting the GuageField is currently not supported\n");
    }

    gauge_field.save((output_dir + "rank" + std::to_string(rank) + "_" +
                      gauge_field.dParams.format() + "_" +
                      params.gauge_field_filename));
  }
  return;
}
}  // namespace klft
