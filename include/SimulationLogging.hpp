#pragma once
#include <fstream>

#include "GLOBAL.hpp"

namespace klft {
// define a struct to hold parameters related to the simulation logging
struct SimulationLoggingParams {
  size_t log_interval;       // interval between logs
  std::string log_filename;  // filename for the log
  bool write_to_file;        // whether to write logs to file
  size_t flush;  // interval to flush logs to file ,0 to flush at the end of the
                 // simulation
  bool flushed;  // check if the logs were flushed at least once -> used to
                 // add the header line to the file
  // define flags for the different types of logs
  bool log_delta_H;
  bool log_acceptance;
  bool log_accept;
  bool log_time;
  bool log_observable_time;

  // define vectors to hold the logs
  std::vector<size_t> log_steps;
  std::vector<real_t> delta_H;
  std::vector<real_t> acceptance;
  std::vector<bool> accept;
  std::vector<real_t> time;
  std::vector<real_t> observable_time;

  // constructor to initialize the parameters
  SimulationLoggingParams()
      : log_interval(0),
        flush(25),
        flushed(false),
        write_to_file(false),
        log_delta_H(false),
        log_acceptance(false),
        log_accept(false),
        log_time(false),
        log_observable_time(false) {}
};

// define a function to log simulation information
inline void addLogData(
    SimulationLoggingParams& params,
    const size_t step,
    const real_t _delta_H = 0.0,
    const real_t _acceptance = 0.0,
    const bool _accept = false,
    const real_t _time = 0.0,
    const real_t _observable_time =
        0.0) {  // TODO: add overloads for different passed parameters
  if (params.log_interval == 0 || step % params.log_interval != 0 ||
      step == 0) {
    return;
  }

  if (KLFT_VERBOSITY > 1) {
    printf("Logging Simulation Data\n");
    printf("step: %zu\n", step);
  }

  if (params.log_delta_H) {
    params.delta_H.push_back(_delta_H);
    if (KLFT_VERBOSITY > 1) {
      printf("delta_H: %11.6f\n", _delta_H);
    }
  }

  if (params.log_acceptance) {
    params.acceptance.push_back(_acceptance);
    if (KLFT_VERBOSITY > 1) {
      printf("acceptance: %11.6f\n", _acceptance);
    }
  }

  if (params.log_accept) {
    params.accept.push_back(_accept);
    if (KLFT_VERBOSITY > 1) {
      printf("accept: %s\n", params.accept.back() ? "true" : "false");
    }
  }

  if (params.log_time) {
    params.time.push_back(_time);
    if (KLFT_VERBOSITY > 1) {
      printf("time: %11.6f\n", _time);
    }
  }

  if (params.log_observable_time) {
    params.observable_time.push_back(_observable_time);
    if (KLFT_VERBOSITY > 1) {
      printf("observable_time: %11.6f\n", _observable_time);
    }
  }

  params.log_steps.push_back(step);
}

inline void clearSimulationLogs(SimulationLoggingParams& params) {
  // clear the logs
  params.log_steps.clear();
  params.delta_H.clear();
  params.acceptance.clear();
  params.accept.clear();
  params.time.clear();
  params.observable_time.clear();
}

inline void forceflushSimulationLogs(SimulationLoggingParams& params,
                                     const bool clear_after_flush = false) {
  // check if write_to_file is enabled
  if (!params.write_to_file) {
    if (KLFT_VERBOSITY > 0) {
      printf("write_to_file is not enabled\n");
    }
    return;
  }
  // open the log file
  std::ofstream file(params.log_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open log file %s\n", params.log_filename.c_str());
    return;
  }

  bool HEADER = !params.flushed;  // write header only once
  // write header if required
  if (HEADER) {
    file << "# step";
    if (params.log_acceptance) {
      file << ", acceptance";
    }
    if (params.log_delta_H) {
      file << ", delta_H";
    }
    if (params.log_accept) {
      file << ", accept";
    }
    if (params.log_time) {
      file << ", time";
    }
    if (params.log_observable_time) {
      file << ", observable_time";
    }
    file << "\n";
  }

  // write the logs
  for (size_t i = 0; i < params.log_steps.size(); ++i) {
    file << params.log_steps[i];
    if (params.log_acceptance) {
      file << ", " << params.acceptance[i];
    }
    if (params.log_delta_H) {
      file << ", " << params.delta_H[i];
    }
    if (params.log_accept) {
      file << ", " << (params.accept[i] ? 1 : 0);
    }
    if (params.log_time) {
      file << ", " << params.time[i];
    }
    if (params.log_observable_time) {
      file << ", " << params.observable_time[i];
    }
    file << "\n";
  }

  // close the file
  file.close();
  if (clear_after_flush) {
    clearSimulationLogs(params);  // clear the logs after flushing
  }
  params.flushed = true;  // set flushed to true after writing
}

inline void flushSimulationLogs(SimulationLoggingParams& params,
                                const size_t step,
                                const bool clear_after_flush = false) {
  if (params.flush != 0 && step % params.flush == 0) {
    forceflushSimulationLogs(params, clear_after_flush);
  }
}

struct PTBCSimulationLoggingParams {
  size_t log_interval;       // interval between logs
  std::string log_filename;  // filename for the log
  bool write_to_file;        // whether to write logs to file
  size_t flush;  // interval to flush logs to file ,0 to flush at the end of the
                 // simulation
  bool flushed;  // check if the logs were flushed at least once -> used to
                 // add the header line to the file

  // define flags for the different types of logs
  bool log_swap_accepts;
  bool log_delta_H_swap;
  bool log_defects;

  // define vectors to hold the logs
  std::vector<size_t> log_steps;
  std::vector<bool> starting_defect_value;
  std::vector<std::vector<real_t>> delta_H_swap;
  std::vector<std::vector<real_t>> defects;
  std::vector<std::vector<real_t>> prev_defects;
  std::vector<std::vector<bool>> accepts;

  // constructor to initialize the parameters
  PTBCSimulationLoggingParams()
      : log_interval(0),
        flush(25),
        flushed(false),
        write_to_file(false),
        log_delta_H_swap(false),
        log_swap_accepts(true),
        log_defects(true) {}
};

// ---------- PTBC logging: add / flush / clear ----------

inline void addPTBCLogData(PTBCSimulationLoggingParams& p,
                           const size_t step,
                           const bool starting_defect_value,
                           const std::vector<bool>* _accepts = nullptr,
                           const std::vector<real_t>* _delta_H_swap = nullptr,
                           const std::vector<real_t>* _defects = nullptr,
                           const std::vector<real_t>* _prev_defects = nullptr) {
  // obey interval (skip step 0, like your other logger)
  if (p.log_interval == 0 || step % p.log_interval != 0 || step == 0)
    return;

  if (KLFT_VERBOSITY > 1) {
    printf("Logging PTBC Data (step %zu)\n", step);
  }

  // Always push a step entry if we're logging this step
  p.log_steps.push_back(step);

  // For each enabled stream, push an entry (empty if nullptr provided)

  if (p.log_swap_accepts) {
    if (_accepts)
      p.accepts.push_back(*_accepts);
    else
      p.accepts.emplace_back();  // empty vector keeps indices aligned
    if (KLFT_VERBOSITY > 1 && _accepts) {
      printf("accepts: ");
      for (bool b : *_accepts)
        printf("%d ", b ? 1 : 0);
      printf("\n");
    }
  }

  if (p.log_delta_H_swap) {
    if (_delta_H_swap)
      p.delta_H_swap.push_back(*_delta_H_swap);
    else
      p.delta_H_swap.emplace_back();
    if (KLFT_VERBOSITY > 1 && _delta_H_swap) {
      printf("delta_H_swap size: %zu\n", _delta_H_swap->size());
    }
  }

  if (p.log_defects) {
    p.starting_defect_value.push_back(starting_defect_value);
    if (_defects) {
      p.defects.push_back(*_defects);
    } else {
      p.defects.emplace_back();
    }
    if (_prev_defects) {
      p.prev_defects.push_back(*_prev_defects);
    } else {
      p.prev_defects.emplace_back();
    }
  }
}

inline void clearPTBCSimulationLogs(PTBCSimulationLoggingParams& p) {
  p.log_steps.clear();
  p.starting_defect_value.clear();
  p.delta_H_swap.clear();
  p.defects.clear();
  p.prev_defects.clear();
  p.accepts.clear();
}

// helper: join vectors for writing
template <class T>
static inline void _write_vec(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      os << ' ';
    os << v[i];
  }
  os << "]";
}

// bool specialization (std::vector<bool> is bit-packed)
static inline void _write_vec_bool(std::ostream& os,
                                   const std::vector<bool>& v) {
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i)
      os << ' ';
    os << (v[i] ? 1 : 0);
  }
  os << "]";
}

inline void forceflushPTBCSimulationLogs(PTBCSimulationLoggingParams& p,
                                         const bool clear_after_flush = false) {
  if (!p.write_to_file) {
    if (KLFT_VERBOSITY > 0)
      printf("PTBC write_to_file is not enabled\n");
    return;
  }

  std::ofstream file(p.log_filename, std::ios::app);
  if (!file.is_open()) {
    printf("Error: could not open PTBC log file %s\n", p.log_filename.c_str());
    return;
  }

  const bool HEADER = !p.flushed;
  if (HEADER) {
    file << "# step";
    if (p.log_swap_accepts)
      file << ", accepts";
    if (p.log_delta_H_swap)
      file << ", delta_H_swap";
    if (p.log_defects) {
      file << ", starting_defect_value";
      file << ", prev_defects";
      file << ", defects";
    }
    file << "\n";
  }

  // We assume you call addPTBCLogData once per step → vectors are aligned
  const size_t n = p.log_steps.size();
  for (size_t i = 0; i < n; ++i) {
    file << p.log_steps[i];

    if (p.log_swap_accepts) {
      if (i < p.accepts.size()) {
        file << ", ";
        _write_vec_bool(file, p.accepts[i]);
      } else {
        file << ", []";
      }
    }
    if (p.log_delta_H_swap) {
      if (i < p.delta_H_swap.size()) {
        file << ", ";
        _write_vec<real_t>(file, p.delta_H_swap[i]);
      } else {
        file << ", []";
      }
    }
    if (p.log_defects) {
      if (i < p.starting_defect_value.size()) {
        file << ", ";
        file << (p.starting_defect_value[i] ? "1" : "0");
      }
      if (i < p.prev_defects.size()) {
        file << ", ";
        _write_vec<real_t>(file, p.prev_defects[i]);
      } else {
        file << ", []";
      }
      if (i < p.defects.size()) {
        file << ", ";
        _write_vec<real_t>(file, p.defects[i]);
      } else {
        file << ", []";
      }
    }
    file << "\n";
  }

  file.close();
  if (clear_after_flush)
    clearPTBCSimulationLogs(p);
  p.flushed = true;
}

inline void flushPTBCSimulationLogs(PTBCSimulationLoggingParams& p,
                                    const size_t step,
                                    const bool clear_after_flush = false) {
  if (p.flush != 0 && step % p.flush == 0) {
    forceflushPTBCSimulationLogs(p, clear_after_flush);
  }
}

}  // namespace klft
