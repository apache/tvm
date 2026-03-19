#include <tvm/ffi/function.h>
#include <tvm/runtime/builtin_fp16.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using tvm::Device;
using tvm::runtime::DataType;
using tvm::runtime::Tensor;

namespace {

std::unordered_map<std::string, std::string> ReadKeyValueFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }
  std::unordered_map<std::string, std::string> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    size_t sep = line.find('=');
    if (sep == std::string::npos) {
      throw std::runtime_error("bad key=value line in " + path + ": " + line);
    }
    out.emplace(line.substr(0, sep), line.substr(sep + 1));
  }
  return out;
}

std::vector<uint8_t> ReadBinaryFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }
  in.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(size);
  if (size != 0 && !in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size))) {
    throw std::runtime_error("failed to read " + path);
  }
  return bytes;
}

std::vector<int64_t> ParseShape(const std::string& text) {
  std::vector<int64_t> shape;
  std::stringstream ss(text);
  std::string part;
  while (std::getline(ss, part, 'x')) {
    if (!part.empty()) {
      shape.push_back(std::stoll(part));
    }
  }
  if (shape.empty()) {
    throw std::runtime_error("invalid shape string: " + text);
  }
  return shape;
}

Tensor LoadFloat16Tensor(const std::string& path, const std::vector<int64_t>& shape) {
  size_t elems = 1;
  for (int64_t dim : shape) {
    elems *= static_cast<size_t>(dim);
  }
  std::vector<uint8_t> bytes = ReadBinaryFile(path);
  size_t expected_nbytes = elems * sizeof(uint16_t);
  if (bytes.size() != expected_nbytes) {
    std::ostringstream os;
    os << "size mismatch for " << path << ": got " << bytes.size() << ", want "
       << expected_nbytes;
    throw std::runtime_error(os.str());
  }
  Tensor tensor = Tensor::Empty(shape, DataType::Float(16), Device{kDLCPU, 0});
  tensor.CopyFromBytes(bytes.data(), bytes.size());
  return tensor;
}

double MaxAbsErrFloat16(const Tensor& actual, const Tensor& expected) {
  size_t elems = 1;
  for (int64_t dim : actual.Shape()) {
    elems *= static_cast<size_t>(dim);
  }
  std::vector<uint16_t> actual_buf(elems);
  std::vector<uint16_t> expected_buf(elems);
  actual.CopyToBytes(actual_buf.data(), actual_buf.size() * sizeof(uint16_t));
  expected.CopyToBytes(expected_buf.data(), expected_buf.size() * sizeof(uint16_t));
  double max_err = 0.0;
  for (size_t i = 0; i < elems; ++i) {
    double a = static_cast<double>(__gnu_h2f_ieee(actual_buf[i]));
    double b = static_cast<double>(__gnu_h2f_ieee(expected_buf[i]));
    max_err = std::max(max_err, std::abs(a - b));
  }
  return max_err;
}

int64_t CountNonFiniteFloat16(const Tensor& tensor) {
  size_t elems = 1;
  for (int64_t dim : tensor.Shape()) {
    elems *= static_cast<size_t>(dim);
  }
  std::vector<uint16_t> buf(elems);
  tensor.CopyToBytes(buf.data(), buf.size() * sizeof(uint16_t));
  int64_t count = 0;
  for (uint16_t raw : buf) {
    double value = static_cast<double>(__gnu_h2f_ieee(raw));
    if (!std::isfinite(value)) {
      count += 1;
    }
  }
  return count;
}

bool MaybeConfigureChainBlob(const tvm::ffi::Module& executable, const std::string& bundle_dir) {
  auto apply_chain_blob = executable->GetFunction("rknpu_bridge_apply_chain_blob", true);
  if (apply_chain_blob.has_value()) {
    (*apply_chain_blob)();
    return true;
  }
  auto set_chain_blob = tvm::ffi::Function::GetGlobal("runtime.rknpu_bridge_set_chain_blob");
  if (!set_chain_blob.has_value()) {
    throw std::runtime_error("runtime.rknpu_bridge_set_chain_blob not found");
  }
  std::vector<uint8_t> blob = ReadBinaryFile(bundle_dir + "/chain_blob.bin");
  (*set_chain_blob)(tvm::ffi::Bytes(
      std::string(reinterpret_cast<const char*>(blob.data()), blob.size())));
  return false;
}

std::string JsonEscape(const std::string& text) {
  std::ostringstream os;
  for (char c : text) {
    switch (c) {
      case '\\':
        os << "\\\\";
        break;
      case '"':
        os << "\\\"";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        os << c;
        break;
    }
  }
  return os.str();
}

std::string IntArrayJson(const std::vector<int64_t>& values) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  os << "]";
  return os.str();
}

int64_t ExtractSummedJsonIntField(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?\\d+)");
  int64_t total = 0;
  for (std::sregex_iterator it(json.begin(), json.end(), pattern), end; it != end; ++it) {
    total += std::stoll((*it)[1].str());
  }
  return total;
}

struct RunnerConfig {
  std::string bundle_dir;
  int warmup = 0;
  int iters = 1;
  std::string json_out;
};

RunnerConfig ParseArgs(int argc, char** argv) {
  if (argc < 2) {
    throw std::runtime_error("usage: rknpu_vm_cpp_runner <bundle_dir> [--warmup N] [--iters N] [--json-out PATH]");
  }
  RunnerConfig cfg;
  cfg.bundle_dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
      }
      ++i;
      return argv[i];
    };
    if (arg == "--warmup") {
      cfg.warmup = std::stoi(require_value("--warmup"));
    } else if (arg == "--iters") {
      cfg.iters = std::stoi(require_value("--iters"));
    } else if (arg == "--json-out") {
      cfg.json_out = require_value("--json-out");
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (cfg.iters <= 0) {
    cfg.iters = 1;
  }
  if (cfg.warmup < 0) {
    cfg.warmup = 0;
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    RunnerConfig cfg = ParseArgs(argc, argv);
    const std::string& bundle_dir = cfg.bundle_dir;
    const auto bundle = ReadKeyValueFile(bundle_dir + "/bundle.txt");
    const std::string entry = bundle.count("entry") ? bundle.at("entry") : "main";

    const auto reset_stats = tvm::ffi::Function::GetGlobal("runtime.rknpu_bridge_reset_stats");
    const auto get_stats_json =
        tvm::ffi::Function::GetGlobal("runtime.rknpu_bridge_get_stats_json");

    tvm::ffi::Module executable = tvm::ffi::Module::LoadFromFile(bundle_dir + "/exec.so");
    const bool used_embedded_chain_blob = MaybeConfigureChainBlob(executable, bundle_dir);
    auto vm_load = executable->GetFunction("vm_load_executable");
    if (!vm_load.has_value()) {
      throw std::runtime_error("vm_load_executable missing from exported library");
    }
    tvm::ffi::Module vm = (*vm_load)().cast<tvm::ffi::Module>();
    auto vm_initialization = vm->GetFunction("vm_initialization");
    if (!vm_initialization.has_value()) {
      throw std::runtime_error("vm_initialization missing from VM module");
    }
    (*vm_initialization)(static_cast<int>(kDLCPU), 0, 2);

    auto main_fn = vm->GetFunction(entry);
    if (!main_fn.has_value()) {
      throw std::runtime_error("entry function missing from VM module: " + entry);
    }

    int num_inputs = std::stoi(bundle.at("num_inputs"));
    int num_outputs = std::stoi(bundle.at("num_outputs"));
    if (num_outputs != 1) {
      throw std::runtime_error("runner currently supports exactly one tensor output");
    }

    std::vector<Tensor> inputs;
    inputs.reserve(static_cast<size_t>(num_inputs));
    std::vector<tvm::ffi::AnyView> packed_inputs(static_cast<size_t>(num_inputs));
    for (int i = 0; i < num_inputs; ++i) {
      const std::string file_key = "input" + std::to_string(i) + "_file";
      const std::string shape_key = "input" + std::to_string(i) + "_shape";
      Tensor tensor = LoadFloat16Tensor(bundle_dir + "/" + bundle.at(file_key), ParseShape(bundle.at(shape_key)));
      inputs.push_back(tensor);
      packed_inputs[static_cast<size_t>(i)] = inputs.back();
    }
    const std::string expected_file = bundle.at("output0_expected_file");
    Tensor expected_y = LoadFloat16Tensor(bundle_dir + "/" + expected_file, ParseShape(bundle.at("output0_shape")));

    std::vector<int64_t> warmup_wall_ns;
    std::vector<int64_t> warmup_runtime_total_ns;
    std::vector<int64_t> warmup_runtime_submit_ns;
    std::vector<int64_t> warmup_runtime_hw_ns;
    for (int i = 0; i < cfg.warmup; ++i) {
      if (reset_stats.has_value()) {
        (*reset_stats)();
      }
      auto t0 = std::chrono::steady_clock::now();
      tvm::ffi::Any warmup_result;
      (*main_fn).CallPacked(packed_inputs.data(), static_cast<int32_t>(packed_inputs.size()),
                            &warmup_result);
      auto t1 = std::chrono::steady_clock::now();
      warmup_wall_ns.push_back(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
      std::string stats_json =
          get_stats_json.has_value() ? std::string((*get_stats_json)().cast<tvm::ffi::String>())
                                     : "{}";
      warmup_runtime_total_ns.push_back(ExtractSummedJsonIntField(stats_json, "total_ns"));
      warmup_runtime_submit_ns.push_back(
          ExtractSummedJsonIntField(stats_json, "submit_ns"));
      warmup_runtime_hw_ns.push_back(
          ExtractSummedJsonIntField(stats_json, "hw_elapsed_ns"));
    }

    Tensor actual_y;
    double last_wall_ms = 0.0;
    double max_err = 0.0;
    int64_t output_non_finite = 0;
    struct TimedSample {
      int64_t wall_ns;
      int64_t runtime_total_ns;
      int64_t runtime_submit_ns;
      int64_t runtime_hw_ns;
      std::string stats_json;
    };
    std::vector<TimedSample> samples;
    samples.reserve(static_cast<size_t>(cfg.iters));
    for (int i = 0; i < cfg.iters; ++i) {
      if (reset_stats.has_value()) {
        (*reset_stats)();
      }
      auto t0 = std::chrono::steady_clock::now();
      tvm::ffi::Any result;
      (*main_fn).CallPacked(packed_inputs.data(), static_cast<int32_t>(packed_inputs.size()), &result);
      auto t1 = std::chrono::steady_clock::now();
      actual_y = result.cast<Tensor>();
      max_err = MaxAbsErrFloat16(actual_y, expected_y);
      output_non_finite = CountNonFiniteFloat16(actual_y);
      std::string stats_json =
          get_stats_json.has_value() ? std::string((*get_stats_json)().cast<tvm::ffi::String>())
                                     : "{}";
      int64_t wall_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      samples.push_back(TimedSample{
          wall_ns,
          ExtractSummedJsonIntField(stats_json, "total_ns"),
          ExtractSummedJsonIntField(stats_json, "submit_ns"),
          ExtractSummedJsonIntField(stats_json, "hw_elapsed_ns"),
          stats_json,
      });
      last_wall_ms = wall_ns / 1e6;
    }

    const std::string last_stats_json = samples.empty() ? "{}" : samples.back().stats_json;

    std::ostringstream json;
    json << "{";
    json << "\"bundle\":\"" << JsonEscape(bundle_dir) << "\",";
    json << "\"entry\":\"" << JsonEscape(entry) << "\",";
    json << "\"chain_blob_source\":\"" << (used_embedded_chain_blob ? "embedded" : "sidecar")
         << "\",";
    json << "\"warmup\":{";
    json << "\"iterations_run\":" << cfg.warmup << ",";
    json << "\"wall_ns_samples\":" << IntArrayJson(warmup_wall_ns) << ",";
    json << "\"runtime_total_ns_samples\":" << IntArrayJson(warmup_runtime_total_ns) << ",";
    json << "\"runtime_submit_ns_samples\":" << IntArrayJson(warmup_runtime_submit_ns) << ",";
    json << "\"runtime_hw_ns_samples\":" << IntArrayJson(warmup_runtime_hw_ns);
    json << "},";
    json << "\"timed_samples\":[";
    for (size_t i = 0; i < samples.size(); ++i) {
      if (i != 0) {
        json << ",";
      }
      json << "{";
      json << "\"wall_ns\":" << samples[i].wall_ns << ",";
      json << "\"runtime_total_ns\":" << samples[i].runtime_total_ns << ",";
      json << "\"runtime_submit_ns\":" << samples[i].runtime_submit_ns << ",";
      json << "\"runtime_hw_ns\":" << samples[i].runtime_hw_ns << ",";
      json << "\"runtime_bridge_stats\":" << samples[i].stats_json;
      json << "}";
    }
    json << "],";
    json << "\"max_err\":" << max_err << ",";
    json << "\"output_non_finite\":" << output_non_finite << ",";
    json << "\"runtime_bridge_stats_last\":" << last_stats_json;
    json << "}";

    if (!cfg.json_out.empty()) {
      std::ofstream out(cfg.json_out);
      if (!out) {
        throw std::runtime_error("failed to open json out: " + cfg.json_out);
      }
      out << json.str();
    }

    std::cout << "CPP_VM_RUN"
              << " bundle=" << bundle_dir
              << " chain_blob=" << (used_embedded_chain_blob ? "embedded" : "sidecar")
              << " wall_ms=" << last_wall_ms
              << " max_err=" << max_err
              << " output_non_finite=" << output_non_finite
              << " iters=" << cfg.iters << "\n";
    std::cout << last_stats_json << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "rknpu_vm_cpp_runner error: " << e.what() << "\n";
    return 1;
  }
}
