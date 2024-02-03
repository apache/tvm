# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""tvm.contrib.msc.framework.tensorrt.codegen.sources"""

from typing import Dict

from tvm.contrib.msc.core.codegen import get_base_sources


def get_trt_common_h_code() -> str:
    """Create trt_common header file codes

    Returns
    -------
    source: str
        The trt_common header source.
    """

    return """#ifndef TVM_CONTRIB_MSC_UTILS_TRT_COMMON_H_
#define TVM_CONTRIB_MSC_UTILS_TRT_COMMON_H_

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "NvInfer.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace nvinfer1;

#ifndef TRT_VERSION_GE
#define TRT_VERSION_GE(major, minor, patch)                            \\
  ((TRT_MAJOR > major) || (TRT_MAJOR == major && TRT_MINOR > minor) || \\
   (TRT_MAJOR == major && TRT_MINOR == minor && TRT_PATCH >= patch))
#endif

#if TRT_VERSION_GE(8, 0, 0)
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

#define CHECK(status)                                    \\
  do {                                                   \\
    auto ret = (status);                                 \\
    if (ret != 0) {                                      \\
      std::cout << "Cuda failure: " << ret << std::endl; \\
      abort();                                           \\
    }                                                    \\
  } while (0)

class TRTLogger : public ILogger {
 public:
  TRTLogger() : TRTLogger(Severity::kINFO) {}
  explicit TRTLogger(Severity severity) { severity_ = severity; }
  void log(Severity severity, const char* msg) noexcept override {
    if (severity > severity_) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cout << "[MSC.INTERNAL_ERROR]: " << msg << std::endl;
        break;
      case Severity::kERROR:
        std::cout << "[MSC.ERROR]: " << msg << std::endl;
        break;
      case Severity::kWARNING:
        std::cout << "[MSC.WARNING]: " << msg << std::endl;
        break;
      case Severity::kINFO:
        std::cout << "[MSC.INFO]: " << msg << std::endl;
        break;
      case Severity::kVERBOSE:
        std::cout << "[MSC.VERBOSE]: " << msg << std::endl;
        break;
      default:
        std::cout << "[MSC.UNKNOWN]: " << msg << std::endl;
        break;
    }
  }

  void setLogSeverity(Severity severity) { severity_ = severity; }

 private:
  Severity severity_;
};

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
#if TRT_VERSION_GE(8, 0, 0)
      delete obj;
#else
      obj->destroy();
#endif
    }
  }
};

template <typename T>
using TRTPtr = std::unique_ptr<T, InferDeleter>;

class TRTUtils {
 public:
  static const std::string TensorInfo(ILayer* layer, size_t id = 0);

  static std::map<std::string, Weights> LoadWeights(const std::string& file);

#if TRT_VERSION_GE(6, 0, 0)
  static bool SerializeEngineToFile(const std::string& file, TRTPtr<IBuilder>& builder,
                                    TRTPtr<INetworkDefinition>& network,
                                    TRTPtr<IBuilderConfig>& config, TRTLogger& logger);
#else
  static bool SerializeEngineToFile(const std::string& file, TRTPtr<IBuilder>& builder,
                                    TRTPtr<INetworkDefinition>& network, TRTLogger& logger);

#endif

  static bool DeserializeEngineFromFile(const std::string& file,
                                        std::shared_ptr<ICudaEngine>& engine, TRTLogger& logger);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_UTILS_TRT_COMMON_H_
"""


def get_trt_common_cc_code() -> str:
    """Create trt_common cc file codes

    Returns
    -------
    source: str
        The trt_common cc source.
    """

    return """#include "trt_common.h"

namespace tvm {
namespace contrib {
namespace msc {

const std::string TRTUtils::TensorInfo(ILayer* layer, size_t id) {
  std::string info = "S:";
  Dims dims = layer->getOutput(id)->getDimensions();
  for (int i = 0; i < dims.nbDims; i++) {
    info += std::to_string(dims.d[i]) + ';';
  }
  DataType dtype = layer->getOutput(id)->getType();
  info += " D:";
  if (dtype == DataType::kFLOAT) {
    info += "float32";
  } else if (dtype == DataType::kHALF) {
    info += "float16";
  } else if (dtype == DataType::kINT32) {
    info += "int32";
  } else if (dtype == DataType::kINT8) {
    info += "int8";
  } else if (dtype == DataType::kBOOL) {
    info += "bool";
  } else {
    info += "unknown";
  }
  return info;
}

std::map<std::string, Weights> TRTUtils::LoadWeights(const std::string& file) {
  std::map<std::string, Weights> weightMap;
  // Open weights file
  std::ifstream input(file, std::ios::binary);
  assert(input.is_open() && ("Failed to open file " + file).c_str());

  // Read number of weight blobs
  int32_t count;
  input >> count;
  assert(count > 0 && "Invalid weight map file.");
  std::cout << "Find " << count << " weigths in the file : " << file << std::endl;

  while (count--) {
    Weights wt{DataType::kFLOAT, nullptr, 0};
    uint32_t type, size;
    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> type >> size;
    wt.type = static_cast<DataType>(type);

    // Load blob
    if (wt.type == DataType::kFLOAT) {
      uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
      for (uint32_t x = 0; x < size; ++x) {
        input >> std::hex >> val[x];
      }
      wt.values = val;
    } else if (wt.type == DataType::kHALF) {
      uint16_t* val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
      for (uint32_t x = 0; x < size; ++x) {
        input >> std::hex >> val[x];
      }
      wt.values = val;
    }
    wt.count = size;
    weightMap[name] = wt;
  }
  input.close();
  return weightMap;
}

#if TRT_VERSION_GE(6, 0, 0)
bool TRTUtils::SerializeEngineToFile(const std::string& file, TRTPtr<IBuilder>& builder,
                                     TRTPtr<INetworkDefinition>& network,
                                     TRTPtr<IBuilderConfig>& config, TRTLogger& logger) {
#if TRT_VERSION_GE(8, 0, 0)
  auto plan = TRTPtr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
#else
  auto engine = TRTPtr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
  if (!engine) {
    logger.log(ILogger::Severity::kERROR, "Failed to build engine");
    return false;
  }
  auto plan = TRTPtr<IHostMemory>(engine->serialize());
#endif
  if (!plan) {
    logger.log(ILogger::Severity::kERROR, "Failed to serialize network");
    return false;
  }
  std::ofstream ofs(file, std::ios::out | std::ios::binary);
  assert(ofs.is_open() && ("Failed to open file " + file).c_str());
  ofs.write((char*)(plan->data()), plan->size());
  ofs.close();
  return true;
}
#else
bool TRTUtils::SerializeEngineToFile(const std::string& file, TRTPtr<IBuilder>& builder,
                                     TRTPtr<INetworkDefinition>& network, TRTLogger& logger) {
  auto engine = TRTPtr<ICudaEngine>(builder->buildCudaEngine(*network));
  if (!engine) {
    logger.log(ILogger::Severity::kERROR, "Failed to build engine");
    return false;
  }
  auto plan = TRTPtr<IHostMemory>(engine->serialize());
  if (!plan) {
    logger.log(ILogger::Severity::kERROR, "Failed to serialize network");
    return false;
  }
  std::ofstream ofs(file, std::ios::out | std::ios::binary);
  assert(ofs.is_open() && ("Failed to open file " + file).c_str());
  ofs.write((char*)(plan->data()), plan->size());
  ofs.close();
  return true;
}
#endif

bool TRTUtils::DeserializeEngineFromFile(const std::string& file,
                                         std::shared_ptr<ICudaEngine>& engine, TRTLogger& logger) {
  std::vector<char> stream;
  size_t size{0};
  std::ifstream input(file, std::ifstream::binary);
  assert(input.is_open() && ("Failed to open file " + file).c_str());
  if (input.good()) {
    input.seekg(0, input.end);
    size = input.tellg();
    input.seekg(0, input.beg);
    stream.resize(size);
    input.read(stream.data(), size);
    input.close();
  }
  logger.log(ILogger::Severity::kINFO,
             ("size of engine from " + file + " is " + std::to_string(size)).c_str());
  auto runtime = TRTPtr<IRuntime>(createInferRuntime(logger));
  engine = std::shared_ptr<ICudaEngine>(
      runtime->deserializeCudaEngine(stream.data(), size, nullptr), InferDeleter());
  input.close();
  return true;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
"""


def get_trt_quantize_h_code():
    """Create trt_quantize header file codes

    Returns
    -------
    source: str
        The trt_quantize header source.
    """

    return """#ifndef TVM_CONTRIB_MSC_UTILS_TRT_QUANTIZE_H_
#define TVM_CONTRIB_MSC_UTILS_TRT_QUANTIZE_H_

#include <cassert>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"
#include "base.h"
#include "trt_common.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace nvinfer1;

class CalibrateHelper {
 public:
  CalibrateHelper(const std::string& range_file, const std::string& folder, int max_size = -1);

  ~CalibrateHelper() {
    for (const auto& buffer : cpu_buffers_) {
      free(buffer);
    }
    for (const auto& buffer : gpu_buffers_) {
      CHECK(cudaFree(buffer));
    }
  }

  bool GetBatch(void* bindings[], const char* names[], int nbBindings);

  const void* ReadCache(size_t& length);

  void WriteCache(const void* cache, size_t length);

 private:
  std::unique_ptr<DatasetReader> reader_;
  std::string range_file_;
  std::vector<char> cache_;
  std::vector<void*> cpu_buffers_;
  std::vector<void*> gpu_buffers_;
};

#define CALIBRATE_MEMBERS(Calibrator)                                                       \\
 public:                                                                                    \\
  Calibrator(const std::string& range_file, const std::string& folder, int max_size = -1) { \\
    helper_.reset(new CalibrateHelper(range_file, folder, max_size));                       \\
  }                                                                                         \\
                                                                                            \\
  virtual ~Calibrator() {}                                                                  \\
                                                                                            \\
  int getBatchSize() const noexcept override { return 1; }                                  \\
                                                                                            \\
  bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {  \\
    return helper_->GetBatch(bindings, names, nbBindings);                                  \\
  }                                                                                         \\
                                                                                            \\
  const void* readCalibrationCache(size_t& length) noexcept override {                      \\
    return helper_->ReadCache(length);                                                      \\
  }                                                                                         \\
                                                                                            \\
  void writeCalibrationCache(const void* cache, size_t length) noexcept override {          \\
    return helper_->WriteCache(cache, length);                                              \\
  }                                                                                         \\
                                                                                            \\
 private:                                                                                   \\
  std::unique_ptr<CalibrateHelper> helper_;

class MSCInt8EntropyCalibrator : public IInt8EntropyCalibrator {
  CALIBRATE_MEMBERS(MSCInt8EntropyCalibrator)
};

class MSCInt8EntropyCalibrator2 : public IInt8EntropyCalibrator2 {
  CALIBRATE_MEMBERS(MSCInt8EntropyCalibrator2)
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_UTILS_TRT_QUANTIZE_H_
"""


def get_trt_quantize_cc_code():
    """Create trt_quantize cc file codes

    Returns
    -------
    source: str
        The trt_quantize cc source.
    """

    return """#include "trt_quantize.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace nvinfer1;

CalibrateHelper::CalibrateHelper(const std::string& range_file, const std::string& folder,
                                 int max_size) {
  range_file_ = range_file;
  reader_.reset(new DatasetReader(folder, max_size));
  const auto& tensor_names = reader_->GetTensorNames();
  cpu_buffers_.resize(tensor_names.size());
  gpu_buffers_.resize(tensor_names.size());
  for (size_t i = 0; i < tensor_names.size(); i++) {
    size_t tensor_size = reader_->GetTensorSize(tensor_names[i]);
    cpu_buffers_[i] = malloc(tensor_size);
    CHECK(cudaMalloc(&gpu_buffers_[i], tensor_size));
  }
}

bool CalibrateHelper::GetBatch(void* bindings[], const char* names[], int nbBindings) {
  if (!reader_->ReadNext(cpu_buffers_.data())) {
    return false;
  }
  for (size_t i = 0; i < nbBindings; i++) {
    CHECK(cudaMemcpy(gpu_buffers_[i], cpu_buffers_[i], reader_->GetTensorSize(names[i]),
                     cudaMemcpyHostToDevice));
    bindings[i] = gpu_buffers_[i];
  }
  return true;
}

const void* CalibrateHelper::ReadCache(size_t& length) {
  cache_.clear();
  std::ifstream in_file(range_file_, std::ifstream::binary);
  if (!in_file.is_open()) {
    return nullptr;
  }
  in_file >> std::noskipws;
  std::copy(std::istream_iterator<char>(in_file), std::istream_iterator<char>(),
            std::back_inserter(cache_));
  length = cache_.size();
  return length > 0 ? &cache_[0] : nullptr;
}

void CalibrateHelper::WriteCache(const void* cache, size_t length) {
  std::ofstream output(range_file_, std::ios::binary);
  output.write(reinterpret_cast<const char*>(cache), length);
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
"""


def get_trt_sources() -> Dict[str, str]:
    """Create trt sources for cpp codegen

    Returns
    -------
    sources: dict<str,str>
        The trt utils sources.
    """

    sources = get_base_sources()
    sources.update(
        {
            "trt_common.h": get_trt_common_h_code(),
            "trt_common.cc": get_trt_common_cc_code(),
            "trt_quantize.h": get_trt_quantize_h_code(),
            "trt_quantize.cc": get_trt_quantize_cc_code(),
        }
    )
    return sources
