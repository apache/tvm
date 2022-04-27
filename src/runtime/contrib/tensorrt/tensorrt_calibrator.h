/* * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.

 * file runtime/contrib/tensorrt/tensorrt_builder.h
 * brief Contains TensorRTBuilder class which can be used to convert a relay
 * program into a TRT engine which can be used for inference.
*/

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_CALIBRATOR_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_CALIBRATOR_H_

#include <string>
#include <vector>

#include "../../cuda/cuda_common.h"
#include "NvInfer.h"

namespace tvm {
namespace runtime {

class TensorRTCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  TensorRTCalibrator(int batch_size, const std::vector<std::string>& input_names)
      : batch_size_(batch_size), num_batches_calibrated_(0), input_names_(input_names) {}

  ~TensorRTCalibrator() {
    // Free calibration data
    for (auto& inputs : data_) {
      for (size_t i = 0; i < inputs.size(); ++i) {
        delete[] inputs[i];
      }
    }
    // Free buffers
    for (size_t i = 0; i < buffers_.size(); ++i) {
      CUDA_CALL(cudaFree(buffers_[i]));
    }
  }

  void AddBatchData(const std::vector<void*>& bindings, const std::vector<size_t>& binding_sizes) {
    // Copy data from GPU
    std::vector<float*> data_host(bindings.size(), nullptr);
    for (size_t i = 0; i < bindings.size(); ++i) {
      data_host[i] = new float[batch_size_ * binding_sizes[i]];
      CUDA_CALL(cudaMemcpy(static_cast<void*>(data_host[i]), bindings[i],
                           batch_size_ * binding_sizes[i] * sizeof(float), cudaMemcpyDeviceToHost));
    }
    data_.push_back(data_host);
    data_sizes_.push_back(binding_sizes);
  }

  int getBatchSize() const noexcept override { return batch_size_; }

  /*!
   * \brief TensorRT will call this method to get next batch of data to
   * calibrate with.
   */
  bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
    AllocateBuffersIfNotAllocated();
    CHECK_EQ(input_names_.size(), nbBindings);
    for (size_t i = 0; i < input_names_.size(); ++i) {
      CHECK_EQ(input_names_[i], names[i]);
      CUDA_CALL(cudaMemcpy(buffers_[i], data_[num_batches_calibrated_][i],
                           batch_size_ * data_sizes_[num_batches_calibrated_][i] * sizeof(float),
                           cudaMemcpyHostToDevice));
      bindings[i] = buffers_[i];
    }
    num_batches_calibrated_++;
    // TODO(trevmorr): Free data from previous batch?
    return (num_batches_calibrated_ < static_cast<int>(data_.size()));
  }

  const void* readCalibrationCache(size_t& length) noexcept override {
    if (calibration_cache_.empty()) return nullptr;
    length = calibration_cache_.size();
    return calibration_cache_.data();
  }

  void writeCalibrationCache(const void* cache, size_t length) noexcept override {
    calibration_cache_.assign(static_cast<const char*>(cache), length);
  }

 private:
  /*! \brief Batch size. */
  int batch_size_;
  /*! \brief Number of batches already fed to calibrator. */
  int num_batches_calibrated_;
  /*! \brief Storage for calibration cache. */
  std::string calibration_cache_;

  /*! \brief Data to be used for calibration. */
  std::vector<std::vector<float*>> data_;
  /*! \brief Number of elements for data to be used for calibration. */
  std::vector<std::vector<size_t>> data_sizes_;

  /*! \brief Device buffers to be used for calibration. */
  std::vector<void*> buffers_;

  /*! \brief Names of inputs */
  const std::vector<std::string> input_names_;

  /*! \brief Allocate device memory buffers. data_sizes_ must already have one
   * entry. */
  void AllocateBuffersIfNotAllocated() {
    if (!buffers_.empty()) return;
    CHECK_GE(data_sizes_.size(), 1);
    const int num_inputs = data_sizes_[0].size();
    buffers_.assign(num_inputs, nullptr);
    for (int i = 0; i < num_inputs; ++i) {
      CUDA_CALL(cudaMalloc(&buffers_[i], data_sizes_[0][i] * sizeof(float)));
    }
  }
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_CALIBRATOR_H_
