/*
 * Licensed to the Apache Software Foundation (ASF) under one
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
 */

/*!
 * \file ethosn_device.cc
 * \brief Arm(R) Ethos(TM)-N NPU device integration.
 */

#include "ethosn_device.h"

#include <dlpack/dlpack.h>
#include <poll.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/tir/expr.h>
#include <unistd.h>

#include <algorithm>
#include <memory>

#include "ethosn_driver_library/Buffer.hpp"
#include "ethosn_runtime.h"
#include "ethosn_support_library/Support.hpp"

#if defined ETHOSN_HW

#include "ethosn_driver_library/Inference.hpp"
#include "ethosn_driver_library/Network.hpp"

namespace tvm {
namespace runtime {
namespace ethosn {

namespace dl = ::ethosn::driver_library;

bool WaitForInference(dl::Inference* inference, int timeout) {
  // Wait for inference to complete
  int fd = inference->GetFileDescriptor();
  struct pollfd fds;
  memset(&fds, 0, sizeof(fds));
  fds.fd = fd;
  fds.events = POLLIN;  // Wait for any available input.

  const int ms_per_seconds = 1000;
  int poll_result = poll(&fds, 1, timeout * ms_per_seconds);
  if (poll_result > 0) {
    dl::InferenceResult result;
    if (read(fd, &result, sizeof(result)) != sizeof(result)) {
      return false;
    }
    if (result != dl::InferenceResult::Completed) {
      return false;
    }
  } else if (poll_result == 0) {
    return false;
  } else {
    return false;
  }
  return true;
}

void CreateBuffers(std::vector<std::shared_ptr<dl::Buffer>>* fm,
                   const std::vector<DLTensor*>& tensors, const std::vector<uint32_t>& tensor_sizes,
                   bool input) {
  for (size_t i = 0; i < tensors.size(); i++) {
    auto* data = static_cast<uint8_t*>(tensors[i]->data);
    if (input) {
      (*fm)[i] = std::make_shared<dl::Buffer>(data, tensor_sizes[i], dl::DataFormat::NHWC);
    } else {
      (*fm)[i] = std::make_shared<dl::Buffer>(tensor_sizes[i], dl::DataFormat::NHWC);
    }
  }
}

bool Inference(tvm::runtime::TVMArgs args, dl::Network* npu,
               const std::vector<uint32_t>& input_order, const std::vector<uint32_t>& output_order,
               const std::vector<uint32_t>& input_sizes,
               const std::vector<uint32_t>& output_sizes) {
  // Unpack parameters
  size_t n_inputs = input_order.size();
  size_t n_outputs = output_order.size();
  std::vector<DLTensor*> inputs(n_inputs);
  for (size_t i = 0; i < n_inputs; i++) {
    inputs[i] = args[input_order[i]];
  }
  std::vector<DLTensor*> outputs(n_outputs);
  size_t output_offset = n_inputs;
  for (size_t i = 0; i < n_outputs; i++) {
    outputs[i] = args[output_order[i] + output_offset];
  }

  // Set up input buffers
  std::vector<std::shared_ptr<dl::Buffer>> ifm(n_inputs);
  CreateBuffers(&ifm, inputs, input_sizes, true);

  // Set up output buffers
  std::vector<std::shared_ptr<dl::Buffer>> ofm(n_outputs);
  CreateBuffers(&ofm, outputs, output_sizes, false);

  // Raw pointers for the inference
  dl::Buffer* ifm_raw[n_inputs];
  for (size_t i = 0; i < n_inputs; i++) {
    ifm_raw[i] = ifm[i].get();
  }
  dl::Buffer* ofm_raw[n_outputs];
  for (size_t i = 0; i < n_outputs; i++) {
    ofm_raw[i] = ofm[i].get();
  }

  // Execute the inference.
  std::unique_ptr<dl::Inference> result(
      npu->ScheduleInference(ifm_raw, n_inputs, ofm_raw, n_outputs));
  bool inferenceCompleted = WaitForInference(result.get(), 60);
  if (inferenceCompleted) {
    for (size_t i = 0; i < n_outputs; i++) {
      DLTensor* tensor = outputs[i];
      dl::Buffer* source_buffer = ofm_raw[i];
      uint8_t* dest_buffer = static_cast<uint8_t*>(tensor->data);
      size_t size = source_buffer->GetSize();
      uint8_t* source_buffer_data = source_buffer->Map();
      std::copy(source_buffer_data, source_buffer_data + size, dest_buffer);
      source_buffer->Unmap();
    }
  }
  return inferenceCompleted;
}

}  // namespace ethosn
}  // namespace runtime
}  // namespace tvm

#else
/* If USE_ETHOSN_HW=OFF, we mock the inference call with a known-good output.
 * That output can be set by using relay.ethos-n.test.infra.inference_result
 * which will set the values the mocked inference will return the next time
 * it's called.
 */

#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace ethosn {

namespace sl = ::ethosn::support_library;

std::vector<tvm::runtime::NDArray> test_outputs;

TVM_REGISTER_GLOBAL("relay.ethos-n.test.infra.inference_result")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      test_outputs.clear();
      for (int argc = 0; argc < args.size(); argc++) {
        const DLTensor* tensor = args[argc];
        auto shape = std::vector<int64_t>(tensor->shape, tensor->shape + tensor->ndim);
        test_outputs.emplace_back(
            tvm::runtime::NDArray::Empty(shape, tensor->dtype, tensor->device));
        test_outputs[test_outputs.size() - 1].CopyFrom(tensor);
      }
    });

// Allow the ethos-n support code to be tested without a device
bool Inference(tvm::runtime::TVMArgs args, dl::Network* /* npu */,
               const std::vector<uint32_t>& input_order, const std::vector<uint32_t>& output_order,
               const std::vector<uint32_t>& input_sizes,
               const std::vector<uint32_t>& output_sizes) {
  std::vector<DLTensor*> outputs;
  for (int argc = input_order.size(); argc < args.size(); argc++) {
    outputs.push_back(args[argc]);
  }
  bool rc = false;
  if (test_outputs.size() == outputs.size()) {
    for (auto i = 0u; i < outputs.size(); i++) {
      test_outputs[i].CopyTo(outputs[i]);
    }
    rc = true;
  }
  // Clear after first usage; on-exit destructor of NDArray fails
  test_outputs.clear();
  return rc;
}

}  // namespace ethosn
}  // namespace runtime
}  // namespace tvm

#endif
