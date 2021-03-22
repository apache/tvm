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
 * \brief Ethos-N NPU device integration.
 */

#include <dlpack/dlpack.h>
#include <poll.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/tir/expr.h>
#include <unistd.h>

#include <algorithm>
#include <memory>

#include "ethosn_driver_library/Buffer.hpp"
#include "ethosn_support_library/Support.hpp"

#if defined ETHOSN_HW

#include "ethosn_driver_library/Inference.hpp"
#include "ethosn_driver_library/Network.hpp"

namespace tvm {
namespace runtime {
namespace ethosn {

namespace sl = ::ethosn::support_library;
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

template <typename T>
void CopyOutput(dl::Buffer* source_buffers[], std::vector<DLTensor*>* outputs) {
  for (DLTensor* tensor : *outputs) {
    dl::Buffer* source_buffer = source_buffers[0];
    uint8_t* source_buffer_data = source_buffer->GetMappedBuffer();
    size_t size = source_buffer->GetSize();
    T* dest_pointer = static_cast<T*>(tensor->data);
    std::copy_backward(source_buffer_data, source_buffer_data + size, dest_pointer + size);
    source_buffers++;
  }
}

void CreateBuffers(std::vector<std::shared_ptr<dl::Buffer> >* fm,
                   const std::vector<DLTensor*>& tensors) {
  int index = 0;
  for (auto buffer : tensors) {
    auto* data = static_cast<uint8_t*>(buffer->data);
    // The NPU only needs the size of the tensor * uint8_t.
    auto data_size = static_cast<uint32_t>(GetDataSize(*buffer));
    (*fm)[index++] = std::make_shared<dl::Buffer>(data, data_size, dl::DataFormat::NHWC);
  }
}

bool Inference(tvm::runtime::TVMArgs args, sl::CompiledNetwork* network,
               const std::vector<uint32_t>& input_order,
               const std::vector<uint32_t>& output_order) {
  // Unpack parameters
  uint8_t argc = 0;
  std::vector<DLTensor*> inputs(input_order.size());
  for (uint8_t i = 0; i < network->GetInputBufferInfos().size(); i++) {
    inputs[input_order[i]] = args[argc++];
  }
  auto out_infos = network->GetOutputBufferInfos();
  std::vector<DLTensor*> outputs(output_order.size());
  for (uint8_t i = 0; i < network->GetOutputBufferInfos().size(); i++) {
    outputs[output_order[i]] = args[argc++];
  }

  // Set up input buffers
  std::vector<std::shared_ptr<dl::Buffer> > ifm(inputs.size());
  CreateBuffers(&ifm, inputs);

  // Set up output buffers
  std::vector<std::shared_ptr<dl::Buffer> > ofm(outputs.size());
  CreateBuffers(&ofm, outputs);

  // Raw pointers for the inference
  dl::Buffer* ifm_raw[inputs.size()];
  for (size_t i = 0; i < inputs.size(); i++) {
    ifm_raw[i] = ifm[i].get();
  }
  dl::Buffer* ofm_raw[outputs.size()];
  for (size_t i = 0; i < outputs.size(); i++) {
    ofm_raw[i] = ofm[i].get();
  }

  auto npu = std::make_unique<dl::Network>(*network);

  // Execute the inference.
  std::unique_ptr<dl::Inference> result(
      npu->ScheduleInference(ifm_raw, sizeof(ifm_raw) / sizeof(ifm_raw[0]), ofm_raw,
                             sizeof(ofm_raw) / sizeof(ofm_raw[0])));
  bool inferenceCompleted = WaitForInference(result.get(), 60);
  if (inferenceCompleted) {
    switch ((outputs)[0]->dtype.bits) {
      case 8: {
        dl::Buffer** ofms = &ofm_raw[0];
        for (DLTensor* tensor : outputs) {
          uint8_t* source_buffer_data = (*ofms++)->GetMappedBuffer();
          uint8_t* dest_pointer = static_cast<uint8_t*>(tensor->data);
          if (source_buffer_data != dest_pointer) {
            CopyOutput<uint8_t>(ofm_raw, &outputs);
            break;
          }
        }
        break;
      }
      case 16:
        CopyOutput<uint16_t>(ofm_raw, &outputs);
        break;
      case 32:
        CopyOutput<uint32_t>(ofm_raw, &outputs);
        break;
      default:
        break;
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
bool Inference(tvm::runtime::TVMArgs args, sl::CompiledNetwork* network,
               const std::vector<uint32_t>& input_order,
               const std::vector<uint32_t>& output_order) {
  std::vector<DLTensor*> outputs;
  for (int argc = network->GetInputBufferInfos().size(); argc < args.size(); argc++) {
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
