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
 * \file edgetpu_runtime.cc
 */
#include "edgetpu_runtime.h"

#include <edgetpu.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

void EdgeTPURuntime::Init(const std::string& tflite_model_bytes, Device dev) {
  const char* buffer = tflite_model_bytes.c_str();
  size_t buffer_size = tflite_model_bytes.size();
  // Load compiled model as a FlatBufferModel

  // According to tflite_runtime.cc, the buffer for tflite::FlatBufferModel
  // should be allocated on flatBuffersBuffer_ to make share it must be kept alive
  // for interpreters.
  flatBuffersBuffer_ = std::make_unique<char[]>(buffer_size);
  std::memcpy(flatBuffersBuffer_.get(), buffer, buffer_size);
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(flatBuffersBuffer_.get(), buffer_size);
  // Build resolver
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // Init EdgeTPUContext object
  edgetpu_context_ = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  // Add custom edgetpu ops to resolver
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  // Build interpreter
  TfLiteStatus status = tflite::InterpreterBuilder(*model, resolver)(&interpreter_);
  CHECK_TFLITE_STATUS(status) << "Failed to build interpreter.";
  // Bind EdgeTPU context with interpreter.
  interpreter_->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context_.get());
  interpreter_->SetNumThreads(1);
  // Allocate tensors
  status = interpreter_->AllocateTensors();
  CHECK_TFLITE_STATUS(status) << "Failed to allocate tensors.";

  device_ = dev;
}

Module EdgeTPURuntimeCreate(const std::string& tflite_model_bytes, Device dev) {
  auto exec = make_object<EdgeTPURuntime>();
  exec->Init(tflite_model_bytes, dev);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.edgetpu_runtime.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = EdgeTPURuntimeCreate(args[0], args[1]);
});
}  // namespace runtime
}  // namespace tvm
