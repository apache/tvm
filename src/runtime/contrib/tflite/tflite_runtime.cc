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
 * \file tflite_runtime.cc
 */
#include "tflite_runtime.h"

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

#define TVM_DTYPE_DISPATCH(type, DType, ...)    \
  if (type == DataType::Float(64)) {            \
    typedef double DType;                       \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Float(32)) {     \
    typedef float DType;                        \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Float(16)) {     \
    typedef uint16_t DType;                     \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Int(64)) {       \
    typedef int64_t DType;                      \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Int(32)) {       \
    typedef int32_t DType;                      \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Int(16)) {       \
    typedef int16_t DType;                      \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::Int(8)) {        \
    typedef int8_t DType;                       \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::UInt(64)) {      \
    typedef uint64_t DType;                     \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::UInt(32)) {      \
    typedef uint32_t DType;                     \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::UInt(16)) {      \
    typedef uint16_t DType;                     \
    { __VA_ARGS__ }                             \
  } else if (type == DataType::UInt(8)) {       \
    typedef uint8_t DType;                      \
    { __VA_ARGS__ }                             \
  } else {                                      \
    LOG(FATAL) << "unknown data type " << type; \
  }

DataType TfLiteDType2TVMDType(TfLiteType dtype) {
  switch (dtype) {
    case kTfLiteFloat32:
      return DataType::Float(32);
    case kTfLiteInt32:
      return DataType::Int(32);
    case kTfLiteInt64:
      return DataType::Int(64);
    case kTfLiteInt16:
      return DataType::Int(16);
    case kTfLiteInt8:
      return DataType::Int(8);
    case kTfLiteUInt8:
      return DataType::UInt(8);
    case kTfLiteFloat16:
      return DataType::Float(16);
    default:
      LOG(FATAL) << "tflite data type not support yet: " << dtype;
  }
}

void TFLiteRuntime::Init(const std::string& tflite_model_bytes, Device dev) {
  const char* buffer = tflite_model_bytes.c_str();
  size_t buffer_size = tflite_model_bytes.size();
  // The buffer used to construct the model must be kept alive for
  // dependent interpreters to be used.
  flatBuffersBuffer_ = std::make_unique<char[]>(buffer_size);
  std::memcpy(flatBuffersBuffer_.get(), buffer, buffer_size);
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(flatBuffersBuffer_.get(), buffer_size);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // Build interpreter
  TfLiteStatus status = tflite::InterpreterBuilder(*model, resolver)(&interpreter_);
  CHECK_TFLITE_STATUS(status) << "Failed to build interpreter.";
  // Allocate tensors
  status = interpreter_->AllocateTensors();
  CHECK_TFLITE_STATUS(status) << "Failed to allocate tensors.";

  device_ = dev;
}

void TFLiteRuntime::Invoke() { interpreter_->Invoke(); }

void TFLiteRuntime::SetInput(int index, DLTensor* data_in) {
  DataType dtype(data_in->dtype);
  TVM_DTYPE_DISPATCH(dtype, DType, {
    DType* dest = interpreter_->typed_input_tensor<DType>(index);
    DType* src = static_cast<DType*>(data_in->data);
    ICHECK(data_in->strides == NULL);
    int64_t size = 1;
    for (int64_t i = 0; i < data_in->ndim; ++i) {
      size *= data_in->shape[i];
    }
    for (int64_t i = 0; i < size; ++i) {
      dest[i] = src[i];
    }
  });
}

void TFLiteRuntime::SetNumThreads(int num_threads) { interpreter_->SetNumThreads(num_threads); }

NDArray TFLiteRuntime::GetOutput(int index) const {
  TfLiteTensor* output = interpreter_->tensor(interpreter_->outputs()[index]);
  DataType dtype = TfLiteDType2TVMDType(output->type);
  TfLiteIntArray* dims = output->dims;
  int64_t size = 1;
  std::vector<int64_t> shape;
  for (int i = 0; i < dims->size; ++i) {
    shape.push_back(dims->data[i]);
    size *= dims->data[i];
  }
  NDArray ret = NDArray::Empty(shape, dtype, device_);
  TVM_DTYPE_DISPATCH(dtype, DType, {
    DType* dest = static_cast<DType*>(ret->data);
    DType* src = interpreter_->typed_output_tensor<DType>(index);
    for (int64_t i = 0; i < size; ++i) {
      dest[i] = src[i];
    }
  });
  return ret;
}

PackedFunc TFLiteRuntime::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int in_idx = args[0];
      ICHECK_GE(in_idx, 0);
      this->SetInput(in_idx, args[1]);
    });
  } else if (name == "get_output") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetOutput(args[0]); });
  } else if (name == "invoke") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Invoke(); });
  } else if (name == "set_num_threads") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int num_threads = args[0];
      CHECK_GE(num_threads, 1);
      this->SetNumThreads(num_threads);
    });
  } else {
    return PackedFunc();
  }
}

Module TFLiteRuntimeCreate(const std::string& tflite_model_bytes, Device dev) {
  auto exec = make_object<TFLiteRuntime>();
  exec->Init(tflite_model_bytes, dev);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.tflite_runtime.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = TFLiteRuntimeCreate(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("target.runtime.tflite").set_body_typed(TFLiteRuntimeCreate);
}  // namespace runtime
}  // namespace tvm
