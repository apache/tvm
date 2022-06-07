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

#ifdef TF_TVMDSOOP_ENABLE_GPU
#include <cuda_runtime.h>
#endif
#include <dlpack/dlpack.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
typedef tensorflow::gtl::InlinedVector<tensorflow::int64, 4> ShapeContainer;

using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMArgsSetter;
using tvm::runtime::TVMRetValue;

// Op utility trait for diffrent device type template
template <typename DEVICE_TYPE>
class TVMDSOOpTrait;

// Buffer information used for actual computation.
// Each buffer is associated with one TensorFlow tensor
// whose underlying buffer is record into "origin_buf".
// For input tensor, we copy data from origin_buf to buf
// and for output tensor, copy data from buf to origin_buf
class TensorAsBuf {
 public:
  tensorflow::Tensor inline_tensor;
  tensorflow::Tensor* tensor;

  size_t size;
  size_t offset;

  int device_type;

  char* origin_buf;
  char* buf;

  void CopyToOrigin() {
    if (buf == origin_buf) {
      return;
    }
    if (device_type == kDLCPU) {
      memcpy(origin_buf, buf + offset, size);
#ifdef TF_TVMDSOOP_ENABLE_GPU
    } else if (device_type == kDLCUDA) {
      cudaMemcpy(origin_buf, buf + offset, size, cudaMemcpyDeviceToDevice);
#endif
    } else {
      LOG(FATAL) << "Only support CPU and CUDA now. Device " << device_type
                 << " is not implemented currently";
    }
  }

  void CopyFromOrigin() {
    if (buf == origin_buf) {
      return;
    }
    if (device_type == kDLCPU) {
      memcpy(buf + offset, origin_buf, size);
#ifdef TF_TVMDSOOP_ENABLE_GPU
    } else if (device_type == kDLCUDA) {
      cudaMemcpy(buf + offset, origin_buf, size, cudaMemcpyDeviceToDevice);
#endif
    } else {
      LOG(FATAL) << "Only support CPU and CUDA now. Device " << device_type
                 << " is not implemented currently";
    }
  }
};

tensorflow::Status GetDLPackDtype(const tensorflow::Tensor& tf_tensor, DLDataType* res) {
  auto dtype = tf_tensor.dtype();

  if (dtype == tensorflow::DT_HALF) {
    *res = {kDLFloat, 16, 1};
  } else if (dtype == tensorflow::DT_FLOAT) {
    *res = {kDLFloat, 32, 1};
  } else if (dtype == tensorflow::DT_DOUBLE) {
    *res = {kDLFloat, 64, 1};
  } else if (dtype == tensorflow::DT_INT8) {
    *res = {kDLInt, 8, 1};
  } else if (dtype == tensorflow::DT_INT16) {
    *res = {kDLInt, 16, 1};
  } else if (dtype == tensorflow::DT_INT32) {
    *res = {kDLInt, 32, 1};
  } else if (dtype == tensorflow::DT_INT64) {
    *res = {kDLInt, 64, 1};
  } else if (dtype == tensorflow::DT_UINT8) {
    *res = {kDLUInt, 8, 1};
  } else if (dtype == tensorflow::DT_UINT16) {
    *res = {kDLUInt, 16, 1};
  } else if (dtype == tensorflow::DT_UINT32) {
    *res = {kDLUInt, 32, 1};
  } else if (dtype == tensorflow::DT_UINT64) {
    *res = {kDLUInt, 64, 1};
  } else {
    return tensorflow::Status(tensorflow::error::INTERNAL, "Fail to get dlpack datatype");
  }
  return tensorflow::Status::OK();
}

// Ensure buffer used for actual computation take 64byte alignment
void EnsureAlignment(OpKernelContext* ctx, const tensorflow::Tensor& tensor, TensorAsBuf* out) {
  char* buf = const_cast<char*>(tensor.tensor_data().data());
  out->origin_buf = buf;
  out->size = tensor.TotalBytes();

  int alignment = 64;
  char* aligned = reinterpret_cast<char*>(((uint64_t)buf + alignment - 1) & (~(alignment - 1)));
  if (buf == aligned) {
    out->tensor = const_cast<tensorflow::Tensor*>(&tensor);
    out->buf = buf;
    out->offset = 0;
  } else {
    tensorflow::TensorShape buf_shape;
    tensorflow::int64 dims[1] = {(tensorflow::int64)(tensor.TotalBytes() + alignment)};
    tensorflow::TensorShapeUtils::MakeShape(dims, 1, &buf_shape);

    out->tensor = &out->inline_tensor;
    ctx->allocate_temp(tensor.dtype(), buf_shape, out->tensor);

    buf = const_cast<char*>(out->tensor->tensor_data().data());
    char* buf_aligned = reinterpret_cast<char*>(((uint64_t)buf + alignment) & (~(alignment - 1)));
    out->buf = buf;
    out->offset = buf_aligned - buf;
  }
}

// Create DLPack tensor from TensorFlow tensor
tensorflow::Status MakeDLTensor(const TensorAsBuf& src, const DLDevice& dev, int64_t* tf_shape,
                                DLTensor* out) {
  DLDataType dlpack_type;
  const tensorflow::Tensor& tensor = *src.tensor;

  auto status = GetDLPackDtype(tensor, &dlpack_type);
  if (!status.ok()) {
    return status;
  }
  out->device = dev;
  out->ndim = tensor.shape().dims();
  out->shape = tf_shape;
  out->strides = nullptr;
  out->byte_offset = 0;
  out->dtype = dlpack_type;
  out->data = src.buf + src.offset;
  return tensorflow::Status::OK();
}

template <>
class TVMDSOOpTrait<CPUDevice> {
 public:
  static const int device_type = kDLCPU;

  static int device_id(OpKernelContext* context) { return 0; }

  static void make_shape_from_tensor(const tensorflow::Tensor& shape_tensor,
                                     tensorflow::TensorShape* output_shape) {
    tensorflow::int64 num_dims = shape_tensor.NumElements();
    const tensorflow::int64* dims = shape_tensor.flat<tensorflow::int64>().data();
    tensorflow::TensorShapeUtils::MakeShape(dims, num_dims, output_shape);
  }
};

#ifdef TF_TVMDSOOP_ENABLE_GPU
template <>
class TVMDSOOpTrait<GPUDevice> {
 public:
  static const int device_type = kDLCUDA;

  static int device_id(OpKernelContext* context) {
    auto device_base = context->device();
    auto gpu_device_info = device_base->tensorflow_gpu_device_info();
    return gpu_device_info->gpu_id;
  }

  static void make_shape_from_tensor(const tensorflow::Tensor& shape_tensor,
                                     tensorflow::TensorShape* output_shape) {
    tensorflow::int64 num_dims = shape_tensor.NumElements();
    const tensorflow::int64* flat = shape_tensor.flat<tensorflow::int64>().data();
    tensorflow::int64* dims = new tensorflow::int64[num_dims];
    cudaMemcpy(dims, flat, sizeof(tensorflow::int64) * num_dims, cudaMemcpyDeviceToHost);
    tensorflow::TensorShapeUtils::MakeShape(dims, num_dims, output_shape);
    delete[] dims;
  }
};
#endif

template <typename DEVICE_TYPE>
class TVMDSOOp : public OpKernel {
 private:
  tvm::runtime::PackedFunc tvm_func;
  std::string lib_path;
  std::string func_name;

  tensorflow::DataType output_dtype;

  bool has_static_output_shape;
  std::vector<tensorflow::int64> static_output_shape;

  void initAttributes(OpKernelConstruction* context) {
    context->GetAttr("lib_path", &lib_path);
    context->GetAttr("func_name", &func_name);
    context->GetAttr("output_dtype", &output_dtype);

    context->GetAttr("has_static_output_shape", &has_static_output_shape);
    context->GetAttr("static_output_shape", &static_output_shape);
  }

 public:
  explicit TVMDSOOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get attr
    initAttributes(context);

    // Load TVM function from dynamic library
    tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(lib_path);
    tvm_func = mod_dylib.GetFunction(func_name);
    ICHECK(tvm_func != nullptr);
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    // the last input is output shape spec
    const int num_inputs = context->num_inputs() - 1;
    const int num_total_args = num_inputs + 1;
    std::vector<DLTensor> args(num_total_args);
    std::vector<TensorAsBuf> buf_info(num_inputs);
    std::vector<ShapeContainer> shapes(num_inputs);

    tensorflow::Status status;
    int device_id = TVMDSOOpTrait<DEVICE_TYPE>::device_id(context);
    int device_type = TVMDSOOpTrait<DEVICE_TYPE>::device_type;

    DLDevice dl_dev = {DLDeviceType(device_type), device_id};

    // Get output shape
    tensorflow::TensorShape output_shape;
    auto& output_shape_tensor = context->input(num_inputs);
    if (has_static_output_shape) {
      // use static output shape
      const tensorflow::int64* dims = static_output_shape.data();
      tensorflow::TensorShapeUtils::MakeShape(dims, static_output_shape.size(), &output_shape);
    } else if (output_shape_tensor.dims() == 1) {
      // use shape tensor values as output shape
      TVMDSOOpTrait<DEVICE_TYPE>::make_shape_from_tensor(output_shape_tensor, &output_shape);
    } else {
      // use input tensor shape by default
      output_shape = context->input(0).shape();
    }

    for (int i = 0; i < num_inputs; ++i) {
      // Grab the input tensor
      auto& input_tensor = context->input(i);

      // Create shape container, should keep ref during execution
      shapes[i] = input_tensor.shape().dim_sizes();
      auto shape_ptr = reinterpret_cast<int64_t*>(shapes[i].data());

      TensorAsBuf& input = buf_info[i];
      input.device_type = device_type;

      EnsureAlignment(context, input_tensor, &input);
      input.CopyFromOrigin();

      status = MakeDLTensor(input, dl_dev, shape_ptr, &args[i]);
      OP_REQUIRES_OK(context, status);
    }

    // Allocate output tensor
    tensorflow::Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    // shape dimension buf should keel alive on stack
    auto output_shape_dim_buf = output_tensor->shape().dim_sizes();
    auto output_shape_ptr = reinterpret_cast<int64_t*>(output_shape_dim_buf.data());

    TensorAsBuf output;
    output.device_type = device_type;
    EnsureAlignment(context, *output_tensor, &output);

    status = MakeDLTensor(output, dl_dev, output_shape_ptr, &args[num_inputs]);
    OP_REQUIRES_OK(context, status);

    // Prepare PackedFunc arguments
    std::vector<TVMValue> tvm_values(num_total_args);
    std::vector<int> tvm_type_codes(num_total_args);
    TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    for (int k = 0; k < num_total_args; ++k) {
      setter(k, &args[k]);
    }
    TVMRetValue rv;
    tvm_func.CallPacked(TVMArgs(tvm_values.data(), tvm_type_codes.data(), num_total_args), &rv);

    output.CopyToOrigin();
  }
};

#ifdef TF_TVMDSOOP_ENABLE_GPU
REGISTER_KERNEL_BUILDER(Name("TvmDsoOp").Device(tensorflow::DEVICE_CPU), TVMDSOOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("TvmDsoOp").Device(tensorflow::DEVICE_GPU), TVMDSOOp<GPUDevice>);
#else
REGISTER_KERNEL_BUILDER(Name("TvmDsoOp").Device(tensorflow::DEVICE_CPU), TVMDSOOp<CPUDevice>);
#endif
