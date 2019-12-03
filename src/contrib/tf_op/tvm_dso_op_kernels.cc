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

#include <cstdio>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "tensorflow/core/framework/op_kernel.h"

#include "index_seq.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
typedef gtl::InlinedVector<int64, 4> ShapeContainer;


template <typename DEVICE_TYPE>
class TVMDSOOpTrait;


class TensorAsBuf {
  public:
    Tensor inline_tensor;
    Tensor* tensor;

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
        } else {
            cudaMemcpy(origin_buf, buf + offset, size, cudaMemcpyDeviceToDevice);
        }
    }

    void CopyFromOrigin() {
        if (buf == origin_buf) {
            return;
        }
        if (device_type == kDLCPU) {
            memcpy(buf + offset, origin_buf, size); 
        } else {
            cudaMemcpy(buf + offset, origin_buf, size, cudaMemcpyDeviceToDevice);
        }
    }
};


int GetDLPackDtype(const Tensor& tf_tensor, DLDataType* res) {
    auto dtype = tf_tensor.dtype();
    if (dtype == DT_FLOAT) {
      res->code = kDLFloat;
      res->bits = 32;
      res->lanes = 1;
    } else if (dtype == DT_INT64) {
      res->code = kDLInt;
      res->bits = 64;
      res->lanes = 1;
    } else if (dtype == DT_INT32) {
      res->code = kDLInt;
      res->bits = 32;
      res->lanes = 1;
    } else {
      return -1;
    }
    return 0;
}


void EnsureAlignment(OpKernelContext* ctx, const Tensor& tensor, TensorAsBuf* out) {
    char* buf = (char*) tensor.tensor_data().data();
    out->origin_buf = buf;
    out->size = tensor.TotalBytes(); 

    int alignment = 64;
    char* aligned = (char*)(((uint64_t)buf + alignment - 1) & (~ (alignment - 1)));
    if (buf == aligned) {
        out->tensor = const_cast<Tensor*>(&tensor);
        out->buf = buf;
        out->offset = 0;
    } else {
        TensorShape buf_shape;
        int64 dims[1] = { (int64)(tensor.TotalBytes() + alignment) }; 
        TensorShapeUtils::MakeShape(dims, 1, &buf_shape);
        
        out->tensor = &out->inline_tensor;
        ctx->allocate_temp(tensor.dtype(), buf_shape, out->tensor);
        
        buf = (char*)(out->tensor->tensor_data().data());
        char* buf_aligned = (char*)(((uint64_t)buf + alignment) & (~ (alignment - 1)));
        out->buf = buf;
        out->offset = buf_aligned - buf;
    }
}


int MakeDLTensor(const TensorAsBuf& src, const DLContext& ctx, int64_t* tf_shape, DLTensor* out) {
    DLDataType dlpack_type;
    const Tensor& tensor = *src.tensor;

    int status = GetDLPackDtype(tensor, &dlpack_type);
    if (status != 0) {
        return status;
    }
    out->ctx = ctx;
    out->ndim = tensor.shape().dims();
    out->shape = tf_shape;
    out->strides = NULL;
    out->byte_offset = 0;
    out->dtype = dlpack_type;    
    out->data = src.buf + src.offset;
    return 0;
}


template <>
class TVMDSOOpTrait<CPUDevice> {
  public:
    static const int device_type = kDLCPU;

    static int device_id(OpKernelContext* context) {
        return 0;
    }

};


template <>
class TVMDSOOpTrait<GPUDevice> {
  public:
    static const int device_type = kDLGPU;

    static int device_id(OpKernelContext* context) {
        auto device_base = context->device();
        auto gpu_device_info = device_base->tensorflow_gpu_device_info();
        return gpu_device_info->gpu_id;
    }
};


template <typename DEVICE_TYPE, int NUM_INPUTS>
class TVMDSOOp : public OpKernel {

private:
  tvm::runtime::PackedFunc tvm_func;
  string lib_path;
  string func_name;

  DataType output_dtype;
  
  bool has_static_output_shape;
  std::vector<int64> static_output_shape;

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
    LOG(INFO) << "Verify dynamic loading from " << lib_path << " device_type=" << TVMDSOOpTrait<DEVICE_TYPE>::device_type;
    tvm_func = mod_dylib.GetFunction(func_name);
    CHECK(tvm_func != nullptr);
  }
  
  void Compute(OpKernelContext* context) override {

    DLTensor args[NUM_INPUTS + 1];
    TensorAsBuf buf_info[NUM_INPUTS];
    ShapeContainer shapes[NUM_INPUTS];

    int status;
    int device_id = TVMDSOOpTrait<DEVICE_TYPE>::device_id(context);
    int device_type = TVMDSOOpTrait<DEVICE_TYPE>::device_type;
    
    DLContext dl_ctx = { DLDeviceType(device_type), device_id };

    // Get output shape
    TensorShape output_shape;
    auto& output_shape_tensor = context->input(NUM_INPUTS);
    if (has_static_output_shape) {
      // use static output shape
      const int64* dims = static_output_shape.data();
      TensorShapeUtils::MakeShape(dims, static_output_shape.size(), &output_shape);
    } else if (output_shape_tensor.dims() == 1) {
      // use shape tensor values as output shape
      const int64* dims = output_shape_tensor.flat<int64>().data();
      TensorShapeUtils::MakeShape(dims, 1, &output_shape);
    } else {
      // use input tensor shape by default
      output_shape = context->input(0).shape();
    }
    
    for (int i = 0; i < NUM_INPUTS; ++i) {
        // Grab the input tensor
        auto& input_tensor = context->input(i);

        // Create shape container, should keep ref during execution
        shapes[i] = input_tensor.shape().dim_sizes();
        auto shape_ptr = (int64_t*) shapes[i].data();

        TensorAsBuf& input = buf_info[i];
        input.device_type = device_type;

        EnsureAlignment(context, input_tensor, &input);
        input.CopyFromOrigin();

        status = MakeDLTensor(input, dl_ctx, shape_ptr, &args[i]);
        OP_REQUIRES(context, status == 0, Status(error::INTERNAL, "Fail to create dlpack tensor for input"));
    }

    // Allocate output tensor
    Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output_shape_dim_buf = output_tensor->shape().dim_sizes(); // should keep alive on stack 
    auto output_shape_ptr = (int64_t*) output_shape_dim_buf.data();
    
    TensorAsBuf output;
    output.device_type = device_type;
    EnsureAlignment(context, *output_tensor, &output);

    status = MakeDLTensor(output, dl_ctx, output_shape_ptr, &args[NUM_INPUTS]);
    OP_REQUIRES(context, status == 0, Status(error::INTERNAL, "Fail to create dlpack tensor for output"));
   
    apply_variadic_by_ptrs(tvm_func, args);
   
    output.CopyToOrigin(); 
  }
};



#define REGISTER_TFTVM_KERNEL(n) \
    REGISTER_KERNEL_BUILDER(Name("TvmDsoOp" #n).Device(DEVICE_CPU), TVMDSOOp<CPUDevice, n>); \
    REGISTER_KERNEL_BUILDER(Name("TvmDsoOp" #n).Device(DEVICE_GPU), TVMDSOOp<GPUDevice, n>); \

REGISTER_TFTVM_KERNEL(1)
REGISTER_TFTVM_KERNEL(2)
REGISTER_TFTVM_KERNEL(3)
REGISTER_TFTVM_KERNEL(4)
REGISTER_TFTVM_KERNEL(5)
REGISTER_TFTVM_KERNEL(6)
REGISTER_TFTVM_KERNEL(7)
REGISTER_TFTVM_KERNEL(8)

