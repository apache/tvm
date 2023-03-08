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
 * \file src/runtime/contrib/onednn/softmax.cc
 * \brief Use external onednn softmax function
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

//#include "cudnn_utils.h"
//#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_common.h"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

namespace tvm {
namespace contrib {

using namespace runtime;
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

void softmax_impl(TVMArgs args, TVMRetValue* ret) {
  DLTensor* x = args[0];
  DLTensor* y = args[1];
  int axis = args[2];
  int ndim = x->ndim;
  int64_t* shape = x->shape;
  if (axis < 0) axis += ndim;
  ICHECK(axis >= 0 && axis < ndim);
  std::cout << "#### onednn softmax_impl x->ndim "
            << x->ndim << " y->ndim " << y->ndim  << " axis " << axis << std::endl;
  std::cout << "shape " << shape[0] << " " << shape[1] << std::endl;

  // Create execution dnnl::engine.
    dnnl::engine engine(dnnl::engine::kind::gpu, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    //const memory::dim N = 3, // batch size
    //        IC = 1000; // channels

    // Source (src) and destination (dst) tensors dimensions.
    //memory::dims src_dims = {N, IC};
    memory::dims src_dims = {shape[0], shape[1]};

    // Allocate buffer.
    //std::vector<float> src_data(product(src_dims));

    //std::generate(src_data.begin(), src_data.end(), []() {
    //    static int i = 0;
    //    return std::cos(i++ / 10.f);
    //});

    // Create src memory descriptor and memory object.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nc);
    auto dst_md = memory::desc(src_dims, dt::f32, tag::nc);
    //auto src_mem = memory(src_md, engine);
    std::cout << "## debug 80" << std::endl;
    //cl::BufferDescriptor* desc_x = x->data;
    //auto src_mem = dnnl::ocl_interop::make_memory(src_md, engine, desc_x->buffer);
    auto src_mem = memory(src_md, engine, x->data);
    
    //auto dst_mem = dnnl::ocl_interop::make_memory(dst_md, engine, dnnl::ocl_interop::memory_kind::usm, y->data);
    auto dst_mem = memory(dst_md, engine, y->data);
    std::cout << "## debug 84" << std::endl;

    // Write data to memory object's handle.
    //write_to_dnnl_memory(src_data.data(), src_mem);

    // Softmax axis.
    //const int axis = 1;

    // Create primitive descriptor.
    auto softmax_pd = softmax_forward::primitive_desc(engine,
            prop_kind::forward_training, algorithm::softmax_accurate, src_md,
            dst_md, axis);
    std::cout << "## debug 100" << std::endl;
    // Create the primitive.
    auto softmax_prim = softmax_forward(softmax_pd);
    std::cout << "## debug 103" << std::endl;

    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, dst_mem});

    std::cout << "## debug 110" << std::endl;

    // Primitive execution.
    softmax_prim.execute(engine_stream, softmax_args);
    std::cout << "## debug 114" << std::endl;

    // Wait for the computation to finalize.
    engine_stream.wait();


#if 0
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  entry_ptr->softmax_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);

  // Set mode and shape descriptor
  if (axis == ndim - 1) {
    int64_t N = 1;
    for (int i = 0; i < ndim - 1; ++i) {
      N *= shape[i];
    }
    entry_ptr->softmax_entry.mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->softmax_entry.shape_desc, CUDNN_TENSOR_NCHW,
                                          entry_ptr->softmax_entry.data_type, static_cast<int>(N),
                                          static_cast<int>(shape[ndim - 1]), 1, 1));
  } else {
    int64_t pre_axis_dim = 1;
    int64_t post_axis_dim = 1;
    for (int i = 0; i < ndim; ++i) {
      if (i < axis) {
        pre_axis_dim *= shape[i];
      } else if (i > axis) {
        post_axis_dim *= shape[i];
      }
    }
    entry_ptr->softmax_entry.mode = CUDNN_SOFTMAX_MODE_CHANNEL;
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->softmax_entry.shape_desc, CUDNN_TENSOR_NCHW, entry_ptr->softmax_entry.data_type,
        static_cast<int>(pre_axis_dim), static_cast<int>(shape[axis]),
        static_cast<int>(post_axis_dim), 1));
  }

  auto alpha = CuDNNDataType::GetConst<1>(entry_ptr->softmax_entry.data_type);
  auto beta = CuDNNDataType::GetConst<0>(entry_ptr->softmax_entry.data_type);
  CUDNN_CALL(cudnnSoftmaxForward(entry_ptr->handle, alg, entry_ptr->softmax_entry.mode, alpha,
                                 entry_ptr->softmax_entry.shape_desc, x->data, beta,
                                 entry_ptr->softmax_entry.shape_desc, y->data));
#endif
}

TVM_REGISTER_GLOBAL("tvm.contrib.onednn.softmax.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      softmax_impl(args, ret);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.onednn.log_softmax.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) { softmax_impl(args, ret); });

}  // namespace contrib
}  // namespace tvm
