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
 * \brief Registration of NN operators
 * \file nn.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/nn/bias_add.h>
#include <tvm/topi/nn/bnn.h>
#include <tvm/topi/nn/dense.h>
#include <tvm/topi/nn/dilate.h>
#include <tvm/topi/nn/flatten.h>
#include <tvm/topi/nn/group_norm.h>
#include <tvm/topi/nn/instance_norm.h>
#include <tvm/topi/nn/layer_norm.h>
#include <tvm/topi/nn/local_response_norm.h>
#include <tvm/topi/nn/mapping.h>
#include <tvm/topi/nn/pooling.h>
#include <tvm/topi/nn/rms_norm.h>
#include <tvm/topi/nn/softmax.h>

namespace tvm {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

/* Ops from nn.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.relu").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = relu<float>(args[0].cast<te::Tensor>());
});

TVM_FFI_REGISTER_GLOBAL("topi.nn.leaky_relu")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = leaky_relu(args[0].cast<te::Tensor>(), args[1].cast<double>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.prelu").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = prelu(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(), args[2].cast<int>());
});

TVM_FFI_REGISTER_GLOBAL("topi.nn.pad").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = pad(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
            args[2].cast<Array<PrimExpr>>(), args[3].cast<PrimExpr>());
});

TVM_FFI_REGISTER_GLOBAL("topi.nn.space_to_batch_nd")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = space_to_batch_nd(args[0].cast<te::Tensor>(), args[1].cast<Array<Integer>>(),
                              args[2].cast<Array<PrimExpr>>(), args[3].cast<Array<PrimExpr>>(),
                              args[4].cast<PrimExpr>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.batch_to_space_nd")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = batch_to_space_nd(args[0].cast<te::Tensor>(), args[1].cast<Array<Integer>>(),
                              args[2].cast<Array<PrimExpr>>(), args[3].cast<Array<PrimExpr>>(),
                              args[4].cast<std::string>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.nll_loss").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nll_loss(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(), args[2].cast<te::Tensor>(),
                 args[3].cast<std::string>(), args[4].cast<int>());
});

/* Ops from nn/dense.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.dense").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::dense(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                  args[2].cast<te::Tensor>(), args[3].cast<DataType>());
});

/* Ops from nn/bias_add.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.bias_add").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::bias_add(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(), args[2].cast<int>());
});

/* Ops from nn/dilate.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.dilate").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::dilate(args[0].cast<te::Tensor>(), args[1].cast<Array<Integer>>(),
                   args[2].cast<double>());
});

/* Ops from nn/flatten.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.flatten").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::flatten(args[0].cast<te::Tensor>());
});

/* Ops from nn/mapping.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.scale_shift_nchw")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::scale_shift_nchw(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                 args[2].cast<te::Tensor>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.scale_shift_nhwc")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::scale_shift_nhwc(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                                 args[2].cast<te::Tensor>());
    });

/* Ops from nn/pooling.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.pool_grad")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::pool_grad(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                          args[2].cast<Array<PrimExpr>>(), args[3].cast<Array<PrimExpr>>(),
                          args[4].cast<Array<PrimExpr>>(),
                          static_cast<nn::PoolType>(args[5].cast<int>()), args[6].cast<bool>(),
                          args[7].cast<std::string>(), args[8].cast<bool>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.global_pool")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::global_pool(args[0].cast<te::Tensor>(),
                            static_cast<nn::PoolType>(args[1].cast<int>()),
                            args[2].cast<std::string>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.adaptive_pool1d")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::adaptive_pool1d(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
                                static_cast<nn::PoolType>(args[2].cast<int>()),
                                args[3].cast<std::string>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.adaptive_pool")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::adaptive_pool(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
                              static_cast<nn::PoolType>(args[2].cast<int>()),
                              args[3].cast<std::string>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.adaptive_pool3d")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::adaptive_pool3d(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
                                static_cast<nn::PoolType>(args[2].cast<int>()),
                                args[3].cast<std::string>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.pool1d").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::pool1d(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
                   args[2].cast<Array<PrimExpr>>(), args[3].cast<Array<PrimExpr>>(),
                   args[4].cast<Array<PrimExpr>>(), static_cast<nn::PoolType>(args[5].cast<int>()),
                   args[6].cast<bool>(), args[7].cast<std::string>(), args[8].cast<bool>());
});

TVM_FFI_REGISTER_GLOBAL("topi.nn.pool2d").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::pool2d(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
                   args[2].cast<Array<PrimExpr>>(), args[3].cast<Array<PrimExpr>>(),
                   args[4].cast<Array<PrimExpr>>(), static_cast<nn::PoolType>(args[5].cast<int>()),
                   args[6].cast<bool>(), args[7].cast<std::string>(), args[8].cast<bool>());
});

TVM_FFI_REGISTER_GLOBAL("topi.nn.pool3d").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::pool3d(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
                   args[2].cast<Array<PrimExpr>>(), args[3].cast<Array<PrimExpr>>(),
                   args[4].cast<Array<PrimExpr>>(), static_cast<nn::PoolType>(args[5].cast<int>()),
                   args[6].cast<bool>(), args[7].cast<std::string>(), args[8].cast<bool>());
});

/* Ops from nn/softmax.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.softmax").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::softmax(args[0].cast<te::Tensor>(), args[1].cast<int>());
});

TVM_FFI_REGISTER_GLOBAL("topi.nn.log_softmax")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::log_softmax(args[0].cast<te::Tensor>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.lrn").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::lrn(args[0].cast<te::Tensor>(), args[1].cast<int>(), args[2].cast<int>(),
                args[3].cast<double>(), args[4].cast<double>(), args[5].cast<double>());
});

/* Ops from nn/bnn.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.binarize_pack")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::binarize_pack(args[0].cast<te::Tensor>(), args[1].cast<int>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.nn.binary_dense")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::binary_dense(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>());
    });

/* Ops from nn/layer_norm.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.layer_norm")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::layer_norm(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                           args[2].cast<te::Tensor>(), args[3].cast<Array<Integer>>(),
                           args[4].cast<double>());
    });

/* Ops from nn/group_norm.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.group_norm")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::group_norm(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                           args[2].cast<te::Tensor>(), args[3].cast<int>(), args[4].cast<int>(),
                           args[5].cast<Array<Integer>>(), args[6].cast<double>());
    });

/* Ops from nn/instance_norm.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.instance_norm")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = nn::instance_norm(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                              args[2].cast<te::Tensor>(), args[3].cast<int>(),
                              args[4].cast<Array<Integer>>(), args[5].cast<double>());
    });

/* Ops from nn/rms_norm.h */
TVM_FFI_REGISTER_GLOBAL("topi.nn.rms_norm").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  *rv = nn::rms_norm(args[0].cast<te::Tensor>(), args[1].cast<te::Tensor>(),
                     args[2].cast<Array<Integer>>(), args[3].cast<double>());
});

}  // namespace topi
}  // namespace tvm
