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
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/nn/batch_matmul.h>
#include <tvm/topi/nn/bias_add.h>
#include <tvm/topi/nn/bnn.h>
#include <tvm/topi/nn/dense.h>
#include <tvm/topi/nn/dilate.h>
#include <tvm/topi/nn/flatten.h>
#include <tvm/topi/nn/local_response_norm.h>
#include <tvm/topi/nn/mapping.h>
#include <tvm/topi/nn/pooling.h>
#include <tvm/topi/nn/softmax.h>

namespace tvm {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

/* Ops from nn.h */
TVM_REGISTER_GLOBAL("topi.nn.relu").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = relu<float>(args[0]);
});

TVM_REGISTER_GLOBAL("topi.nn.leaky_relu").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = leaky_relu(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.nn.prelu").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = prelu(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("topi.nn.pad").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = pad(args[0], args[1], args[2], args[3]);
});

/* Ops from nn/dense.h */
TVM_REGISTER_GLOBAL("topi.nn.dense").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::dense(args[0], args[1], args[2], args[3]);
});

/* Ops from nn/bias_add.h */
TVM_REGISTER_GLOBAL("topi.nn.bias_add").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::bias_add(args[0], args[1], args[2]);
});

/* Ops from nn/batch_matmul.h */
TVM_REGISTER_GLOBAL("topi.nn.batch_matmul").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::batch_matmul(args[0], args[1]);
});

/* Ops from nn/dilate.h */
TVM_REGISTER_GLOBAL("topi.nn.dilate").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::dilate(args[0], args[1], args[2]);
});

/* Ops from nn/flatten.h */
TVM_REGISTER_GLOBAL("topi.nn.flatten").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::flatten(args[0]);
});

/* Ops from nn/mapping.h */
TVM_REGISTER_GLOBAL("topi.nn.scale_shift_nchw").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::scale_shift_nchw(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("topi.nn.scale_shift_nhwc").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::scale_shift_nhwc(args[0], args[1], args[2]);
});

/* Ops from nn/pooling.h */
TVM_REGISTER_GLOBAL("topi.nn.pool").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::pool(args[0], args[1], args[2], args[3],
                 static_cast<nn::PoolType>(static_cast<int>(args[4])), args[5], args[6], args[7]);
});

TVM_REGISTER_GLOBAL("topi.nn.pool_grad").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::pool_grad(args[0], args[1], args[2], args[3], args[4],
                      static_cast<nn::PoolType>(static_cast<int>(args[5])), args[6], args[7],
                      args[8]);
});

TVM_REGISTER_GLOBAL("topi.nn.global_pool").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::global_pool(args[0], static_cast<nn::PoolType>(static_cast<int>(args[1])), args[2]);
});

TVM_REGISTER_GLOBAL("topi.nn.adaptive_pool").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::adaptive_pool(args[0], args[1], static_cast<nn::PoolType>(static_cast<int>(args[2])),
                          args[3]);
});

TVM_REGISTER_GLOBAL("topi.nn.adaptive_pool3d").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::adaptive_pool3d(args[0], args[1], static_cast<nn::PoolType>(static_cast<int>(args[2])),
                            args[3]);
});

TVM_REGISTER_GLOBAL("topi.nn.pool1d").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::pool1d(args[0], args[1], args[2], args[3],
                   static_cast<nn::PoolType>(static_cast<int>(args[4])), args[5], args[6], args[7]);
});

TVM_REGISTER_GLOBAL("topi.nn.pool3d").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::pool3d(args[0], args[1], args[2], args[3],
                   static_cast<nn::PoolType>(static_cast<int>(args[4])), args[5], args[6], args[7]);
});

/* Ops from nn/softmax.h */
TVM_REGISTER_GLOBAL("topi.nn.softmax").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::softmax(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.nn.log_softmax").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::log_softmax(args[0]);
});

TVM_REGISTER_GLOBAL("topi.nn.lrn").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::lrn(args[0], args[1], args[2], static_cast<double>(args[3]),
                static_cast<double>(args[4]), static_cast<double>(args[5]));
});

/* Ops from nn/bnn.h */
TVM_REGISTER_GLOBAL("topi.nn.binarize_pack").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::binarize_pack(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.nn.binary_dense").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = nn::binary_dense(args[0], args[1]);
});

}  // namespace topi
}  // namespace tvm
