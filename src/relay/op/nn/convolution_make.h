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
 * \file src/relay/op/nn/convolution_make.h
 * \brief utilities for creating convolution ops
 */
#ifndef TVM_RELAY_OP_NN_CONVOLUTION_MAKE_H_
#define TVM_RELAY_OP_NN_CONVOLUTION_MAKE_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {

template <typename T>
inline Expr MakeConv(Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                     Array<IndexExpr> dilation, int groups, IndexExpr channels,
                     Array<IndexExpr> kernel_size, std::string data_layout,
                     std::string kernel_layout, std::string out_layout, DataType out_dtype,
                     std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

template <typename T>
inline Expr MakeConvWinograd(Expr data, Expr weight, int tile_size, Array<IndexExpr> strides,
                             Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
                             IndexExpr channels, Array<IndexExpr> kernel_size,
                             std::string data_layout, std::string kernel_layout,
                             std::string out_layout, DataType out_dtype, std::string op_name) {
  auto attrs = make_object<T>();
  attrs->tile_size = tile_size;
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

template <typename T>
inline Expr MakeConvGemm(Expr data, Expr weight, Array<IndexExpr> strides, Array<IndexExpr> padding,
                         Array<IndexExpr> dilation, int groups, IndexExpr channels,
                         Array<IndexExpr> kernel_size, std::string data_layout,
                         std::string kernel_layout, std::string out_layout, DataType out_dtype,
                         std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

template <typename T>
inline Expr MakeConvTranspose(Expr data, Expr weight, Array<IndexExpr> strides,
                              Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
                              IndexExpr channels, Array<IndexExpr> kernel_size,
                              std::string data_layout, std::string kernel_layout,
                              std::string out_layout, Array<IndexExpr> output_padding,
                              DataType out_dtype, std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->output_padding = std::move(output_padding);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

template <typename T>
inline Expr MakeDeformableConv(Expr data, Expr offset, Expr weight, Array<IndexExpr> strides,
                               Array<IndexExpr> padding, Array<IndexExpr> dilation,
                               int deformable_groups, int groups, int channels,
                               Array<IndexExpr> kernel_size, std::string data_layout,
                               std::string kernel_layout, std::string out_layout,
                               DataType out_dtype, std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = strides;
  attrs->padding = padding;
  attrs->dilation = dilation;
  attrs->deformable_groups = deformable_groups;
  attrs->groups = groups;
  attrs->channels = channels;
  attrs->kernel_size = kernel_size;
  attrs->data_layout = data_layout;
  attrs->kernel_layout = kernel_layout;
  attrs->out_layout = out_layout;
  attrs->out_dtype = out_dtype;
  const Op& op = Op::Get(op_name);
  return Call(op, {data, offset, weight}, Attrs{attrs}, {});
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_NN_CONVOLUTION_MAKE_H_
