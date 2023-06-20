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
 * \file src/relay/op/nn/pooling.h
 * \brief utilities for creating pool ops
 */
#ifndef TVM_RELAY_OP_NN_POOLING_H_
#define TVM_RELAY_OP_NN_POOLING_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>

#include <utility>

namespace tvm {
namespace relay {

template <typename T>
inline Expr MakeMaxPool(Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                        Array<IndexExpr> dilation, Array<IndexExpr> padding, String layout,
                        String out_layout, bool ceil_mode, String op_name) {
  auto attrs = make_object<T>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->dilation = std::move(dilation);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  attrs->ceil_mode = ceil_mode;
  static const Op& op = Op::Get(op_name);
  return Call(op, {data}, Attrs(attrs), {});
}

template <typename T>
inline Expr MakeAvgPool(Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                        Array<IndexExpr> dilation, Array<IndexExpr> padding, String layout,
                        String out_layout, bool ceil_mode, bool count_include_pad, String op_name) {
  auto attrs = make_object<T>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->dilation = std::move(dilation);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->out_layout = std::move(out_layout);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  static const Op& op = Op::Get(op_name);
  return Call(op, {data}, Attrs(attrs), {});
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_NN_POOLING_H_
