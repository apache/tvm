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
 *
 * \file tvm/relay/op/make_op.h
 * \brief Header of internal operator functions
 * to assist in creating ops in C++
 */
#ifndef TVM_RELAY_OP_MAKE_OP_H_
#define TVM_RELAY_OP_MAKE_OP_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

// Include Templated Make Functions
#include "nn/convolution_make.h"
#include "nn/pooling.h"

namespace tvm {
namespace relay {

Expr MakeBroadCastTo(Expr data, Expr shape);

Expr MakeCast(Expr data, DataType dtype);

Expr MakeClip(Expr a, double a_min, double a_max);

Expr MakeConcatenate(Expr data, int axis);

Expr MakeDense(Expr data, Expr weight, IndexExpr units, DataType out_dtype);

Expr MakeExpandDims(Expr data, int axis, int num_newaxis);

Expr MakeFull(Expr fill_value, Expr shape, DataType dtype);

Expr MakeLayoutTransform(Expr data, String src_layout, String dst_layout);

Expr MakeOnes(Expr shape, DataType dtype);

Expr MakePad(Expr data, Array<Array<IndexExpr>> pad_width, double pad_value, String pad_mode);

Expr MakeReduce(Expr data, Array<Integer> axis, bool keepdims, bool exclude, String op_name);

Expr MakeRepeat(Expr data, int repeats, int axis);

Expr MakeReshape(Expr data, Array<Integer> newshape);

Expr MakeSplit(Expr data, ObjectRef indices_or_sections, int axis);

Expr MakeSqueeze(Expr data, Array<Integer> axis);

Expr MakeStack(Expr data, int axis);

Expr MakeStridedSlice(Expr data, Expr begin, Expr end, Expr strides, String slice_mode);

Expr MakeTile(Expr data, Array<Integer> reps);

Expr MakeTopK(Expr data, int k, int axis, String ret_type, bool is_ascend, DataType dtype);

Expr MakeVariance(Expr data, Expr mean, Array<Integer> axis, bool keepdims, bool exclude);

Expr MakeZeros(Expr shape, DataType dtype);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_MAKE_OP_H_
