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
 * \brief Registration of transform operators
 * \file transform.cc
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/topi/einsum.h>
#include <tvm/topi/transform.h>
#include <tvm/topi/utils.h>

#include <iostream>

#include "tvm/ir/expr.h"

namespace tvm {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_REGISTER_GLOBAL("topi.expand_dims").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = expand_dims(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("topi.transpose").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = transpose(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.flip").set_body([](TVMArgs args, TVMRetValue* rv) {
  // pass empty seq_lengths tensor to reverse_sequence
  *rv = reverse_sequence(args[0], Tensor(), args[1]);
});

TVM_REGISTER_GLOBAL("topi.reverse_sequence").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = reverse_sequence(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("topi.reshape").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = reshape(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.sliding_window").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = sliding_window(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("topi.squeeze").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = squeeze(args[0], ArrayOrInt(args[1]));
});

TVM_REGISTER_GLOBAL("topi.concatenate").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = concatenate(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.stack").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = stack(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.shape").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = shape(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.ndarray_size").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = ndarray_size(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.split").set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args[1].type_code() == kDLInt || args[1].type_code() == kDLUInt) {
    *rv = split_sections(args[0], args[1], args[2]);
  } else {
    *rv = split(args[0], args[1], args[2]);
  }
});

TVM_REGISTER_GLOBAL("topi.layout_transform").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = layout_transform(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("topi.take").set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args.size() == 4) {
    std::string mode = args[3];
    int batch_dims = args[2];
    *rv = take(args[0], args[1], batch_dims, mode);
  } else {
    ICHECK_EQ(args.size(), 5) << "topi.take expects 4 or 5 arguments";
    int batch_dims = args[2];
    int axis = args[3];
    std::string mode = args[4];
    *rv = take(args[0], args[1], batch_dims, axis, mode);
  }
});

TVM_REGISTER_GLOBAL("topi.sequence_mask").set_body([](TVMArgs args, TVMRetValue* rv) {
  double pad_val = args[2];
  int axis = args[3];
  *rv = sequence_mask(args[0], args[1], pad_val, axis);
});

TVM_REGISTER_GLOBAL("topi.where").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = where(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("topi.arange").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = arange(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("topi.meshgrid").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = meshgrid(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.repeat").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = repeat(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("topi.tile").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = tile(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.gather").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = gather(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("topi.gather_nd").set_body([](TVMArgs args, TVMRetValue* rv) {
  int batch_dims = args[2];
  *rv = gather_nd(args[0], args[1], batch_dims);
});

TVM_REGISTER_GLOBAL("topi.unravel_index").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = unravel_index(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("topi.sparse_to_dense").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = sparse_to_dense(args[0], args[1], args[2], args[3]);
});

TVM_REGISTER_GLOBAL("topi.matmul").set_body([](TVMArgs args, TVMRetValue* rv) {
  switch (args.size()) {
    case 2:
      *rv = matmul(args[0], args[1]);
      break;
    case 3:
      *rv = matmul(args[0], args[1], args[2]);
      break;
    case 4:
      *rv = matmul(args[0], args[1], args[2], args[3]);
      break;
    default:
      ICHECK(0) << "topi.matmul expects 2, 3 or 4 arguments";
  }
});

TVM_REGISTER_GLOBAL("topi.tensordot").set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args.size() == 2) {
    *rv = tensordot(args[0], args[1]);
  } else if (args.size() == 3) {
    *rv = tensordot(args[0], args[1], args[2]);
  } else {
    Array<PrimExpr> axes = args[3];
    *rv = tensordot(args[0], args[1], args[2], axes);
  }
});

TVM_REGISTER_GLOBAL("topi.strided_slice").set_body([](TVMArgs args, TVMRetValue* rv) {
  Tensor x = args[0];
  Array<PrimExpr> begin = args[1];
  Array<PrimExpr> end = args[2];
  Array<PrimExpr> strides = args[3];
  Array<Integer> axes = args[4];
  bool assume_inbound = args[6];
  if (IsConstIntArray(begin) && IsConstIntArray(end) && IsConstIntArray(strides) &&
      IsConstIntArray(x->shape)) {
    Array<Integer> begin_static = args[1];
    Array<Integer> end_static = args[2];
    Array<Integer> strides_static = args[3];
    std::string slice_mode = args[5];
    if (axes.size()) {
      *rv = strided_slice_with_axes(x, begin_static, end_static, strides_static, axes, slice_mode);
    } else {
      *rv = strided_slice(x, begin_static, end_static, strides_static, slice_mode);
    }
  } else {
    if (axes.size()) {
      *rv = dynamic_strided_slice_with_axes(x, begin, end, strides, axes, assume_inbound);
    } else {
      *rv = dynamic_strided_slice(x, begin, end, strides, assume_inbound);
    }
  }
});

TVM_REGISTER_GLOBAL("topi.dynamic_strided_slice").set_body([](TVMArgs args, TVMRetValue* rv) {
  te::Tensor begin = args[1];
  te::Tensor end = args[2];
  te::Tensor strides = args[3];
  *rv = dynamic_strided_slice(args[0], begin, end, strides);
});

TVM_REGISTER_GLOBAL("topi.relax_dynamic_strided_slice").set_body([](TVMArgs args, TVMRetValue* rv) {
  te::Tensor begin = args[1];
  te::Tensor end = args[2];
  te::Tensor strides = args[3];
  Array<PrimExpr> output_shape = args[4];
  *rv = relax::dynamic_strided_slice(args[0], begin, end, strides, output_shape);
});

TVM_REGISTER_GLOBAL("topi.one_hot").set_body([](TVMArgs args, TVMRetValue* rv) {
  int depth = args[3];
  int axis = args[4];
  DataType dtype = args[5];
  *rv = one_hot(args[0], args[1], args[2], depth, axis, dtype);
});

TVM_REGISTER_GLOBAL("topi.matrix_set_diag").set_body([](TVMArgs args, TVMRetValue* rv) {
  int k1 = args[2];
  int k2 = args[3];
  bool super_diag_right_align = args[4];
  bool sub_diag_right_align = args[5];
  *rv = matrix_set_diag(args[0], args[1], k1, k2, super_diag_right_align, sub_diag_right_align);
});

TVM_REGISTER_GLOBAL("topi.adv_index").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = adv_index(args[0], args[1]);
});

}  // namespace topi
}  // namespace tvm
