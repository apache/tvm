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
#include <topi/transform.h>
#include <topi/util.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

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
  *rv = layout_transform(args[0], args[1], args[2]);
});

TVM_REGISTER_GLOBAL("topi.take").set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args.size() == 3) {
    std::string mode = args[2];
    *rv = take(args[0], args[1], mode);
  } else {
    int axis = args[2];
    std::string mode = args[3];
    *rv = take(args[0], args[1], axis, mode);
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
  *rv = gather_nd(args[0], args[1]);
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
      CHECK(0) << "topi.matmul expects 2, 3 or 4 arguments";
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
  *rv = strided_slice(args[0], args[1], args[2], args[3], args[4]);
});

TVM_REGISTER_GLOBAL("topi.one_hot").set_body([](TVMArgs args, TVMRetValue* rv) {
  int depth = args[3];
  int axis = args[4];
  DataType dtype = args[5];
  *rv = one_hot(args[0], args[1], args[2], depth, axis, dtype);
});

}  // namespace topi
