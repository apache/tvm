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
 * \file tl/op/bulk_copy.cc
 * \brief Bulk copy operator.
 *
 */

#include "bulk_copy.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "../target/utils.h"
#include "cuda.h"

namespace tvm {
namespace tl {

using namespace tir;

static int to_CUtensorMapDataType(DataType dtype) {
  CUtensorMapDataType tp;
  if (dtype.is_float()) {
    switch (dtype.bits()) {
      case 64:
        tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
        break;
      case 32:
        tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        break;
      case 16:
        tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
        break;
      case 8:
        tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        break;
      default:
        ICHECK(0) << dtype;
    }
  } else if (dtype.is_bfloat16()) {
    tp = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (dtype.is_int()) {
    switch (dtype.bits()) {
      case 64:
        tp = CU_TENSOR_MAP_DATA_TYPE_INT64;
        break;
      case 32:
        tp = CU_TENSOR_MAP_DATA_TYPE_INT32;
        break;
      case 16:
        tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
        break;
      case 8:
        tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        break;
      default:
        ICHECK(0) << dtype;
    }
  } else if (dtype.is_uint()) {
    switch (dtype.bits()) {
      case 64:
        tp = CU_TENSOR_MAP_DATA_TYPE_UINT64;
        break;
      case 32:
        tp = CU_TENSOR_MAP_DATA_TYPE_UINT32;
        break;
      case 16:
        tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
        break;
      case 8:
        tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        break;
      default:
        ICHECK(0) << dtype;
    }
  } else {
    ICHECK(0) << dtype;
  }
  return static_cast<int>(tp);
}

template <typename T>
static Array<T> ReverseArray(Array<T> array) {
  return Array<T>{array.rbegin(), array.rend()};
}

Stmt Copy::LowerBulkCopy(const LowerArgs& T, arith::Analyzer* analyzer) const {
  bool is_load = src.scope() == "global";
  Buffer global_tensor = is_load ? src : dst;
  Buffer shared_tensor = is_load ? dst : src;
  Layout shared_layout;
  if (T.layout_map.count(shared_tensor)) {
    shared_layout = T.layout_map[shared_tensor];
    shared_tensor = T.buffer_remap[shared_tensor];
  }
  if (T.layout_map.count(global_tensor)) {
    ICHECK(T.layout_map.count(global_tensor) == 0) << "Cannot support global layout.";
  }
  ICHECK(shared_tensor.scope() == "shared" || shared_tensor.scope() == "shared.dyn");
  ICHECK(global_tensor.scope() == "global");

  TMADesc desc;

  // Verify copy rank
  desc.rank = global_tensor->shape.size();
  ICHECK(desc.rank >= 1 && desc.rank <= 5) << desc.rank;

  // Verify datatype
  ICHECK(global_tensor->dtype == shared_tensor->dtype);
  desc.data_type = to_CUtensorMapDataType(global_tensor->dtype);

  // Global Tensor Shape and Stride
  auto global_range = is_load ? src_range : dst_range;
  desc.global_addr = global_tensor->data;
  desc.global_shape = ReverseArray(global_tensor->shape);
  Array<PrimExpr> global_coords = ReverseArray(global_range.Map([](Range r) { return r->min; }));
  if (!global_tensor->strides.empty()) {
    desc.global_stride = ReverseArray(global_tensor->strides);
  } else {
    // Create stride from shape
    PrimExpr stride = 1;
    desc.global_stride.reserve(desc.rank);
    for (size_t i = 0; i < desc.rank; i++) {
      desc.global_stride.push_back(stride);
      stride *= desc.global_shape[i];
    }
  }
  // The first stride element should be 1
  ICHECK(is_one(desc.global_stride[0])) << desc.global_stride;
  // Make global stride in bytes
  desc.global_stride =
      desc.global_stride.Map([&](PrimExpr e) { return e * global_tensor->dtype.bytes(); });

  // Smem Box
  desc.smem_box = ReverseArray(global_range.Map([](Range r) { return r->extent; }));
  desc.smem_stride = Array<PrimExpr>(desc.rank, PrimExpr(1));

  // L2 & OOB
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // Detect smem layout
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  if (!shared_layout.defined()) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else {
    ICHECK(shared_layout->InputDim() == 2) << "Cannot detect TMA layout.";
    auto stride = as_const_int(shared_layout->InputShape()[0]);
    auto continuous = as_const_int(shared_layout->InputShape()[1]);
    ICHECK(stride != nullptr && continuous != nullptr);
    if (StructuralEqual()(shared_layout, makeHalfBankSwizzleLayout(*stride, *continuous,
                                                                   shared_tensor->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (StructuralEqual()(
                   shared_layout,
                   makeFullBankSwizzleLayout(*stride, *continuous, shared_tensor->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else {
      ICHECK(0) << "Cannot detect TMA layout.";
    }
  }

  auto inner_box_dim = as_const_int(desc.smem_box[0]);
  ICHECK(inner_box_dim != nullptr);
  int instruction_dim = *inner_box_dim;
  if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B)) {
    instruction_dim = 64 / src->dtype.bytes();
  } else if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B)) {
    instruction_dim = 128 / src->dtype.bytes();
  }
  ICHECK((*inner_box_dim) % instruction_dim == 0);
  desc.smem_box.Set(0, PrimExpr(instruction_dim));

  Call create_descriptor = Call(DataType::Handle(), CreateTMADescriptorOp(), desc.EncodeCallArgs());

  Array<PrimExpr> args;
  args.reserve(desc.rank + 3);
  args.push_back(create_descriptor);
  if (is_load) args.push_back(0);  // mbarrier id placeholder
  auto op = is_load ? TMALoadOp() : TMAStoreOp();

  if ((*inner_box_dim) != instruction_dim) {
    Var loop_var("i");
    int loop_extent = (*inner_box_dim) / instruction_dim;
    PrimExpr total_elements = 1;
    for (auto e : desc.smem_box) total_elements *= e;
    PrimExpr shared_addr = shared_tensor.access_ptr(is_load ? 2 : 1, DataType::Handle(), 1,
                                                    total_elements * loop_var, total_elements);
    args.push_back(shared_addr);
    global_coords.Set(0, global_coords[0] + instruction_dim * loop_var);
    for (auto coord : global_coords) args.push_back(coord);
    Call TMA_copy = Call(DataType::Handle(), op, args);
    For loop = For(loop_var, 0, loop_extent, ForKind::kUnrolled, Evaluate(TMA_copy));
    return loop;
  } else {
    PrimExpr shared_addr = shared_tensor.access_ptr(is_load ? 2 : 1);
    args.push_back(shared_addr);
    for (auto coord : global_coords) args.push_back(coord);
    Call TMA_copy = Call(DataType::Handle(), op, args);
    return Evaluate(TMA_copy);
  }
}

Array<PrimExpr> TMADesc::EncodeCallArgs() const {
  Array<PrimExpr> args;
  args.reserve(rank * 4 + 7);

  args.push_back(data_type);
  args.push_back(static_cast<int>(rank));
  args.push_back(global_addr);
  for (auto e : global_shape) args.push_back(e);
  for (auto e : global_stride) args.push_back(e);
  for (auto e : smem_box) args.push_back(e);
  for (auto e : smem_stride) args.push_back(e);
  args.push_back(interleave);
  args.push_back(swizzle);
  args.push_back(l2_promotion);
  args.push_back(oob_fill);

  return args;
}

DataType cuTensorMapType() { return DataType::UInt(8, 128); }

TIR_DEFINE_TL_BUILTIN(CreateTMADescriptorOp)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(TMALoadOp).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(TMAStoreOp)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(MBarrierWaitParity)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(LDMatrixOp)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

}  // namespace tl
}  // namespace tvm