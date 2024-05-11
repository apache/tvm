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
#include "builtin.h"
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
  if (!TargetIsHopper(T.target)) return Stmt();
  bool is_load;
  if (src.scope() == "global" && (dst.scope() == "shared.dyn" || dst.scope() == "shared")) {
    // Use the Hopper TMA bulk copy instructions
    is_load = true;
  } else if (dst.scope() == "global" && (src.scope() == "shared.dyn" || src.scope() == "shared")) {
    is_load = false;
  } else {
    return Stmt();
  }
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

  Stmt tma_copy;

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
    tma_copy = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                   Evaluate(Call(DataType::Handle(), op, args)));
  } else {
    PrimExpr shared_addr = shared_tensor.access_ptr(is_load ? 2 : 1);
    args.push_back(shared_addr);
    for (auto coord : global_coords) args.push_back(coord);
    tma_copy = Evaluate(Call(DataType::Handle(), op, args));
  }
  tma_copy = IfThenElse(EQ(T.thread_var, 0), tma_copy);

  if (!is_load) {
    // TODO: Add this async proxy fence with a seperate pass
    auto fence_stmt = Evaluate(Call(DataType::Handle(), FenceProxyAsyncOp(), {}));
    tma_copy = SeqStmt({fence_stmt, tma_copy});
  }

  return tma_copy;
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

Conv2DIm2ColOp::Conv2DIm2ColOp(Array<PrimExpr> args, BufferMap vmap) {
  src = vmap[GetVarFromAccessPtr(args[0])];
  dst = vmap[GetVarFromAccessPtr(args[1])];
  nhw_step = args[2];
  c_step = args[3];
  kernel = args[4].as<IntImm>().value()->value;
  stride = args[5].as<IntImm>().value()->value;
  dilation = args[6].as<IntImm>().value()->value;
  padding = args[7].as<IntImm>().value()->value;
}

Stmt Conv2DIm2ColOp::Lower(const LowerArgs& T, arith::Analyzer* analyzer) const {
  ICHECK(TargetIsHopper(T.target));
  ICHECK(src.scope() == "global" && (dst.scope() == "shared.dyn" || dst.scope() == "shared"));
  ICHECK(src->shape.size() == 4);
  ICHECK(dst->shape.size() == 2);
  ICHECK(src->dtype == dst->dtype);
  Layout shared_layout;
  if (T.layout_map.count(dst)) {
    shared_layout = T.layout_map[dst];
  }

  TMAIm2ColDesc desc;
  desc.rank = src->shape.size();
  desc.data_type = to_CUtensorMapDataType(src->dtype);
  desc.global_addr = src->data;
  desc.global_shape = ReverseArray(src->shape);
  if (!src->strides.empty()) {
    desc.global_stride = ReverseArray(src->strides);
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
  desc.global_stride = desc.global_stride.Map([&](PrimExpr e) { return e * src->dtype.bytes(); });
  desc.elem_stride = {1, stride, stride, 1};
  desc.lower_corner = {-padding, -padding};
  desc.upper_corner = {-padding, -padding};
  desc.smem_box_pixel = Downcast<IntImm>(dst->shape[0])->value;
  desc.smem_box_channel = Downcast<IntImm>(dst->shape[1])->value;
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  if (!shared_layout.defined()) {
    desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else {
    ICHECK(shared_layout->InputDim() == 2) << "Cannot detect TMA layout.";
    auto stride = as_const_int(shared_layout->InputShape()[0]);
    auto continuous = as_const_int(shared_layout->InputShape()[1]);
    ICHECK(stride != nullptr && continuous != nullptr);
    if (StructuralEqual()(shared_layout,
                          makeHalfBankSwizzleLayout(*stride, *continuous, dst->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (StructuralEqual()(shared_layout, makeFullBankSwizzleLayout(*stride, *continuous,
                                                                          dst->dtype.bits()))) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else {
      ICHECK(0) << "Cannot detect TMA layout.";
    }
  }

  Call create_desc = Call(DataType::Handle(), CreateTMAIm2ColDescriptorOp(), desc.EncodeCallArgs());

  Array<PrimExpr> global_coords;  // c, w, h, n
  Array<PrimExpr> image_offset;   // w, h
  global_coords.reserve(desc.rank);

  ICHECK(analyzer->CanProveEqual(FloorMod(desc.global_shape[0], desc.smem_box_channel), 0))
      << "Currently can only support divisible channel case";

  global_coords.push_back(FloorMod(c_step * desc.smem_box_channel, desc.global_shape[0]));
  image_offset.push_back(
      dilation * FloorMod(FloorDiv(c_step * desc.smem_box_channel, desc.global_shape[0]), kernel));
  image_offset.push_back(dilation *
                         FloorDiv(c_step * desc.smem_box_channel, desc.global_shape[0] * kernel));

  PrimExpr h_dim = FloorDiv(src->shape[1] + 2 * padding - (kernel - 1) * dilation - 1, stride) + 1;
  PrimExpr w_dim = FloorDiv(src->shape[2] + 2 * padding - (kernel - 1) * dilation - 1, stride) + 1;
  global_coords.push_back(stride * FloorMod(nhw_step * desc.smem_box_pixel, w_dim) - padding);
  global_coords.push_back(
      stride * FloorMod(FloorDiv(nhw_step * desc.smem_box_pixel, w_dim), h_dim) - padding);
  global_coords.push_back(FloorDiv(nhw_step * desc.smem_box_pixel, w_dim * h_dim));

  Array<PrimExpr> args;
  args.reserve(desc.rank * 2 + 1);
  args.push_back(create_desc);
  args.push_back(0);  // mbar placeholder
  auto dst_buffer = T.buffer_remap.count(dst) ? T.buffer_remap[dst] : dst;
  auto shared_addr = dst_buffer.access_ptr(2);
  args.push_back(shared_addr);
  for (auto coord : global_coords) args.push_back(coord);
  for (auto offset : image_offset) args.push_back(offset);

  Stmt tma_copy =
      IfThenElse(EQ(T.thread_var, 0), Evaluate(Call(DataType::Handle(), TMALoadIm2ColOp(), args)));
  return tma_copy;
}

Array<PrimExpr> TMAIm2ColDesc::EncodeCallArgs() const {
  Array<PrimExpr> args;
  args.reserve(rank * 5 + 5);

  args.push_back(data_type);
  args.push_back(static_cast<int>(rank));
  args.push_back(global_addr);
  for (auto e : global_shape) args.push_back(e);
  for (auto e : global_stride) args.push_back(e);
  for (auto e : elem_stride) args.push_back(e);
  for (auto e : lower_corner) args.push_back(e);
  for (auto e : upper_corner) args.push_back(e);
  args.push_back(smem_box_pixel);
  args.push_back(smem_box_channel);
  args.push_back(interleave);
  args.push_back(swizzle);
  args.push_back(l2_promotion);
  args.push_back(oob_fill);

  return args;
}

TIR_REGISTER_TL_OP(Conv2DIm2ColOp, c2d_im2col)
    .set_num_inputs(8)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

}  // namespace tl
}  // namespace tvm