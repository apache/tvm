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
 * \file tl/op/bulk_copy.h
 * \brief Bulk copy operator.
 *
 */

#ifndef TVM_TL_OP_BULK_COPY_H_
#define TVM_TL_OP_BULK_COPY_H_

#include "elem.h"

namespace tvm {
namespace tl {

using namespace tir;

struct TMADesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape, global_stride;
  Array<PrimExpr> smem_box, smem_stride;
  PrimExpr global_addr;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;

  Array<PrimExpr> EncodeCallArgs() const;
};

DataType cuTensorMapType();

struct TMAIm2ColDesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape, global_stride, elem_stride; // rank
  Array<PrimExpr> lower_corner, upper_corner; // rank - 2
  PrimExpr global_addr;
  int smem_box_pixel, smem_box_channel;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;

  Array<PrimExpr> EncodeCallArgs() const;
};

class Conv2DIm2ColOp : public Operator {
 public:
  Conv2DIm2ColOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const final;
  static const Op& Get();

 private:
  Buffer src, dst;
  int stride, padding, dilation, kernel;
  PrimExpr nhw_step, c_step;
};

}  // namespace tl
}  // namespace tvm

#endif  //  TVM_TL_OP_BULK_COPY_H_