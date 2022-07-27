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
 * \file strided_slice.h
 * \brief Utility functions for strided_slice op
 */
#ifndef TVM_TOPI_DETAIL_STRIDED_SLICE_H_
#define TVM_TOPI_DETAIL_STRIDED_SLICE_H_

#include <tvm/tir/expr.h>

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "constant_utils.h"

namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

inline int64_t CanonicalizeIndex(int64_t index, int64_t extent, int64_t stride) {
  int64_t begin_range = stride < 0 ? -1 : 0;
  int64_t end_range = stride < 0 ? extent - 1 : extent;
  if (index < 0) {
    index += extent;
  }
  return std::min(std::max(index, begin_range), end_range);
}

inline std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> ConvertToVec(
    const Array<Integer>& begin, const Array<Integer>& end, const Array<Integer>& strides,
    std::string slice_mode) {
  std::vector<int64_t> stride_vec(strides.size(), 1);
  if (slice_mode == "end") {
    for (size_t i = 0; i < strides.size(); ++i) {
      ICHECK(strides[i].defined());
      stride_vec[i] = GetConstInt(strides[i]);
    }
  }
  const int64_t max_range = std::numeric_limits<int64_t>::max();
  std::vector<int64_t> begin_vec;
  for (size_t i = 0; i < begin.size(); ++i) {
    if (!begin[i].defined()) {
      // value=None
      begin_vec.push_back(stride_vec[i] > 0 ? 0 : max_range);
    } else {
      begin_vec.push_back(GetConstInt(begin[i]));
    }
  }
  std::vector<int64_t> end_vec;
  for (size_t i = 0; i < end.size(); ++i) {
    // allow end to be None
    if (!end[i].defined()) {
      end_vec.push_back(stride_vec[i] < 0 ? 0 : max_range);
    } else if (slice_mode == "size") {
      int64_t end_val = GetConstInt(end[i]);
      if (end_val < 0) {
        end_vec.push_back(stride_vec[i] < 0 ? 0 : max_range);
      } else {
        end_vec.push_back(begin_vec[i] + end_val);
      }
    } else {
      end_vec.push_back(GetConstInt(end[i]));
    }
  }
  return std::make_tuple(begin_vec, end_vec, stride_vec);
}

inline Array<PrimExpr> StridedSliceCanonicalizeBegin(const Array<PrimExpr>& ishape,
                                                     const std::vector<int64_t>& begin,
                                                     const std::vector<int64_t>& strides,
                                                     const Array<Integer>& axes, DataType dtype,
                                                     std::string slice_mode = "end") {
  Array<PrimExpr> begin_expr;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (ishape[axes[i].IntValue()]->IsInstance<tvm::IntImmNode>()) {
      int64_t dim_i = GetConstInt(ishape[axes[i].IntValue()]);
      int64_t begin_i = CanonicalizeIndex(begin[i], dim_i, strides[i]);
      begin_expr.push_back(make_const(dtype, begin_i));
    } else {
      auto idim = ishape[axes[i].IntValue()];
      auto b_expr = make_const(dtype, begin[i]);
      PrimExpr b = begin[i] < 0 ? b_expr + idim : b_expr;
      auto s = strides[i];
      if (s < 0) {
        b = tvm::min(b, idim - 1);
      } else {
        b = tvm::if_then_else(b < 0, 0, b);
      }
      begin_expr.push_back(b);
    }
  }
  return begin_expr;
}

inline Array<PrimExpr> StridedSliceOutputShape(const Array<PrimExpr>& ishape,
                                               const std::vector<int64_t>& begin,
                                               const std::vector<int64_t>& end,
                                               const std::vector<int64_t>& strides,
                                               const Array<Integer>& axes, std::string slice_mode,
                                               const Array<PrimExpr>& begin_canonicalized,
                                               bool use_any = false) {
  const size_t src_tensor_dim = ishape.size();
  Array<PrimExpr> out_shape;
  for (size_t i = 0; i < src_tensor_dim; ++i) {
    out_shape.push_back(ishape[i]);
  }

  for (size_t i = 0; i < axes.size(); ++i) {
    if (ishape[axes[i].IntValue()]->IsInstance<tvm::IntImmNode>()) {
      const int64_t dim_i = GetConstInt(ishape[axes[i].IntValue()]);
      ICHECK(begin_canonicalized[i]->IsInstance<tvm::IntImmNode>());
      int64_t begin_i = GetConstInt(begin_canonicalized[i]);
      int64_t end_i = CanonicalizeIndex(end[i], dim_i, strides[i]);
      int interval = std::abs(end_i - begin_i);
      int slice_size =
          static_cast<int>((interval + std::abs(strides[i]) - 1) / std::abs(strides[i]));
      ICHECK(strides[i] < 0 ? (end_i <= begin_i) : (begin_i <= end_i))
          << ": Input [Begin=" << begin[i] << ", End=" << end[i] << "] is invalid for axis=" << i;
      out_shape.Set(axes[i].IntValue(), cast(out_shape[i].dtype(), PrimExpr(slice_size)));
    } else if (use_any) {
      out_shape.Set(axes[i].IntValue(), tvm::tir::Any());
    } else {
      out_shape.Set(axes[i].IntValue(), tvm::tir::Var("dim", out_shape[i]->dtype));
    }
  }

  return out_shape;
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_STRIDED_SLICE_H_
