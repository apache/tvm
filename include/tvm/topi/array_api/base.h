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
#ifndef TVM_TOPI_ARRAY_API_BASE_H_
#define TVM_TOPI_ARRAY_API_BASE_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace topi {
namespace array_api {

class NotDerivable : public tvm::Error {
 public:
  explicit NotDerivable(const std::string& msg) : Error(msg) {}
};

inline bool ProveEqual(arith::Analyzer* analyzer, PrimExpr a, PrimExpr b) {
  if (analyzer) {
    return analyzer->CanProveEqual(a, b);
  } else {
    return tir::ExprDeepEqual()(a, b);
  }
}

inline Array<PrimExpr> BroadcastShape(const Array<PrimExpr>& a, const Array<PrimExpr>& b,
                                      arith::Analyzer* analyzer) {
  int64_t ndim_a = a.size();
  int64_t ndim_b = b.size();
  int64_t ndim = std::max(ndim_a, ndim_b);
  std::vector<PrimExpr> out_shape;
  out_shape.reserve(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    PrimExpr dim_a = i < ndim_a ? a[ndim_a - 1 - i] : IntImm(DataType::Int(64), 1);
    PrimExpr dim_b = i < ndim_b ? b[ndim_b - 1 - i] : IntImm(DataType::Int(64), 1);
    if (tir::is_one(dim_a)) {
      out_shape.push_back(dim_b);
    } else if (tir::is_one(dim_b)) {
      out_shape.push_back(dim_a);
    } else if (ProveEqual(analyzer, dim_a, dim_b)) {
      out_shape.push_back(dim_a);
    } else if (tir::is_const_int(dim_a) && tir::is_const_int(dim_b)) {
      LOG(FATAL) << "ValueError: Cannot broadcast shapes: " << a << ", " << b;
    } else {
      throw NotDerivable("NotDeducible: symbolic dimension broadcasting is not supported");
    }
  }
  std::reverse(out_shape.begin(), out_shape.end());
  return out_shape;
}

inline Array<PrimExpr> BroadcastShape(const Array<PrimExpr>& a, const Array<PrimExpr>& b,
                                      const Array<PrimExpr>& c, arith::Analyzer* analyzer) {
  Array<PrimExpr> shape;
  shape = BroadcastShape(a, b, analyzer);
  shape = BroadcastShape(shape, c, analyzer);
  return shape;
}

inline Array<PrimExpr> BroadcastIndices(const Array<PrimExpr>& indices,
                                        const Array<PrimExpr>& x_shape,
                                        const Array<PrimExpr>& out_shape) {
  static const IntImm zero = IntImm(DataType::Int(64), 0);
  int ndim_x = x_shape.size();
  int ndim_indices = indices.size();
  std::vector<PrimExpr> out_indices;
  out_indices.reserve(ndim_x);
  for (int i = 0; i < ndim_x; ++i) {
    if (i < ndim_indices) {
      if (tir::is_one(x_shape[ndim_x - 1 - i]) && !tir::is_one(out_shape[ndim_indices - 1 - i])) {
        out_indices.push_back(zero);
      } else {
        out_indices.push_back(indices[ndim_indices - 1 - i]);
      }
    } else {
      out_indices.push_back(zero);
    }
  }
  std::reverse(out_indices.begin(), out_indices.end());
  return out_indices;
}

inline std::string _StringifyIntVector(const std::vector<int>& vec) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << vec[i];
  }
  os << "]";
  return os.str();
}

inline int NormalizeAxis(int ndim, int axis) {
  if (axis >= ndim || axis < -ndim) {
    LOG(FATAL) << "ValueError: Expect axis to be in range [" << -ndim << ", " << ndim
               << "), but got: " << axis;
  }
  return axis < 0 ? axis + ndim : axis;
}

inline std::vector<int64_t> NormalizeAxes(int ndim, const std::vector<int>& axes) {
  std::vector<int64_t> out_axes;
  out_axes.reserve(axes.size());
  for (int axis : axes) {
    if (axis >= ndim || axis < -ndim) {
      LOG(FATAL) << "ValueError: Expect axis to be in range [" << -ndim << ", " << ndim
                 << "), but got: " << _StringifyIntVector(axes);
    }
    out_axes.push_back(axis < 0 ? axis + ndim : axis);
  }
  if (std::set<int>(out_axes.begin(), out_axes.end()).size() != out_axes.size()) {
    LOG(FATAL) << "ValueError: Duplicate axes in: " << _StringifyIntVector(axes)
               << " where ndim = " << ndim;
  }
  std::sort(out_axes.begin(), out_axes.end());
  return out_axes;
}

inline std::vector<int64_t> NormalizeAxes(int ndim, const Array<IntImm>& axes) {
  std::vector<int> converted_axes;
  converted_axes.reserve(axes.size());
  for (const IntImm& axis : axes) {
    converted_axes.push_back(axis->value);
  }
  return NormalizeAxes(ndim, converted_axes);
}

inline PrimExpr ProdShape(const Array<PrimExpr>& shape) {
  if (shape.empty()) {
    return IntImm(DataType::Int(64), 1);
  }
  PrimExpr prod = shape[0];
  for (uint32_t i = 1; i < shape.size(); ++i) {
    prod *= shape[i];
  }
  return prod;
}

inline PrimExpr RavelIndex(const Array<PrimExpr>& indices, const Array<PrimExpr>& shape) {
  CHECK_EQ(indices.size(), shape.size())
      << "ValueError: indices and shape must have equal size, "
      << "but got " << indices.size() << " and " << shape.size() << " respectively";
  if (indices.empty()) {
    return IntImm(DataType::Int(64), 1);
  }
  PrimExpr i = indices[0];
  for (uint32_t j = 1; j < indices.size(); ++j) {
    i = i * shape[j] + indices[j];
  }
  return i;
}

inline Array<PrimExpr> UnravelIndex(PrimExpr i, const Array<PrimExpr>& shape) {
  int ndim = shape.size();
  std::vector<PrimExpr> indices;
  indices.reserve(ndim);
  for (int j = ndim - 1; j >= 0; --j) {
    indices.push_back(tvm::floormod(i, shape[j]));
    i = tvm::floordiv(i, shape[j]);
  }
  std::reverse(indices.begin(), indices.end());
  return indices;
}

inline std::vector<int64_t> ArrayToVector(const Array<IntImm>& array) {
  std::vector<int64_t> vec;
  vec.reserve(array.size());
  for (const IntImm& expr : array) {
    vec.push_back(expr->value);
  }
  return vec;
}

inline Array<IntImm> VectorToArray(const std::vector<int64_t>& vec) {
  Array<IntImm> array;
  array.reserve(vec.size());
  for (int64_t value : vec) {
    array.push_back(IntImm(DataType::Int(64), value));
  }
  return array;
}

}  // namespace array_api
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_ARRAY_API_BASE_H_
