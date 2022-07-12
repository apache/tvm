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
 * \brief Detail broadcast.
 * \file topi/detail/broadcast.h
 */
#ifndef TVM_TOPI_DETAIL_BROADCAST_H_
#define TVM_TOPI_DETAIL_BROADCAST_H_

#include <tvm/te/operation.h>
#include <tvm/topi/detail/constant_utils.h>

#include <algorithm>
#include <deque>
#include <string>

namespace tvm {
namespace topi {
namespace detail {

struct BroadcastHelper {
  std::deque<tvm::PrimExpr> common_shape;
  std::deque<tvm::tir::Var> all_vars;
  std::deque<tvm::tir::Var> vars1;
  std::deque<tvm::tir::Var> vars2;
};

static inline DataType CommonType(DataType type1, DataType type2) {
  ICHECK(type1.is_scalar() && type2.is_scalar());
  ICHECK(type1.code() == type2.code());
  return DataType(type1.code(), std::max(type1.bits(), type2.bits()), /*lanes=*/1);
}

inline BroadcastHelper BroadcastShape(const tvm::Array<tvm::PrimExpr>& shape1,
                                      const tvm::Array<tvm::PrimExpr>& shape2) {
  BroadcastHelper bh;
  int s1_size = shape1.size();
  int s2_size = shape2.size();
  tvm::PrimExpr one(1);
  int i;

  auto cast_if_needed = [](DataType to_type, PrimExpr expr) {
    return to_type != expr.dtype() ? cast(to_type, expr) : expr;
  };

  for (i = 1; i <= std::min(s1_size, s2_size); ++i) {
    // TODO(@icemelon9): Need to revisit this part
    const IntImmNode* static_size1 = shape1[s1_size - i].as<IntImmNode>();
    const IntImmNode* static_size2 = shape2[s2_size - i].as<IntImmNode>();
    DataType common_type = CommonType(shape1[s1_size - i].dtype(), shape2[s2_size - i].dtype());

    bh.all_vars.push_front(tvm::tir::Var("dim", common_type));
    if (topi::detail::EqualCheck(shape1[s1_size - i], shape2[s2_size - i])) {
      bh.common_shape.push_front(cast_if_needed(common_type, shape1[s1_size - i]));
      bh.vars1.push_front(bh.all_vars[0]);
      bh.vars2.push_front(bh.all_vars[0]);
    } else if (topi::detail::EqualCheck(one, shape1[s1_size - i])) {
      ICHECK(!topi::detail::EqualCheck(one, shape2[s2_size - i]));
      bh.common_shape.push_front(cast_if_needed(common_type, shape2[s2_size - i]));
      bh.vars2.push_front(bh.all_vars[0]);
    } else if (topi::detail::EqualCheck(one, shape2[s2_size - i])) {
      bh.common_shape.push_front(cast_if_needed(common_type, shape1[s1_size - i]));
      bh.vars1.push_front(bh.all_vars[0]);
    } else if (!static_size1 && !static_size2) {
      bh.common_shape.push_front(
          cast_if_needed(common_type, max(shape1[s1_size - i], shape2[s2_size - i])));
      bh.vars1.push_front(bh.all_vars[0]);
      bh.vars2.push_front(bh.all_vars[0]);
    } else if (!static_size1) {
      bh.common_shape.push_front(cast_if_needed(common_type, shape2[s2_size - i]));
      bh.vars2.push_front(bh.all_vars[0]);
      bh.vars1.push_front(bh.all_vars[0]);
    } else if (!static_size2) {
      bh.common_shape.push_front(cast_if_needed(common_type, shape1[s1_size - i]));
      bh.vars1.push_front(bh.all_vars[0]);
      bh.vars2.push_front(bh.all_vars[0]);
    } else {
      ICHECK(false) << "Incompatible broadcast dims: " << shape1[s1_size - i] << " and "
                    << shape2[s2_size - i]
                    << " in: " << tvm::Array<tvm::PrimExpr>(shape1.begin(), shape1.end()) << " and "
                    << tvm::Array<tvm::PrimExpr>(shape2.begin(), shape2.end());
    }
  }
  // Remaining dimensions whether on shape1 or shape2 can always be completed
  auto max_size = std::max(s1_size, s2_size);
  auto& shape = (s1_size > s2_size) ? shape1 : shape2;
  auto& vars = (s1_size > s2_size) ? bh.vars1 : bh.vars2;
  for (; i <= max_size; ++i) {
    bh.all_vars.push_front(tvm::tir::Var("v", shape[max_size - 1].dtype()));
    bh.common_shape.push_front(shape[max_size - i]);
    vars.push_front(bh.all_vars[0]);
  }
  return bh;
}

inline tvm::Array<tvm::PrimExpr> InputIndexFromBroadcast(
    const tvm::Array<tvm::tir::Var>& ovars, const tvm::te::Tensor& T,
    const std::deque<tvm::tir::Var>& my_vars, const std::deque<tvm::tir::Var>& all_vars) {
  tvm::Array<tvm::PrimExpr> ivars;
  ICHECK_EQ(ovars.size(), all_vars.size());
  // N^2, could use a map but NBD.
  size_t expected_dims = T->shape.size();
  for (size_t i = 0; i < ovars.size(); ++i) {
    bool found = false;
    for (size_t j = 0; j < my_vars.size(); ++j) {
      if (all_vars[i].same_as(my_vars[j])) {
        ivars.push_back(ovars[i]);
        found = true;
        break;
      }
    }
    // Only inject 0 here if we have not yet reached the dimension of I
    // (i.e. this must be a 1)
    if (!found && (ovars.size() - i) <= expected_dims) {
      ivars.push_back(tvm::tir::make_zero(ovars[i].dtype()));
    }
  }
  ICHECK(expected_dims == ivars.size());
  return ivars;
}

template <typename FBinaryExpr>
inline tvm::te::Tensor WithBroadcast(FBinaryExpr op, const tvm::te::Tensor& A,
                                     const tvm::te::Tensor& B, const std::string& name = "tensor",
                                     const std::string& tag = "") {
  auto bh = BroadcastShape(A->shape, B->shape);
  auto l = [&](tvm::Array<tvm::tir::Var> ovars) {
    return op(A(InputIndexFromBroadcast(ovars, A, bh.vars1, bh.all_vars)),
              B(InputIndexFromBroadcast(ovars, B, bh.vars2, bh.all_vars)));
  };
  return tvm::te::compute(tvm::Array<tvm::PrimExpr>(bh.common_shape.begin(), bh.common_shape.end()),
                          l, name, tag);
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_DETAIL_BROADCAST_H_
