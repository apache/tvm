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
 * \file type_relations.cc
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#include "./type_relations.h"

#include <tvm/arith/analyzer.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/tir/op.h>

#include <numeric>

namespace tvm {
namespace relay {

bool IdentityRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  for (size_t i = 1; i < types.size(); ++i) {
    reporter->Assign(types[i], types[0]);
  }
  return true;
}

bool EqualCheck(const IndexExpr& lhs, const IndexExpr& rhs) {
  IndexExpr diff = lhs - rhs;
  if (const int64_t* pdiff = tir::as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  // symbolic
  tvm::arith::Analyzer ana;
  diff = ana.Simplify(diff);
  if (const int64_t* pdiff = tir::as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  return false;
}

bool EqualConstInt(const IndexExpr& lhs, int64_t value) {
  if (const int64_t* pvalue = tir::as_const_int(lhs)) {
    return pvalue[0] == value;
  }
  return false;
}

TensorType ConcreteBroadcast(const TensorType& t1, const TensorType& t2, DataType output_dtype) {
  std::vector<IndexExpr> oshape;
  size_t ndim1 = t1->shape.size();
  size_t ndim2 = t2->shape.size();
  size_t i = 1;
  for (; i <= std::min(ndim1, ndim2); ++i) {
    IndexExpr s1 = t1->shape[ndim1 - i];
    IndexExpr s2 = t2->shape[ndim2 - i];
    if (EqualConstInt(s1, 1)) {
      oshape.push_back(s2);
    } else if (EqualConstInt(s2, 1)) {
      oshape.push_back(s1);
    } else if (s1.as<AnyNode>()) {
      // s1 == 1 || s1 == s2
      oshape.push_back(s2);
    } else if (s2.as<AnyNode>()) {
      // s2 == 1 || s2 == s1
      oshape.push_back(s1);
    } else if (EqualCheck(s1, s2)) {
      oshape.push_back(s1);
    } else {
      throw CompileError(ErrorBuilder() << "Incompatible broadcast type " << t1 << " and " << t2);
    }
  }

  size_t max_ndim = std::max(ndim1, ndim2);
  auto& rshape = (ndim1 > ndim2) ? t1->shape : t2->shape;
  for (; i <= max_ndim; ++i) {
    oshape.push_back(rshape[max_ndim - i]);
  }
  return TensorType(Array<IndexExpr>(oshape.rbegin(), oshape.rend()), output_dtype);
}

bool BroadcastRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  // DLOG(INFO) << "In1:" << types[0] << ",In2:" << types[1]
  //                 << ",Out:" << types[2] << std::endl;
  if (auto* t0 = types[0].as<TensorTypeNode>()) {
    if (auto* t1 = types[1].as<TensorTypeNode>()) {
      if (t0->dtype != t1->dtype) {
        reporter->GetDiagCtx().Emit(Diagnostic::Error(t0->span)
                                    << "data types " << t0->dtype << " and " << t1->dtype
                                    << " do not match in BroadcastRel");
      }
      reporter->Assign(
          types[2], ConcreteBroadcast(GetRef<TensorType>(t0), GetRef<TensorType>(t1), t0->dtype));
      return true;
    }
  }
  return false;
}

bool BroadcastCompRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  // DLOG(INFO) << "In1:" << types[0] << ",In2:" << types[1]
  //                 << ",Out:" << types[2] << std::endl;
  if (auto* t0 = types[0].as<TensorTypeNode>()) {
    if (auto* t1 = types[1].as<TensorTypeNode>()) {
      if (t0->dtype != t1->dtype) {
        reporter->GetDiagCtx().Emit(Diagnostic::Error(t0->span)
                                    << "data types " << t0->dtype << " and " << t1->dtype
                                    << " do not match in BroadcastCompRel");
      }
      reporter->Assign(types[2], ConcreteBroadcast(GetRef<TensorType>(t0), GetRef<TensorType>(t1),
                                                   DataType::Bool()));
      return true;
    }
  }
  return false;
}

bool IdentityCompRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  if (const auto* t0 = types[0].as<TensorTypeNode>()) {
    Type out_type = TensorType(t0->shape, DataType::Bool());
    reporter->Assign(types[1], out_type);
    return true;
  }
  return false;
}

Array<IndexExpr> RankShape(const Array<IndexExpr>& shape) {
  if (shape.size() == 0) {
    return {};
  } else {
    return {tvm::Integer(shape.size())};
  }
}

bool ShapeOfRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  ICHECK_EQ(num_inputs, 1);
  auto tt = types[0].as<TensorTypeNode>();
  if (tt == nullptr) {
    return false;
  }
  const auto* param = attrs.as<ShapeOfAttrs>();
  ICHECK(param != nullptr);
  auto rank_shape = RankShape(tt->shape);
  reporter->Assign(types[1], TensorType(rank_shape, param->dtype));
  return true;
}

}  // namespace relay
}  // namespace tvm
