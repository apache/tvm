/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_relations.cc
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/logging.h>
#include <tvm/relay/op.h>
#include <tvm/ir_pass.h>
#include <numeric>
#include "./type_relations.h"

namespace tvm {
namespace relay {

TensorType ToTensorType(const Type& t) {
  if (const auto* tt_node = t.as<TensorTypeNode>()) {
    return GetRef<TensorType>(tt_node);
  } else {
    return TensorType(nullptr);
  }
}

bool IdentityRel(const Array<Type>& types,
                 int num_inputs,
                 const Attrs& attrs,
                 const TypeReporter& reporter) {
  for (size_t i = 1; i < types.size(); ++i) {
    reporter->Assign(types[i], types[0]);
  }
  return true;
}

bool EqualCheck(const IndexExpr& lhs,
                const IndexExpr& rhs) {
  IndexExpr diff = lhs - rhs;
  if (const int64_t* pdiff = as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  // symbolic
  diff = tvm::ir::CanonicalSimplify(diff);
  if (const int64_t* pdiff = as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  return false;
}

bool EqualConstInt(const IndexExpr& lhs, int64_t value) {
  if (const int64_t* pvalue = as_const_int(lhs)) {
    return pvalue[0] == value;
  }
  return false;
}

Type ConcreteBroadcast(const TensorType& t1,
                       const TensorType& t2,
                       DataType output_dtype) {
  std::vector<IndexExpr> oshape;
  size_t ndim1 = t1->shape.size();
  size_t ndim2 = t2->shape.size();
  size_t i = 1;
  for (; i <= std::min(ndim1, ndim2); ++i) {
    IndexExpr s1 = t1->shape[ndim1 - i];
    IndexExpr s2 = t2->shape[ndim2 - i];
    if (EqualCheck(s1, s2)) {
      oshape.push_back(s1);
    } else if (EqualConstInt(s1, 1)) {
      oshape.push_back(s2);
    } else if (EqualConstInt(s2, 1)) {
      oshape.push_back(s1);
    } else {
      RELAY_ERROR(
          "Incompatible broadcast type "
              << t1 << " and " << t2).Raise();
    }
  }

  size_t max_ndim = std::max(ndim1, ndim2);
  auto& rshape = (ndim1 > ndim2) ? t1->shape : t2->shape;
  for (; i <= max_ndim; ++i) {
    oshape.push_back(rshape[max_ndim - i]);
  }
  return TensorTypeNode::make(Array<IndexExpr>(
      oshape.rbegin(), oshape.rend()), output_dtype);
}

bool BroadcastRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  RELAY_LOG(INFO) << "In1:" << types[0] << ",In2:" << types[1]
                  << ",Out:" << types[2] << std::endl;
  if (auto t0 = ToTensorType(types[0])) {
    if (auto t1 = ToTensorType(types[1])) {
      CHECK_EQ(t0->dtype, t1->dtype);
      reporter->Assign(types[2],
        ConcreteBroadcast(t0, t1, t0->dtype));
      return true;
    }
  }
  return false;
}

bool BroadcastCompRel(const Array<Type>& types,
                      int num_inputs,
                      const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  RELAY_LOG(INFO) << "In1:" << types[0] << ",In2:" << types[1]
                  << ",Out:" << types[2] << std::endl;
  if (auto t0 = ToTensorType(types[0])) {
    if (auto t1 = ToTensorType(types[1])) {
      CHECK_EQ(t0->dtype, t1->dtype);
      reporter->Assign(types[2], ConcreteBroadcast(t0, t1, ::tvm::Bool()));
      return true;
    }
  }
  return false;
}

}  // namespace relay
}  // namespace tvm
