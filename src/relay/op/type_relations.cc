/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_relations.cc
 * \brief A set of utilities and common functionality
 * for type relations.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/logging.h>
#include <tvm/relay/op.h>
#include "../pass/incomplete_type.h"

namespace tvm {
namespace relay {

TensorType as_ttype(const Type& t) {
  if (auto tt_node = t.as<TensorTypeNode>()) {
    return GetRef<TensorType>(tt_node);
  } else {
    return TensorType(nullptr);
  }
}

// TODO(@jroesch) what size value do we extract?
int to_int(const tvm::Expr& e) {
  CHECK(e.defined());
  auto imm = e.as<tvm::ir::IntImm>();
  CHECK(imm) << "TYPE: " << imm << imm->type << std::endl;
  return imm->value;
}

Array<Type> IdentityRel(const Array<Type>& types, int num_args) {
  CHECK_EQ(types.size(), 2);
  auto t1 = as_ttype(types[0]);
  if (t1 && types[1].as<IncompleteTypeNode>()) {
    return {t1, t1};
  } else {
    return types;
  }
}

static Type ConcreteBroadcast(const TensorType& t1, const TensorType& t2,
                              DataType output_dtype) {
  RELAY_LOG(INFO) << "ConcreteBroadcast: t1=" << t1 << " t2=" << t2
                  << std::endl;
  auto sh1 = t1->shape;
  auto sh2 = t2->shape;
  RELAY_LOG(INFO) << "ConcreteBroadcast: sh1=" << sh1 << " sh2=" << sh2
                  << std::endl;
  if (sh1.size() == 0 && sh2.size() == 0) {
    return TensorTypeNode::make({}, output_dtype);
    // We have non-zero shapes so broadcast rules apply.
  } else {
    auto suffix_len = static_cast<int>(std::min(sh1.size(), sh2.size()));
    auto full_len = static_cast<int>(std::max(sh1.size(), sh2.size()));

    auto rev_sh1 = sh1.rbegin();
    auto rev_sh2 = sh2.rbegin();

    while (rev_sh1 != sh1.rend() && rev_sh2 != sh2.rend()) {
      auto dim1 = to_int(*rev_sh1);
      auto dim2 = to_int(*rev_sh2);
      if ((dim1 != dim2) && ((dim1 != 1) && (dim2 != 1))) {
        CHECK(false) << "Dimension mistmatch " << "dim1: " << dim1 << " dim2: " << dim2 << std::endl;
      }
      rev_sh1++;
      rev_sh2++;
    }

    Array<HalideIR::Expr> larger;
    Array<HalideIR::Expr> smaller;

    for (int i = 0; i < (full_len - suffix_len); i++) {
      smaller.push_back(tvm::ir::IntImm::make(HalideIR::Int(64), 1));
    }

    if (sh1.size() < sh2.size()) {
      for (auto sh : sh1) {
        smaller.push_back(sh);
      }
      larger = sh2;
    } else if (sh1.size() > sh2.size()) {
      for (auto sh : sh1) {
        larger.push_back(sh);
      }
      smaller = sh2;
    } else {
      larger = sh1;
      smaller = sh2;
    }

    CHECK_EQ(larger.size(), smaller.size());

    Array<HalideIR::Expr> out_shape;
    for (size_t i = 0; i < smaller.size(); i++) {
      auto left = smaller[i].as<tvm::ir::IntImm>();
      auto right = larger[i].as<tvm::ir::IntImm>();
      CHECK(left);
      CHECK(right);
      int64_t dim = std::max(left->value, right->value);
      out_shape.push_back(tvm::ir::IntImm::make(HalideIR::Int(64), dim));
    }

    return TensorTypeNode::make(out_shape, output_dtype);
  }
}

Array<Type> BroadcastRel(const Array<Type>& types, int num_args) {
  CHECK_EQ(types.size(), 3);
  RELAY_LOG(INFO) << "In1: " << types[0] << "In2: " << types[1] << "Out: " << types[2] << std::endl;
  if (auto t1 = as_ttype(types[0])) {
    if (auto t2 = as_ttype(types[1])) {
      CHECK_EQ(t1->dtype, t2->dtype);
      return {t1, t2, ConcreteBroadcast(t1, t2, t1->dtype)};
    }
  }

  return types;
}

/* A relation which specifies broadcasting rules for operations which
   compute boolean results.
*/
Array<Type> BroadcastCompRel(const Array<Type>& types, int num_args) {
  CHECK_EQ(types.size(), 3);
  if (auto t1 = as_ttype(types[0])) {
    if (auto t2 = as_ttype(types[1])) {
      return {t1, t2, ConcreteBroadcast(t1, t2, HalideIR::Bool())};
    }
  }

  return types;
}

}  // namespace relay
}  // namespace tvm
