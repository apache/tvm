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
 * \file relax/op/tuple.cc
 *
 *  builtin intrinsic operators for manipulating tuples
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/registry.h>

#include <optional>

namespace tvm {
namespace relax {

namespace {
/*! \brief Utility function for NormalizeTupleGetItem and tuple_get_item
 *
 * \param index The index at which the tuple is accessed
 *
 * \return The known index, if static, otherwise std::nullopt.
 */
std::optional<int> FindStaticIndex(const Expr& index) {
  if (auto index_sinfo = index->struct_info_.as<PrimStructInfoNode>()) {
    if (auto known_index = index_sinfo->value.as<IntImmNode>()) {
      return known_index->value;
    }
  }
  return std::nullopt;
}
}  // namespace

StructInfo InferStructInfoTupleGetItem(const Call& call, const BlockBuilder&) {
  CHECK_EQ(call->args.size(), 2) << "Operator " << call->op
                                 << " expects exactly two arguments [tuple, index], "
                                 << "but received " << call->args.size()
                                 << " arguments in expression " << call;
  auto tuple = call->args[0];
  auto index = call->args[1];

  auto tuple_sinfo = tuple->struct_info_.as<TupleStructInfoNode>();
  CHECK(tuple_sinfo) << "Operator " << call->op
                     << " expects its first argument to specify a tuple, "
                     << "but expression " << call << " has tuple argument " << tuple
                     << ", which has struct info " << tuple->struct_info_;

  auto index_sinfo = index->struct_info_.as<PrimStructInfoNode>();
  CHECK(index_sinfo && index_sinfo->dtype == DataType::Int(64))
      << "TupleGetItem requires the index to be a R.Prim('int64'), "
      << "but expression " << call << " has index argument " << index << ", which has struct info "
      << index->struct_info_;

  auto known_index = index_sinfo->value.as<IntImmNode>();

  if (known_index) {
    // The exact index used to access the tuple is known.  We can
    // apply bounds-checking, and can provide the exact StructInfo of
    // the accessed element.
    int int_index = known_index->value;

    CHECK_GE(int_index, 0) << "IndexError: "
                           << "Operator " << call->op << " attempted to access tuple " << tuple
                           << " at index " << index << ".  "
                           << "However, the index " << index << " is known to be " << int_index
                           << ", and negative indices are not allowed.";

    CHECK_LT(int_index, tuple_sinfo->fields.size())
        << "IndexError: "
        << "Operator " << call->op << " attempted to access tuple " << tuple << " at index "
        << index << ".  "
        << "However, tuple " << tuple << " is of size " << tuple_sinfo->fields.size()
        << ", the index expression has a known value of " << int_index
        << ", outside the bounds of the tuple";
    return tuple_sinfo->fields[int_index];

  } else {
    // The exact index used to access the tuple is unknown.  We can't
    // apply bounds checking, but we can check that an index might
    // exist.  We can't provide an exact StructInfo for the accessed
    // type, but we can provide the common base type of all items in
    // the tuple.
    CHECK_GT(tuple_sinfo->fields.size(), 0)
        << "IndexError: "
        << "The exact value of index " << index << " is unknown, "
        << "but expression " << tuple << " has struct info " << tuple->struct_info_ << ".  "
        << "This is a tuple of length zero, and there is no index such that 0 <= index < 0.";

    StructInfo reduce_lca = tuple_sinfo->fields[0];
    for (size_t i = 1; i < tuple_sinfo->fields.size(); i++) {
      reduce_lca = StructInfoLCA(reduce_lca, tuple_sinfo->fields[1]);
    }
    return reduce_lca;
  }
}

Expr NormalizeTupleGetItem(const BlockBuilder&, const Call& call) {
  ICHECK_EQ(call->args.size(), 2);
  auto tuple = call->args[0];
  auto index = call->args[1];

  if (auto index_sinfo = index->struct_info_.as<PrimStructInfoNode>()) {
    if (auto known_index = index_sinfo->value.as<IntImmNode>()) {
      return TupleGetItem(tuple, known_index->value);
    }
  }
  return std::move(call);
}

RELAY_REGISTER_OP("relax.tuple_get_item_dyn")
    .set_num_inputs(2)
    .add_argument("tuple", "Expr (R.Tuple([...]))", "The tuple to access")
    .add_argument("index", "Expr (R.Prim(dtype='int64'))",
                  "The index at which to access the tuple.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTupleGetItem)
    .set_attr<FNormalize>("FNormalize", NormalizeTupleGetItem)
    .set_attr<Bool>("FPurity", Bool(true));

Expr tuple_get_item(Expr tuple, Expr index) {
  auto opt_static_index = FindStaticIndex(index);
  auto known_tuple = tuple.as<TupleNode>();

  if (opt_static_index && known_tuple) {
    // Both the tuple and index are known.  We can return the accessed
    // expression directly.
    return known_tuple->fields[opt_static_index.value()];
  } else if (opt_static_index) {
    // The index is known, but the tuple is bound to a variable.  We
    // can return a static TupleGetItem, which is useful in many
    // passes.
    return TupleGetItem(tuple, opt_static_index.value());
  } else {
    // The index isn't known, so fall back to the most general case.
    // If a later pass (e.g. BindParams) provides a statically-known
    // index, then this will be normalized back to a TupleGetItem at
    // that point.
    static const auto op = Op::Get("relax.tuple_get_item_dyn");
    return Call(op, {tuple, index});
  }
}

TVM_REGISTER_GLOBAL("relax.tuple_get_item").set_body_typed(tuple_get_item);

}  // namespace relax
}  // namespace tvm
