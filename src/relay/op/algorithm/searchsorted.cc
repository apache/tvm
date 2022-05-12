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
 * \file searchsorted.cc
 * \brief SearchSorted operators
 */
#include <tvm/relay/attrs/algorithm.h>
#include <tvm/relay/op.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(SearchSortedAttrs);

bool SearchSortedRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  const SearchSortedAttrs* param = attrs.as<SearchSortedAttrs>();
  ICHECK_EQ(types.size(), 3);
  const auto* sorted_sequence = types[0].as<TensorTypeNode>();
  const auto* values = types[1].as<TensorTypeNode>();
  ICHECK(sorted_sequence) << "Expects TensorType in the first input";
  ICHECK(values) << "Expects TensorType in the second input";
  ICHECK_GT(values->shape.size(), 0) << "The rank of `values` must be greater than one";

  if (sorted_sequence->shape.size() > 1) {
    ICHECK_EQ(sorted_sequence->shape.size(), values->shape.size())
        << "Ranks of `sorted_sequence` and values must be the same if `sorted_sequence` is "
           "multi-dimensional.";

    for (size_t i = 0; i < values->shape.size() - 1; ++i) {
      if (sorted_sequence->shape[i].as<IntImmNode>() && values->shape[i].as<IntImmNode>()) {
        ICHECK_EQ(sorted_sequence->shape[i].as<IntImmNode>()->value,
                  values->shape[i].as<IntImmNode>()->value)
            << "`sorted_sequence and `values` do not have the same shape along outer axes";
      }
    }
  }

  reporter->Assign(types[2], TensorType(values->shape, param->dtype));
  return true;
}

Expr MakeSearchSorted(Expr sorted_sequence, Expr values, Bool right, DataType dtype) {
  auto attrs = make_object<SearchSortedAttrs>();
  static const Op& op = Op::Get("searchsorted");
  attrs->dtype = dtype;
  attrs->right = right;
  return Call(op, {sorted_sequence, values}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.searchsorted").set_body_typed(MakeSearchSorted);

RELAY_REGISTER_OP("searchsorted")
    .describe(
        R"doc(Find indices where elements should be inserted to maintain order.
If `sorted_sequence` is N-dimensional, the innermost dimension of
`values` are searched in the corresponding dimension of `sorted_sequence`.
)doc" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .set_attrs_type<SearchSortedAttrs>()
    .add_argument("sorted_sequence", "Tensor",
                  "Monotonically increasing sequence on the innermost dimension.")
    .add_argument("values", "Tensor", "Values to search for.")
    .set_support_level(6)
    .add_type_rel("SearchSorted", SearchSortedRel);

}  // namespace relay
}  // namespace tvm
