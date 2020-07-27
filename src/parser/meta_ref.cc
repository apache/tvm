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
 * \file src/parser/meta_ref.cc
 * \brief An operator which allows forward referencing a yet-to-be parsed meta table reference.
 */

#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <topi/elemwise.h>

#include "./meta_ref.h"

namespace tvm {
namespace parser {

TVM_REGISTER_NODE_TYPE(MetaRefAttrs);

bool MetaRefRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
    LOG(FATAL) << "need to expand before type checking";
//   CHECK_EQ(types.size(), 3u);
//   auto size_type = types[0];
//   auto tensor_type = size_type.as<TensorTypeNode>();
//   CHECK(tensor_type != nullptr);
//   CHECK_EQ(tensor_type->dtype, DataType::Int(64));
//   CHECK_EQ(tensor_type->shape.size(), 0);
//   auto align_type = types[1];
//   auto align_ttype = align_type.as<TensorTypeNode>();
//   CHECK(align_ttype != nullptr);
//   CHECK_EQ(align_ttype->dtype, DataType::Int(64));
//   CHECK_EQ(align_ttype->shape.size(), 0);
//   auto mod = reporter->GetModule();
//   CHECK(mod.defined());
//   auto storage_name = mod->GetGlobalTypeVar("Storage");
//   auto storage = TypeCall(storage_name, {});
//   reporter->Assign(types[2], storage);
//   return true;
}

RELAY_REGISTER_OP("parser.MetaRef")
    .describe(R"code(A reference into the meta table.)code" TVM_ADD_FILELINE)
    .set_num_inputs(0)
    .set_support_level(10)
    .add_type_rel("MetaRef", MetaRefRel)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<TNonComputational>("TNonComputational", true);

Expr MetaRef(std::string type_key, uint64_t node_index) {
    static const Op& op = Op::Get("parser.MetaRef");
    auto attrs = make_object<MetaRefAttrs>();
    attrs->type_key = type_key;
    attrs->node_index = node_index;
    return Call(op, {}, Attrs(attrs), {});
}

}  // namespace parser
}  // namespace tvm
