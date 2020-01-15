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
 * \file src/tvm/ir/expr.cc
 * \brief The expression AST nodes for the common IR infra.
 */
#include <tvm/runtime/registry.h>
#include <tvm/ir/expr.h>

namespace tvm {

IntImm::IntImm(DataType dtype, int64_t value) {
  CHECK(dtype.is_scalar())
      << "ValueError: IntImm can only take scalar.";
  CHECK(dtype.is_int() || dtype.is_uint())
      << "ValueError: IntImm can only take scalar.";
  if (dtype.is_uint()) {
    CHECK_GE(value, 0U);
  }
  ObjectPtr<IntImmNode> node = make_object<IntImmNode>();
  node->dtype = dtype;
  node->value = value;
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("make.IntImm")
.set_body_typed([](DataType dtype, int64_t value) {
  return IntImm(dtype, value);
});

GlobalVar::GlobalVar(std::string name_hint) {
  ObjectPtr<GlobalVarNode> n = make_object<GlobalVarNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(GlobalVarNode);

TVM_REGISTER_GLOBAL("relay._make.GlobalVar")
.set_body_typed([](std::string name){
  return GlobalVar(name);
});

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<GlobalVarNode>([](const ObjectRef& ref, NodePrinter* p) {
    auto* node = static_cast<const GlobalVarNode*>(ref.get());
    p->stream << "GlobalVar(" << node->name_hint << ")";
  });

}  // namespace tvm
