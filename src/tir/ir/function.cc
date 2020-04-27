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
 * \file src/tir/ir/function.cc
 * \brief The function data structure.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace tir {

// Get the function type of a PrimFunc
PrimFunc::PrimFunc(Array<tir::Var> params,
                   Stmt body,
                   Type ret_type,
                   Map<tir::Var, Buffer> buffer_map,
                   DictAttrs attrs) {
  // Assume void-return type for now
  // TODO(tvm-team) consider type deduction from body.
  if (!ret_type.defined()) {
    ret_type = VoidType();
  }
  auto n = make_object<PrimFuncNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->buffer_map = std::move(buffer_map);
  n->attrs = std::move(attrs);
  n->checked_type_ = n->func_type_annotation();
  data_ = std::move(n);
}

FuncType PrimFuncNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(GetType(param));
  }
  return FuncType(param_types, ret_type, {}, {});
}

TVM_REGISTER_NODE_TYPE(PrimFuncNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<PrimFuncNode>([](const ObjectRef& ref, ReprPrinter* p) {
  // TODO(tvm-team) redirect to Text printer once we have a good text format.
  auto* node = static_cast<const PrimFuncNode*>(ref.get());
  p->stream << "PrimFunc(" << node->params << ") ";
  if (node->attrs.defined()) {
    p->stream << "attrs=" << node->attrs;
  }
  p->stream << " {\n";
  p->indent += 2;
  p->Print(node->body);
  p->indent -= 2;
  p->stream << "}\n";
});


TVM_REGISTER_GLOBAL("tir.PrimFunc")
.set_body_typed([](Array<tir::Var> params,
                   Stmt body,
                   Type ret_type,
                   Map<tir::Var, Buffer> buffer_map,
                   DictAttrs attrs) {
  return PrimFunc(params, body, ret_type, buffer_map, attrs);
});

}  // namespace tir
}  // namespace tvm
