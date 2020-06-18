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
 * \file src/relay/ir/function.cc
 * \brief Function in relay.
 */
#include <tvm/relay/function.h>

namespace tvm {
namespace relay {

Function::Function(tvm::Array<Var> params,
                   Expr body,
                   Type ret_type,
                   tvm::Array<TypeVar> type_params,
                   DictAttrs attrs) {
  ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
  CHECK(params.defined());
  CHECK(type_params.defined());
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}

FuncType FunctionNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    Type param_type = (param->type_annotation.defined()) ? param->type_annotation
      : IncompleteType(Kind::kType);
    param_types.push_back(param_type);
  }

  Type ret_type = (this->ret_type.defined()) ? this->ret_type
    : IncompleteType(Kind::kType);
  return FuncType(param_types, ret_type, this->type_params, {});
}

TVM_REGISTER_NODE_TYPE(FunctionNode);

TVM_REGISTER_GLOBAL("relay.ir.Function")
.set_body_typed([](tvm::Array<Var> params,
                   Expr body,
                   Type ret_type,
                   tvm::Array<TypeVar> ty_params,
                   tvm::DictAttrs attrs) {
  return Function(params, body, ret_type, ty_params, attrs);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<FunctionNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const FunctionNode*>(ref.get());
  p->stream << "FunctionNode(" << node->params << ", " << node->ret_type
            << ", " << node->body << ", " << node->type_params << ", "
            << node->attrs << ")";
});

}  // namespace relay
}  // namespace tvm
