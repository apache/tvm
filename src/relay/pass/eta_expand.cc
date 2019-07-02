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
 * Copyright (c) 2019 by Contributors
 *
 * \file eta_expand.cc
 *
 * \brief Add abstraction over a function. For example, abs will become (fun x -> abs x).
 *
 */
#include <tvm/relay/type.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

Expr EtaExpand(const Expr& e, const Module& mod) {
  tvm::Array<Var> original_params;
  tvm::Array<Expr> params;
  tvm::Array<Var> args;
  tvm::Array<TypeVar> original_type_params;
  Type ret_type;

  if (e->is_type<GlobalVarNode>()) {
    auto gvar_node = e.as_derived<GlobalVarNode>();
    auto func = mod->Lookup(GetRef<GlobalVar>(gvar_node));
    original_params = func->params;
    original_type_params = func->type_params;
    ret_type = func->ret_type;
  } else {
    CHECK(e->is_type<FunctionNode>());
    auto func = GetRef<Function>(e.as_derived<FunctionNode>());
    original_params = func->params;
    original_type_params = func->type_params;
    ret_type = func->ret_type;
  }

  for (size_t i = 0; i < original_params.size(); ++i) {
    auto var = VarNode::make("a", original_params[i]->type_annotation);
    params.push_back(var);
    args.push_back(var);
  }

  auto new_func =
      FunctionNode::make(args, CallNode::make(e, params), ret_type, original_type_params);

  return new_func;
}

namespace transform {

Pass EtaExpand() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(EtaExpand(f, m));
    };
  Pass expanded = CreateFunctionPass(pass_func, 1, "EtaExpand", {});
  return Sequential({expanded, InferType()});
}

TVM_REGISTER_API("relay._transform.EtaExpand")
.set_body_typed(EtaExpand);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
