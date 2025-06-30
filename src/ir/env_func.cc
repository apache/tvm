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
 * \file env_func.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/env_func.h>
#include <tvm/tir/expr.h>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK({ EnvFuncNode::RegisterReflection(); });

using ffi::Any;
using ffi::Function;
using ffi::PackedArgs;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<EnvFuncNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const EnvFuncNode*>(node.get());
      p->stream << "EnvFunc(" << op->name << ")";
    });

ObjectPtr<Object> CreateEnvNode(const std::string& name) {
  auto f = tvm::ffi::Function::GetGlobal(name);
  ICHECK(f.has_value()) << "Cannot find global function \'" << name << '\'';
  ObjectPtr<EnvFuncNode> n = make_object<EnvFuncNode>();
  n->func = *f;
  n->name = name;
  return n;
}

EnvFunc EnvFunc::Get(const String& name) { return EnvFunc(CreateEnvNode(name)); }

TVM_FFI_REGISTER_GLOBAL("ir.EnvFuncGet").set_body_typed(EnvFunc::Get);

TVM_FFI_REGISTER_GLOBAL("ir.EnvFuncCall").set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
  EnvFunc env = args[0].cast<EnvFunc>();
  ICHECK_GE(args.size(), 1);
  env->func.CallPacked(args.Slice(1), rv);
});

TVM_FFI_REGISTER_GLOBAL("ir.EnvFuncGetFunction").set_body_typed([](const EnvFunc& n) {
  return n->func;
});

TVM_REGISTER_NODE_TYPE(EnvFuncNode)
    .set_creator(CreateEnvNode)
    .set_repr_bytes([](const Object* n) -> std::string {
      return static_cast<const EnvFuncNode*>(n)->name;
    });

}  // namespace tvm
