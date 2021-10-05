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

#include <gtest/gtest.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/te/operation.h>

TEST(Relay, SelfReference) {
  using namespace tvm;
  auto tensor_type = relay::TensorType({}, DataType::Bool());
  auto x = relay::Var("x", relay::Type());
  auto f = relay::Function(tvm::Array<relay::Var>{x}, x, relay::Type(), {});
  ICHECK(f->IsInstance<BaseFuncNode>());
  auto y = relay::Var("y", tensor_type);
  auto call = relay::Call(f, Array<relay::Expr>{y});
  auto fx = relay::Function(tvm::Array<relay::Var>{y}, call, relay::Type(), {});
  auto mod = IRModule::FromExpr(fx);
  mod = relay::transform::InferType()(mod);
  auto type_fx = mod->Lookup("main");

  auto expected = relay::FuncType(tvm::Array<relay::Type>{tensor_type}, tensor_type, {}, {});
  ICHECK(tvm::StructuralEqual()(type_fx->checked_type(), expected));
}
