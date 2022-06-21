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
#include <tvm/ir/module.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/te/operation.h>

using namespace tvm;
using namespace transform;

Pass MutateModulePass() {
  auto pass_func = [=](IRModule mod, PassContext pc) -> IRModule {
    GlobalVar var = mod->GetGlobalVar("dummyFunction");
    mod->Remove(var);
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 1, "ImmutableModulev1", {});
}

Pass DoNotMutateModulePass() {
  auto pass_func = [=](IRModule mod, PassContext pc) -> IRModule {
    IRModule result(mod->functions, mod->type_definitions, mod->Imports(), mod->source_map,
                    mod->attrs);
    GlobalVar var = result->GetGlobalVar("dummyFunction");
    result->Remove(var);
    return result;
  };
  return tvm::transform::CreateModulePass(pass_func, 1, "ImmutableModulev2", {});
}

IRModule preamble() {
  auto x = relay::Var("x", relay::Type());
  auto f = relay::Function(tvm::Array<relay::Var>{x}, x, relay::Type(), {});
  ICHECK(f->IsInstance<BaseFuncNode>());

  auto global_var = GlobalVar("dummyFunction");
  auto mod = IRModule::FromExpr(f, {{global_var, f}}, {});
  return mod;
}

TEST(Relay, ModuleIsMutated) {
  IRModule mod = preamble();

  EXPECT_THROW(
      {
        auto pass_ctx = relay::transform::PassContext::Create();
        pass_ctx->config.Set("testing.immutable_module", Bool(true));
        {
          tvm::With<relay::transform::PassContext> ctx_scope(pass_ctx);
          mod = MutateModulePass()(mod);
        }
      },
      runtime::InternalError);
}

TEST(Relay, ModuleIsNotMutated) {
  IRModule mod = preamble();

  auto pass_ctx = relay::transform::PassContext::Create();
  pass_ctx->config.Set("testing.immutable_module", Bool(true));
  {
    tvm::With<relay::transform::PassContext> ctx_scope(pass_ctx);
    mod = DoNotMutateModulePass()(mod);
  }
}
