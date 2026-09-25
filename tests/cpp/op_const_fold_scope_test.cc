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
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/script/ir_builder/ir.h>
#include <tvm/s_tir/script/ir_builder/ir.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/script/ir_builder/ir.h>

#include <stdexcept>
#include <thread>

namespace tvm {
TEST(OpConstFoldScope, NativeAndFFIState) {
  PrimExpr one = IntImm::Int32(1);
  EXPECT_TRUE(prim::OpConstFoldEnabled());
  EXPECT_TRUE((one + one)->IsInstance<IntImmNode>());
  auto query = ffi::Function::GetGlobal("prim.op_const_fold_enabled").value();
  try {
    With<prim::OpConstFoldScope> disabled(false);
    EXPECT_FALSE(query().cast<bool>());
    EXPECT_TRUE((one + one)->IsInstance<prim::AddNode>());
    EXPECT_TRUE((one + 0)->IsInstance<prim::AddNode>());
    sym::Analyzer analyzer;
    EXPECT_EQ(analyzer->Simplify(one + one).as<IntImmNode>()->value, 2);
    PrimVar x("x", PrimType::Int(32));
    analyzer->MarkGlobalNonNegValue(x);
    EXPECT_TRUE(analyzer->CanProve(x >= 0));
    PrimVar i("i", PrimType::Int(32));
    PrimVar n("n", PrimType::Int(32));
    analyzer->transitive_comparisons.Bind(i, Range::FromMinExtent(0, n));
    EXPECT_EQ(analyzer->transitive_comparisons.TryCompare(i, n), sym::CompareResult::kLT);
    {
      With<prim::OpConstFoldScope> enabled(true);
      EXPECT_TRUE(query().cast<bool>());
      EXPECT_TRUE((one + one)->IsInstance<IntImmNode>());
    }
    EXPECT_FALSE(prim::OpConstFoldEnabled());
    std::thread child([] {
      EXPECT_TRUE(prim::OpConstFoldEnabled());
      With<prim::OpConstFoldScope> disabled(false);
      EXPECT_FALSE(prim::OpConstFoldEnabled());
    });
    child.join();
    EXPECT_FALSE(prim::OpConstFoldEnabled());
    throw std::runtime_error("unwind");
  } catch (const std::runtime_error&) {
  }
  EXPECT_TRUE(query().cast<bool>());
}

TEST(OpConstFoldScope, NativeFunctionFrames) {
  namespace B = script::ir_builder;
  auto global = [](const char* name) { return ffi::Function::GetGlobal(name).value(); };
  auto enter = global("script.ir_builder.IRBuilderEnter");
  auto exit = global("script.ir_builder.IRBuilderExit");
  // Builder factories are internal symbols; use their registry entries, then enter
  // the actual native frames directly through With and virtual dispatch.
  for (const char* dialect : {"tirx", "s_tir", "relax"}) {
    auto builder = global("script.ir_builder.IRBuilder")();
    enter(builder);
    std::string prefix = std::string("script.ir_builder.") + dialect + ".";
    bool is_relax = std::string(dialect) == "relax";
    auto make_frame = global((prefix + "DeclFunction").c_str());
    auto frame = (is_relax ? make_frame(true, false, false) : make_frame(false, false))
                     .cast<B::IRBuilderFrame>();
    {
      With<B::IRBuilderFrame> declaration(frame);
      EXPECT_FALSE(prim::OpConstFoldEnabled());
      global((std::string("script.ir_builder.") + (is_relax ? "relax" : "tirx") + ".FuncName")
                 .c_str())("main");
    }
    EXPECT_TRUE(prim::OpConstFoldEnabled());
    {
      With<B::IRBuilderFrame> body(frame);
      EXPECT_FALSE(prim::OpConstFoldEnabled());
      if (is_relax) {
        global("script.ir_builder.relax.FuncRetValue")(IntImm::Int32(1));
      } else {
        global("script.ir_builder.tirx.Evaluate")(IntImm::Int32(1) + IntImm::Int32(2));
      }
    }
    EXPECT_TRUE(prim::OpConstFoldEnabled());
    exit(builder);
  }
}
}  // namespace tvm
