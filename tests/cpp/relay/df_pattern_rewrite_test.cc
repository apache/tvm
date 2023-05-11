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
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/function.h>
#include <tvm/relay/parser.h>

#include "../../../src/relay/transforms/simplify_expr.h"

namespace tvm {
namespace relay {
namespace {

// Demonstrates rewriting a deeply nested sub-graph with specific
// attributes on the inner-most operator call.
class TestRewriter : public DFPatternRewrite {
 public:
  TestRewriter() {
    x_ = IsWildcard();
    const1_ = IsWildcard();
    const2_ = IsWildcard();
    const3_ = IsWildcard();
    const4_ = IsWildcard();

    auto biasadd = IsOp("nn.bias_add");
    auto relu = IsOp("nn.relu");
    auto conv2d = IsOp("nn.conv2d");

    Map<String, ObjectRef> attrs;
    attrs.Set("groups", Integer(304));
    auto maybedepthwise = conv2d({x_, const1_}).HasAttr(attrs);

    pattern_ =
        relu({biasadd({conv2d({relu({biasadd({maybedepthwise, const2_})}), const3_}), const4_})});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    LOG(INFO) << "depthwise conv2d detected!";
    auto attrs = runtime::make_object<InitOpAttrs>();
    attrs->shape = Array<Integer>({Integer(1), Integer(256), Integer(128), Integer(128)});
    attrs->dtype = DataType::Float(32);
    return Call(Op::Get("zeros"), {}, Attrs(attrs));
  }

  DFPattern x_, const1_, const2_, const3_, const4_;
};

TEST(DFPatternRewrite, DeeplyNestedWithCallAttributes) {
  constexpr const char* kModel = R"(
    #[version = "0.0.5"]
    def @main(%data : Tensor[(1, 304, 128, 128), float32],
             %weight1 : Tensor[(304, 1, 3, 3), float32],
             %bias1 : Tensor[(304), float32],
             %weight2 : Tensor[(256, 304, 1, 1), float32],
             %bias2 : Tensor[(256), float32]) -> Tensor[(1, 256, 128, 128), float32] {
      %0 = nn.conv2d(%data, %weight1, padding=[1, 1, 1, 1], groups=304, channels=304, kernel_size=[3, 3]);
      %1 = nn.bias_add(%0, %bias1);
      %2 = nn.relu(%1);
      %3 = nn.conv2d(%2, %weight2, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
      %4 = nn.bias_add(%3, %bias2);
      nn.relu(%4)
    }
  )";

  IRModule module = ParseModule("string", kModel);
  DFPatternRewriteComposer composer;
  composer.AddRewrite<TestRewriter>();
  Function in_function = Downcast<Function>(module->Lookup("main"));
  LOG(INFO) << "input function:\n" << PrettyPrint(in_function);
  Function out_function =
      Downcast<Function>(RewritePatterns(composer.MakeCallbacks(), in_function, module));
  LOG(INFO) << "output function:\n" << PrettyPrint(out_function);
  const auto* call_node = out_function->body.as<CallNode>();
  ASSERT_TRUE(call_node != nullptr);
  ASSERT_TRUE(call_node->op == Op::Get("zeros"));
}

}  // namespace
}  // namespace relay
}  // namespace tvm
