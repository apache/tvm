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
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/builtin_fp16.h>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

class DivToMulRewrite : public MixedModeMutator {
  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    if (const CallNode* call_node = post.as<CallNode>()) {
      if (call_node->op == Op::Get("divide")) {
        auto rhs = call_node->args[1].as<ConstantNode>();
        if (rhs != nullptr) {
          auto inv =
              runtime::NDArray::Empty(rhs->data.Shape(), rhs->data.DataType(), rhs->data->device);
          std::string dtype = DLDataType2String(rhs->data.DataType());
          if (dtype == "float32") {
            float rhs_val = static_cast<float*>(rhs->data->data)[0];
            // Check for division by zero
            if (rhs_val == 0.) {
              return post;
            }
            static_cast<float*>(inv->data)[0] = 1. / rhs_val;
          } else if (dtype == "float64") {
            double rhs_val = static_cast<double*>(rhs->data->data)[0];
            // Check for division by zero
            if (rhs_val == 0.) {
              return post;
            }
            static_cast<double*>(inv->data)[0] = 1. / rhs_val;
          } else if (dtype == "float16") {
            // Do f16 math in f32
            float rhs_val = __gnu_h2f_ieee(static_cast<uint16_t*>(rhs->data->data)[0]);
            // Check for division by zero
            if (rhs_val == 0.) {
              return post;
            }
            static_cast<uint16_t*>(inv->data)[0] = __gnu_f2h_ieee(1. / rhs_val);
          } else {
            // Cannot do 1/int because it will truncate
            return post;
          }
          return Multiply(call_node->args[0], Constant(inv));
        }
      }
    }
    return post;
  }
};

namespace transform {

Pass DivToMul() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(DivToMulRewrite().Mutate(f));
      };
  return CreateFunctionPass(pass_func, 0, "DivToMul", {"InferType", "FoldConstant"});
}

TVM_REGISTER_GLOBAL("relay._transform.DivToMul").set_body_typed(DivToMul);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
