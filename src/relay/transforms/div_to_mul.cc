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

template <typename T>
inline bool const_has_values(size_t size, const ConstantNode* const_node, const std::vector<T>&& values) {
  for (size_t i = 0; i < size; i++) {
    T data = static_cast<T*>(const_node->data->data)[i];
    for (const T& v: values) {
      if (data == v) return true;
    }
  }
  return false;
}

inline size_t get_num_elements_const(const ConstantNode* const_node) {
  const auto& shape = const_node -> data.Shape();

  size_t cnt_elements = 1;
  for (const auto& dim: shape) {
    cnt_elements *= dim;
  }

  return cnt_elements;
}

class DivToMulRewrite : public MixedModeMutator {
  Expr Rewrite_(const CallNode* pre, const Expr& post) final {
    if (const CallNode* call_node = post.as<CallNode>()) {
      if (call_node->op == Op::Get("divide")) {
        auto rhs = call_node->args[1].as<ConstantNode>();
        if (rhs != nullptr) {
          auto one =
              runtime::NDArray::Empty({}, rhs->data.DataType(), rhs->data->device);
          size_t num_ele = get_num_elements_const(rhs);
          std::string dtype = DLDataType2String(rhs->data.DataType());

          bool const_has_zero_flag = false;
          if (dtype == "float32") {
              static_cast<float*>(one->data)[0] = 1.;
              const_has_zero_flag = const_has_values<float>(num_ele, rhs, {0.});
          } else if (dtype == "float64") {
              static_cast<double*>(one->data)[0] = 1.;
              const_has_zero_flag = const_has_values<double>(num_ele, rhs, {0.});
          } else if (dtype == "float16") {
              static_cast<uint16_t*>(one->data)[0] = __gnu_f2h_ieee(1.);
              const_has_zero_flag = const_has_values<uint16_t>(num_ele, rhs, {0});
          } else {
              LOG(WARNING) << "Unknown dtype not handled for div_to_mull: " << rhs->data.DataType(); 
              return post;
          }
  
          if (const_has_zero_flag) {
            return post;
          }
          
          // rely on constant folding to fold things 
          return Multiply(call_node->args[0], Divide(Constant(one), call_node->args[1]));
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
