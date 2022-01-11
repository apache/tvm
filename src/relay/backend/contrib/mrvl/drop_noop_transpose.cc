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
 * \file src/relay/backend/contrib/mrvl/drop_noop_trnaspose.cc
 * \brief Marvell MLIP specific API
 */

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>

#include <limits>
#include <utility>

#include "../../../op/tensor/transform.h"
#include "../../../transforms/pattern_utils.h"
#include "../../../transforms/simplify_expr.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace mrvl {

/*!
 * \brief DropNoOpTranspose, which does not change any axes.
 */
class DropNoOpTranspose : public DFPatternRewrite {
 public:
  DropNoOpTranspose() {
    x_ = IsWildcard();
    auto trans = IsOp("transpose");
    pattern_ = trans({x_});
  }

  Expr Callback(const Expr& pre, const Expr& post,
                const Map<DFPattern, Array<Expr>>& node_map) const override {
    auto get_axes_from_call = [](const Call trans_call, int ndim) {
      std::vector<int> attr_axes;
      if (auto attr = trans_call->attrs.as<TransposeAttrs>()) {
        if (attr->axes.defined()) {
          for (int i = 0; i < ndim; ++i) {
            int64_t axis = attr->axes[i];
            axis += (axis < 0) ? ndim : 0;
            attr_axes.push_back(axis);
          }
        } else {
          // Empty axes means reverse
          for (int i = ndim - 1; i >= 0; --i) {
            attr_axes.push_back(i);
          }
        }
      } else if (auto attr = trans_call->attrs.as<LayoutTransformAttrs>()) {
        Layout src_layout(attr->src_layout);
        Layout dst_layout(attr->dst_layout);
        for (int i = 0; i < ndim; ++i) {
          attr_axes.push_back(src_layout.IndexOf(dst_layout[i]));
        }
      } else {
        CHECK(false) << "Mrvl-TVM-ERROR: Expected transpose or layout_transform, but got "
                     << Downcast<Op>(trans_call->op)->name;
      }
      return std::move(attr_axes);
    };

    auto x = node_map[x_][0];

    // check axes
    int ndim = Downcast<TensorType>(pre->checked_type())->shape.size();

    // Collect axes from the transpose
    Call trans_call = Downcast<Call>(post);
    std::vector<int> actual_axes = get_axes_from_call(trans_call, ndim);
    bool drop = true;
    for (int i = 0; i < ndim; ++i) {
      if (actual_axes[i] != i) {
        drop = false;
        break;
      }
    }

    // x is result of the node just before the pattern
    if (drop) return x;

    // keep the transpose node
    return post;
  }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
};

Expr DropNoopTranspose(const Expr& expr, const IRModule& mod) {
  // the rewrites will be applied in the given order, and repeated until fixed point
  DFPatternRewriteComposer composer;
  composer.AddRewrite<DropNoOpTranspose>();
  return RewritePatterns(composer.MakeCallbacks(), expr, mod);
}

}  // namespace mrvl
}  // namespace contrib

namespace transform {

Pass DropNoopTranspose() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(tvm::relay::contrib::mrvl::DropNoopTranspose(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "DropNoopTranspose", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.DropNoopTranspose").set_body_typed(DropNoopTranspose);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
