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

#include "./meta_schedule_layout_rewrite.h"

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <deque>
#include <mutex>
#include <vector>

#include "../backend/te_compiler.h"

namespace tvm {
namespace relay {

class LayoutIndexQueue {
 public:
  static LayoutIndexQueue* Global() {
    static LayoutIndexQueue inst;
    return &inst;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
  }

 private:
  friend class MetaScheduleLayoutRewriter;
  std::mutex mutex_;
  std::deque<tir::IndexMap> queue_;
};

void MetaScheduleLayoutRewriter::LayoutQueuePush(const tir::IndexMap& index_map) {
  LayoutIndexQueue* self = LayoutIndexQueue::Global();
  {
    std::lock_guard<std::mutex> lock(self->mutex_);
    self->queue_.push_back(index_map);
  }
}

bool IsSupportedOp(const OpNode* op) {
  static std::vector<std::string> target_ops{
      "nn.conv2d",  //
      "nn.contrib_conv2d_winograd_without_weight_transform",
      "nn.conv3d",
      "nn.matmul",
      "nn.dense",
      "nn.batch_matmul",
  };
  return std::find(target_ops.begin(), target_ops.end(), op->name) != target_ops.end();
}

#define TVM_RELAY_LAYOUT_WITH_ORIGINAL_SHAPE(Attr, AttrType, OriginalShape, Result) \
  if (const AttrType* ptr = Attr.as<AttrType>()) {                                  \
    ObjectPtr<AttrType> n = make_object<AttrType>(*ptr);                            \
    n->meta_schedule_original_shape = OriginalShape;                                \
    Result = Attrs(n);                                                              \
  }

// Mutate ops in a function
class MetaScheduleFuncMutator : public ExprMutator {
 public:
  explicit MetaScheduleFuncMutator(std::deque<tir::IndexMap>&& layout_queue)
      : layout_queue_(std::move(layout_queue)) {}

  Expr VisitExpr_(const CallNode* call) {
    Expr expr = ExprMutator::VisitExpr_(call);
    if (layout_queue_.empty()) {
      return expr;
    }
    if (const auto* call = expr.as<CallNode>()) {
      if (const auto* op = call->op.as<OpNode>()) {
        if (IsSupportedOp(op)) {
          ICHECK_EQ(call->args.size(), 2);
          tir::IndexMap index_map = layout_queue_.front();
          layout_queue_.pop_front();
          Array<PrimExpr> shape;
          if (call->args[1]->IsInstance<VarNode>()) {
            Var var = Downcast<Var>(call->args[1]);
            shape = Downcast<TensorType>(var->type_annotation)->shape;
          } else if (const ConstantNode* cnst = call->args[1].as<ConstantNode>()) {
            shape = cnst->tensor_type()->shape;
          } else {
            LOG(FATAL) << "Unexpected input " << call->args[1];
          }
          Attrs attrs{nullptr};
          TVM_RELAY_LAYOUT_WITH_ORIGINAL_SHAPE(call->attrs, Conv2DAttrs, shape, attrs);
          TVM_RELAY_LAYOUT_WITH_ORIGINAL_SHAPE(call->attrs, Conv2DWinogradAttrs, shape, attrs);
          TVM_RELAY_LAYOUT_WITH_ORIGINAL_SHAPE(call->attrs, Conv3DAttrs, shape, attrs);
          TVM_RELAY_LAYOUT_WITH_ORIGINAL_SHAPE(call->attrs, MatmulAttrs, shape, attrs);
          TVM_RELAY_LAYOUT_WITH_ORIGINAL_SHAPE(call->attrs, DenseAttrs, shape, attrs);
          TVM_RELAY_LAYOUT_WITH_ORIGINAL_SHAPE(call->attrs, BatchMatmulAttrs, shape, attrs);
          ICHECK(attrs.defined()) << "TypeError: Unknown attribute: " << call->attrs;
          expr = Call(call->op,
                      {call->args[0], MakeMetaScheduleLayoutTransform(call->args[1], index_map)},
                      attrs);
        }
      }
    }
    return expr;
  }

 private:
  std::deque<tir::IndexMap> layout_queue_;
};

#undef TVM_RELAY_LAYOUT_WITH_ORIGINAL_SHAPE

Expr MetaScheduleLayoutRewriter::VisitExpr_(const CallNode* call) {
  Expr expr = ExprMutator::VisitExpr_(call);
  call = expr.as<CallNode>();
  if (call != nullptr) {
    if (const auto* func = call->op.as<FunctionNode>()) {
      LayoutIndexQueue* self = LayoutIndexQueue::Global();
      self->queue_.clear();
      tec::PrimFuncFor(GetRef<Function>(func), Target::Current(), GlobalVarSupply(NameSupply("")));
      if (!self->queue_.empty()) {
        std::deque<tir::IndexMap> queue = std::move(self->queue_);
        self->queue_.clear();
        return MetaScheduleFuncMutator(std::move(queue)).VisitExpr(expr);
      }
    }
  }
  return expr;
}

namespace transform {

Pass MetaScheduleLayoutRewrite() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) -> Function {
    return Downcast<Function>(MetaScheduleLayoutRewriter().Mutate(std::move(f)));
  };
  return CreateFunctionPass(pass_func, 3, "MetaScheduleLayoutRewrite", {"InferType"});
}

#define TVM_RELAY_META_SCHEDULE_LAYOUT_REWRITE_GET_ORIGINAL_SHAPE(Attrs, AttrType) \
  if (const auto* p = Attrs.as<AttrType>()) {                                      \
    return p->meta_schedule_original_shape;                                        \
  }

TVM_REGISTER_GLOBAL("relay.attrs.get_meta_schedule_original_shape")
    .set_body_typed([](const Attrs& attrs) -> Array<PrimExpr> {
      TVM_RELAY_META_SCHEDULE_LAYOUT_REWRITE_GET_ORIGINAL_SHAPE(attrs, Conv2DAttrs);
      TVM_RELAY_META_SCHEDULE_LAYOUT_REWRITE_GET_ORIGINAL_SHAPE(attrs, Conv2DWinogradAttrs);
      TVM_RELAY_META_SCHEDULE_LAYOUT_REWRITE_GET_ORIGINAL_SHAPE(attrs, Conv3DAttrs);
      TVM_RELAY_META_SCHEDULE_LAYOUT_REWRITE_GET_ORIGINAL_SHAPE(attrs, MatmulAttrs);
      TVM_RELAY_META_SCHEDULE_LAYOUT_REWRITE_GET_ORIGINAL_SHAPE(attrs, DenseAttrs);
      TVM_RELAY_META_SCHEDULE_LAYOUT_REWRITE_GET_ORIGINAL_SHAPE(attrs, BatchMatmulAttrs);
      LOG(FATAL) << "TypeError: Unknown attribute: " << attrs;
      throw;
    });
TVM_REGISTER_GLOBAL("relay._transform.MetaScheduleLayoutRewrite")
    .set_body_typed(MetaScheduleLayoutRewrite);

#undef TVM_RELAY_META_SCHEDULE_LAYOUT_REWRITE_GET_ORIGINAL_SHAPE

}  // namespace transform
}  // namespace relay
}  // namespace tvm
