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
 *
 * \file combine_parallel_dense.cc
 * \brief Combine parallel dense ops into a single dense.
 *
 * This pass replaces dense ops that share the same input node, same shape,
 * and don't have "units" defined with a single batch matrix multiplication.
 * The inputs of the new batch_matmul is the stack of the original inputs.
 * Elemwise and broadcast ops following dense are also combined if possible.
 *
 * This prevents launching multiple kernels in networks with multiple
 * dense branches, such as BERT.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "./combine_parallel_op_batch.h"
#include "./expr_subst.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*
 * Class that find and combine parallel dense ops into batch_matmul.
 */
class ParallelDenseToBatchCombiner : public ParallelOpBatchCombiner {
 public:
  explicit ParallelDenseToBatchCombiner(uint64_t min_num_branches)
      : ParallelOpBatchCombiner("nn.dense", "nn.batch_matmul", min_num_branches) {}

 protected:
  Call MakeCombinedOp(const Group& branches) {
    Array<Expr> new_args;
    size_t num_args = branches[0][0]->args.size();
    for (size_t i = 0; i < num_args; i++) {
      Array<Expr> arg_from_all_branches;
      for (const auto& branch : branches) {
        arg_from_all_branches.push_back(branch[0]->args[i]);
      }

      new_args.push_back(MakeStack(Tuple(arg_from_all_branches), 0));
    }

    CHECK_EQ(num_args, 2);
    const auto* origin_attrs = branches[0][0]->attrs.as<DenseAttrs>();
    ICHECK(origin_attrs);
    return Downcast<Call>(
        MakeBatchMatmul(new_args[0], new_args[1], origin_attrs->out_dtype, false, true));
  }

  virtual bool CanOpsBeCombined(const CallNode* a, const CallNode* b) {
    StructuralEqual eq;
    const auto* attrs_a = a->attrs.as<DenseAttrs>();
    const auto* attrs_b = b->attrs.as<DenseAttrs>();
    ICHECK(attrs_a);
    ICHECK(attrs_b);
    const auto* weight_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* weight_b = b->args[1]->type_as<TensorTypeNode>();

    return eq(attrs_a->out_dtype, attrs_b->out_dtype) &&
           eq(weight_a->shape[0], weight_b->shape[0]) && eq(weight_a->shape[1], weight_b->shape[1]);
  }
};

/*
 * Class that find and combine parallel dense ops into one dense op
 * whose num of output units equals to sum of each sub-ops.
 */
class ParallelDenseToDenseCombiner : public ParallelOpCombiner {
 public:
  explicit ParallelDenseToDenseCombiner(uint64_t min_num_branches)
      : ParallelOpCombiner("nn.dense", min_num_branches) {}

 protected:
  bool IsSupportedOp(const CallNode* n) { return true; }

  bool CanOpsBeCombined(const CallNode* a, const CallNode* b) {
    StructuralEqual eq;
    const auto* attrs_a = a->attrs.as<DenseAttrs>();
    const auto* attrs_b = b->attrs.as<DenseAttrs>();
    const auto* weight_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* weight_b = b->args[1]->type_as<TensorTypeNode>();
    ICHECK(attrs_a != nullptr && attrs_b != nullptr && weight_a != nullptr && weight_b != nullptr);
    // output dims (weight->shape[0]) can be different
    return eq(attrs_a->out_dtype, attrs_b->out_dtype) && eq(weight_a->shape[1], weight_b->shape[1]);
  }

  Call MakeCombinedOp(const Group& branches) {
    const Op& dense_op = Op::Get("nn.dense");
    Expr input = branches[0][0]->args[0];
    // concat all weights into one
    auto [new_weight, new_output_dims] = TransformWeight(branches);
    const auto* origin_attrs = branches[0][0]->attrs.as<DenseAttrs>();
    ICHECK(origin_attrs);
    const auto dense_attrs = make_object<DenseAttrs>();
    dense_attrs->units = new_output_dims;
    dense_attrs->out_dtype = origin_attrs->out_dtype;
    return Call(dense_op, {input, new_weight}, Attrs{dense_attrs}, {});
  }

  bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) {
    StructuralEqual eq;
    auto ta = a->args[index]->type_as<TensorTypeNode>();
    auto tb = b->args[index]->type_as<TensorTypeNode>();
    auto toutput_a = a->type_as<TensorTypeNode>();
    auto toutput_b = b->type_as<TensorTypeNode>();
    ICHECK(ta != nullptr && tb != nullptr && toutput_a != nullptr && toutput_b != nullptr);

    if (!eq(ta->dtype, tb->dtype) || ta->shape.size() != tb->shape.size()) {
      return false;
    }
    if (toutput_a->shape.size() < ta->shape.size() || toutput_b->shape.size() < tb->shape.size()) {
      return false;  // not broadcast/elemwise
    }
    if (ta->shape.size() > 0) {
      for (size_t i = 0; i < ta->shape.size() - 1; i++) {
        // shape dims must match except last dim
        if (!eq(ta->shape[i], tb->shape[i])) return false;
      }
    }
    return true;
  }

  Call MakeCombinedCallFromFollowingOps(const Expr& data, const Group& branches, size_t depth,
                                        size_t parent_index) {
    Array<Expr> new_args;
    const CallNode* call = branches[0][depth];
    for (size_t i = 0; i < call->args.size(); i++) {
      if (i == parent_index) {
        new_args.push_back(data);
        continue;
      }
      size_t arg_ndim = call->args[i]->type_as<TensorTypeNode>()->shape.size();
      size_t concat_axis = arg_ndim == 0 ? 0 : arg_ndim - 1;
      Array<Expr> tuple;
      for (const auto& branch : branches) {
        auto parent = branch[depth]->args[parent_index];
        auto& parent_shape = parent->type_as<TensorTypeNode>()->shape;
        auto out_dim = tir::as_const_int(parent_shape[parent_shape.size() - 1]);
        ICHECK(out_dim != nullptr);

        auto arg = branch[depth]->args[i];
        auto& arg_shape = arg->type_as<TensorTypeNode>()->shape;
        bool repeat_last_dim = false;
        if (arg_ndim == 0) {
          repeat_last_dim = true;
          arg = MakeExpandDims(arg, -1, 1);
        } else {
          auto arg_last_dim = tir::as_const_int(arg_shape[arg_shape.size() - 1]);
          ICHECK(arg_last_dim != nullptr);
          if (*out_dim > 1 && *arg_last_dim == 1) {
            repeat_last_dim = true;
          }
        }
        if (repeat_last_dim) {
          // ensure broadcast is valid after concat args
          arg = MakeRepeat(arg, *out_dim, concat_axis);
        }
        tuple.push_back(arg);
      }
      auto concat = MakeConcatenate(Tuple(tuple), concat_axis);
      new_args.push_back(std::move(concat));
    }
    return Call(call->op, new_args, call->attrs, {});
  }

  void UpdateGroupOutput(const Expr& data, const Group& branches, size_t depth,
                         ExprSubstMap* subst_map) {
    int index = 0;
    const auto dense_op = Op::Get("nn.dense");
    for (const auto& branch : branches) {
      const CallNode* call = branch[depth];
      auto& out_shape = call->type_as<TensorTypeNode>()->shape;

      const CallNode* dense = branch[0];
      ICHECK(dense->op.same_as(dense_op));
      auto& dense_shape = dense->type_as<TensorTypeNode>()->shape;
      auto dense_out_dims = tir::as_const_int(dense_shape[1]);
      ICHECK(dense_out_dims != nullptr);

      // dense can be followed by shape-changing operations, so the slicing axis is
      // not necessarily the last one.
      // TODO(masahi): The following logic is incorrect if (1) there is no axis in
      // out_shape[i] that directly corresponds to the output channel of dense or (2) there
      // is another axis that happens to have the same size as the output channel of dense.
      // Such cases might arise due to reshape / transpose / split etc. Revisit this logic
      // when we encounter them in practice.
      auto slice_axis = -1;
      for (size_t i = out_shape.size() - 1; i >= 0; --i) {
        ICHECK(tir::as_const_int(out_shape[i]));
        if (*tir::as_const_int(out_shape[i]) == *dense_out_dims) {
          slice_axis = i;
          break;
        }
      }
      ICHECK(slice_axis != -1);

      Array<Integer> begin(out_shape.size(), 0);
      Array<Integer> end(out_shape.size(), -1);
      Array<Integer> strides(out_shape.size(), 1);
      begin.Set(slice_axis, index);
      end.Set(slice_axis, *dense_out_dims);
      index += *dense_out_dims;
      auto slice = MakeStridedSlice(data, begin, end, strides, "size");
      subst_map->insert({GetRef<Expr>(branch[depth]), slice});
    }
  }

 private:
  std::tuple<Expr, IndexExpr> TransformWeight(const Group& branches) {
    int64_t out_dims = 0;
    Array<Expr> weights;
    for (const auto& branch : branches) {
      auto weight = branch[0]->args[1];
      weights.push_back(weight);
      out_dims += *tir::as_const_int(weight->type_as<TensorTypeNode>()->shape[0]);
    }
    return std::make_tuple(MakeConcatenate(Tuple(weights), 0),
                           tir::make_const(DataType::Int(32), out_dims));
  }
};

/*! \brief Combine parallel dense if number of branches >= min_num_branches */
Expr CombineParallelDense(const Expr& expr, uint64_t min_num_branches, bool to_batch) {
  if (to_batch) {
    return ParallelDenseToBatchCombiner(min_num_branches).Combine(expr);
  } else {
    return ParallelDenseToDenseCombiner(min_num_branches).Combine(expr);
  }
}

namespace transform {

Pass CombineParallelDense(uint64_t min_num_branches, bool to_batch_matmul) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CombineParallelDense(f, min_num_branches, to_batch_matmul));
      };
  return CreateFunctionPass(pass_func, 4, "CombineParallelDense", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.CombineParallelDense").set_body_typed(CombineParallelDense);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
