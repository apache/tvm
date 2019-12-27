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
 * \file dispatch_global_func.cc
 * 
 * \brief API for dispatch global function with dynamic input shape.
 */

#include <tvm/relay/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/attrs/reduce.h>

namespace tvm {
namespace relay {

using namespace runtime;

using InputShapeDict = Map<std::string, Array<IndexExpr>>;
using BucketDict = Map<std::string, Map<Integer, Array<Array<Integer>>>>;
using ConditionList = Array<Array<Expr>>;

/*! \brief Dispatch a global var for dynamic input shape
 * Helper class to build dispatching tree.
 */
class DispatchTreeBuilder {
 public:
  DispatchTreeBuilder(const Module& mod,
                      const std::string& func_name,
                      const ConditionList& conditions)
    : mod_(mod), func_name_(func_name), conditions_(conditions) {
    func_ = mod_->Lookup(func_name_);
    param_vars_ = FreeVars(func_->body);
    for (auto param_var : param_vars_) {
      param_exprs_.push_back(Expr(param_var));
    }
    num_copied_funcs_ = 0;
  }

  Expr BuildDispatchTree(unsigned level, unsigned pos);

 private:
  Module mod_;
  Function func_;
  Array<Var> param_vars_;
  Array<Expr> param_exprs_;
  std::string func_name_;
  ConditionList conditions_;
  int num_copied_funcs_;
};

// Helper function to build dispatch tree.
// This function build a tree from bottom to up:
//
//   1. For leaf node, if it is the rightmost one, insert function call.
//      Otherwise, insert an IfNode with function call and recursive call.
//
//   2. For non-leaf level, if it is the rightmost one, recursively call
//      the leftmost node on the next level. Otherwise, recursively call
//      the next node on the same level and the leftmost node on the next
//      level.
Expr DispatchTreeBuilder::BuildDispatchTree(unsigned level, unsigned pos) {
  if (level == conditions_.size() - 1) {
    GlobalVar copied_global_var =
      GlobalVarNode::make(func_name_ + "_copy_" +
                            std::to_string(num_copied_funcs_));
    Function new_func = FunctionNode::make(param_vars_,
                                           func_->body,
                                           func_->ret_type,
                                           func_->type_params,
                                           func_->attrs);
    mod_->Add(copied_global_var, new_func);
    num_copied_funcs_ += 1;
    if (pos == conditions_[level].size() - 1) {
      return CallNode::make(copied_global_var, param_exprs_);
    } else {
      return IfNode::make(conditions_[level][pos],
                          CallNode::make(copied_global_var, param_exprs_),
                          BuildDispatchTree(level, pos + 1));
    }
  } else {
    if (pos == conditions_[level].size() - 1) {
      return BuildDispatchTree(level + 1, 0);
    } else {
      return IfNode::make(conditions_[level][pos],
                          BuildDispatchTree(level + 1, 0),
                          BuildDispatchTree(level, pos + 1));
    }
  }
}

/*! \brief Dispatch a global var for dynamic input shape
 *
 * This function needs the following steps:
 *
 *   1. Given symbolic input shape, return a BucketDict with using dispatching
 *      logic provided by dispatch_func.
 *
 *   2. Generate a condition expression list given from BucketDict.
 *
 *   3. Recursively build a dispatching tree with condition list.
 *
 * Note that currently only continuous range is valid as bucket, and is
 * represented as a pair of (low, high). If high is -1, it means the high
 * end is positive infinite.
 */
Module AddDispatchFunc(const Module& mod,
                          const std::string& func_name,
                          const InputShapeDict& input_shape,
                          const PackedFunc& dispatch_func) {
  // Generate BucketDict and validate buckets
  auto ret = dispatch_func(input_shape);
  auto buckets = ret.AsObjectRef<BucketDict>();
  for (auto const& str_bucket_pair : buckets) {
    for (auto const& index_bucket : str_bucket_pair.second) {
      for (const Array<Integer>& bucket : index_bucket.second) {
        CHECK_EQ(bucket.size(), 2) << "Each bucket must be a pair of "
            "[low, high). Set high=-1 to indicate positive infinite.";
      }
    }
  }

  const Function& func = mod->Lookup(func_name);
  Array<Var> param_vars = FreeVars(func->body);
  Array<Expr> param_exprs;
  for (auto param_var : param_vars) {
    param_exprs.push_back(Expr(param_var));
  }

  DLContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  ConditionList conditions;
  for (size_t i = 0; i < param_vars.size(); ++i) {
    Var param_var = param_vars[i];
    Expr param_expr = param_exprs[i];
    if (input_shape.count(param_var->name_hint())) {
      auto shapeof_attrs = make_node<ShapeOfAttrs>();
      shapeof_attrs->dtype = DataType(kDLInt, 64, 1);
      Call dshape = CallNode::make(Op::Get("shape_of"),
                                   Array<Expr>({param_expr}),
                                   Attrs(shapeof_attrs),
                                   {});
      // Generate condition for each bucket.
      auto bucket_map = buckets[param_var->name_hint()];
      for (auto const& bucket : bucket_map) {
        int64_t index = bucket.first->value;
        auto list_ranges = bucket.second;
        NDArray take_indices;
        take_indices = NDArray::Empty({1}, DataType::Int(32), cpu_ctx);
        int64_t* dims = reinterpret_cast<int64_t*>(take_indices->data);
        dims[0] = index;
        auto take_attrs = make_node<TakeAttrs>();
        Call dim_val =
          CallNode::make(Op::Get("take"),
                         Array<Expr>(
                           {dshape, ConstantNode::make(take_indices)}),
                         Attrs(take_attrs), {});
        Array<Expr> cond_list;
        for (auto const& range : list_ranges) {
          int64_t low = range[0]->value;
          int64_t high = range[1]->value;
          Call cond;
          CHECK_GT(low, 0)
            << "Low end of a dispatched interval must be positive.";
          NDArray low_tensor =
            NDArray::Empty({1}, DataType::Int(64), cpu_ctx);
          int64_t* low_tensor_data =
            reinterpret_cast<int64_t*>(low_tensor->data);
          low_tensor_data[0] = low;
          auto cond_attrs = make_node<ReduceAttrs>();
          Call low_cond =
            CallNode::make(Op::Get("greater_equal"),
                           Array<Expr>({dim_val,
                                        ConstantNode::make(low_tensor)}));
          if (high > 0) {
            NDArray high_tensor =
              NDArray::Empty({1}, DataType::Int(64), cpu_ctx);
            int64_t* high_tensor_data =
              reinterpret_cast<int64_t*>(high_tensor->data);
            high_tensor_data[0] = high;
            Call high_cond =
              CallNode::make(Op::Get("less"),
                             Array<Expr>({dim_val,
                                          ConstantNode::make(high_tensor)}));
            cond = CallNode::make(Op::Get("all"),
                                  Array<Expr>({TupleNode::make(
                                    Array<Expr>({low_cond, high_cond}))}),
                                  Attrs(cond_attrs),
                                  {});
          } else {
            cond = CallNode::make(Op::Get("all"),
                                  Array<Expr>({TupleNode::make(
                                    Array<Expr>({low_cond}))}),
                                  Attrs(cond_attrs),
                                  {});
          }
          cond_list.push_back(cond);
        }
        conditions.push_back(cond_list);
      }
    }
  }

  // Build dispatching tree
  // Assume we are dispatching a resnet with dynamic batching input.
  // We use uniform dispatch function. The higher limit is 256 and
  // step is 64. Updated main function would be:
  //
  //    bs = shape_of(input)[0]
  //    if 1 <= bs < 64:
  //        main_copy_0(input)
  //    elif 64 <= bs < 128:
  //        main_copy_1(input)
  //    elif 128 <= bs < 192:
  //        main_copy_2(input)
  //    elif 192 <= bs < 256:
  //        main_copy_3(input)
  //    else:
  //        main_copy_4(input)
  auto new_func_body = DispatchTreeBuilder(mod, func_name, conditions)
    .BuildDispatchTree(0, 0);
  mod->Update(mod->GetGlobalVar(func_name),
              FunctionNode::make(FreeVars(new_func_body),
              new_func_body,
              func->ret_type,
              func->type_params,
              func->attrs));

  return mod;
}

TVM_REGISTER_API("relay._transform.add_dispatch_func")
.set_body_typed<Module(
  Module, std::string, InputShapeDict, PackedFunc)>(AddDispatchFunc);

}  // namespace relay
}  // namespace tvm
