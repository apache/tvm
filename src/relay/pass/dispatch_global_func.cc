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
 *  Copyright (c) 2019 by Contributors
 * \file dispatch_global_func.cc
 * \brief API for dispatch global function with dynamic input shape.
 */

#include <tvm/relay/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/attrs/reduce.h>
#include <sstream>
#include <fstream>
#include <unordered_set>

namespace tvm {
namespace relay {

using namespace runtime;

using InputShapeDict = Map<std::string, Array<Expr>>;
using BucketDict = Map<std::string, Map<Integer, Array<Array<Integer, Integer>>>>;
using ConditionList = Array<Array<Expr>>;

/*! \brief Dispatch a global var for dynamic input shape
 * Helper function to build dispatching tree.
 *
 * This function build a tree from bottom to up:
 *
 *   1. For leaf node, if it is the rightmost one, insert function call.
 *      Otherwise, insert an IfNode with function call and recursive call.
 *
 *   2. For non-leaf level, if it is the rightmost one, recursively call
 *      the leftmost node on the next level. Otherwise, recursively call
 *      the next node on the same level and the leftmost node on the next
 *      level.
 */
Expr BuildDispatchTree(const Module& mod,
                       const Function& func,
                       const Array<Var>& param_vars,
                       const Array<Expr>& param_exprs,
                       const ConditionList& conditions,
                       const std::string& func_name,
                       int num_func_copy,
                       int level,
                       int pos) {
  if (level == conditions.size() - 1) {
    GlobalVar copied_global_var = GlobalVarNode::make(func_name
                                                      + "_copy_"
                                                      + std::to_string(num_func_copy));
    Function new_func = FunctionNode::make(param_vars,
                                           func->body,
                                           func->ret_type,
                                           func->type_params,
                                           func->attrs);
    mod->Add(copied_global_var, new_func);
    if (pos == conditions[level].size() - 1) {
      return CallNode::make(copied_global_var, param_exprs);
    } else {
      return IfNode::make(conditions[level][pos],
                          CallNode::make(copied_global_var, param_exprs),
                          BuildDispatchTree(mod,
                                            func,
                                            param_vars,
                                            param_exprs,
                                            conditions,
                                            func_name,
                                            num_func_copy + 1,
                                            level,
                                            pos + 1));
    }
  } else {
    if (pos == conditions[level].size() - 1) {
      return BuildDispatchTree(mod,
                               func,
                               param_vars,
                               param_exprs,
                               conditions,
                               func_name,
                               num_func_copy,
                               level + 1,
                               0);
    } else {
      return IfNode::make(conditions[level][pos],
                          BuildDispatchTree(mod,
                                            func,
                                            param_vars,
                                            param_exprs,
                                            conditions,
                                            func_name,
                                            num_func_copy,
                                            level + 1,
                                            0),
                          BuildDispatchTree(mod,
                                            func,
                                            param_vars,
                                            param_exprs,
                                            conditions,
                                            func_name,
                                            num_func_copy,
                                            level,
                                            pos + 1));
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
Module DispatchGlobalFunc(const Module& mod,
                          const std::string& func_name,
                          const InputShapeDict& input_shape,
                          const PackedFunc& dispatch_func) {

  // Generate BucketDict and validate buckets
  BucketDict buckets = dispatch_func(input_shape).AsExtension<BucketDict>();
  for (auto const& str_bucket_pair : buckets) {
    for (auto const& index_bucket : str_bucket_pair.second) {
      for (const Array<Integer, Integer>& bucket : index_bucket.second) {
        CHECK_EQ(bucket.size(), 2) << "Each bucket must be a pair of [low, high). "
            "Set high=-1 to indicate positive infinite.";
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
        take_indices = NDArray::Empty({1}, Type2TVMType(Int(32)), cpu_ctx);
        int64_t* dims = reinterpret_cast<int64_t*>(take_indices->data);
        dims[0] = index;
        auto take_attrs = make_node<TakeAttrs>();
        Call dim_val = CallNode::make(Op::Get("take"),
                                      Array<Expr>({dshape, ConstantNode::make(take_indices)}),
                                      Attrs(take_attrs),
                                      {});
        Array<Expr> cond_list;
        for (auto const& range : list_ranges) {
          int64_t low = range[0]->value;
          int64_t high = range[1]->value;
          Call cond;
          CHECK_GT(low, 0) << "Low end of a dispatched interval must be positive.";
          NDArray low_tensor = NDArray::Empty({1}, Type2TVMType(Int(64)), cpu_ctx);
          int64_t* low_tensor_data = reinterpret_cast<int64_t*>(low_tensor->data);
          low_tensor_data[0] = low;
          auto cond_attrs = make_node<ReduceAttrs>();
          Call low_cond = CallNode::make(Op::Get("greater_equal"),
                                         Array<Expr>({dim_val, ConstantNode::make(low_tensor)}));
          if (high > 0) {
            NDArray high_tensor = NDArray::Empty({1}, Type2TVMType(Int(64)), cpu_ctx);
            int64_t* high_tensor_data = reinterpret_cast<int64_t*>(high_tensor->data);
            high_tensor_data[0] = high;
            Call high_cond = CallNode::make(Op::Get("less"),
                                            Array<Expr>({dim_val, ConstantNode::make(high_tensor)}));
            cond = CallNode::make(Op::Get("all"),
                                  Array<Expr>({TupleNode::make(Array<Expr>({low_cond, high_cond}))}),
                                  Attrs(cond_attrs),
                                  {});
          }
          else {
            cond = CallNode::make(Op::Get("all"),
                                  Array<Expr>({TupleNode::make(Array<Expr>({low_cond}))}),
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
  auto new_func_body = BuildDispatchTree(mod,
                                         func,
                                         param_vars,
                                         param_exprs,
                                         conditions,
                                         func_name,
                                         0, 0, 0);
  mod->Update(mod->GetGlobalVar(func_name),
              FunctionNode::make(FreeVars(new_func_body),
              new_func_body,
              func->ret_type,
              func->type_params,
              func->attrs));

  return mod;
}

TVM_REGISTER_API("relay._transform.dispatch_global_func")
.set_body_typed<Module(
  Module, std::string, InputShapeDict, PackedFunc)>(DispatchGlobalFunc);

}  // namespace relay
}  // namespace tvm


