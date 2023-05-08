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
 * \file src/relay/analysis/get_calibration_data.cc
 *
 * \brief To get the calibration data, we need to perform two
 * steps. First, we need to prepare the module that generates
 * the tensor values (GetCalibrateModule). Second, we need to
 * generate the mapping between the values and the functions
 * (GetCalibrateOutputMap).
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>

namespace tvm {
namespace relay {

/*!
 * \brief This function returns a module that will be used by
 * the relay graph executor for collecting the calibration data.
 * To do that, we first make all inputs and outputs of each
 * function into the final output (i.e., the final output is a
 * tuple of tensors). Then, we change the compiler attribute of
 * each function. Finally, we mark all function to be inlined.
 */

class Collector : public ExprRewriter {
 public:
  explicit Collector(const IRModule& module) : module_(module) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    // check if the function implementation is available
    // intrinsic functions are excluded for now
    if (call->op->IsInstance<GlobalVarNode>()) {
      auto var = Downcast<GlobalVar>(call->op);
      ICHECK(module_->ContainGlobalVar(var->name_hint)) << "Function " << var << " is not defined";
      // we only handle functions with Compiler attribute set
      auto func = Downcast<Function>(module_->Lookup(var));
      if (func->GetAttr<String>(attr::kCompiler)) {
        // collect all the inputs and outputs
        for (const auto& it : call->args) new_outputs_.push_back(it);
        new_outputs_.push_back(post);
      }
    }
    return post;
  }

  Array<Expr> GetNewOutputs() { return new_outputs_; }

 private:
  const IRModule& module_;
  Array<Expr> new_outputs_;
};

Expr FlattenOutputTuple(const Array<Expr>& exprs) {
  Array<Expr> fields;
  for (const auto& it : exprs) {
    ICHECK(it->checked_type_.defined());
    if (auto* tn = it->checked_type_.as<TupleTypeNode>()) {
      // TODO(seanlatias): for now input argument cannot be a tuple
      ICHECK(it->IsInstance<CallNode>());
      for (size_t i = 0; i < tn->fields.size(); i++) {
        fields.push_back(TupleGetItem(it, i));
      }
    } else {
      fields.push_back(it);
    }
  }
  return Tuple(fields);
}

IRModule GetCalibrateModule(IRModule module) {
  auto glob_funcs = module->functions;
  // module is mutable, hence, we make a copy of it.
  module.CopyOnWrite();
  for (const auto& pair : glob_funcs) {
    if (auto opt = pair.second.as<Function>()) {
      // we only collect the outputs for main function
      if (pair.first->name_hint == "main") {
        auto func = opt.value();
        Collector collector(module);
        PostOrderRewrite(func->body, &collector);
        auto new_outputs = collector.GetNewOutputs();
        Expr tuple = FlattenOutputTuple(new_outputs);
        func = Function(func->params, tuple, tuple->checked_type_, func->type_params, func->attrs);
        module->Update(pair.first, func);
      }
    }
  }
  // reset the attribute of functions for running graph executor
  for (const auto& pair : glob_funcs) {
    if (auto opt = pair.second.as<Function>()) {
      auto func = opt.value();
      if (func->GetAttr<String>(attr::kCompiler)) {
        // we need to inline the functions in order to run grpah runtime
        func = WithAttr(std::move(func), attr::kInline, tvm::Integer(1));
        // reset the compiler attribute to null for llvm execution
        func = WithAttr(std::move(func), attr::kCompiler, NullValue<ObjectRef>());
        module->Update(pair.first, func);
      }
    }
  }
  return module;
}

/*!
 * \brief This function generates the output mapping between
 * the calibration data and each function. The key is a
 * GlobalVar that corresponds to each function and the value
 * is an array of integers. The size of the array is always
 * three. The first value is the offset the points to the start.
 * The second value is the number of inputs. The third value
 * is the number of outputs.
 */

class OutputMapper : public ExprRewriter {
 public:
  OutputMapper(Map<GlobalVar, Array<Integer>>* output_map, const IRModule& module, size_t* offset)
      : output_map_(output_map), module_(module), offset_(offset) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (call->op->IsInstance<GlobalVarNode>()) {
      auto var = Downcast<GlobalVar>(call->op);
      ICHECK(module_->ContainGlobalVar(var->name_hint)) << "Function " << var << " is not defined";
      ICHECK_EQ(output_map_->count(var), 0)
          << "Repeated function call " << var << " is not supported.";
      auto func = Downcast<Function>(module_->Lookup(var));
      // we only handle functions with Compiler attribute set
      if (func->GetAttr<String>(attr::kCompiler)) {
        Array<Integer> info;
        // the first value is the offset
        info.push_back(Integer(*offset_));
        // the second value is the number of inputs
        info.push_back(Integer(call->args.size()));
        // the third value is the number of outputs
        // we need to check if the output is a tuple
        size_t out_size = 1;
        if (auto* tn = func->body.as<TupleNode>()) {
          info.push_back(Integer(tn->fields.size()));
          out_size = tn->fields.size();
        } else {
          info.push_back(Integer(1));
        }
        output_map_->Set(var, info);
        // calculate the offset for the next function
        *offset_ = *offset_ + call->args.size() + out_size;
      }
    }
    return post;
  }

 private:
  Map<GlobalVar, Array<Integer>>* output_map_;
  const IRModule& module_;
  size_t* offset_;
};

Map<GlobalVar, Array<Integer>> GetCalibrateOutputMap(const IRModule& module) {
  Map<GlobalVar, Array<Integer>> output_map;
  size_t offset = 0;
  auto glob_funcs = module->functions;
  for (const auto& pair : glob_funcs) {
    if (const auto* func = pair.second.as<FunctionNode>()) {
      if (pair.first->name_hint == "main") {
        OutputMapper output_mapper(&output_map, module, &offset);
        PostOrderRewrite(func->body, &output_mapper);
      }
    }
  }

  return output_map;
}

TVM_REGISTER_GLOBAL("relay.analysis.get_calibrate_module").set_body_typed([](IRModule mod) {
  return GetCalibrateModule(mod);
});

TVM_REGISTER_GLOBAL("relay.analysis.get_calibrate_output_map")
    .set_body_typed([](const IRModule& mod) { return GetCalibrateOutputMap(mod); });

}  // namespace relay
}  // namespace tvm
