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
 * steps. First, we need to prepare the module that generate
 * the tensor values (GetCalibrateModule). Second, we need to
 * generate the mapping between the values and the subgraphs
 * (GetCalibrateOutputMap).
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/analysis.h>

namespace tvm {
namespace relay {

/*!
 * \brief This function returns a module that will be used by
 * the relay graph runtime for collecting the calibration data.
 * To do that, we first make all inputs and outputs of each
 * subgrpah into the final output (i.e., the final output is a
 * tuple of tensors). Then, we change the compiler attribute of
 * each subgraph. Finally, we mark all subgraph to be inlined.
 */

IRModule GetCalibrateModule(IRModule module) {
  class OutputCollector : public ExprRewriter {
   public:
    explicit OutputCollector(const Map<GlobalVar, BaseFunc>& glob_funcs)
      : glob_funcs(glob_funcs) {}

    Expr Rewrite_(const CallNode* call, const Expr& post) final {
      if (call->op->IsInstance<GlobalVarNode>()) {
        auto var = Downcast<GlobalVar>(call->op);
        // check if it is a subgraph of the original graph
        if (glob_funcs.count(var) > 0) {
          for (size_t i = 0; i < call->args.size(); i++)
            new_outputs.push_back(call->args[i]);
          // need to flatten the output if it is a tuple
          auto* fn = glob_funcs[var].as<FunctionNode>();
          if (auto* tn = fn->body.as<TupleNode>()) {
            for (size_t i = 0; i < tn->fields.size(); i++) {
              new_outputs.push_back(TupleGetItem(post, i));
            }
          } else {
            new_outputs.push_back(post);
          }
        }
      }
      return post;
    }

    Array<Expr> GetNewOutputs() {
      return new_outputs;
    }

   private:
    const Map<GlobalVar, BaseFunc>& glob_funcs;
    Array<Expr> new_outputs;
  };

  auto glob_funcs = module->functions;
  // module is mutable, hence, we make a copy of it.
  module.CopyOnWrite();
  for (const auto& pair : glob_funcs) {
    if (auto* fn = pair.second.as<FunctionNode>()) {
      auto func = GetRef<Function>(fn);
      auto* gl_var = pair.first.as<GlobalVarNode>();
      // we only collect the outputs for main function
      if (gl_var->name_hint == "main") {
        OutputCollector output_collector(glob_funcs);
        PostOrderRewrite(func->body, &output_collector);
        auto new_outputs = output_collector.GetNewOutputs();
        if (!new_outputs.empty()) {
          Array<Expr> fields;
          for (const auto& output : new_outputs) {
            fields.push_back(output);
          }
          auto tuple = Tuple(fields);
          func = Function(func->params, tuple, tuple->checked_type_,
                          func->type_params, func->attrs);
        }
      // inline the function if it is not main function
      } else {
        func = WithAttr(std::move(func), attr::kInline, tvm::Integer(1));
      }
      // reset the compiler attribute to null
      func = WithAttr(std::move(func), attr::kCompiler, NullValue<ObjectRef>());
      module->Update(pair.first, func);
    }
  }
  return module;
}

/*!
 * \brief This function generates the output mapping between
 * the calibration data and each subgraph. The key is a
 * GlobalVar that corresponds to each subgraph and the value
 * is an array of integers. The size of the array is always
 * three. The first value is the offset the points to the start.
 * The second value is the number of inputs. The third value
 * is the number of outputs.
 */

Map<GlobalVar, Array<Integer>> GetCalibrateOutputMap(const IRModule& module) {
  class OutputMapper : public ExprRewriter {
   public:
    OutputMapper(Map<GlobalVar, Array<Integer>>* output_map,
                 const Map<GlobalVar, BaseFunc>& glob_funcs,
                 int* offset)
      : output_map(output_map), glob_funcs(glob_funcs), offset(offset) {}

    Expr Rewrite_(const CallNode* call, const Expr& post) final {
      if (call->op->IsInstance<GlobalVarNode>()) {
        auto var = Downcast<GlobalVar>(call->op);
        Array<Integer> info;
        // the first value is the offset
        info.push_back(Integer(*offset));
        // the second value is the number of inputs
        info.push_back(Integer(call->args.size()));
        // the third value is the number of outputs
        // we need to check if the output is a tuple
        int out_size = 1;
        auto* fn = glob_funcs[var].as<FunctionNode>();
        if (auto* tn = fn->body.as<TupleNode>()) {
          info.push_back(Integer(tn->fields.size()));
          out_size = tn->fields.size();
        } else {
          info.push_back(Integer(1));
        }
        output_map->Set(var, info);
        // calculate the offset for the next function
        *offset = *offset + call->args.size() + out_size;
      }
      return post;
    }

   private:
    Map<GlobalVar, Array<Integer>>* output_map;
    const Map<GlobalVar, BaseFunc>& glob_funcs;
    int* offset;
  };

  Map<GlobalVar, Array<Integer>> output_map;
  int offset = 0;
  auto glob_funcs = module->functions;
  for (const auto& pair : glob_funcs) {
    if (auto* fn = pair.second.as<FunctionNode>()) {
      auto* gl_var = pair.first.as<GlobalVarNode>();
      if (gl_var->name_hint == "main") {
        OutputMapper output_mapper(&output_map, glob_funcs, &offset);
        auto func = GetRef<Function>(fn);
        PostOrderRewrite(func->body, &output_mapper);
      }
    }
  }

  return output_map;
}

TVM_REGISTER_GLOBAL("relay.analysis.get_calibrate_module")
    .set_body_typed([](IRModule mod) {
      return GetCalibrateModule(mod);
});

TVM_REGISTER_GLOBAL("relay.analysis.get_calibrate_output_map")
    .set_body_typed([](const IRModule& mod) {
      return GetCalibrateOutputMap(mod);
});

}  // namespace relay
}  // namespace tvm
