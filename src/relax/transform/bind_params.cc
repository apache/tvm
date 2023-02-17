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

#include <tvm/driver/driver_api.h>
#include <tvm/ir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

#include <utility>

namespace tvm {
namespace relax {

/*!
 * \brief Bind params to function by using name
 * \param func Relax function
 * \param params params dict
 * \return Function
 */
inline Function BindParamsByName(Function func, const Map<String, runtime::NDArray>& params) {
  std::unordered_map<std::string, Var> name_dict;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> repeat_var;
  for (auto arg : func->params) {
    const auto& name = arg->name_hint();
    if (name_dict.count(name)) {
      repeat_var.insert(name_dict[name]);
    } else {
      name_dict[name] = arg;
    }
  }

  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> bind_dict;
  for (auto& kv : params) {
    if (name_dict.count(kv.first) == 0) {
      continue;
    }
    auto arg = name_dict.at(kv.first);
    if (repeat_var.count(arg)) {
      LOG(FATAL) << "ValueError: Multiple args in the function have name " << kv.first;
    }
    bind_dict[arg] = Constant(kv.second);
  }
  Expr bound_expr = Bind(func, bind_dict);
  Function ret = Downcast<Function>(bound_expr);
  ICHECK(ret.defined()) << "The returning type is expected to be a Relax Function."
                        << "\n";
  return ret;
}

/*!
 * \brief Bind params to a specific function in a module
 * \param m The module
 * \param func_name The name of the specific function
 * \param param The param dict
 * \return The module after binding params.
 */
IRModule BindParam(IRModule m, String func_name, Map<String, runtime::NDArray> param) {
  IRModuleNode* new_module = m.CopyOnWrite();
  Map<GlobalVar, BaseFunc> functions = m->functions;
  for (const auto& func_pr : functions) {
    if (const auto* relax_f = func_pr.second.as<FunctionNode>()) {
      if (relax_f->GetLinkageType() == LinkageType::kExternal) {
        // Use global_symbol if it's external linkage
        Optional<String> gsymbol = relax_f->GetAttr<String>(tvm::attr::kGlobalSymbol);
        if (gsymbol.defined() && gsymbol.value() == func_name) {
          Function f_after_bind = BindParamsByName(GetRef<Function>(relax_f), param);
          new_module->Update(func_pr.first, f_after_bind);
        }
      } else {
        // Use global var's name_hint if it's internal linkage
        if (func_pr.first->name_hint == func_name) {
          Function f_after_bind = BindParamsByName(GetRef<Function>(relax_f), param);
          new_module->Update(func_pr.first, f_after_bind);
        }
      }
    }
  }
  return GetRef<IRModule>(new_module);
}

namespace transform {

Pass BindParams(String func_name, Map<String, runtime::NDArray> params) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return BindParam(std::move(mod), func_name, params); };
  return CreateModulePass(pass_func, 0, "BindParams", {});
}

TVM_REGISTER_GLOBAL("relax.transform.BindParams").set_body_typed(BindParams);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
