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

#include <tuple>
#include <utility>

namespace tvm {
namespace relax {

void MatchSymbolicVar(const Expr& arg, const Expr& constant,
                      Map<tir::Var, PrimExpr>* symbolic_var_map, arith::Analyzer* analyzer_) {
  auto opt_arg_sinfo = MatchStructInfo<TensorStructInfo>(arg);
  CHECK(opt_arg_sinfo)
      << "The struct info of the bound parameter is expected to be TensorStructInfo, but got: "
      << GetStructInfo(arg);
  auto opt_const_sinfo = MatchStructInfo<TensorStructInfo>(constant);
  // As the constant is generated by internal codes, we use ICHECK here.
  ICHECK(opt_const_sinfo)
      << "The struct info of the bound weight is expected to be TensorStructInfo, but got: "
      << GetStructInfo(constant);

  TensorStructInfo arg_sinfo = opt_arg_sinfo.value();
  TensorStructInfo const_sinfo = opt_const_sinfo.value();
  ICHECK(!const_sinfo->IsUnknownDtype());
  ICHECK(!const_sinfo->IsUnknownNdim());
  ICHECK(const_sinfo->shape.defined());

  // dtype mismatch
  if (!arg_sinfo->IsUnknownDtype() && arg_sinfo->dtype != const_sinfo->dtype) {
    LOG(FATAL) << "The dtype of the bound parameter is expected to be " << arg_sinfo->dtype
               << ", but got: " << const_sinfo->dtype;
  }
  // ndim mismatch
  if (!arg_sinfo->IsUnknownNdim() && arg_sinfo->ndim != const_sinfo->ndim) {
    LOG(FATAL) << "The ndim of the bound parameter is expected to be " << arg_sinfo->ndim
               << ", but got: " << const_sinfo->ndim;
  }
  if (!arg_sinfo->shape.defined()) return;
  const auto* arg_shape = arg_sinfo->shape.value().as<ShapeExprNode>();
  const auto* const_shape = const_sinfo->shape.value().as<ShapeExprNode>();

  CHECK(arg_shape && const_shape)
      << "The shape of the bound parameter and weight is expected to be ShapeExprNode for now";

  for (int i = 0; i < arg_sinfo->ndim; ++i) {
    const PrimExpr& const_dim = const_shape->values[i];
    ICHECK(tir::is_const_int(const_dim));
    if (const auto* shape_var = arg_shape->values[i].as<tir::VarNode>()) {
      auto it = symbolic_var_map->find(GetRef<tir::Var>(shape_var));
      if (it == symbolic_var_map->end()) {
        symbolic_var_map->Set(GetRef<tir::Var>(shape_var), const_dim);
      } else {
        CHECK(analyzer_->CanProveEqual((*it).second, const_dim))
            << "The shape of the bound parameter is expected to be " << (*it).second
            << ", but got: " << const_dim;
      }
    }
  }
}

std::tuple<Map<Var, Expr>, Map<tir::Var, PrimExpr>> NormalizeBindings(
    const Function& func, const Map<ObjectRef, ObjectRef>& untyped_params) {
  ICHECK(func.defined());
  ICHECK(untyped_params.defined());

  // Map from string to the variable(s) with that name.
  std::unordered_map<std::string, Array<relax::Var>> string_lookup;
  std::unordered_set<const relax::VarNode*> var_set;
  for (const auto& param : func->params) {
    string_lookup[param->name_hint()].push_back(param);
    var_set.insert(param.get());
  }

  Map<relax::Var, relax::Expr> relax_var_remap;

  auto normalize_key = [&](ObjectRef obj) -> relax::Var {
    if (auto opt_str = obj.as<String>()) {
      std::string str = opt_str.value();
      auto it = string_lookup.find(str);
      CHECK(it != string_lookup.end())
          << "Function does not have parameter with name \"" << str << "\".  "
          << "Function parameters are named "
          << func->params.Map([](const auto& param) { return param->name_hint(); });
      CHECK_EQ(it->second.size(), 1)
          << "Function contains multiple parameters with name \"" << str << "\".  "
          << "The Relax variables " << it->second << " are all named \"" << str << "\"";
      auto var = it->second[0];
      CHECK(!relax_var_remap.count(var))
          << "Remap of variable " << var << " was defined multiple times";

      return var;
    } else if (auto opt_var = obj.as<relax::Var>()) {
      auto var = opt_var.value();
      CHECK(!relax_var_remap.count(var))
          << "Remap of variable " << var << " was defined multiple times";
      CHECK(var_set.count(var.get()))
          << "Function does not use Relax variable " << var << " as a parameter.  "
          << "Function parameters are " << func->params;
      return var;
    } else {
      LOG(FATAL)
          << "Expected bound parameter to be a relax::Var, "
          << " or a string that uniquely identifies a relax::Var param within the function.  "
          << "However, received object " << obj << " of type " << obj->GetTypeKey();
    }
  };
  auto normalize_value = [&](ObjectRef obj) -> relax::Expr {
    if (auto opt = obj.as<relax::Expr>()) {
      return opt.value();
    } else if (auto opt = obj.as<runtime::NDArray>()) {
      return Constant(opt.value());
    } else {
      LOG(FATAL) << "Cannot coerce object of type " << obj->GetTypeKey()
                 << " into relax expression";
    }
  };

  for (const auto& [key, value] : untyped_params) {
    relax_var_remap.Set(normalize_key(key), normalize_value(value));
  }

  arith::Analyzer analyzer;
  Map<tir::Var, PrimExpr> symbolic_var_map = InferSymbolicVarMap(relax_var_remap, &analyzer);

  // for (const auto& [bind_param, bind_expr] : relax_var_remap) {
  //   MatchSymbolicVar(bind_param, bind_expr, &symbolic_var_map, &analyzer);
  // }

  return {relax_var_remap, symbolic_var_map};
}

/*!
 * \brief Bind params to function by using name
 * \param func Relax function
 * \param params params dict
 * \return Function
 */
Function FunctionBindParams(Function func, const Map<ObjectRef, ObjectRef>& untyped_params) {
  auto [bind_dict, symbolic_var_map] = NormalizeBindings(func, untyped_params);

  Expr bound_expr = Bind(func, bind_dict, symbolic_var_map);
  return Downcast<Function>(bound_expr);
}

/*!
 * \brief Bind params to a specific function in a module
 * \param m The module
 * \param func_name The name of the specific function
 * \param param The param dict
 * \return The module after binding params.
 */
IRModule BindParam(IRModule m, String func_name, Map<ObjectRef, ObjectRef> bind_params) {
  IRModuleNode* new_module = m.CopyOnWrite();
  Map<GlobalVar, BaseFunc> functions = m->functions;
  for (const auto& func_pr : functions) {
    if (const auto* relax_f = func_pr.second.as<FunctionNode>()) {
      if (relax_f->GetLinkageType() == LinkageType::kExternal) {
        // Use global_symbol if it's external linkage
        Optional<String> gsymbol = relax_f->GetAttr<String>(tvm::attr::kGlobalSymbol);
        if (gsymbol.defined() && gsymbol.value() == func_name) {
          Function f_after_bind = FunctionBindParams(GetRef<Function>(relax_f), bind_params);
          new_module->Update(func_pr.first, f_after_bind);
        }
      } else {
        // Use global var's name_hint if it's internal linkage
        if (func_pr.first->name_hint == func_name) {
          Function f_after_bind = FunctionBindParams(GetRef<Function>(relax_f), bind_params);
          new_module->Update(func_pr.first, f_after_bind);
        }
      }
    }
  }
  return GetRef<IRModule>(new_module);
}

TVM_REGISTER_GLOBAL("relax.FunctionBindParams").set_body_typed(FunctionBindParams);

namespace transform {

Pass BindParams(String func_name, Map<ObjectRef, ObjectRef> params) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return BindParam(std::move(mod), func_name, params); };
  return CreateModulePass(pass_func, 0, "BindParams", {});
}

TVM_REGISTER_GLOBAL("relax.transform.BindParams").set_body_typed(BindParams);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
