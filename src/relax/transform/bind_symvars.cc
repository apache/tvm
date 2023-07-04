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
#include <tvm/relax/struct_info.h>
#include <utility>

namespace tvm {
namespace relax {

class ExprBinder : public ExprMutator {
 public:
  explicit ExprBinder(const tvm::Map<tir::Var, PrimExpr>& symvar_map)
      : symvar_map_(symvar_map) {}

 private:
  Expr VisitExpr_(const FunctionNode* op) final {
    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (const Var& param : op->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      if (!param.same_as(new_param)) {
        this->var_remap_[param->vid] = new_param;
        all_params_unchanged = false;
      }
    }

    Expr body = this->VisitWithNewScope(op->body, params);

    // FuncStructInfo does not depend on Expr
    if (all_params_unchanged && body.same_as(op->body)) {
      return GetRef<Expr>(op);
    } else {
      // purity won't be affected, no need to update annotation
      return Function(params, body, VisitExprDepStructInfoField(op->ret_struct_info), op->is_pure,
                      op->attrs);
    }
  }

  PrimExpr VisitPrimExpr(const PrimExpr& expr) final {

    if (const tir::VarNode* var = expr.as<tir::VarNode>()) {
      auto it = symvar_map_.find(GetRef<tir::Var>(var));
      if (it != symvar_map_.end()) {
        return (*it).second;
      }
    }else if(const auto* var = expr.as<tir::AddNode>() ){
      return VisitPrimExpr(var->a) + VisitPrimExpr(var->b);
    }else if(const auto* var = expr.as<tir::SubNode>() ){
      return VisitPrimExpr(var->a) - VisitPrimExpr(var->b);
    }else if(const auto* var = expr.as<tir::MulNode>() ){
      return VisitPrimExpr(var->a) * VisitPrimExpr(var->b);
    }else if(const auto* var = expr.as<tir::DivNode>() ){
      return tir::Cast(DataType::Int(64), tir::Div(VisitPrimExpr(var->a), VisitPrimExpr(var->b)));
    }else if(const auto* var = expr.as<tir::ModNode>() ){
      return tir::Cast(DataType::Int(64), tir::Mod(VisitPrimExpr(var->a), VisitPrimExpr(var->b)));
    }else if(const auto* var = expr.as<tir::FloorDivNode>() ){
      return tir::Cast(DataType::Int(64), tir::FloorDiv(VisitPrimExpr(var->a), VisitPrimExpr(var->b)));
    }else if(const auto* var = expr.as<tir::FloorModNode>() ){
      return tir::Cast(DataType::Int(64), tir::FloorMod(VisitPrimExpr(var->a), VisitPrimExpr(var->b)));
    }
  
    return ExprMutator::VisitPrimExpr(expr);
  }


 private:
  const tvm::Map<tir::Var, PrimExpr>& symvar_map_;
};

/*!
 * \brief Bind values to symbolic variables in a specific function 
 * \param m The module
 * \param func_name The name of the specific function
 * \param binding_map User-provided map for symbolic variables and their values to bind
 * \return The module after binding symvars.
 */
IRModule BindSymVars(IRModule m, String func_name, Map<String, Integer> binding_map) {
  IRModuleNode* new_module = m.CopyOnWrite();
  Function func = Downcast<Function>(m->Lookup(func_name));
  Map<tir::Var, PrimExpr> smap;
  Array<PrimExpr> shape_values;
  for(auto param:func->params){
    if(const auto* tsinfo = GetStructInfoAs<TensorStructInfoNode>(param)){
      const auto* shape = tsinfo->shape.as<ShapeExprNode>();
      ICHECK(shape != nullptr) << "Shape should be defined.";
      shape_values = shape->values;
    }else if(const auto* ssinfo = GetStructInfoAs<ShapeStructInfoNode>(param)){
      shape_values = ssinfo->values.value();  
    }else{
        LOG(FATAL) << "Function params should have either TensorStructInfo or ShapeStructInfo.";
    }

    for(auto val:shape_values){
      if(const auto* v = val.as<tir::VarNode>()){
         if(binding_map.find(v->name_hint)!=binding_map.end())
           smap.Set(GetRef<tir::Var>(v), binding_map[v->name_hint]);
      }
    }
  }
  GlobalVar gv = m->GetGlobalVar(func_name);
  Expr bound_expr= ExprBinder(smap).VisitExpr(func);
  Function bound_func = Downcast<Function>(bound_expr);
  new_module->Update(gv, bound_func);
  return GetRef<IRModule>(new_module);
}

namespace transform {

Pass BindSymVars(String func_name, Map<String, Integer> binding_map) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return relax::BindSymVars(std::move(mod), func_name, binding_map); 
      };
  return CreateModulePass(pass_func, 0, "BindSymVars", {});
}

TVM_REGISTER_GLOBAL("relax.transform.BindSymVars").set_body_typed(BindSymVars);

}  // namespace transform

}  // namespace relax
}  // namespace tvm