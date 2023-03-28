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
 * \file src/relax/backend/aot/aot_lower_main.cc
 * \brief Lower the Relax main func into an AOT TIR main func.
 */
#include "./aot_lower_main.h"

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/name_transforms.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include "../../../relay/backend/name_transforms.h"
#include "../../../runtime/meta_data.h"

namespace tvm {
namespace relax {
namespace aot {

/*! \brief Lower a Relax main function into an equivalent TIR function which can be compiled
 * for execution.
 *
 * \note Not all Relax expressions have a 1:1 correspondance with a TIR PrimExpr/Stmt. Because of
 * this, the translation approach taken lowers the Relax function on a per-Binding basis, with
 * special rules to handle each possible binding type.
 *
 * A primary concept in the translation is the relax::Expr -> tir::Vars map. This associates all
 * the encountered relax::Vars and relax::Constants with a sequence of tir::Vars they translate to.
 * It is a one-to-many map to allow for the representation of tuple-types, which aren't explicitly
 * supported in TIR. So, if a relax::Var is bound to a relax::Tuple then it will translate to
 * a vector of tir::Vars with one tir::Var per tuple field.
 *
 * Where a relax::Binding can be entirely represented by updating the Expr->Vars map, no explicit
 * binding (in the form of a tir::LetStmt) is emitted. This both avoids emitting unnecessary code
 * as well as allowing for tuple-types to propagate throughout the program without being explicitly
 * realized (which wouldn't be possible in TIR). Instead, tuples are only 'realized' when they
 * are directly used as arguments. In that case, the tuple is flattened into the tir::Vars it
 * translates to, with each tir::Var passed as a separate argument.
 */
class AOTMainLowerer : public ExprVisitor {
 public:
  using ExprVisitor::VisitExpr_;
  explicit AOTMainLowerer(tvm::CompilationConfig config) : config_(config) {}

  IRModule Lower(IRModule mod, String mod_name) {
    IRModule lowered_mod = GetRef<IRModule>(mod.CopyOnWrite());
    auto main_base_func = lowered_mod->Lookup("main");
    auto main_func = GetRef<Function>(main_base_func.as<FunctionNode>());
    ICHECK(main_func.defined()) << "Main function must be a relax::Function.";

    // Translate the function params to tir::Vars
    Array<tir::Var> main_params;
    Map<relax::Var, tir::Var> param_map;
    for (auto rvar : main_func->params) {
      tir::Var tvar = CreateIOVar(rvar);
      main_params.push_back(tvar);
      param_map.Set(rvar, tvar);
    }

    const auto* relax_body = main_func->body.as<SeqExprNode>();
    ICHECK(relax_body) << "Body of the main function must be a relax::SeqExpr.";
    VisitExpr(GetRef<SeqExpr>(relax_body));
    tir::Stmt tir_body = seq_stmt_map_[relax_body];

    // Remove the Relax main and replace it with the lowered TIR version
    mod->Remove(lowered_mod->GetGlobalVar("main"));
    auto tir_main_func = CreateMainFunc(mod_name, main_params, tir_body, main_func->attrs);
    tir_main_func = UpdateParamTypeAttr_(tir_main_func, param_map, "input_vars");
    tir_main_func = UpdateParamTypeAttr_(tir_main_func, param_map, "output_vars");

    lowered_mod->Update(GlobalVar(runtime::symbol::tvm_module_main), tir_main_func);
    lowered_mod = tir::transform::RemoveNoOp()(lowered_mod);
    return lowered_mod;
  }

 private:
  struct BindingStmt {
    BindingStmt() : seq{}, let{} {}
    BindingStmt(const std::vector<tir::Stmt>& seq, const tir::LetStmt& let) : seq(seq), let(let) {}

    std::vector<tir::Stmt> seq;
    tir::LetStmt let;
  };

  /*! \brief The compilation configuration. */
  CompilationConfig config_;
  /*! \brief input and output variables belonging to the main function signature */
  Map<tir::Var, tir::Buffer> main_buffer_map_;
  /*! \brief This is per IO var name counter to aid the generating unique names */
  std::vector<tir::Buffer> allocs_;
  /*! \brief A map between tir::Vars and the buffers they're bound to. */
  Map<tir::Var, tir::Buffer> alloc_buffers_;
  /*! \brief The constants to be allocated. */
  std::vector<std::pair<tir::Var, relax::Constant>> alloc_constants_;
  /*! \brief A map between relax::SeqExprs and the tir::Stmt they translate to. */
  std::unordered_map<const SeqExprNode*, tir::Stmt> seq_stmt_map_;
  /*! \brief A map between relax::Bindings and the BindingStmts they translate to. */
  std::unordered_map<const BindingNode*, BindingStmt> binding_stmt_map_;
  /*! \brief A map between relax::Exprs and the tir::Vars they translate to. */
  std::unordered_map<const relax::ExprNode*, std::vector<tir::Var>> expr_var_map_;
  /*! \brief The number of tensors in the program (used to name the tensor vars). */
  int tensor_count_;

  tir::PrimFunc CreateMainFunc(String mod_name, Array<tir::Var> params, tir::Stmt body,
                               const tvm::DictAttrs& attrs) {
    // Allocates
    for (const auto& buffer : allocs_) {
      body = tir::Allocate(buffer->data, buffer->dtype, buffer->shape, tir::const_true(), body);
    }
    // AllocateConsts
    for (const auto& it : alloc_constants_) {
      relay::Shape shape;
      for (int i = 0; i < it.second->data->ndim; i++) {
        shape.push_back(Integer(it.second->data->shape[i]));
      }
      body = tir::AllocateConst(it.first, DataType(it.second->data->dtype), shape, it.second->data,
                                body);
    }

    // Define the PrimFunc attributes
    Map<String, ObjectRef> dict_attrs = attrs->dict;
    String run_func_name = runtime::get_name_mangled(mod_name, runtime::symbol::tvm_module_main);
    dict_attrs.Set("global_symbol", run_func_name);
    dict_attrs.Set("runner_function", Bool(true));
    dict_attrs.Set(tvm::attr::kTarget, config_->host_target);

    // Make the PrimFunc
    return tir::PrimFunc(params, body, VoidType(), main_buffer_map_, DictAttrs(dict_attrs));
  }

  tir::Var CreateIOVar(const relax::Var& var) {
    std::string name = runtime::SanitizeName(var->name_hint());
    auto shape = var->struct_info_.as<TensorStructInfoNode>()->shape.value();
    auto type = var->checked_type();
    tir::Var tvar = tir::Var(name, DataType::Handle());
    auto tensor_type = type.as<DynTensorTypeNode>();
    ICHECK(tensor_type) << "Expected DynTensorType node but was " << type->GetTypeKey();
    DataType elem_type = tensor_type->dtype;
    auto tensor_shape = shape.as<ShapeExprNode>()->values;
    tir::Var buffer_var =
        tir::Var(name + "_buffer_var", PointerType(PrimType(elem_type), "global"));
    tir::Buffer buffer =
        tir::Buffer(buffer_var, elem_type, tensor_shape, {}, IntImm(DataType::Int(64), 0),
                    name + "_buffer", 16, 1, tir::BufferType::kDefault);
    main_buffer_map_.Set(tvar, buffer);
    SetVar(var, buffer_var);
    return tvar;
  }

  tir::PrimFunc UpdateParamTypeAttr_(const tir::PrimFunc& func,
                                     const Map<relax::Var, tir::Var>& param_map,
                                     const std::string& param_type) {
    auto param_type_attr = func->GetAttr<Array<relax::Var>>(param_type);
    ICHECK(param_type_attr.defined()) << "Main function is missing the " << param_type << " attr.";
    auto relax_vars = param_type_attr.value();
    Array<tir::Var> tir_vars;
    for (const auto& rvar : relax_vars) {
      tir_vars.push_back(param_map.at(rvar));
    }
    return WithAttr(func, param_type, tir_vars);
  }

  void VisitExpr_(const SeqExprNode* expr) override {
    tir::Stmt body{};
    for (auto block_it = expr->blocks.rbegin(); block_it != expr->blocks.rend(); ++block_it) {
      auto block = *block_it;
      this->VisitBindingBlock(block);
      for (auto bind_it = block->bindings.rbegin(); bind_it != block->bindings.rend(); ++bind_it) {
        BindingStmt stmt = binding_stmt_map_[(*bind_it).get()];
        body = CombineBindingStmt_(stmt, body);
      }
    }

    seq_stmt_map_[expr] = body;
  }

  tir::SeqStmt CombineBindingStmt_(const BindingStmt& stmt, const tir::Stmt& body) const {
    tir::Stmt new_body = body;
    if (stmt.let.defined()) {
      new_body = tir::LetStmt(stmt.let->var, stmt.let->value, body, stmt.let->span);
    }
    if (stmt.seq.size() == 0) {
      if (const auto* seq = new_body.as<tir::SeqStmtNode>()) {
        return GetRef<tir::SeqStmt>(seq);
      }
      return tir::SeqStmt({new_body});
    } else {
      Array<tir::Stmt> new_seq{stmt.seq};
      if (new_body.defined()) {
        new_seq.push_back(new_body);
      }
      return tir::SeqStmt(new_seq);
    }
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    this->VisitExpr(binding->var);
    this->VisitExpr(binding->value);
    // Handle each type of binding differently
    if (const auto* value = binding->value.as<VarNode>()) {
      binding_stmt_map_[binding] = MakeBindingStmt_(binding->var, value);
    } else if (const auto* value = binding->value.as<ConstantNode>()) {
      binding_stmt_map_[binding] = MakeBindingStmt_(binding->var, value);
    } else if (const auto* value = binding->value.as<TupleNode>()) {
      binding_stmt_map_[binding] = MakeBindingStmt_(binding->var, value);
    } else if (const auto* value = binding->value.as<TupleGetItemNode>()) {
      binding_stmt_map_[binding] = MakeBindingStmt_(binding->var, value);
    } else if (const auto* value = binding->value.as<CallNode>()) {
      binding_stmt_map_[binding] = MakeBindingStmt_(binding->var, value);
    } else {
      LOG(FATAL) << "Encountered unsupported node";
    }
  }

  BindingStmt MakeBindingStmt_(const relax::Var& var, const VarNode* value) {
    // Where a relax::Var is bound to another relax::Var, rather than emitting a
    // tir::LetStmt to represent the binding, the lhs relax::Var is associated
    // with the tir::Var that the rhs relax::Var translates to.
    // This is necessary so that we don't have to 'realize' relax::Tuples outside
    // of where they are directly passed as arguments (in which case they can be
    // flattened).
    SetVars(var, GetVars(value));
    return BindingStmt{{}, {}};
  }

  /*!
   * \brief Make a BindingStmt from a relax::Var = relax::Constant binding.
   * \note An empty BindingStmt is emitted and instead the binding is
   * represented using SetVars to associate the relax::Var with the tir::Var
   * that the relax::Constant translates to (see the relax::ConstantNode* VisitExpr_
   * implementation for more detail).
   *
   * It would be possible to emit a tir::LetStmt to represent the binding explicitly,
   * but as relax::Constants are already translated to a function-scoped binding via
   * tir::AllocConstant, this would be unnecessary.
   */
  BindingStmt MakeBindingStmt_(const relax::Var& var, const ConstantNode* value) {
    SetVar(var, GetVar(value));
    return BindingStmt{{}, {}};
  }

  /*!
   * \brief Make a BindingStmt from a relax::Var = relax::Tuple binding.
   * \note An empty BindingStmt is emitted and instead the binding is
   * represented using SetVars to associate the relax::Var with the tir::Vars
   * that the relax::Tuple translates to.
   *
   * TIR has no way to represent tuples explicitly, so these bindings can't be
   * translated directly. Where the relax::Tuples are ultimately used as arguments,
   * they are flattened into the tir::Vars they translate to.
   */
  BindingStmt MakeBindingStmt_(const relax::Var& var, const TupleNode* value) {
    std::vector<tir::Var> vars;
    for (const auto& field : value->fields) {
      vars.insert(vars.end(), GetVars(field).begin(), GetVars(field).end());
    }
    SetVars(var, vars);
    return BindingStmt{{}, {}};
  }

  BindingStmt MakeBindingStmt_(const relax::Var var, const TupleGetItemNode* value) {
    SetVars(var, {GetVars(value->tuple)[value->index]});
    return BindingStmt{{}, {}};
  }

  BindingStmt MakeBindingStmt_(const relax::Var var, const CallNode* value) {
    static const Op& alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    static const Op& alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");

    // Handle alloc_storage ops specially because they translate to tir::Allocate
    if (value->op == alloc_storage_op) {
      DataType dtype = value->args[3].as<DataTypeImmNode>()->value;
      String storage_scope = value->args[2].as<StringImmNode>()->value;
      tir::Var buffer_var("sid_" + std::to_string(allocs_.size()),
                          PointerType(PrimType(dtype), storage_scope));
      SetVar(var, buffer_var);
      int alloc_size;
      if (const auto* shape_expr = value->args[0].as<ShapeExprNode>()) {
        alloc_size = shape_expr->values[0].as<IntImmNode>()->value;
      } else {
        alloc_size = static_cast<int64_t*>(value->args[0].as<ConstantNode>()->data->data)[0];
      }
      auto buffer = tir::Buffer(buffer_var, dtype, {alloc_size}, {1}, 0, buffer_var->name_hint, 16,
                                1, tir::BufferType::kDefault);
      allocs_.push_back(buffer);
      alloc_buffers_.Set(buffer_var, buffer);
      return BindingStmt{{}, {}};
    }

    // Handle alloc_tensor ops specially because they translate to tir::BufferLoads
    if (value->op == alloc_tensor_op) {
      DataType dtype = value->args[3].as<DataTypeImmNode>()->value;
      PrimExpr offset = value->args[1].as<PrimValueNode>()->value;
      auto storage_var = GetVar(value->args[0]);
      auto scope = storage_var->type_annotation.as<PointerTypeNode>()->storage_scope;
      auto var_name = "tid_" + std::to_string(tensor_count_++);
      auto var_dtype = PointerType(PrimType(dtype), scope);
      tir::Var tensor_var{var_name, var_dtype};
      SetVar(var, tensor_var);
      auto buffer = alloc_buffers_[storage_var];
      auto load_node = tir::BufferLoad(buffer, {offset});
      auto address_of_load = tir::Call(DataType::Handle(), tir::builtin::address_of(), {load_node});
      return BindingStmt{{}, {tensor_var, address_of_load, tir::Evaluate(0)}};
    }

    // TIR functions
    if (const auto* gv = value->op.as<GlobalVarNode>()) {
      return CreateFuncCall_(gv->name_hint, value->args);
    }

    // External functions
    if (const auto* ef = value->op.as<ExternFuncNode>()) {
      return CreateFuncCall_(ef->global_symbol, value->args);
    }

    LOG(FATAL) << "Unsupported op";
    return BindingStmt{{}, {}};
  }

  BindingStmt CreateFuncCall_(std::string func_name, const Array<Expr>& func_args) {
    tvm::Array<PrimExpr> args{tvm::tir::StringImm(func_name)};

    // Pack the inputs
    for (const Expr& arg : func_args) {
      std::vector<tir::Var> vars = GetVars(arg);
      args.insert(args.end(), vars.begin(), vars.end());
    }
    // NOTE: LowerTVMBuiltin expects some device_context placeholder.
    args.push_back(tir::make_zero(DataType::Handle()));
    // Create a function call using the CPacked API
    // TODO(mbaret): Lower to raw Call and lower to a specific API in a later pass
    tir::Stmt func_call = tir::Evaluate(
        tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::tvm_call_cpacked(), args));

    return BindingStmt{{func_call}, {}};
  }

  void VisitExpr_(const ConstantNode* constant) override {
    // In ANF, all arguments must be trivial (either relax::Var or relax::Constant).
    // This means that constants must be visited explicitly as we won't necessarily
    // visit all of them just through visiting relax::Bindings because they
    // may only appear as an argument.

    // relax::Constant translates to tir::AllocConstant, however the former is
    // not scoped whereas the latter is. Because of this, rather that translating
    // relax::Constants 'in-place', they're translated to a function-scoped
    // tir::AllocConstant which binds the allocated constant to a tir::Var. The
    // relax::Constant is then translated to that tir::Var.
    Type type = PointerType(PrimType(DataType(constant->data->dtype)));
    std::string name = "constant_" + std::to_string(alloc_constants_.size());
    tir::Var constant_var(name, type);
    alloc_constants_.emplace_back(constant_var, GetRef<Constant>(constant));
    SetVar(constant, constant_var);
  }

  /*!
   * \brief Get the tir::Vars associated with a Relax expression.
   * \param expr The Relax expression to translate.
   * \return The tir::Vars the Relax expression translates to.
   */
  const std::vector<tir::Var>& GetVars(const relax::ExprNode* expr) const {
    ICHECK(expr_var_map_.find(expr) != expr_var_map_.end());
    return expr_var_map_.at(expr);
  }

  /*! \brief Utility overload of GetVars. */
  const std::vector<tir::Var>& GetVars(const relax::Expr& expr) const {
    return GetVars(expr.get());
  }

  /*!
   * \brief Get the tir::Var associated with a Relax expression.
   * \param expr The Relax expression to translate.
   * \return The tir::Var the Relax expression translates to.
   * \note Will raise an error if the expression translates to multiple tir::Vars.
   */
  const tir::Var GetVar(const relax::ExprNode* expr) const {
    auto vars = GetVars(expr);
    ICHECK_EQ(vars.size(), 1);
    return vars[0];
  }

  /*! \brief Utility overload of GetVar. */
  const tir::Var GetVar(const relax::Expr& expr) const { return GetVar(expr.get()); }

  /*!
   * \brief Set the tir::Vars that a Relax expression translates to.
   * \param expr The Relax expression to translate.
   * \param vars The tir::Vars the expression translates to.
   * \note A Relax expression may be a tuple, so in that case it will translate
   * to multiple tir::Vars (TIR has no tuple types).
   */
  void SetVars(const relax::Expr& expr, const std::vector<tir::Var>& vars) {
    expr_var_map_[expr.get()] = vars;
  }

  /*!
   * \brief Set the tir::Var that a Relax expression translates to.
   * \param expr The Relax expression to translate.
   * \param var The tir::Var the expression translates to.
   */
  void SetVar(const relax::Expr& expr, const tir::Var& var) { SetVar(expr.get(), var); }

  /*! \brief Utility overload of SetVar. */
  void SetVar(const relax::ExprNode* expr, const tir::Var& var) { expr_var_map_[expr] = {var}; }
};

transform::Pass AOTLowerMain(String mod_name, tvm::CompilationConfig config) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [=](IRModule module, transform::PassContext ctx) {
        return AOTMainLowerer(config).Lower(module, mod_name);
      };

  return tvm::transform::CreateModulePass(pass_func, 0, "relax.aot.AOTLowerMain", {"InferType"});
}

TVM_REGISTER_GLOBAL("relax.aot.AOTLowerMain")
    .set_body_typed([](const String& mod_name, const tvm::CompilationConfig& config) {
      return AOTLowerMain(mod_name, config);
    });

}  // namespace aot
}  // namespace relax
}  // namespace tvm
