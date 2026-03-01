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
 * \file make_packed_api.cc Lower PrimFunc to use the packed function API.
 */
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>
#include <utility>
#include <vector>

#include "ir_utils.h"
#include "tvm_ffi_binder.h"

namespace tvm {
namespace tir {

namespace {
class ReturnRewriter : public StmtMutator {
 public:
  explicit ReturnRewriter(Var ret_var) : ret_var_(ret_var) {}

  Stmt VisitStmt_(const ForNode* node) override {
    if (node->kind == ForKind::kParallel) in_parallel_ += 1;
    Stmt ret = StmtMutator::VisitStmt_(node);
    if (node->kind == ForKind::kParallel) in_parallel_ -= 1;
    return ret;
  }

  Stmt VisitStmt_(const EvaluateNode* node) override {
    Stmt ret = StmtMutator::VisitStmt_(node);
    const EvaluateNode* eval = ret.as<EvaluateNode>();
    TVM_FFI_ICHECK(eval);
    if (const CallNode* call = eval->value.as<CallNode>()) {
      if (call->op.same_as(builtin::ret())) {
        TVM_FFI_ICHECK_EQ(in_parallel_, 0) << "tir.ret cannot be used in parallel scope.";
        TVM_FFI_ICHECK_EQ(call->args.size(), 1) << "tir.ret expect a single argument.";
        ret = WriteToOut(call->args[0]);
      }
    }
    return ret;
  }

 private:
  struct ConvertedInfo {
    int type_index{-1};
    PrimExpr expr;
  };

  ConvertedInfo ConvertForFFI(PrimExpr val) {
    ConvertedInfo info;

    // convert val's data type to FFI data type, return type code
    DataType dtype = val.dtype();
    if (dtype.is_bool()) {
      info.type_index = ffi::TypeIndex::kTVMFFIBool;
      info.expr = Cast(DataType::Int(64), val);

    } else if (dtype.is_int() || dtype.is_uint()) {
      info.type_index = ffi::TypeIndex::kTVMFFIInt;
      info.expr = Cast(DataType::Int(64), val);
    } else if (dtype.is_float()) {
      info.type_index = ffi::TypeIndex::kTVMFFIFloat;
      info.expr = Cast(DataType::Float(64), val);
    } else if (dtype.is_void()) {
      info.type_index = ffi::TypeIndex::kTVMFFINone;
      info.expr = val;
    } else {
      TVM_FFI_THROW(InternalError) << "data type " << dtype << " not supported yet";
    }
    return info;
  }

  Stmt WriteToOut(PrimExpr val) {
    auto info = ConvertForFFI(val);
    Stmt store_tindex =
        tir::Evaluate(tir::Call(DataType::Int(32), tir::builtin::tvm_struct_set(),
                                {ret_var_, IntImm(DataType::Int(32), 0),
                                 IntImm(DataType::Int(32), tir::builtin::kTVMFFIAnyTypeIndex),
                                 IntImm(DataType::Int(32), info.type_index)}));
    Stmt store_zero_padding =
        tir::Evaluate(tir::Call(DataType::Int(32), tir::builtin::tvm_struct_set(),
                                {ret_var_, IntImm(DataType::Int(32), 0),
                                 IntImm(DataType::Int(32), tir::builtin::kTVMFFIAnyZeroPadding),
                                 IntImm(DataType::Int(32), 0)}));
    Stmt store_val = tir::Evaluate(
        tir::Call(DataType::Int(32), tir::builtin::tvm_struct_set(),
                  {ret_var_, IntImm(DataType::Int(32), 0),
                   IntImm(DataType::Int(32), tir::builtin::kTVMFFIAnyUnionValue), info.expr}));
    Stmt ret_zero = Evaluate(tvm::ret(0));
    return SeqStmt({store_tindex, store_zero_padding, store_val, ret_zero});
  }

  Var ret_var_;
  int in_parallel_{0};
};

class SubroutineCallRewriter : public StmtExprMutator {
 public:
  static ffi::Optional<Stmt> Apply(const ffi::Map<GlobalVar, ffi::String>& packed_func_methods,
                                   Stmt stmt) {
    SubroutineCallRewriter rewriter(packed_func_methods);
    stmt = rewriter.VisitStmt(std::move(stmt));
    if (rewriter.made_change_) {
      return stmt;
    } else {
      return std::nullopt;
    }
  }

 private:
  explicit SubroutineCallRewriter(const ffi::Map<GlobalVar, ffi::String>& packed_func_methods)
      : packed_func_methods(packed_func_methods) {}

  PrimExpr VisitExpr_(const CallNode* op) override {
    auto node = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    if (auto* gvar_ptr = node->op.as<GlobalVarNode>()) {
      auto gvar = ffi::GetRef<GlobalVar>(gvar_ptr);
      if (auto symbol = packed_func_methods.Get(gvar)) {
        ffi::Array<PrimExpr> cpacked_args;
        cpacked_args.push_back(tir::StringImm(symbol.value()));
        for (auto arg : node->args) {
          cpacked_args.push_back(arg);
        }

        // push an empty handle to be compatible with current cpacked convention
        cpacked_args.push_back(tir::make_zero(DataType::Handle()));
        made_change_ = true;
        return tir::Call(node->dtype, tir::builtin::tvm_call_cpacked(), cpacked_args);
      }
    }

    return node;
  }
  const ffi::Map<GlobalVar, ffi::String>& packed_func_methods;
  bool made_change_{false};
};

}  // namespace

/*!
 * \brief Create an assert that lhs == rhs, with multi-part error message.
 * \param kind The error kind (e.g. "TypeError", "ValueError", "RuntimeError").
 * \param lhs Left-hand side of equality check.
 * \param rhs Right-hand side of equality check.
 * \param detail The detail message string.
 * \param func_signature Function signature for context.
 */
inline Stmt MakeAssertEQ(const std::string& kind, PrimExpr lhs, PrimExpr rhs,
                         const std::string& detail, const std::string& func_signature) {
  ffi::Array<StringImm> parts;
  parts.push_back(tvm::tir::StringImm(detail + " when calling:\n  `"));
  parts.push_back(tvm::tir::StringImm(func_signature));
  parts.push_back(tvm::tir::StringImm("`"));
  return AssertStmt(tvm::tir::StringImm(kind), lhs == rhs, parts);
}

/*!
 * \brief Create an assert that ptr is not NULL, with multi-part error message.
 */
inline Stmt MakeAssertNotNull(const std::string& kind, PrimExpr ptr, const std::string& detail,
                              const std::string& func_signature) {
  Call isnull(DataType::Bool(), builtin::isnullptr(), {ptr});
  ffi::Array<StringImm> parts;
  parts.push_back(tvm::tir::StringImm(detail + " when calling:\n  `"));
  parts.push_back(tvm::tir::StringImm(func_signature));
  parts.push_back(tvm::tir::StringImm("`"));
  return AssertStmt(tvm::tir::StringImm(kind), !isnull, parts);
}

/* \brief Return the global_symbol of the function, if it should be updated
 *
 * \param func The function to be inspected
 *
 * \returns The global_symbol to be used for the function at call
 * sites, or std::nullopt if the function is to remain unchanged.
 */
ffi::Optional<ffi::String> RequiresPackedAPI(const PrimFunc& func) {
  // A function with an explicit calling convention has already been
  // lowered, and should not be modified.
  if (auto opt = func->GetAttr<Integer>(tvm::attr::kCallingConv)) {
    if (CallingConv(opt.value()->value) != CallingConv::kDefault) {
      return std::nullopt;
    }
  }

  // Internal function calls do not need the ffi::Function API
  auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  if (!global_symbol.has_value()) {
    return std::nullopt;
  }

  return global_symbol.value();
}

PrimFunc MakePackedAPI(PrimFunc func) {
  auto global_symbol = RequiresPackedAPI(func);
  if (!global_symbol.has_value()) {
    return func;
  }
  std::string name_hint = global_symbol.value();

  Target target = [&]() {
    auto opt = func->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_ICHECK(opt)
        << "MakePackedAPI required the function to be annotated with tvm::attr::kTarget ("
        << tvm::attr::kTarget << "), but the function only has attributes " << func->attrs;
    return opt.value();
  }();
  int target_device_type = target->GetTargetDeviceType();

  // A function without a host target has already been lowered.
  Target target_host;
  if (auto opt = target->GetHost()) {
    target_host = opt.value();
  } else {
    return func;
  }

  auto* func_ptr = func.CopyOnWrite();
  const Stmt nop = Evaluate(0);
  int num_args = static_cast<int>(func_ptr->params.size());

  // Data field definitions
  Var v_self_handle("self_handle", DataType::Handle());
  Var v_packed_args("args", DataType::Handle());
  Var v_num_packed_args("num_args", DataType::Int(32));
  Var v_result("result", PointerType(PrimType(DataType::Void())));

  // The device context
  Var device_id("dev_id");
  Integer device_type(target_device_type);

  // seq_init gives sequence of initialization
  // seq_check gives sequence of later checks after init
  std::vector<Stmt> seq_init, seq_check, arg_buffer_declarations;
  std::unordered_map<const VarNode*, PrimExpr> vmap;

  // Create ArgBinder with all function metadata
  ArgBinder binder(&vmap, name_hint, func_ptr->params, func_ptr->buffer_map, v_packed_args);

  // Num args check (uses binder's func_signature)
  {
    std::ostringstream detail;
    detail << "Expected " << num_args << " arguments";
    seq_init.push_back(MakeAssertEQ("TypeError", v_num_packed_args, num_args, detail.str(),
                                    binder.func_signature()));
  }

  if (num_args > 0) {
    seq_init.push_back(MakeAssertNotNull("TypeError", v_packed_args, "args pointer is NULL",
                                         binder.func_signature()));
  }

  // Bind all packed args: type-check, load values
  std::vector<std::pair<PrimExpr, Var>> var_def;
  for (int i = 0; i < num_args; ++i) {
    auto [arg_value, param] = binder.BindPackedArg(i);
    var_def.emplace_back(arg_value, param);
  }

  // Bind scalar params and DLTensor buffers
  binder.BindAllParams(var_def, device_type, device_id, &arg_buffer_declarations);

  // signature: (void* handle, TVMFFIAny* packed_args, int num_args, TVMFFIAny* v_result)
  ffi::Array<Var> args{v_self_handle, v_packed_args, v_num_packed_args, v_result};

  // reset global symbol to attach prefix
  func = WithAttrs(
      std::move(func),
      {{tvm::attr::kCallingConv, static_cast<int>(CallingConv::kCPackedFunc)},
       {tvm::attr::kTarget, target_host},
       {tvm::attr::kGlobalSymbol, ffi::symbol::tvm_ffi_symbol_prefix + global_symbol.value()}});

  Stmt body = ReturnRewriter(v_result)(func_ptr->body);
  body = AttrStmt(make_zero(DataType::Int(32)), attr::compute_scope,
                  StringImm(name_hint + "_compute_"), body);
  // Set device context
  if (vmap.count(device_id.get())) {
    ffi::Any node = ffi::String("default");
    seq_check.push_back(AttrStmt(node, attr::device_id, device_id, nop));
    seq_check.push_back(AttrStmt(node, attr::device_type, device_type, nop));

    if (runtime::DeviceAPI::NeedSetDevice(target_device_type)) {
      Stmt set_device =
          Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(),
                        {StringImm(runtime::symbol::tvm_set_device), device_type, device_id}));
      body = SeqStmt({set_device, body});
    }
  }

  // Return error code of zero on success
  body = SeqStmt({body, Evaluate(ret(Integer(0)))});

  body = MergeNest(
      {seq_init, binder.init_nest(), seq_check, binder.asserts(), arg_buffer_declarations}, body);
  func_ptr->body = body;
  func_ptr->params = args;

  ffi::Array<Var> undefined = UndefinedVars(func_ptr->body, func_ptr->params);
  TVM_FFI_ICHECK_EQ(undefined.size(), 0)
      << "In PrimFunc " << name_hint << " variables " << undefined
      << " are used, but are not passed in as API arguments";

  func_ptr->buffer_map = ffi::Map<Var, Buffer>();
  func_ptr->ret_type = PrimType(DataType::Int(32));

  // return the function.
  return func;
}

namespace transform {

Pass MakePackedAPI() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    ffi::Map<GlobalVar, ffi::String> packed_func_methods;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        auto prim_func = opt.value();
        if (auto global_symbol = RequiresPackedAPI(prim_func)) {
          packed_func_methods.Set(gvar, global_symbol.value());
        }
      }
    }

    IRModuleNode* mptr = mod.CopyOnWrite();
    IRModule updates;

    for (const auto& [gvar, base_func] : mptr->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        auto func = opt.value();
        auto orig_func = func;

        if (auto body = SubroutineCallRewriter::Apply(packed_func_methods, func->body)) {
          func.CopyOnWrite()->body = body.value();
        }

        func = MakePackedAPI(std::move(func));

        if (!func.same_as(orig_func)) {
          updates->Add(gvar, func);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.MakePackedAPI", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.MakePackedAPI", []() { return MakePackedAPI(); });
}
}  // namespace transform
}  // namespace tir
}  // namespace tvm
