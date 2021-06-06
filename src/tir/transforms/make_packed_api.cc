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
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
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

#include "arg_binder.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

class ReturnRewriter : public StmtMutator {
 public:
  explicit ReturnRewriter(Var ret_var, Var ret_tcode) : ret_var_(ret_var), ret_tcode_(ret_tcode) {}

  Stmt VisitStmt_(const ForNode* node) override {
    if (node->kind == ForKind::kParallel) in_parallel_ += 1;
    Stmt ret = StmtMutator::VisitStmt_(node);
    if (node->kind == ForKind::kParallel) in_parallel_ -= 1;
    return ret;
  }

  Stmt VisitStmt_(const EvaluateNode* node) override {
    Stmt ret = StmtMutator::VisitStmt_(node);
    const EvaluateNode* eval = ret.as<EvaluateNode>();
    ICHECK(eval);
    if (const CallNode* call = eval->value.as<CallNode>()) {
      if (call->op.same_as(builtin::ret())) {
        ICHECK_EQ(in_parallel_, 0) << "tir.ret cannot be used in parallel scope.";
        ICHECK_EQ(call->args.size(), 1) << "tir.ret expect a single argument.";
        ret = WriteToOut(call->args[0], ret_var_, ret_tcode_);
      }
    }
    return ret;
  }

 private:
  std::pair<int, PrimExpr> ConvertForFFI(PrimExpr val) {
    // convert val's data type to FFI data type, return type code
    DataType dtype = val.dtype();
    if (dtype.is_int() || dtype.is_uint()) {
      return {kTVMArgInt, Cast(DataType::Int(64), val)};
    } else if (dtype.is_float()) {
      return {kTVMArgFloat, Cast(DataType::Float(64), val)};
    } else if (dtype.is_void()) {
      return {kTVMNullptr, val};
    } else {
      LOG(FATAL) << "data type " << dtype << " not supported yet";
    }
    return {kTVMNullptr, val};
  }

  Stmt WriteToOut(PrimExpr val, Var ret_var, Var ret_tcode) {
    auto p = ConvertForFFI(val);
    int tcode = p.first;
    val = p.second;
    Stmt store_val = Store(ret_var_, val, 0, const_true());
    Stmt store_tcode = Store(ret_tcode_, tcode, 0, const_true());
    Stmt ret_zero = Evaluate(tvm::ret(0));
    return SeqStmt({store_val, store_tcode, ret_zero});
  }

  Var ret_var_;
  Var ret_tcode_;
  int in_parallel_{0};
};

Stmt RewriteReturn(Stmt body, Var ret_var, Var ret_tcode) {
  ReturnRewriter rewriter(ret_var, ret_tcode);
  return rewriter(body);
}

inline Stmt MakeAssertEQ(PrimExpr lhs, PrimExpr rhs, std::string msg) {
  return AssertStmt(lhs == rhs, tvm::tir::StringImm(msg), Evaluate(0));
}

PrimFunc MakePackedAPI(PrimFunc&& func, int num_unpacked_args) {
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol) << "MakePackedAPI: Expect PrimFunc to have the global_symbol attribute";

  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined()) << "MakePackedAPI: Require the target attribute";
  int target_device_type = target.value()->kind->device_type;

  std::string name_hint = global_symbol.value();

  auto* func_ptr = func.CopyOnWrite();
  const Stmt nop = Evaluate(0);
  int num_args = static_cast<int>(func_ptr->params.size());
  ICHECK_LE(num_unpacked_args, num_args);

  int num_packed_args = num_args - num_unpacked_args;
  // Data field definitions
  // The packed fields
  Var v_packed_args("args", DataType::Handle());
  Var v_packed_arg_type_ids("arg_type_ids", DataType::Handle());
  Var v_num_packed_args("num_args", DataType::Int(32));
  Var v_out_ret_value("out_ret_value", DataType::Handle());
  Var v_out_ret_tcode("out_ret_tcode", DataType::Handle());
  Var v_resource_handle("resource_handle", DataType::Handle());
  // The arguments of the function.
  Array<Var> args;
  // The device context
  Var device_id("dev_id");
  Integer device_type(target_device_type);
  // seq_init gives sequence of initialization
  // seq_check gives sequence of later checks after init
  std::vector<Stmt> seq_init, seq_check;
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  ArgBinder binder(&vmap);
  // ---------------------------
  // local function definitions
  // load i-th argument as type t
  auto f_arg_value = [&](DataType t, int i) {
    Array<PrimExpr> call_args{v_packed_args, IntImm(DataType::Int(32), i),
                              IntImm(DataType::Int(32), builtin::kTVMValueContent)};
    // load 64 bit version
    DataType api_type = APIType(t);
    PrimExpr res = Call(api_type, builtin::tvm_struct_get(), call_args);
    // cast to the target version.
    if (api_type != t) {
      res = Cast(t, res);
    }
    return res;
  };

  // ---------------------------
  // start of logics
  // add signiture for packed arguments.
  if (num_packed_args != 0) {
    args.push_back(v_packed_args);
    args.push_back(v_packed_arg_type_ids);
    args.push_back(v_num_packed_args);
    std::ostringstream os;

    os << name_hint << ": num_args should be " << num_packed_args;
    seq_init.emplace_back(MakeAssertEQ(v_num_packed_args, num_packed_args, os.str()));
  }

  // Need to re-declare vars, in case some arguments also appears in the buffer.
  std::vector<std::pair<Var, Var> > var_def;
  std::vector<std::pair<Var, Buffer> > buffer_def;

  for (int i = 0; i < static_cast<int>(func_ptr->params.size()); ++i) {
    Var param = func_ptr->params[i];
    Var v_arg = Var("arg" + std::to_string(i), param->dtype);

    auto it = func_ptr->buffer_map.find(param);
    if (it != func_ptr->buffer_map.end()) {
      buffer_def.emplace_back(v_arg, (*it).second);
    } else {
      var_def.emplace_back(v_arg, param);
    }
    if (i < num_packed_args) {
      // Value loads
      seq_init.emplace_back(LetStmt(v_arg, f_arg_value(v_arg.dtype(), i), nop));
      // type code checks
      Var tcode(v_arg->name_hint + ".code", DataType::Int(32));
      seq_init.emplace_back(LetStmt(tcode,
                                    Load(DataType::Int(32), v_packed_arg_type_ids,
                                         IntImm(DataType::Int(32), i), const_true(1)),
                                    nop));
      DataType t = v_arg.dtype();
      if (t.is_handle()) {
        std::ostringstream msg;
        msg << name_hint << ": Expect arg[" << i << "] to be pointer";
        seq_check.emplace_back(AssertStmt(tcode == kTVMOpaqueHandle || tcode == kTVMNDArrayHandle ||
                                              tcode == kTVMDLTensorHandle || tcode == kTVMNullptr,
                                          tvm::tir::StringImm(msg.str()), nop));
      } else if (t.is_int() || t.is_uint()) {
        std::ostringstream msg;
        msg << name_hint << ": Expect arg[" << i << "] to be int";
        seq_check.emplace_back(AssertStmt(tcode == kDLInt, tvm::tir::StringImm(msg.str()), nop));
      } else {
        ICHECK(t.is_float());
        std::ostringstream msg;
        msg << name_hint << ": Expect arg[" << i << "] to be float";
        seq_check.emplace_back(AssertStmt(tcode == kDLFloat, tvm::tir::StringImm(msg.str()), nop));
      }
    } else {
      args.push_back(v_arg);
    }
  }

  // allow return value if the function is packed.
  if (num_packed_args != 0) {
    args.push_back(v_out_ret_value);
    args.push_back(v_out_ret_tcode);
    args.push_back(v_resource_handle);
  }

  size_t expected_nargs = num_unpacked_args + (num_packed_args != 0 ? 6 : 0);
  ICHECK_EQ(args.size(), expected_nargs);

  // Arg definitions are defined before buffer binding to avoid the use before
  // def errors.
  //
  // For example, for auto broadcasting, checks are required to guarantee that
  // either 0 or the original stride will be correctly used. Checks here have
  // to use the args that may have no let binding yet. Therefore, hoisting let
  // binding for args before buffer declaration is needed.
  for (const auto& kv : var_def) {
    binder.Bind(kv.second, kv.first, kv.first->name_hint, true);
  }

  for (const auto& kv : buffer_def) {
    binder.BindDLTensor(kv.second, device_type, device_id, kv.first, kv.first->name_hint);
  }

  if (num_unpacked_args == 0) {
    func = WithAttr(std::move(func), tvm::attr::kCallingConv, Integer(CallingConv::kCPackedFunc));
  }

  Stmt body = RewriteReturn(func_ptr->body, v_out_ret_value, v_out_ret_tcode);
  body = AttrStmt(make_zero(DataType::Int(32)), attr::compute_scope,
                  StringImm(name_hint + "_compute_"), body);
  // Set device context
  if (vmap.count(device_id.get())) {
    PrimExpr node = StringImm("default");
    seq_check.push_back(AttrStmt(node, attr::device_id, device_id, nop));
    seq_check.push_back(AttrStmt(node, attr::device_type, device_type, nop));

    if (runtime::DeviceAPI::NeedSetDevice(target_device_type)) {
      Stmt set_device =
          Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(),
                        {StringImm(runtime::symbol::tvm_set_device), device_type, device_id}));
      body = SeqStmt({set_device, body});
    }
  }
  func_ptr->body = MergeNest({seq_init, binder.init_nest(), seq_check, binder.asserts()}, body);
  func_ptr->params = args;

  Array<Var> undefined = UndefinedVars(func_ptr->body, func_ptr->params);
  if (undefined.size() != 0) {
    std::ostringstream os;
    for (Var v : undefined) {
      os << " \'" << v->name_hint << "\' ";
    }
    os << " is not bound to any variables";
    LOG(FATAL) << "Not all Vars are passed in api_args: " << os.str();
  }

  func_ptr->buffer_map = Map<Var, Buffer>();
  func_ptr->checked_type_ = func_ptr->func_type_annotation();
  func_ptr->ret_type = PrimType(DataType::Int(32));

  // return the function.
  return std::move(func);
}

namespace transform {

Pass MakePackedAPI(int num_unpacked_args) {
  auto pass_func = [num_unpacked_args](IRModule m, PassContext ctx) {
    IRModuleNode* mptr = m.CopyOnWrite();
    std::vector<std::pair<GlobalVar, PrimFunc> > updates;

    for (const auto& kv : mptr->functions) {
      if (auto* n = kv.second.as<PrimFuncNode>()) {
        PrimFunc func = GetRef<PrimFunc>(n);
        if (func->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(CallingConv::kDefault)) ==
            CallingConv::kDefault) {
          auto updated_func = MakePackedAPI(std::move(func), num_unpacked_args);
          updates.push_back({kv.first, updated_func});
        }
      }
    }

    for (const auto& pair : updates) {
      mptr->AddUnchecked(pair.first, pair.second);
    }
    return m;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.MakePackedAPI", {});
}

TVM_REGISTER_GLOBAL("tir.transform.MakePackedAPI").set_body_typed(MakePackedAPI);
}  // namespace transform
}  // namespace tir
}  // namespace tvm
