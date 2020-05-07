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
#include <tvm/tir/expr.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/buffer.h>
#include <tvm/target/target.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container.h>

#include <vector>
#include <utility>
#include <unordered_set>

#include "ir_util.h"
#include "arg_binder.h"

namespace tvm {
namespace tir {

inline Stmt MakeAssertEQ(PrimExpr lhs, PrimExpr rhs, std::string msg) {
  return AssertStmtNode::make(lhs == rhs, tvm::tir::StringImmNode::make(msg),
                              EvaluateNode::make(0));
}

PrimFunc MakePackedAPI(PrimFunc&& func,
                       int num_unpacked_args) {
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  CHECK(global_symbol)
      << "MakePackedAPI: Expect PrimFunc to have the global_symbol attribute";

  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  CHECK(target.defined())
      << "MakePackedAPI: Require the target attribute";
  int target_device_type = target.value()->device_type;

  std::string name_hint = global_symbol.value();

  auto* func_ptr = func.CopyOnWrite();
  const Stmt nop = EvaluateNode::make(0);
  int num_args = static_cast<int>(func_ptr->params.size());
  CHECK_LE(num_unpacked_args, num_args);

  int num_packed_args = num_args - num_unpacked_args;
  // Data field definitions
  // The packed fields
  Var v_packed_args("args", DataType::Handle());
  Var v_packed_arg_type_ids("arg_type_ids", DataType::Handle());
  Var v_num_packed_args("num_args", DataType::Int(32));
  Var v_out_ret_value("out_ret_value", DataType::Handle());
  Var v_out_ret_tcode("out_ret_tcode", DataType::Handle());
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
    Array<PrimExpr> call_args{
      v_packed_args,
      IntImm(DataType::Int(32), i),
      IntImm(DataType::Int(32), intrinsic::kTVMValueContent)};
    // load 64 bit version
    DataType api_type = APIType(t);
    PrimExpr res = CallNode::make(
        api_type, intrinsic::tvm_struct_get, call_args,
        CallNode::PureIntrinsic);
    // cast to the target version.
    if (api_type != t) {
      res = CastNode::make(t, res);
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
    seq_init.emplace_back(
        MakeAssertEQ(v_num_packed_args, num_packed_args, os.str()));
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
      seq_init.emplace_back(LetStmtNode::make(
          v_arg, f_arg_value(v_arg.dtype(), i), nop));
      // type code checks
      Var tcode(v_arg->name_hint + ".code", DataType::Int(32));
      seq_init.emplace_back(LetStmtNode::make(
          tcode, LoadNode::make(
              DataType::Int(32), v_packed_arg_type_ids,
              IntImm(DataType::Int(32), i), const_true(1)),
          nop));
      DataType t = v_arg.dtype();
      if (t.is_handle()) {
        std::ostringstream msg;
        msg << name_hint << ": Expect arg[" << i << "] to be pointer";
        seq_check.emplace_back(
            AssertStmtNode::make(tcode == kTVMOpaqueHandle ||
                                 tcode == kTVMNDArrayHandle ||
                                 tcode == kTVMDLTensorHandle ||
                                 tcode == kTVMNullptr,
                                 tvm::tir::StringImmNode::make(msg.str()), nop));
      } else if (t.is_int() || t.is_uint()) {
        std::ostringstream msg;
        msg << name_hint << ": Expect arg[" << i << "] to be int";
        seq_check.emplace_back(
            AssertStmtNode::make(tcode == kDLInt, tvm::tir::StringImmNode::make(msg.str()), nop));
      } else {
        CHECK(t.is_float());
        std::ostringstream msg;
        msg << name_hint << ": Expect arg[" << i << "] to be float";
        seq_check.emplace_back(
            AssertStmtNode::make(tcode == kDLFloat, tvm::tir::StringImmNode::make(msg.str()), nop));
      }
    } else {
      args.push_back(v_arg);
    }
  }

  // allow return value if the function is packed.
  if (num_packed_args != 0) {
    args.push_back(v_out_ret_value);
    args.push_back(v_out_ret_tcode);
  }

  size_t expected_nargs = num_unpacked_args + (num_packed_args != 0 ? 5 : 0);
  CHECK_EQ(args.size(), expected_nargs);

  // Arg definitions are defined before buffer binding to avoid the use before
  // def errors.
  //
  // For example, for auto broadcasting, checks are required to guarantee that
  // either 0 or the original stride will be correctly used. Checks here have
  // to use the args that may have no let bining yet. Therefore, hoisting let
  // binding for args before buffer declaration is needed.
  for (const auto& kv : var_def) {
    binder.Bind(kv.second, kv.first, kv.first->name_hint, true);
  }

  for (const auto& kv : buffer_def) {
    binder.BindDLTensor(kv.second, device_type, device_id,
                        kv.first, kv.first->name_hint);
  }

  if (num_unpacked_args == 0) {
    func = WithAttr(std::move(func), tvm::attr::kCallingConv, Integer(CallingConv::kCPackedFunc));
  }

  auto body = AttrStmtNode::make(
      make_zero(DataType::Int(32)), attr::compute_scope,
      StringImmNode::make(name_hint + "_compute_"), func_ptr->body);
  // Set device context
  if (vmap.count(device_id.get())) {
    PrimExpr node = StringImmNode::make("default");
    seq_check.push_back(AttrStmtNode::make(
        node, attr::device_context_id, device_id, nop));
    seq_check.push_back(AttrStmtNode::make(
        node, attr::device_context_type, device_type, nop));

    if (runtime::DeviceAPI::NeedSetDeviceContext(target_device_type)) {
      Stmt set_device = EvaluateNode::make(CallNode::make(
              DataType::Int(32), intrinsic::tvm_call_packed,
              {StringImmNode::make(runtime::symbol::tvm_set_device),
               device_type, device_id}, CallNode::Intrinsic));
      body = SeqStmt({set_device, body});
    }
  }
  func_ptr->body = MergeNest(
      {seq_init, binder.init_nest(), seq_check, binder.asserts()}, body);
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
        if (func->GetAttr<Integer>(
                tvm::attr::kCallingConv,
                Integer(CallingConv::kDefault)) == CallingConv::kDefault) {
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

  return tvm::transform::CreateModulePass(
      pass_func, 0, "tir.MakePackedAPI", {});
}

TVM_REGISTER_GLOBAL("tir.transform.MakePackedAPI")
.set_body_typed(MakePackedAPI);
}  // namespace transform
}  // namespace tir
}  // namespace tvm
