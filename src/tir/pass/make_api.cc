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
 * \file make_api.cc Build API function.
 */
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/buffer.h>
#include <tvm/runtime/device_api.h>
#include <vector>
#include <utility>
#include <unordered_set>

#include "ir_util.h"
#include "arg_binder.h"

namespace tvm {
namespace tir {

inline Stmt MakeAssertEQ(PrimExpr lhs, PrimExpr rhs, std::string msg) {
  return AssertStmtNode::make(lhs == rhs, msg, EvaluateNode::make(0));
}

LoweredFunc MakeAPI(Stmt body,
                    std::string name,
                    Array<ObjectRef> api_args,
                    int num_unpacked_args,
                    bool is_restricted) {
  const Stmt nop = EvaluateNode::make(0);
  int num_args = static_cast<int>(api_args.size());
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
  Var device_type("dev_type"), device_id("dev_id");
  // seq_init gives sequence of initialization
  // seq_check gives sequence of later checks after init
  std::vector<Stmt> seq_init, seq_check;
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  ArgBinder binder(&vmap);
  // ---------------------------
  // local function definitions
  // load i-th argument as type t
  auto f_arg_value = [&](DataType t, int i) {
    Array<PrimExpr> call_args{v_packed_args,
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
  // get declaration of argument i
  auto f_arg_decl = [&](int i) {
    std::ostringstream os;
    os << "arg" << i;
    const VarNode* v = api_args[i].as<VarNode>();
    return Var(os.str(), v ? v->dtype: DataType::Handle());
  };
  // ---------------------------
  // start of logics
  // add signiture for packed arguments.
  if (num_packed_args != 0) {
    args.push_back(v_packed_args);
    args.push_back(v_packed_arg_type_ids);
    args.push_back(v_num_packed_args);
    std::ostringstream os;

    os << name << ": num_args should be " << num_packed_args;
    seq_init.emplace_back(
        MakeAssertEQ(v_num_packed_args, num_packed_args, os.str()));
  }

  // Save the input variables and buffers that will be bound later.
  std::vector<std::pair<Var, Var> > var_defs;
  std::vector<std::pair<Buffer, Var> > buf_defs;
  for (int i = 0; i < static_cast<int>(api_args.size()); ++i) {
    Var v_arg = f_arg_decl(i);
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
        msg << name << ": Expect arg[" << i << "] to be pointer";
        seq_check.emplace_back(
            AssertStmtNode::make(tcode == kTVMOpaqueHandle ||
                             tcode == kTVMNDArrayHandle ||
                             tcode == kTVMDLTensorHandle ||
                             tcode == kTVMNullptr, msg.str(), nop));
      } else if (t.is_int() || t.is_uint()) {
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be int";
        seq_check.emplace_back(AssertStmtNode::make(tcode == kDLInt, msg.str(), nop));
      } else {
        CHECK(t.is_float());
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be float";
        seq_check.emplace_back(
            AssertStmtNode::make(tcode == kDLFloat, msg.str(), nop));
      }
    } else {
      args.push_back(v_arg);
    }
    // add checks for functions.
    if (api_args[i].as<VarNode>()) {
      var_defs.emplace_back(std::make_pair(Downcast<Var>(api_args[i]), v_arg));
    } else {
      // Buffer checks
      CHECK(api_args[i].as<BufferNode>())
          << "api_args can only be Buffer or Var";
      buf_defs.emplace_back(std::make_pair(Downcast<Buffer>(api_args[i]), v_arg));
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
  for (const auto& arg : var_defs) {
    binder.Bind(arg.first, arg.second, arg.second->name_hint, true);
  }

  for (const auto& buf_arg : buf_defs) {
    binder.BindDLTensor(buf_arg.first, device_type, device_id,
                        buf_arg.second, buf_arg.second->name_hint);
  }

  ObjectPtr<LoweredFuncNode> n = make_object<LoweredFuncNode>();
  n->name = name;
  n->args = args;
  n->handle_data_type = binder.def_handle_dtype();
  n->is_packed_func = num_unpacked_args == 0;
  n->is_restricted = is_restricted;
  body = AttrStmtNode::make(
      make_zero(DataType::Int(32)), attr::compute_scope,
      StringImmNode::make(name + "_compute_"), body);
  // Set device context
  if (vmap.count(device_id.get())) {
    PrimExpr node = StringImmNode::make("default");
    CHECK(vmap.count(device_type.get()));
    seq_check.push_back(AttrStmtNode::make(
        node, attr::device_context_id, device_id, nop));
    seq_check.push_back(AttrStmtNode::make(
        node, attr::device_context_type, device_type, nop));
    Stmt set_device = IfThenElseNode::make(
        device_type != kDLCPU, EvaluateNode::make(CallNode::make(
            DataType::Int(32), intrinsic::tvm_call_packed,
            {StringImmNode::make(runtime::symbol::tvm_set_device),
             device_type, device_id}, CallNode::Intrinsic)));
    body = SeqStmt({set_device, body});
  }
  n->body = MergeNest(
      {seq_init, binder.init_nest(), seq_check, binder.asserts()}, body);
  LoweredFunc f(n);
  Array<Var> undefined = UndefinedVars(f->body, f->args);
  if (undefined.size() != 0) {
    std::ostringstream os;
    for (Var v : undefined) {
      os << " \'" << v->name_hint << "\' ";
    }
    os << " does not appear in api_args";
    LOG(FATAL) << "Not all Vars are passed in api_args: " << os.str();
  }
  return f;
}

class DeviceTypeBinder: public StmtExprMutator {
 public:
  explicit DeviceTypeBinder(int device_type)
      : device_type_(device_type) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::device_context_type) {
      if (const VarNode* var = op->value.as<VarNode>()) {
        var_ = var;
        PrimExpr value = make_const(op->value.dtype(), device_type_);
        Stmt body = StmtExprMutator::VisitStmt_(op);
        var_ = nullptr;
        std::ostringstream os;
        os << "device_type need to be " << device_type_;
        return AssertStmtNode::make(op->value == value, os.str(), body);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    // eager simplify if guard.
    Stmt res = StmtExprMutator::VisitStmt_(op);
    op = res.as<IfThenElseNode>();
    if (is_zero(op->condition)) {
      if (op->else_case.defined()) return op->else_case;
      return EvaluateNode::make(0);
    }
    if (is_one(op->condition)) {
      return op->then_case;
    }
    return res;
  }

  PrimExpr VisitExpr_(const NENode* op) final {
    // eager check NE for device check
    PrimExpr res = StmtExprMutator::VisitExpr_(op);
    op = res.as<NENode>();
    if (tir::Equal(op->a, op->b)) {
      return make_const(op->dtype, false);
    }
    return res;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == var_) {
      return make_const(op->dtype, device_type_);
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

 public:
  const VarNode* var_{nullptr};
  int device_type_;
};

LoweredFunc BindDeviceType(LoweredFunc f,
                           int device_type) {
  auto n = make_object<LoweredFuncNode>(*f.operator->());
  n->body = DeviceTypeBinder(device_type)(n->body);
  return LoweredFunc(n);
}

}  // namespace tir
}  // namespace tvm
