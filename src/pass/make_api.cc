/*!
 *  Copyright (c) 2017 by Contributors
 * \file make_api.cc Build API function.
 */
#include <tvm/ir_pass.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/buffer.h>
#include <tvm/runtime/device_api.h>
#include <vector>
#include <utility>
#include <unordered_set>

#include "./ir_util.h"
#include "./arg_binder.h"
#include "../arithmetic/compute_expr.h"

namespace tvm {
namespace ir {

inline Stmt MakeAssertEQ(Expr lhs, Expr rhs, std::string msg) {
  return AssertStmt::make(lhs == rhs, msg, Evaluate::make(0));
}

LoweredFunc MakeAPI(Stmt body,
                    std::string name,
                    Array<NodeRef> api_args,
                    int num_unpacked_args,
                    bool is_restricted) {
  const Stmt nop = Evaluate::make(0);
  int num_args = static_cast<int>(api_args.size());
  CHECK_LE(num_unpacked_args, num_args);
  int num_packed_args = num_args - num_unpacked_args;
  // Data field definitions
  // The packed fields
  Var v_packed_args("args", Handle());
  Var v_packed_arg_type_ids("arg_type_ids", Handle());
  Var v_num_packed_args("num_args", Int(32));
  // The arguments of the function.
  Array<Var> args;
  // The device context
  Var device_type("dev_type"), device_id("dev_id");
  // seq_init gives sequence of initialization
  // seq_check gives sequence of later checks after iniit
  std::vector<Stmt> seq_init, seq_check;
  std::unordered_map<const Variable*, Expr> vmap;
  ArgBinder binder(&vmap);
  // ---------------------------
  // local function defintiions
  // load i-th argument as type t
  auto f_arg_value = [&](Type t, int i) {
    Array<Expr> call_args{v_packed_args,
                          IntImm::make(Int(32), i),
                          IntImm::make(Int(32), intrinsic::kTVMValueContent)};
    // load 64 bit version
    Type api_type = APIType(t);
    Expr res = Call::make(
        api_type, intrinsic::tvm_struct_get, call_args,
        Call::PureIntrinsic);
    // cast to the target version.
    if (api_type != t) {
      res = Cast::make(t, res);
    }
    return res;
  };
  // get declaration of argument i
  auto f_arg_decl = [&](int i) {
    std::ostringstream os;
    os << "arg" << i;
    const Variable* v = api_args[i].as<Variable>();
    return Var(os.str(), v ? v->type: Handle());
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
  for (int i = 0; i < static_cast<int>(api_args.size()); ++i) {
    Var v_arg = f_arg_decl(i);
    if (i < num_packed_args) {
      // Value loads
      seq_init.emplace_back(LetStmt::make(
          v_arg, f_arg_value(v_arg.type(), i), nop));
      // type code checks
      Var tcode(v_arg->name_hint + ".code", Int(32));
      seq_init.emplace_back(LetStmt::make(
          tcode, Load::make(
              Int(32), v_packed_arg_type_ids, IntImm::make(Int(32), i), const_true(1)),
          nop));
      Type t = v_arg.type();
      if (t.is_handle()) {
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be pointer";
        seq_check.emplace_back(
            AssertStmt::make(tcode == kHandle ||
                             tcode == kArrayHandle ||
                             tcode == kNull, msg.str(), nop));
      } else if (t.is_int() || t.is_uint()) {
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be int";
        seq_check.emplace_back(AssertStmt::make(tcode == kDLInt, msg.str(), nop));
      } else {
        CHECK(t.is_float());
        std::ostringstream msg;
        msg << name << ": Expect arg[" << i << "] to be float";
        seq_check.emplace_back(
            AssertStmt::make(tcode == kDLFloat, msg.str(), nop));
      }
    } else {
      args.push_back(v_arg);
    }
    // add checks for functions.
    if (api_args[i].as<Variable>()) {
      binder.Bind(Var(api_args[i].node_), v_arg, v_arg->name_hint, true);
    } else {
      // Buffer checks
      CHECK(api_args[i].as<BufferNode>())
          << "api_args can only be Buffer or Var";
      Buffer buf(api_args[i].node_);
      binder.BindDLTensor(
          buf, device_type, device_id, v_arg, v_arg->name_hint);
    }
  }

  std::shared_ptr<LoweredFuncNode> n = std::make_shared<LoweredFuncNode>();
  n->name = name;
  n->args = args;
  n->handle_data_type = binder.def_handle_dtype();
  n->is_packed_func = num_unpacked_args == 0;
  n->is_restricted = is_restricted;
  body = AttrStmt::make(
      make_zero(Int(32)), attr::compute_scope,
      StringImm::make(name + "_compute_"), body);
  // Set device context
  if (vmap.count(device_id.get())) {
    Expr node = StringImm::make("default");
    CHECK(vmap.count(device_type.get()));
    seq_check.push_back(AttrStmt::make(
        node, attr::device_context_id, device_id, nop));
    seq_check.push_back(AttrStmt::make(
        node, attr::device_context_type, device_type, nop));
    Stmt set_device = IfThenElse::make(
        device_type != kDLCPU, Evaluate::make(Call::make(
            Int(32), intrinsic::tvm_call_packed,
            {StringImm::make(runtime::symbol::tvm_set_device),
             device_type, device_id}, Call::Intrinsic)));
    body = Block::make(set_device, body);
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
    os << " does not appeared in api_args";
    LOG(FATAL) << "Not all Vars are passed in api_args: " << os.str();
  }
  return f;
}

class DeviceTypeBinder: public IRMutator {
 public:
  explicit DeviceTypeBinder(int device_type)
      : device_type_(device_type) {}

  Stmt Mutate_(const AttrStmt* op, const Stmt &s) final {
    if (op->attr_key == attr::device_context_type) {
      if (const Variable* var = op->value.as<Variable>()) {
        std::unordered_map<const Variable*, Expr> dmap;
        Expr value = make_const(op->value.type(), device_type_);
        dmap[var] = value;
        Stmt body = Substitute(s, dmap);
        std::ostringstream os;
        os << "device_type need to be " << device_type_;
        return AssertStmt::make(op->value == value, os.str(), body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 public:
  int device_type_;
};

LoweredFunc BindDeviceType(LoweredFunc f,
                           int device_type) {
  auto n = std::make_shared<LoweredFuncNode>(*f.operator->());
  n->body = DeviceTypeBinder(device_type).Mutate(n->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
