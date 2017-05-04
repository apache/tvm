/*!
 *  Copyright (c) 2017 by Contributors
 * \file make_api.cc Build API function.
 */
#include <tvm/ir_pass.h>
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/buffer.h>

#include <vector>
#include <utility>
#include <unordered_set>

#include "./ir_util.h"

namespace tvm {
namespace ir {

inline Expr TVMArrayGet(Type t, Var arr, intrinsic::TVMStructFieldKind kind) {
  return TVMStructGet(t, arr, 0, kind);
}

inline Stmt AssertNull(Var handle, std::string msg) {
  return AssertStmt::make(Call::make(
      Bool(1), intrinsic::tvm_handle_is_null,
      {handle}, Call::PureIntrinsic), msg);
}

inline Stmt MakeAssertEQ(Expr lhs, Expr rhs, std::string msg) {
  return AssertStmt::make(lhs == rhs, msg);
}

LoweredFunc MakeAPI(Stmt body,
                    std::string name,
                    Array<NodeRef> api_args,
                    int num_unpacked_args) {
  const Type tvm_shape_type = TVMShapeIndexType();
  const Type tvm_ndim_type = Int(32);
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
  // seq_init gives sequence of initialization
  // seq_check gives sequence of later checks after iniit
  std::vector<Stmt> seq_init, seq_check;
  std::unordered_set<const Variable*> visited;
  // the handle data types
  Map<Var, Expr> handle_data_type;
  // The device context
  Var device_id, device_type;
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
  // Push related into assertions or variable defintion
  // given the symbolic declaration and concrete value
  auto f_push = [&](Expr sym, Expr value, std::string field) {
    if (sym.as<Variable>()) {
      // If sym is a Variable and this Variable is not yet defined
      // add this to defintion.
      Var v(sym.node_);
      if (!visited.count(v.get())) {
        seq_init.emplace_back(LetStmt::make(v, value, nop));
        visited.insert(v.get());
        return true;
      }
    }
    // otherwise, assume sym is already defined, insert assertion.
    std::ostringstream os;
    os << "Field " << field << " has a unsatisfied constraint";
    seq_check.emplace_back(MakeAssertEQ(sym, value, os.str()));
    return false;
  };
  // ---------------------------
  // start of logics
  // add signiture for packed arguments.
  if (num_packed_args != 0) {
    args.push_back(v_packed_args);
    args.push_back(v_packed_arg_type_ids);
    args.push_back(v_num_packed_args);
    std::ostringstream os;
    os << "expected num_args to be " << num_packed_args;
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
        msg << "Expect argument " << i << " to be pointer";
        seq_check.emplace_back(
            AssertStmt::make(tcode == kHandle ||
                             tcode == kArrayHandle ||
                             tcode == kNull, msg.str()));
      } else if (t.is_int() || t.is_uint()) {
        std::ostringstream msg;
        msg << "Expect argument " << i << " to be int";
        seq_check.emplace_back(AssertStmt::make(tcode == kInt, msg.str()));
      } else {
        CHECK(t.is_float());
        std::ostringstream msg;
        msg << "Expect argument " << i << " to be float";
        seq_check.emplace_back(AssertStmt::make(tcode == kFloat, msg.str()));
      }
    } else {
      args.push_back(v_arg);
    }
    // add checks for functions.
    if (api_args[i].as<Variable>()) {
      f_push(Var(api_args[i].node_), v_arg, v_arg->name_hint);
    } else {
      // Buffer checks
      CHECK(api_args[i].as<BufferNode>())
          << "api_args can only be Buffer or Var";
      Buffer buf(api_args[i].node_);
      // dimension checks
      Expr v_ndim = TVMArrayGet(tvm_ndim_type, v_arg, intrinsic::kArrNDim);
      std::ostringstream ndim_err_msg;
      ndim_err_msg << "arg_" << i
                   << ".ndim is expected to equal "
                   << buf->shape.size();
      seq_init.emplace_back(
          MakeAssertEQ(v_ndim,
                       make_const(tvm_ndim_type,
                                  static_cast<int64_t>(buf->shape.size())),
                       ndim_err_msg.str()));
      // type checks
      Type dtype = buf->dtype;
      std::ostringstream type_err_msg;
      type_err_msg << "arg" << i << ".dtype is expected to be " << dtype;
      Expr cond = (TVMArrayGet(UInt(8), v_arg, intrinsic::kArrTypeCode) ==
                   UIntImm::make(UInt(8), dtype.code()) &&
                   TVMArrayGet(UInt(8), v_arg, intrinsic::kArrTypeBits) ==
                   UIntImm::make(UInt(8), dtype.bits()) &&
                   TVMArrayGet(UInt(16), v_arg, intrinsic::kArrTypeLanes) ==
                   UIntImm::make(UInt(16), dtype.lanes()));
      seq_init.emplace_back(AssertStmt::make(cond, type_err_msg.str()));
      // Data Field
      if (f_push(buf->data, TVMArrayGet(Handle(), v_arg, intrinsic::kArrData),
                 v_arg->name_hint + ".data")) {
        Var vptr(buf->data);
        handle_data_type.Set(vptr, make_const(buf->dtype, 0));
      }
      // shape field
      Var v_shape(v_arg->name_hint + ".shape", Handle());
      handle_data_type.Set(v_shape, make_const(tvm_shape_type, 0));
      seq_init.emplace_back(LetStmt::make(
          v_shape, TVMArrayGet(Handle(), v_arg, intrinsic::kArrShape), nop));
      for (size_t k = 0; k < buf->shape.size(); ++k) {
        std::ostringstream field_name;
        field_name << v_shape->name_hint << '[' << k << ']';
        f_push(buf->shape[k],
               cast(buf->shape[k].type(),
                    Load::make(tvm_shape_type, v_shape,
                               IntImm::make(Int(32), k), const_true(1))),
               field_name.str());
      }
      // strides field
      Var v_strides(v_arg->name_hint + ".strides", Handle());
      handle_data_type.Set(v_strides, make_const(tvm_shape_type, 0));
      seq_init.emplace_back(LetStmt::make(
          v_strides, TVMArrayGet(Handle(), v_arg, intrinsic::kArrStrides),
          nop));
      if (buf->strides.size() == 0) {
        std::ostringstream stride_err_msg;
        stride_err_msg << "arg_" << i << ".strides:"
                       << " expected to be nullptr for contiguous array";
        seq_init.emplace_back(AssertNull(v_strides, stride_err_msg.str()));
      } else {
        for (size_t k = 0; k < buf->strides.size(); ++k) {
          std::ostringstream field_name;
          field_name << v_strides->name_hint << '[' << k << ']';
          f_push(buf->strides[k],
                 cast(buf->shape[k].type(),
                      Load::make(tvm_shape_type, v_strides,
                                 IntImm::make(Int(32), k), const_true(1))),
                 field_name.str());
        }
      }
      // Byte_offset field.
      f_push(buf->byte_offset,
             TVMArrayGet(UInt(64), v_arg, intrinsic::kArrByteOffset),
             v_arg->name_hint + ".byte_offset");
      // device info.
      f_push(device_id,
             TVMArrayGet(Int(32), v_arg, intrinsic::kArrDeviceId),
             v_arg->name_hint + ".device_id");
      f_push(device_type,
             TVMArrayGet(Int(32), v_arg, intrinsic::kArrDeviceType),
             v_arg->name_hint + ".device_type");
    }
  }

  std::shared_ptr<LoweredFuncNode> n = std::make_shared<LoweredFuncNode>();
  n->name = name;
  n->args = args;
  n->handle_data_type = handle_data_type;
  n->is_packed_func = num_unpacked_args == 0;

  // Set device context
  if (visited.count(device_id.get())) {
    Expr node = StringImm::make("default");
    CHECK(visited.count(device_type.get()));
    seq_init.push_back(AttrStmt::make(
        node, attr::device_context_id, device_id, nop));
    seq_init.push_back(AttrStmt::make(
        node, attr::device_context_type, device_type, nop));
    Stmt set_device = IfThenElse::make(
        device_type != kCPU, Evaluate::make(Call::make(
            Int(32), intrinsic::tvm_call_packed,
            {StringImm::make(runtime::symbol::tvm_set_device),
             device_type, device_id}, Call::Intrinsic)));
    body = Block::make(set_device, body);
  }
  n->body = MergeNest({seq_init, seq_check}, body);
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
}  // namespace ir
}  // namespace tvm
