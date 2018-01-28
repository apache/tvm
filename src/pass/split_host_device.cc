/*!
 *  Copyright (c) 2017 by Contributors
 * \file split_host_device.cc
 * \brief Split device function from host.
 */
#include <tvm/ir.h>
#include <tvm/lowered_func.h>
#include <tvm/channel.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/runtime/module.h>
#include <unordered_map>

namespace tvm {
namespace ir {

// use/def analysis, also delete unreferenced lets
class IRUseDefAnalysis : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv(op->node.node_);
      CHECK_NE(iv->thread_tag.length(), 0U);
      // thread_extent can appear multiple times
      // use the first appearance as def.
      if (!use_count_.count(iv->var.get())) {
        this->HandleDef(iv->var.get());
        thread_axis_.push_back(iv);
        thread_extent_.push_back(op->value);
      }

      Expr value = op->value;
      if (visit_thread_extent_) {
        value = this->Mutate(value);
      }
      Stmt body = this->Mutate(op->body);
      if (value.same_as(value) && body.same_as(body)) return s;
      return AttrStmt::make(op->node, op->attr_key, value, body);
    } else if (op->attr_key == attr::channel_write_scope ||
               op->attr_key == attr::channel_read_scope) {
      Channel ch(op->node.node_);
      if (!use_count_.count(ch->handle_var.get())) {
        this->HandleDef(ch->handle_var.get());
      }
      return IRMutator::Mutate_(op, s);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const LetStmt *op, const Stmt& s) final {
    this->HandleDef(op->var.get());
    Stmt body = this->Mutate(op->body);
    // eliminate unreferenced let
    if (use_count_.at(op->var.get()) == 0 &&
        !HasSideEffect(op->value)) {
      return body;
    } else {
      Expr value = this->Mutate(op->value);
      if (body.same_as(op->body) &&
          value.same_as(op->value)) {
        return s;
      } else {
        return LetStmt::make(op->var, value, body);
      }
    }
  }

  Stmt Mutate_(const For *op, const Stmt& s) final {
    this->HandleDef(op->loop_var.get());
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt& s) final {
    this->HandleDef(op->buffer_var.get());
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt& s) final {
    this->HandleUse(op->buffer_var);
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Let *op, const Expr& e) final {
    this->HandleDef(op->var.get());
    Expr body = this->Mutate(op->body);
    // eliminate unreferenced let
    if (use_count_.at(op->var.get()) == 0 &&
        !HasSideEffect(op->value)) {
      return body;
    } else {
      Expr value = this->Mutate(op->value);
      if (body.same_as(op->body) &&
          value.same_as(op->value)) {
        return e;
      } else {
        return Let::make(op->var, value, body);
      }
    }
  }

  Expr Mutate_(const Variable *op, const Expr& e) final {
    this->HandleUse(e);
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Load *op, const Expr& e) final {
    this->HandleUse(op->buffer_var);
    return IRMutator::Mutate_(op, e);
  }

  void HandleDef(const Variable* v) {
    CHECK(!def_count_.count(v))
        << "variable " << v->name_hint
        << " has already been defined, the Stmt is not SSA";
    CHECK(!use_count_.count(v))
        << "variable " << v->name_hint
        << " has been used before definition!";
    use_count_[v] = 0;
    def_count_[v] = 1;
  }

  void HandleUse(const Expr& v) {
    CHECK(v.as<Variable>());
    Var var(v.node_);
    auto it = use_count_.find(var.get());
    if (it != use_count_.end()) {
      if (it->second >= 0) {
        ++it->second;
      }
    } else {
      undefined_.push_back(var);
      use_count_[var.get()] = -1;
    }
  }

  // The fields are publically readible to
  // be accessible to the users.
  bool visit_thread_extent_{true};
  Array<Var> undefined_;
  Array<IterVar> thread_axis_;
  Array<Expr> thread_extent_;
  std::unordered_map<const Variable*, int> use_count_;
  std::unordered_map<const Variable*, int> def_count_;
};

class HostDeviceSplitter : public IRMutator {
 public:
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    handle_data_type_[op->buffer_var.get()] = make_const(op->type, 0);
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::pipeline_exec_scope) {
      return SplitDeviceFunc(s);
    }
    return IRMutator::Mutate_(op, s);
  }

  Array<LoweredFunc> Split(LoweredFunc f) {
    CHECK_EQ(f->func_type, kMixedFunc);
    for (auto kv : f->handle_data_type) {
      handle_data_type_[kv.first.get()] = kv.second;
    }
    name_ = f->name;
    std::shared_ptr<LoweredFuncNode> n =
        std::make_shared<LoweredFuncNode>(*f.operator->());
    n->body = this->Mutate(f->body);
    n->func_type = kHostFunc;
    Array<LoweredFunc> ret{LoweredFunc(n)};
    for (LoweredFunc x : device_funcs_) {
      ret.push_back(x);
    }
    return ret;
  }

 private:
  Stmt SplitDeviceFunc(Stmt body) {
    std::ostringstream os;
    os << name_ << "__kernel" << device_funcs_.size();
    std::shared_ptr<LoweredFuncNode> n = std::make_shared<LoweredFuncNode>();
    // isolate the device function.
    IRUseDefAnalysis m;
    m.visit_thread_extent_ = false;
    n->body = m.Mutate(body);
    n->name = os.str();
    n->func_type = kDeviceFunc;
    n->thread_axis = m.thread_axis_;
    // Strictly order the arguments: Var pointers, positional arguments.
    for (Var v : m.undefined_) {
      if (v.type().is_handle()) {
        n->args.push_back(v);
        // mark handle data type.
        auto it = handle_data_type_.find(v.get());
        if (it != handle_data_type_.end()) {
          n->handle_data_type.Set(v, it->second);
        }
      }
    }
    for (Var v : m.undefined_) {
      if (!v.type().is_handle()) {
        n->args.push_back(v);
      }
    }
    LoweredFunc f_device(n);
    Array<Expr> call_args;
    call_args.push_back(StringImm::make(f_device->name));
    for (Var arg : n->args) {
      call_args.push_back(arg);
    }
    for (Expr ext : m.thread_extent_) {
      call_args.push_back(ext);
    }
    device_funcs_.emplace_back(f_device);
    return Evaluate::make(Call::make(
        Int(32), intrinsic::tvm_call_packed,
        call_args, Call::Intrinsic));
  }

  // function name
  std::string name_;
  // the device functions
  std::vector<LoweredFunc> device_funcs_;
  std::unordered_map<const Variable*, Expr> handle_data_type_;
};


Array<Var> UndefinedVars(const Stmt& stmt, const Array<Var>& args) {
  IRUseDefAnalysis m;
  for (Var arg : args) {
    m.use_count_[arg.get()] = 0;
  }
  m.Mutate(stmt);
  return m.undefined_;
}

Array<LoweredFunc> SplitHostDevice(LoweredFunc func) {
  return HostDeviceSplitter().Split(func);
}

}  // namespace ir
}  // namespace tvm
