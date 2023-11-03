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
 * \file src/relay/backend/vm/compiler.cc
 * \brief A compiler from relay::Module to the VM byte code.
 */

#include "compiler.h"

#include <tvm/driver/driver_api.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/parser.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/relay/runtime.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/te/operation.h>

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../../../driver/internal_driver_api.h"
#include "../../../target/metadata_module.h"
#include "../../../target/source/codegen_source_base.h"
#include "../../op/annotation/annotation.h"
#include "../../op/memory/device_copy.h"
#include "../../op/op_common.h"
#include "../../transforms/device_aware_visitors.h"
#include "../../transforms/pass_utils.h"
#include "../utils.h"
#include "./compiler.h"

namespace tvm {
namespace relay {

namespace transform {

Pass LambdaLift();
Pass LabelOps();

Pass MemoryPlan() {
  auto f = tvm::runtime::Registry::Get("relay.transform.MemoryPlan");
  ICHECK(f != nullptr) << "unable to load the memory planning pass";
  return (*f)();
}

Pass LiftConstants() {
  auto f = tvm::runtime::Registry::Get("relay.transform.LiftConstants");
  ICHECK(f != nullptr) << "unable to load the constant lifting pass";
  return (*f)();
}

}  // namespace transform

namespace vm {

using namespace tvm::runtime;
using namespace tvm::runtime::vm;
using namespace relay::transform;

/*! \brief The host device is always stored at device index 0. */
constexpr Index kHostDeviceIndex = 0;

// (@jroesch): VM passes, eventually declare as passes.
bool IsClosure(const Function& func);

// Represent a runtime object that's going to be matched by pattern match expressions
struct MatchValue {
  virtual ~MatchValue() {}
};
using MatchValuePtr = std::shared_ptr<MatchValue>;

// A runtime object that resides in a register
struct RegisterValue : MatchValue {
  // The register num
  RegName register_num;

  explicit RegisterValue(RegName reg) : register_num(reg) {}

  ~RegisterValue() {}
};

// The value is a field of another runtime object
struct AccessField : MatchValue {
  MatchValuePtr parent;
  // Field index
  size_t index;
  // Runtime register num after compiling the access field path
  RegName reg{-1};

  AccessField(MatchValuePtr parent, size_t index) : parent(parent), index(index) {}

  ~AccessField() {}
};

/*!
 * \brief Condition in a decision tree
 */
struct ConditionNode {
  virtual ~ConditionNode() {}
};

using ConditionObjectPtr = std::shared_ptr<ConditionNode>;

/*!
 * \brief A var binding condition
 */
struct VarBinding : ConditionNode {
  Var var;
  MatchValuePtr val;

  VarBinding(Var var, MatchValuePtr val) : var(var), val(val) {}

  ~VarBinding() {}
};

/*!
 * \brief Compare the tag of the object
 */
struct TagCompare : ConditionNode {
  /*! \brief The object to be examined */
  MatchValuePtr obj;

  /*! \brief The expected tag */
  int target_tag;

  TagCompare(MatchValuePtr obj, size_t target) : obj(obj), target_tag(target) {}

  ~TagCompare() {}
};

using TreeObjectPtr = typename relay::TreeNode<ConditionObjectPtr>::pointer;
using TreeLeafNode = relay::TreeLeafNode<ConditionObjectPtr>;
using TreeLeafFatalNode = relay::TreeLeafFatalNode<ConditionObjectPtr>;
using TreeBranchNode = relay::TreeBranchNode<ConditionObjectPtr>;

TreeObjectPtr BuildDecisionTreeFromPattern(MatchValuePtr data, Pattern pattern,
                                           TreeObjectPtr then_branch, TreeObjectPtr else_branch) {
  if (pattern.as<PatternWildcardNode>()) {
    // We ignore wildcard binding since it's not producing new vars
    return then_branch;
  } else if (const auto* pvn = pattern.as<PatternVarNode>()) {
    auto cond = std::make_shared<VarBinding>(pvn->var, data);
    return TreeBranchNode::Make(cond, then_branch, else_branch);
  } else if (const auto* pcn = pattern.as<PatternConstructorNode>()) {
    auto tag = pcn->constructor->tag;

    size_t field_index = 0;
    for (auto& p : pcn->patterns) {
      auto d = std::make_shared<AccessField>(data, field_index);
      then_branch = BuildDecisionTreeFromPattern(d, p, then_branch, else_branch);
      field_index++;
    }
    auto cond = std::make_shared<TagCompare>(data, tag);
    return TreeBranchNode::Make(cond, then_branch, else_branch);
  } else {
    const auto* pt = pattern.as<PatternTupleNode>();
    ICHECK(pt) << "unhandled case: " << AsText(pattern, false);
    size_t field_index = 0;
    for (auto& p : pt->patterns) {
      auto d = std::make_shared<AccessField>(data, field_index++);
      then_branch = BuildDecisionTreeFromPattern(d, p, then_branch, else_branch);
    }
    return then_branch;
  }
}

TreeObjectPtr BuildDecisionTreeFromClause(MatchValuePtr data, Clause clause,
                                          TreeObjectPtr else_branch) {
  return BuildDecisionTreeFromPattern(data, clause->lhs, TreeLeafNode::Make(clause->rhs),
                                      else_branch);
}

TreeObjectPtr BuildDecisionTreeFromClauses(MatchValuePtr data, tvm::Array<Clause> clauses) {
  // When nothing matches, the VM throws fatal error
  TreeObjectPtr else_branch = TreeLeafFatalNode::Make();
  // Start from the last clause
  for (auto it = clauses.rbegin(); it != clauses.rend(); ++it) {
    else_branch = BuildDecisionTreeFromClause(data, *it, else_branch);
  }
  return else_branch;
}

std::vector<int64_t> ToAllocTensorShape(NDArray shape) {
  std::vector<int64_t> raw_shape;
  if (shape->ndim == 0) {
    return raw_shape;
  }
  ICHECK_EQ(shape->ndim, 1u);
  ICHECK_EQ(shape->dtype.code, 0U) << "The dtype of constant shape must be int32 or int64, but got "
                                   << DLDataType2String(shape->dtype);
  ICHECK(shape->dtype.bits == 64 || shape->dtype.bits == 32)
      << "The dtype of constant shape must be int32 or int64, but got"
      << DLDataType2String(shape->dtype);

  if (shape->dtype.bits == 64) {
    int64_t* int_ptr = reinterpret_cast<int64_t*>(shape->data);
    for (auto i = 0; i < shape->shape[0]; i++) {
      raw_shape.push_back(int_ptr[i]);
    }
  } else {  // int32
    int32_t* int_ptr = reinterpret_cast<int32_t*>(shape->data);
    for (auto i = 0; i < shape->shape[0]; i++) {
      raw_shape.push_back(static_cast<int64_t>(int_ptr[i]));
    }
  }
  return raw_shape;
}

class VMFunctionCompiler : DeviceAwareExprFunctor<void(const Expr& n)> {
 public:
  VMFunctionCompiler(VMCompilerContext* context, VirtualDevice host_virtual_device)
      : DeviceAwareExprFunctor(context->module),
        last_register_(0),
        registers_num_(0),
        context_(context),
        host_virtual_device_(std::move(host_virtual_device)) {}

  VMFunction Compile(const GlobalVar& var, const Function& func) {
    VLOG(1) << "Compiling:" << std::endl << PrettyPrint(func);
    std::vector<Index> param_device_indexes;
    if (IsClosure(func)) {
      // After lifting we'll have functions of the form:
      //   fn(closure args) { fn(lifted function args) { body } }
      // But we want the closure's function to be:
      //   fn(closure args, lifter function args) { body }
      // Do that flattening on-the-fly here.
      Function inner_func = Downcast<Function>(func->body);
      std::vector<Var> params;
      params.reserve(func->params.size() + inner_func->params.size());
      param_device_indexes.reserve(func->params.size() + inner_func->params.size());
      for (size_t i = 0; i < func->params.size(); ++i) {
        params.emplace_back(func->params[i]);
        param_device_indexes.push_back(GetDeviceIndex(func->params[i]->virtual_device()));
      }
      for (size_t i = 0; i < inner_func->params.size(); ++i) {
        params.emplace_back(inner_func->params[i]);

        param_device_indexes.push_back(GetDeviceIndex(inner_func->params[i]->virtual_device()));
      }
      std::vector<TypeVar> type_params;
      type_params.reserve(func->type_params.size() + inner_func->type_params.size());
      for (const auto& tyvar : func->type_params) {
        type_params.push_back(tyvar);
      }
      for (const auto& tyvar : inner_func->type_params) {
        type_params.push_back(tyvar);
      }
      Function flattened_func = Function(params, inner_func->body, inner_func->ret_type,
                                         type_params, func->attrs, func->span);
      flattened_func->virtual_device_ = inner_func->virtual_device();
      VisitExpr(flattened_func);
    } else {
      param_device_indexes.reserve(func->params.size());
      for (size_t i = 0; i < func->params.size(); ++i) {
        param_device_indexes.push_back(GetDeviceIndex(func->params[i]->virtual_device()));
      }
      VisitExpr(func);
    }
    return VMFunction(var->name_hint, params_, instructions_, registers_num_,
                      std::move(param_device_indexes));
  }

  /*! \brief Attrs objects for each op. */
  std::map<Index, Map<String, ObjectRef>> op_attrs;

  /*! \brief Attrs objects for each callsite. */
  std::map<Index, Map<String, ObjectRef>> callsite_attrs;

 protected:
  size_t NewRegister() { return registers_num_++; }

  inline void Emit(const Instruction& instr) {
    size_t instruction_index = instructions_.size();
    VLOG(2) << "instruction[" << instruction_index << "] = " << instr;
    ICHECK((int)instr.op < 100) << "Invalid opcode " << (int)instr.op;
    switch (instr.op) {
      case Opcode::AllocADT:
      case Opcode::AllocTensor:
      case Opcode::AllocTensorReg:
      case Opcode::GetField:
      case Opcode::GetTag:
      case Opcode::LoadConst:
      case Opcode::LoadConsti:
      case Opcode::Invoke:
      case Opcode::AllocClosure:
      case Opcode::AllocStorage:
      case Opcode::ShapeOf:
      case Opcode::ReshapeTensor:
      case Opcode::Move:
      case Opcode::InvokeClosure:
      case Opcode::DeviceCopy:
        last_register_ = instr.dst;
        break;
      case Opcode::InvokePacked:
      case Opcode::If:
      case Opcode::Ret:
      case Opcode::Goto:
      case Opcode::Fatal:
      case Opcode::KillRegister:
        break;
    }
    instructions_.push_back(instr);
  }

  /*!
   * \brief Returns the "device index" to represent \p virtual_device for primitives
   * in emitted code. Note that the host device is always at index 0.
   */
  Index GetDeviceIndex(const VirtualDevice& virtual_device) {
    ICHECK(!virtual_device->IsFullyUnconstrained());
    auto itr = std::find(context_->virtual_devices_.begin(), context_->virtual_devices_.end(),
                         virtual_device);
    if (itr != context_->virtual_devices_.end()) {
      return std::distance(context_->virtual_devices_.begin(), itr);
    }

    ICHECK_GT(context_->virtual_devices_.size(), 0);
    ICHECK_NE(virtual_device, host_virtual_device_);  // the host scope is always at index 0

    if (virtual_device->device_type() == context_->virtual_devices_.front()->device_type()) {
      // It's ok if we see distinct scopes which share the host device type. This is because
      // we allow the VirtualDevice for the host to be different from the VirtualDevice for
      // primitive operations which both happen to be on the same device (typically CPU).
      return 0;
    }

    ICHECK(virtual_device != host_virtual_device_);
    Index index = context_->virtual_devices_.size();
    VLOG(2) << "virtual_device[" << index << "] = " << virtual_device;
    context_->virtual_devices_.push_back(virtual_device);

    return index;
  }

  using DeviceAwareExprFunctor<void(const Expr&)>::VisitExpr_;

  void VisitExpr_(const ConstantNode* const_node) final {
    // Check the shape is valid
    NDArray data = const_node->data;
    size_t const_index = context_->constants.size();
    auto con = GetRef<Constant>(const_node);
    Index device_index = GetDeviceIndex(GetVirtualDevice(con));
    VLOG(2) << "constant[" << const_index << "] on device[" << device_index << "]";
    context_->const_device_indexes.push_back(device_index);
    context_->constants.push_back(const_node->data);
    Emit(Instruction::LoadConst(const_index, device_index, NewRegister()));
  }

  void VisitExpr_(const VarNode* var_node) final {
    auto var = GetRef<Var>(var_node);
    auto reg_it = this->var_register_map_.find(var);
    ICHECK(reg_it != this->var_register_map_.end());
    last_register_ = reg_it->second;
  }

  void VisitExpr_(const TupleNode* tuple_node) final {
    auto tuple = GetRef<Tuple>(tuple_node);
    std::vector<Index> fields_registers;

    for (auto& field : tuple->fields) {
      this->VisitExpr(field);
      fields_registers.push_back(last_register_);
    }

    // TODO(@jroesch): use correct tag
    Emit(Instruction::AllocADT(0, tuple->fields.size(), fields_registers, NewRegister()));
  }

  void VisitExpr_(const MatchNode* match_node) final {
    auto match = GetRef<Match>(match_node);

    this->VisitExpr(match->data);
    CompileMatch(match);
  }

  void PreVisitLetBinding_(const Var& var, const Expr& value) final {
    ICHECK(!value.as<FunctionNode>())
        << "unexpected function:" << std::endl
        << PrettyPrint(value) << std::endl
        << "bound to var '" << var->name_hint() << "'. Did you set opt_level = 2?";
    VisitExpr(value);
    var_register_map_.emplace(var, this->last_register_);
  }

  void VisitExpr_(const TupleGetItemNode* get_node) final {
    auto get = GetRef<TupleGetItem>(get_node);
    this->VisitExpr(get->tuple);
    auto tuple_register = last_register_;
    Emit(Instruction::GetField(tuple_register, get->index, NewRegister()));
  }

  void VisitExpr_(const GlobalVarNode* gvar) final {
    auto var = GetRef<GlobalVar>(gvar);
    auto func = context_->module->Lookup(var);
    auto it = context_->global_map.find(var);
    ICHECK(it != context_->global_map.end()) << PrettyPrint(var);
    // Allocate closure with zero free vars
    Emit(Instruction::AllocClosure(it->second, 0, {}, NewRegister()));
  }

  void VisitExpr_(const IfNode* if_node) final {
    this->VisitExpr(if_node->cond);

    size_t test_register = last_register_;

    this->Emit(Instruction::LoadConsti(1, NewRegister()));
    auto after_cond = instructions_.size();
    auto target_register = last_register_;
    this->Emit(Instruction::If(test_register, target_register, 0, 0));
    this->VisitExpr(if_node->true_branch);

    // It saves the result of If-Else expression.
    auto merge_register = NewRegister();
    Emit(Instruction::Move(last_register_, merge_register));
    Emit(Instruction::Goto(0));

    // Finally store how many instructions there are in the
    // true branch.
    auto after_true = this->instructions_.size();

    this->VisitExpr(if_node->false_branch);

    size_t false_register = last_register_;

    // In else-branch, override the then-branch register
    Emit(Instruction::Move(false_register, merge_register));
    // Compute the total number of instructions
    // after generating false.
    auto after_false = this->instructions_.size();

    // Now we will compute the jump targets in order
    // to properly patch the instruction with the
    // the requiste targets.

    // After we emit the true body, and false body,
    // we patch up the if instruction, and goto.
    auto true_offset = 1;
    auto false_offset = after_true - after_cond;
    instructions_[after_cond].if_op.true_offset = true_offset;
    instructions_[after_cond].if_op.false_offset = false_offset;

    // Patch the Goto.
    this->instructions_[after_true - 1].pc_offset = (after_false - after_true) + 1;

    this->last_register_ = merge_register;
  }

  void EmitInvokeTVMOp(const Expr& func, const Expr& inputs, const Expr& outputs,
                       const DictAttrs& attrs) {
    std::vector<Index> argument_registers;

    const auto* global_var_node = func.as<GlobalVarNode>();
    ICHECK(global_var_node) << "Expecting function in invoke_tvm_op to be a global";

    auto input_tuple = inputs.as<TupleNode>();
    ICHECK(input_tuple) << "internal error: invoke_tvm_op inputs must be a tuple,"
                        << "please file a bug in the memory manifestation pass";

    auto output_tuple = outputs.as<TupleNode>();
    ICHECK(output_tuple) << "internal error: invoke_tvm_op outputs must be a tuple,"
                         << "please file a bug in the memory manifestation pass";

    for (auto input : input_tuple->fields) {
      VisitExpr(input);
      argument_registers.push_back(last_register_);
    }

    for (auto output : output_tuple->fields) {
      ICHECK(output->IsInstance<VarNode>()) << "output should be var, found:" << std::endl
                                            << PrettyPrint(output);
      auto reg = var_register_map_.find(Downcast<Var>(output));
      ICHECK(reg != var_register_map_.end())
          << "internal error: all variables should be in the register mapping";
      argument_registers.push_back(reg->second);
    }

    Index op_index;
    auto itr = context_->primitive_map.find(global_var_node->name_hint);
    if (itr == context_->primitive_map.end()) {
      op_index = context_->primitive_map.size();
      context_->primitive_map.emplace(global_var_node->name_hint, op_index);
    } else {
      op_index = itr->second;
    }

    if (attrs.defined() && attrs->dict.defined()) {
      // Capture the dictionary of attributes from the original primitive function so that they
      // can contribute to the hash of the compiled primitive. This way we can distinguish
      // primitives with the same body expression but different attributes which may arbitrarily
      // influence code generation.
      op_attrs[op_index] = attrs->dict;
    }

    Emit(Instruction::InvokePacked(op_index, argument_registers.size(), output_tuple->fields.size(),
                                   argument_registers));
  }

  void DeviceAwareVisitExpr_(const CallNode* call_node) final {
    DeviceCopyProps device_copy_props = GetDeviceCopyProps(call_node);
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);
    ICHECK(!call_lowered_props.lowered_func.defined());
    if (device_copy_props.body.defined()) {
      // TODO(mbs): device_copy cleanup.
      VisitExpr(device_copy_props.body);
      RegName src_reg = last_register_;
      Index src_index = GetDeviceIndex(device_copy_props.src_virtual_device);
      Index dst_index = GetDeviceIndex(device_copy_props.dst_virtual_device);
      // Since scopes distinguish by targets (including any target hosts) but at runtime we
      // deal only with devices, the copy may be unnecessary.
      if (src_index != dst_index) {
        Emit(Instruction::DeviceCopy(src_reg, src_index, dst_index, NewRegister()));
      }
      return;
    }

    // Now we handle the case in which we are using an opaque operator used to define a
    // sub-dialect, such as memory allocation operations.
    if (call_node->op.as<OpNode>()) {
      OpMatch<void> matcher;
      matcher
          .Match("vm.invoke_tvm_op",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 3);
                   EmitInvokeTVMOp(args[0], args[1], args[2], Downcast<DictAttrs>(attrs));
                 })
          .Match("memory.alloc_tensor",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 3);

                   // Get the attributes.
                   auto alloc_attrs = attrs.as<AllocTensorAttrs>();
                   ICHECK(alloc_attrs != nullptr) << "must be the alloc tensor attrs";
                   auto dtype = alloc_attrs->dtype;

                   // The storage will be passed dynamically.
                   this->VisitExpr(args[0]);
                   auto storage_register = last_register_;

                   // The storage will be passed dynamically.
                   this->VisitExpr(args[1]);
                   auto offset_register = last_register_;

                   // If the shape is constant then we will emit a static tensor allocation
                   // instruction. It may be wrapped by an on_device, but it will be on the host
                   // which is assumed by the alloc_tensor instruction anyway.
                   auto const_shape = AsIgnoringOnDevice<ConstantNode>(args[2]);

                   if (const_shape) {
                     NDArray shape = const_shape->data;
                     // TODO(@jroesch): we need to get an RFC done to standarize shape dtype
                     std::vector<int64_t> raw_shape = ToAllocTensorShape(shape);
                     // Add context field.
                     Emit(Instruction::AllocTensor(storage_register, offset_register, raw_shape,
                                                   dtype, NewRegister()));
                   } else {
                     this->VisitExpr(args[2]);
                     auto shape_register = last_register_;
                     Emit(Instruction::AllocTensorReg(storage_register, offset_register,
                                                      shape_register, dtype, NewRegister()));
                   }
                 })
          .Match("memory.alloc_storage",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 3);
                   // Compute the size of the allocation.
                   this->VisitExpr(args[0]);
                   auto size_register = last_register_;

                   auto const_shape = AsIgnoringOnDevice<ConstantNode>(args[1]);
                   std::vector<int64_t> raw_shape;
                   if (const_shape) {
                     NDArray shape = const_shape->data;
                     // TODO(@jroesch): we need to get an RFC done to standarize shape dtype
                     raw_shape = ToAllocTensorShape(shape);
                   }

                   ICHECK(args[2].as<ConstantNode>());  // Always a literal.
                   NDArray alignment_arr = args[2].as<ConstantNode>()->data;
                   ICHECK_EQ(alignment_arr->dtype.code, 0U)
                       << "The dtype of constant shape must be int32 or int64, but got "
                       << DLDataType2String(alignment_arr->dtype);
                   ICHECK_EQ(alignment_arr->dtype.bits, 64U);
                   Index alignment = reinterpret_cast<int64_t*>(alignment_arr->data)[0];

                   // Get the dtype hint from the attributes.
                   auto alloc_attrs = attrs.as<AllocStorageAttrs>();
                   ICHECK(alloc_attrs != nullptr) << "must be the AllocStorage attrs";
                   auto dtype = alloc_attrs->dtype;

                   Emit(Instruction::AllocStorage(size_register, alignment, dtype,
                                                  GetDeviceIndex(alloc_attrs->virtual_device),
                                                  raw_shape, NewRegister()));
                 })
          .Match("vm.shape_of",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 1U);
                   // Get the attributes.
                   const auto* shape_of_attrs = attrs.as<ShapeOfAttrs>();
                   ICHECK(shape_of_attrs) << "Must be the shape_of attrs";
                   ICHECK_EQ(shape_of_attrs->dtype.bits(), 64)
                       << "The dtype of shape of must be int64, but got"
                       << DLDataType2String(shape_of_attrs->dtype);
                   this->VisitExpr(args[0]);
                   Emit(Instruction::ShapeOf(last_register_, NewRegister()));
                 })
          .Match("vm.reshape_tensor",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 2u);
                   this->VisitExpr(args[0]);
                   auto tensor_reg = last_register_;
                   this->VisitExpr(args[1]);
                   auto shape_reg = last_register_;
                   Emit(Instruction::ReshapeTensor(tensor_reg, shape_reg, NewRegister()));
                 })
          .Match("memory.kill",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 1u);
                   this->VisitExpr(args[0]);
                   Emit(Instruction::KillRegister(this->last_register_));
                 });
      matcher(GetRef<Call>(call_node));
      return;
    }

    // In the case it's not one of these specialized operators we will generate code
    // for one of the "standard" cases.
    std::vector<Index> args_registers;

    // Evaluate the call arguments.
    for (auto arg : call_node->args) {
      VisitExpr(arg);
      args_registers.push_back(last_register_);
    }

    if (const auto* global_var_node = call_node->op.as<GlobalVarNode>()) {
      // In the case we are invoking a global we need to find its
      // global ID, and then check whether it is closure invocation
      // or whether it is a standard global, and emit the correct
      // calling convention.
      auto global = GetRef<GlobalVar>(global_var_node);
      auto it = context_->global_map.find(global);
      ICHECK(it != context_->global_map.end()) << PrettyPrint(global);
      VLOG(2) << "VisitExpr_: generating invoke for " << global->name_hint
              << " with func_index=" << it->second;

      // TODO(tvm-team):
      // Think about mixed call into global that is not a relay::Function
      // perhaps establish as an invariance(all functions in mod must be relay::Function)
      auto func = Downcast<Function>(context_->module->Lookup(global));

      if (IsClosure(func)) {
        auto arity = func->params.size();
        Emit(Instruction::AllocClosure(it->second, arity, args_registers, NewRegister()));
      } else {
        Emit(Instruction::Invoke(it->second, args_registers, NewRegister()));
      }
    } else if (const auto* constructor_node = call_node->op.as<ConstructorNode>()) {
      // In the constructor case, we simply need to find its tag
      // and emit a call to allocate the data structure.
      auto constructor = GetRef<Constructor>(constructor_node);
      Emit(Instruction::AllocADT(constructor->tag, call_node->args.size(), args_registers,
                                 NewRegister()));
    } else if (auto var = call_node->op.as<Var>()) {
      // If we are calling a variable, it must be the case that it is a closure so we
      // emit invoke closure here.
      VisitExpr(var.value());
      Emit(Instruction::InvokeClosure(last_register_, args_registers, NewRegister()));
    } else if (auto inner_call = call_node->op.as<Call>()) {
      VisitExpr(inner_call.value());
      Emit(Instruction::InvokeClosure(last_register_, args_registers, NewRegister()));
    } else {
      // Finally if there are any other cases this is a bug.
      LOG(FATAL) << "internal error: unreachable code,"
                 << "should be transformed away by previous passes:" << std::endl
                 << PrettyPrint(GetRef<Expr>(call_node));
    }
  }

  void DeviceAwareVisitExpr_(const FunctionNode* func_node) final {
    if (function_nesting() > 1) {
      ICHECK(func_node->HasNonzeroAttr(attr::kPrimitive))
          << "local functions should have been removed by lambda lifting:" << std::endl
          << "Program: " << AsText(GetRef<Function>(func_node), false) << std::endl
          << "AST: " << GetRef<Function>(func_node);
      return;
    }

    // We're processing a top-level function which has possibly been rejigged to capture
    // both closure and function arguments. Those functions retain their 'Closure' attribute,
    // but we can just process them like any other function here.

    // Assign a register num to each parameter.
    size_t i = 0;
    for (auto param : func_node->params) {
      auto arg_register = NewRegister();
      ICHECK_EQ(i, arg_register);
      var_register_map_.insert({param, arg_register});
      params_.push_back(param->name_hint());
      ++i;
    }

    VisitExpr(func_node->body);

    instructions_.push_back(Instruction::Ret(last_register_));
  }

  /*!
   * \brief Compile a match value
   * Generate byte code that compute the value specified in val
   *
   * \return The register number assigned for the final value
   */
  RegName CompileMatchValue(MatchValuePtr val) {
    if (std::dynamic_pointer_cast<RegisterValue>(val)) {
      auto r = std::dynamic_pointer_cast<RegisterValue>(val);
      return r->register_num;
    } else {
      auto path = std::dynamic_pointer_cast<AccessField>(val);
      auto p = CompileMatchValue(path->parent);
      Emit(Instruction::GetField(p, path->index, NewRegister()));
      path->reg = last_register_;
      return path->reg;
    }
  }

  void CompileTreeNode(TreeObjectPtr tree) {
    if (auto node = std::dynamic_pointer_cast<TreeLeafNode>(tree)) {
      VisitExpr(node->body);
    } else if (std::dynamic_pointer_cast<TreeLeafFatalNode>(tree)) {
      Emit(Instruction::Fatal());
    } else if (auto node = std::dynamic_pointer_cast<TreeBranchNode>(tree)) {
      if (auto cond = std::dynamic_pointer_cast<TagCompare>(node->cond)) {
        // For Tag compariton, generate branches
        auto r = CompileMatchValue(cond->obj);
        Emit(Instruction::GetTag(r, NewRegister()));
        auto operand1 = last_register_;
        Emit(Instruction::LoadConsti(cond->target_tag, NewRegister()));
        auto operand2 = last_register_;

        Emit(Instruction::If(operand1, operand2, 1, 0));
        auto cond_offset = instructions_.size() - 1;
        CompileTreeNode(node->then_branch);
        auto if_reg = last_register_;
        Emit(Instruction::Goto(1));
        auto goto_offset = instructions_.size() - 1;
        CompileTreeNode(node->else_branch);
        auto else_reg = last_register_;
        Emit(Instruction::Move(else_reg, if_reg));
        last_register_ = if_reg;
        auto else_offset = instructions_.size() - 1;
        // Fixing offsets
        instructions_[cond_offset].if_op.false_offset = goto_offset - cond_offset + 1;
        instructions_[goto_offset].pc_offset = else_offset - goto_offset + 1;
      } else {
        // For other non-branch conditions, move to then_branch directly
        auto var_bind = std::dynamic_pointer_cast<VarBinding>(node->cond);
        var_register_map_[var_bind->var] = CompileMatchValue(var_bind->val);
        CompileTreeNode(node->then_branch);
      }
    }
  }

  /*!
   * \brief Compile a pattern match expression
   * It first converts the pattern match expression into a decision tree, the condition
   * could be object comparison or variable binding. If any of the condition fails in a clause,
   * the decision tree switches to check the conditions of next clause and so on. If no clause
   * matches the value, a fatal node is inserted.
   *
   * After the decision tree is built, we convert it into bytecodes using If/Goto.
   */
  void CompileMatch(Match match) {
    auto data = std::make_shared<RegisterValue>(last_register_);
    auto decision_tree = BuildDecisionTreeFromClauses(data, match->clauses);
    CompileTreeNode(decision_tree);
  }

 protected:
  /*! \brief Store the expression a variable points to. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> expr_map_;
  /*! \brief Instructions in the VMFunction. */
  std::vector<Instruction> instructions_;
  /*! \brief Parameter names of the function. */
  std::vector<std::string> params_;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, RegName, ObjectPtrHash, ObjectPtrEqual> var_register_map_;
  /*! \brief Last used register number. */
  size_t last_register_;
  /*! \brief Total number of virtual registers allocated. */
  size_t registers_num_;
  /*! \brief Global shared meta data */
  VMCompilerContext* context_;
  /*! \brief VirtualDevice for data and computation which must reside on a CPU. */
  VirtualDevice host_virtual_device_;
};

PackedFunc VMCompiler::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "lower") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 2);
      this->Lower(args[0], args[1]);
    });
  } else if (name == "codegen") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 0);
      this->Codegen();
    });
  } else if (name == "get_executable") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 0);
      *rv = this->GetExecutable();
    });
  } else if (name == "set_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Map<String, Constant> params = args[0];
      for (const auto& kv : params) {
        this->SetParam(kv.first, kv.second->data);
      }
    });
  } else if (name == "get_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Map<String, Constant> ret;
      for (const auto& kv : params_) {
        ret.Set(kv.first, Constant(kv.second));
      }
      *rv = ret;
    });
  } else if (name == "optimize") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 2);
      *rv = this->OptimizeModule(args[0], args[1]);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
  }
}

void VMCompiler::SetParam(const std::string& name, runtime::NDArray data_in) {
  params_[name] = data_in;
}

void VMCompiler::Lower(IRModule mod, const Array<Target>& raw_targets) {
  VLOG_CONTEXT << "VM Lower";
  Setup(raw_targets);
  LowerImpl(std::move(mod));
}

IRModule VMCompiler::OptimizeModule(IRModule mod, const Array<Target>& raw_targets) {
  VLOG_CONTEXT << "VM Optimize";
  Setup(raw_targets);
  return OptimizeModuleImpl(std::move(mod));
}

runtime::Module VMCompiler::GetExecutable() const {
  if (exec_ == nullptr) {
    LOG(WARNING) << "No executable to return. Did you forget to call VMCompiler::Lower?";
  }
  if (exec_->imports().empty()) {
    LOG(WARNING) << "Executable is empty. Did you forget to call VMCompiler::Codegen?";
  }
  return runtime::Module(exec_);
}

void VMCompiler::Setup(const Array<Target>& raw_targets) {
  ICHECK(exec_ == nullptr) << "Can't reuse VMComplier object for multiple modules";
  exec_ = make_object<Executable>();
  ICHECK(!config_.defined());
  config_ = CompilationConfig(PassContext::Current(), raw_targets);
  VLOG(1) << "Using compilation config:" << std::endl << config_;

  // The first device is always for the host.
  CHECK(context_.virtual_devices_.empty());
  VLOG(1) << "virtual_device[0] = " << config_->host_virtual_device << " (host)";
  context_.virtual_devices_.push_back(config_->host_virtual_device);
}

void VMCompiler::LowerImpl(IRModule mod) {
  // Run the optimizations necessary to target the VM.
  context_.module = OptimizeModuleImpl(std::move(mod));

  // Build the map from global variables bound to Functions to a global index in the
  // VMFunction table.
  size_t num_functions = PopulateGlobalMap();

  // Next we get ready by allocating space for
  // the global state.
  exec_->functions.resize(num_functions);

  for (const auto& pair : context_.module->functions) {
    auto gvar = pair.first;
    if (auto opt = pair.second.as<Function>()) {
      auto func = opt.value();
      if (func->HasNonzeroAttr(attr::kExtern)) {
        // Already compiled during lowering.
        continue;
      }

      VMFunctionCompiler func_compiler(&context_, config_->host_virtual_device);
      auto vm_func = func_compiler.Compile(gvar, func);

      size_t func_index = context_.global_map.at(gvar);
      ICHECK(func_index < exec_->functions.size());
      exec_->functions[func_index] = vm_func;

      // update structural hashes for tvm ops
      for (auto p : func_compiler.op_attrs) {
        exec_->op_attrs.insert(p);
      }
    }
  }

  // Populate virtual devices and the host device index.
  for (const auto& virtual_device : context_.virtual_devices_) {
    ICHECK(!virtual_device->IsFullyUnconstrained());
    ICHECK_GT(virtual_device->device_type(), 0);
    exec_->virtual_devices.push_back(
        std::make_pair(Device{/*device_type=*/virtual_device->device_type(),
                              /*device_id=*/virtual_device->virtual_device_id},
                       virtual_device->memory_scope));
  }
  exec_->host_device_index = kHostDeviceIndex;

  // populate constants
  for (const auto& data : context_.constants) {
    exec_->constants.push_back(data);
  }

  for (auto index : context_.const_device_indexes) {
    exec_->const_device_indexes.push_back(index);
  }

  // update global function map
  for (const auto& gv : context_.global_map) {
    exec_->global_map.insert({gv.first->name_hint, gv.second});
  }

  // update primitive function map
  for (const auto& pair : context_.primitive_map) {
    exec_->primitive_map.insert(pair);
  }

  VLOG(1) << "Compiled to:" << std::endl
          << "-------------------------------------------------" << std::endl
          << exec_->GetVirtualDevices()  //
          << exec_->GetConstants()       //
          << exec_->GetPrimitives()      //
          << exec_->GetBytecode()        //
          << "-------------------------------------------------";

  if (backend::IsAutoSchedulerEnabled()) {
    backend::UpdateAutoSchedulerOpWeights(context_.module);
  }
}

transform::Sequential VMCompiler::MemoryOpt(const CompilationConfig& config) {
  Array<Pass> pass_seqs;
  // Remove unused functions
  Array<runtime::String> entry_functions{"main"};
  pass_seqs.push_back(transform::RemoveUnusedFunctions(entry_functions));
  // Manifest the allocations.
  pass_seqs.push_back(transform::ManifestAlloc(config->host_virtual_device));

  // Compute away possibly introduced constant computation.
  pass_seqs.push_back(transform::FoldConstant());

  // Fuse & lower any new shape functions and device_copies.
  pass_seqs.push_back(FuseAndLowerOperators(config));

  // Manifest the allocations needed for the shape functions.
  pass_seqs.push_back(transform::ManifestAlloc(config->host_virtual_device));

  // Fuse & lower any new allocations.
  pass_seqs.push_back(FuseAndLowerOperators(config));

  // TODO(mbrookhart, jroesch, masahi): this pass is very slow, and is
  // incomplete to provide memory resuse optimizations. Disable it until we can
  // rewrite it in C++ and complete it.
  // // Perform memory planning in order to coalesce/reduce allocations.
  // pass_seqs.push_back(transform::MemoryPlan());

  // Compute away constant computation introduced by coalescing allocations.
  pass_seqs.push_back(transform::FoldConstant());

  // Fuse & lower yet again
  pass_seqs.push_back(FuseAndLowerOperators(config));

  // Create allocations for math introduced by dynamic region math.
  pass_seqs.push_back(transform::ManifestAlloc(config->host_virtual_device));

  // Compute away possibly introduced constant computation.
  pass_seqs.push_back(transform::FoldConstant());

  // Insert kills to free memory.
  pass_seqs.push_back(transform::ManifestLifetimes());

  // Lift constants to the top-level of the block to simplify VM code generation.
  // TODO(@icemelon9, @jroesch): Remove this pass for now because some
  //  instructions need to access to constant
  // pass_seqs.push_back(transform::LiftConstants());

  return transform::Sequential(std::move(pass_seqs));
}

transform::Sequential VMCompiler::FuseAndLowerOperators(const CompilationConfig& config) {
  Array<Pass> pass_seqs;
  // Hoist operators to "primitive" Functions.
  pass_seqs.push_back(FuseOps());
  // Give each "primitive" Function a hash.
  pass_seqs.push_back(LabelOps());
  // Lower "primitive" Functions to PrimFuncs and rewrite calls.
  pass_seqs.push_back(tec::LowerTE(/*module_name=*/"vm_mod", config, [this](const BaseFunc& func) {
    if (func->GetAttr<String>(attr::kCompiler).defined()) {
      backend::UpdateConstants(func, &params_);
    }
  }));
  // Since lowered functions are bound in the IRModule, we can now eliminate any unused
  // let-bound functions.
  pass_seqs.push_back(DeadCodeElimination(/*inline_once=*/false));
  return transform::Sequential(std::move(pass_seqs));
}

IRModule VMCompiler::OptimizeModuleImpl(IRModule mod) {
  backend::BindParamsInModule(mod, params_);
  Array<Pass> pass_seqs = relay::backend::GetPassPrefix(
      /*is_homogeneous=*/config_->optional_homogeneous_target.defined(), /*is_vm=*/true);

  // Always plan devices so the remaining passes don't need to distinguish homogeneous vs
  // heterogeneous execution.
  pass_seqs.push_back(transform::PlanDevices(config_));
  if (config_->optional_homogeneous_target.defined()) {
    // This pass currently only supports the homogeneous case.
    pass_seqs.push_back(transform::SplitArgs(
        config_->optional_homogeneous_target->GetAttr<Integer>("max_function_args", 0)
            .value()
            .IntValue()));
  }

  pass_seqs.push_back(transform::FuseOps());
  pass_seqs.push_back(transform::AnnotateMemoryScope());

  // Do layout rewrite for auto-scheduler.
  transform::PassContext pass_ctx = PassContext::Current();
  if (backend::IsAutoSchedulerEnabled() && config_->optional_homogeneous_target.defined()) {
    Pass major_pass = transform::AutoSchedulerLayoutRewrite();
    bool enable_layout_rewrite_targets =
        config_->optional_homogeneous_target->GetTargetDeviceType() == kDLCPU ||
        config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
    if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
      With<Target> tctx(config_->optional_homogeneous_target);
      pass_seqs.push_back(major_pass);
      // Defuse ops to fold constants, then fuse them again
      pass_seqs.push_back(transform::DefuseOps());
      pass_seqs.push_back(transform::FoldConstant());
      pass_seqs.push_back(transform::FuseOps());
    }
  }
  if (backend::IsMetaScheduleEnabled() && config_->optional_homogeneous_target.defined()) {
    Pass major_pass = transform::MetaScheduleLayoutRewrite();
    bool enable_layout_rewrite_targets =
        config_->optional_homogeneous_target->GetTargetDeviceType() == kDLCPU ||
        config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
    if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
      With<Target> tctx(config_->optional_homogeneous_target);
      pass_seqs.push_back(major_pass);
      // Defuse ops to fold constants, then fuse them again
      pass_seqs.push_back(transform::DefuseOps());
      pass_seqs.push_back(transform::FoldConstant());
      pass_seqs.push_back(transform::FuseOps());
    }
  }

  pass_seqs.push_back(transform::ToANormalForm());
  pass_seqs.push_back(transform::InferType());
  pass_seqs.push_back(transform::LambdaLift());

  // Eliminate dead-code before we lower. We don't track the purity of PrimFuncs, thus after
  // lowering all calls to lowered functions will be kept.
  pass_seqs.push_back(DeadCodeElimination(/*inline_once=*/false));
  pass_seqs.push_back(transform::LabelOps());

  // Lower all functions annotated as "primitive" by FuseOps.
  pass_seqs.push_back(tec::LowerTE(/*module_name=*/"vm_mod", config_, [this](const BaseFunc& func) {
    if (func->GetAttr<String>(attr::kCompiler).defined()) {
      backend::UpdateConstants(func, &params_);
    }
  }));

  // Since lowered functions are bound in the IRModule, we can now eliminate any unused
  // let-bound functions.
  pass_seqs.push_back(DeadCodeElimination(/*inline_once=*/false));

  // At this point it's possible to run PlanDevices again to pick up any additional constraints
  // introduced during lowering. However we'll not do this until more testing has been done.

  // Inline the functions that are lifted to the module scope. We perform this
  // pass after all other optimization passes but before the memory allocation
  // pass. This is because memory allocation pass will insert `invoke_tvm_op`
  // and we use these ops to invoke the symbols in the module generated by
  // external codegen.
  pass_seqs.push_back(transform::Inline());

  pass_seqs.push_back(MemoryOpt(config_));
  pass_seqs.push_back(transform::InferType());

  transform::Sequential seq(pass_seqs);
  tvm::With<relay::transform::PassContext> ctx(pass_ctx);
  if (config_->optional_homogeneous_target.defined()) {
    With<Target> tctx(config_->optional_homogeneous_target);
    return seq(std::move(mod));
  } else {
    return seq(std::move(mod));
  }
}

size_t VMCompiler::PopulateGlobalMap() {
  // Allocate a VMFunction index for every Relay Function we could call.
  // Excludes PrimFuncs and externs, which are managed by the primitive_map_.
  for (const auto& kv : context_.module->functions) {
    if (const auto* function_node = kv.second.as<FunctionNode>()) {
      if (!function_node->HasNonzeroAttr(attr::kExtern)) {
        context_.global_map.emplace(kv.first, context_.global_map.size());
      }
    }
  }
  return context_.global_map.size();
}

void VMCompiler::Codegen() {
  VLOG_CONTEXT << "VM Codegen";
  if (!context_.module.defined()) {
    LOG(WARNING) << "No compiled module to codegen from. Did you forget to call VMCompiler::Lower?";
    return;
  }

  // At this point context_.module will contain only:
  //  - non-external Relay functions, which we've compiled into VMFunctions.
  //  - external Relay functions, which will have definitions within some external runtime module
  //    in the "external_mods" attribute
  //  - PrimFuncs annotated with their targets.
  // Only the PrimFuncs will appear in per_target_modules, and there may legitimately be none.
  Map<Target, IRModule> per_tvm_target_modules = tec::GetPerTargetModules(context_.module);
  for (const auto& kv : per_tvm_target_modules) {
    ICHECK(kv.first->GetTargetDeviceType() != kDLExtDev);
  }

  // Retrieve all external runtime modules accumulated by external codegen (both function-at-a-time
  // and IRModule-at-a-time).
  Array<runtime::Module> external_mods =
      context_.module->GetAttr<Array<runtime::Module>>(tvm::attr::kExternalMods).value_or({});

  // Retrieve any constant bindings accumulated by external codegen (by IRModule-at-a-time passes).
  Map<String, runtime::NDArray> const_name_to_constant =
      context_.module->GetAttr<Map<String, runtime::NDArray>>(tvm::attr::kConstNameToConstant)
          .value_or({});

  VLOG(0) << "have " << per_tvm_target_modules.size() << " targets to build, "
          << external_mods.size() << " external runtime modules, " << const_name_to_constant.size()
          << " external constants, and " << params_.size() << " local constants";

  // Any constant bindings must be merged into the overall 'params' map we've directly accumulated
  // via the TECompiler callback.
  for (const auto& kv : const_name_to_constant) {
    ICHECK_EQ(params_.count(kv.first), 0);
    params_.emplace(kv.first, kv.second);
  }

  runtime::Module lib;
  if (per_tvm_target_modules.empty()) {
    // There is no function handled by TVM. We create a virtual main module
    // to make sure a DSO module will be also available.
    LOG(INFO) << "All lowered functions have been build by BYOC -- generating an empty TVM module";
    lib = codegen::CSourceModuleCreate(";", "", Array<String>{});
  } else {
    lib = tvm::TIRToRuntime(per_tvm_target_modules, config_->host_target);
  }

  lib =
      codegen::CreateMetadataModule(params_, lib, external_mods, config_->host_target,
                                    Runtime::Create("cpp"), Executor::Create("graph"),  // DNS HACK
                                    relay::backend::ExecutorCodegenMetadata());
  exec_->SetLib(lib);
}

runtime::Module CreateVMCompiler() {
  auto exec = make_object<VMCompiler>();
  return runtime::Module(std::move(exec));
}

TVM_REGISTER_GLOBAL("relay._vm._VMCompiler").set_body_typed(CreateVMCompiler);

}  // namespace vm
}  // namespace relay
}  // namespace tvm
