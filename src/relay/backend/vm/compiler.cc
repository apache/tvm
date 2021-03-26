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
#include <tvm/ir/error.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/te/operation.h>

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../../../target/source/codegen_source_base.h"
#include "../../backend/compile_engine.h"
#include "../../op/op_common.h"
#include "../../transforms/pass_utils.h"
#include "../utils.h"
#include "compiler.h"

namespace tvm {
namespace relay {

namespace transform {

Pass LambdaLift();
Pass InlinePrimitives();

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
  RegName rergister_num;

  explicit RegisterValue(RegName reg) : rergister_num(reg) {}

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

/*!
 * \brief Create a default type.
 * \param device_type The device type index.
 * \return the default target for the device.
 */
Target CreateDefaultTarget(int device_type) {
  std::string name = runtime::DeviceName(device_type);
  if (name == "cpu") return Target("llvm");
  if (name == "gpu") return Target("cuda");
  return Target(name);
}

int GetFallbackDevice() {
  transform::PassContext pass_ctx = PassContext::Current();
  Optional<Integer> opt_fallback_dev =
      pass_ctx->GetConfig("relay.fallback_device_type", Integer(static_cast<int>(kDLCPU)));
  auto fallback_dev = opt_fallback_dev.value();
  ICHECK_GT(fallback_dev->value, 0U);
  return fallback_dev->value;
}

class VMFunctionCompiler : ExprFunctor<void(const Expr& expr)> {
 public:
  VMFunctionCompiler(VMCompilerContext* context, TargetsMap targets, Target target_host,
                     ExprDeviceMap expr_device_map)
      : last_register_(0),
        registers_num_(0),
        engine_(CompileEngine::Global()),
        context_(context),
        target_host_(target_host),
        expr_device_map_(std::move(expr_device_map)) {
    for (const auto& it : targets) {
      targets_[it.first->value] = it.second;
    }
  }

  VMFunction Compile(const GlobalVar& var, const Function& func) {
    size_t i = 0;
    // We then assign register num to the free variables
    for (auto param : func->params) {
      auto arg_register = NewRegister();
      ICHECK_EQ(i, arg_register);
      var_register_map_.insert({param, arg_register});
      params_.push_back(param->name_hint());
      ++i;
    }

    if (IsClosure(func)) {
      Function inner_func = Downcast<Function>(func->body);
      for (auto param : inner_func->params) {
        auto arg_register = NewRegister();
        ICHECK_EQ(i, arg_register);
        var_register_map_.insert({param, arg_register});
        params_.push_back(param->name_hint());
        ++i;
      }
      this->VisitExpr(inner_func->body);
    } else {
      this->VisitExpr(func->body);
    }
    instructions_.push_back(Instruction::Ret(last_register_));

    std::vector<Index> params_device_type;
    for (const auto& it : func->params) {
      if (!expr_device_map_.empty()) {
        ICHECK_GT(expr_device_map_.count(it), 0U);
        params_device_type.push_back(expr_device_map_[it].device_type);
      } else {
        ICHECK_EQ(targets_.size(), 1U);
        params_device_type.push_back((targets_.begin())->first);
      }
    }

    return VMFunction(var->name_hint, params_, instructions_, registers_num_, params_device_type);
  }

 protected:
  size_t NewRegister() { return registers_num_++; }

  inline void Emit(const Instruction& instr) {
    DLOG(INFO) << "VMCompiler::Emit: instr=" << instr;
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
        break;
    }
    instructions_.push_back(instr);
  }

  void VisitExpr_(const ConstantNode* const_node) {
    // Check the shape is valid
    NDArray data = const_node->data;
    size_t konst_idx = context_->constants.size();
    if (expr_device_map_.empty()) {
      context_->const_device_type.push_back(targets_.begin()->first);
    } else {
      auto con = GetRef<Constant>(const_node);
      ICHECK_GT(expr_device_map_.count(con), 0U);
      context_->const_device_type.push_back(expr_device_map_[con].device_type);
    }
    context_->constants.push_back(const_node->data);
    Emit(Instruction::LoadConst(konst_idx, NewRegister()));
  }

  void VisitExpr_(const VarNode* var_node) {
    auto var = GetRef<Var>(var_node);
    auto reg_it = this->var_register_map_.find(var);
    ICHECK(reg_it != this->var_register_map_.end());
    last_register_ = reg_it->second;
  }

  void VisitExpr_(const TupleNode* tuple_node) {
    auto tuple = GetRef<Tuple>(tuple_node);
    std::vector<Index> fields_registers;

    for (auto& field : tuple->fields) {
      this->VisitExpr(field);
      fields_registers.push_back(last_register_);
    }

    // TODO(@jroesch): use correct tag
    Emit(Instruction::AllocADT(0, tuple->fields.size(), fields_registers, NewRegister()));
  }

  void VisitExpr_(const MatchNode* match_node) {
    auto match = GetRef<Match>(match_node);

    this->VisitExpr(match->data);
    CompileMatch(match);
  }

  void VisitExpr_(const LetNode* l) final {
    Expr let_binding = GetRef<Expr>(l);
    const LetNode* let;
    while ((let = let_binding.as<LetNode>())) {
      VisitExpr(let->value);
      var_register_map_.insert({let->var, this->last_register_});
      let_binding = let->body;
    }

    VisitExpr(let_binding);
  }

  void VisitExpr_(const TupleGetItemNode* get_node) {
    auto get = GetRef<TupleGetItem>(get_node);
    this->VisitExpr(get->tuple);
    auto tuple_register = last_register_;
    Emit(Instruction::GetField(tuple_register, get->index, NewRegister()));
  }

  void VisitExpr_(const GlobalVarNode* gvar) {
    auto var = GetRef<GlobalVar>(gvar);
    auto func = context_->module->Lookup(var);
    auto it = context_->global_map.find(var);
    ICHECK(it != context_->global_map.end());
    // Allocate closure with zero free vars
    Emit(Instruction::AllocClosure(it->second, 0, {}, NewRegister()));
  }

  void VisitExpr_(const IfNode* if_node) {
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

  void EmitShapeFunc(Function func, Array<Expr> inputs, Array<Expr> outputs) {
    // Lower shape function
    CCacheKey key(func, target_host_);
    auto cfunc = engine_->LowerShapeFunc(key);
    int op_index = -1;
    // pick the only function inside the context
    ICHECK_EQ(cfunc->funcs->functions.size(), 1);
    auto pfunc = Downcast<tir::PrimFunc>((*cfunc->funcs->functions.begin()).second);
    if (context_->seen_funcs.count(pfunc) == 0) {
      op_index = context_->cached_funcs.size();
      context_->cached_funcs.push_back(cfunc);
      context_->seen_funcs[pfunc] = op_index;
    } else {
      op_index = context_->seen_funcs[pfunc];
    }

    // Prepare input and output registers
    std::vector<Index> argument_registers;
    for (auto input : inputs) {
      VisitExpr(input);
      argument_registers.push_back(last_register_);
    }

    for (auto output : outputs) {
      auto reg = var_register_map_.find(Downcast<Var>(output));
      ICHECK(reg != var_register_map_.end())
          << "internal error: all variables should be in the register mapping";
      argument_registers.push_back(reg->second);
    }

    Emit(Instruction::InvokePacked(op_index, argument_registers.size(), outputs.size(),
                                   argument_registers));
  }

  void EmitInvokeTVMOp(const Function& func, const Expr& inputs, const Expr& outputs) {
    std::vector<Index> argument_registers;

    ICHECK(func->GetAttr<Integer>(attr::kPrimitive, 0) != 0)
        << "internal error: invoke_tvm_op requires the first argument to be a relay::Function";

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
      auto reg = var_register_map_.find(Downcast<Var>(output));
      ICHECK(reg != var_register_map_.end())
          << "internal error: all variables should be in the register mapping";
      argument_registers.push_back(reg->second);
    }

    Target target;

    if (func->GetAttr<String>(attr::kCompiler).defined()) {
      target = Target("ext_dev");
    } else {
      // Next generate the invoke instruction.
      if (expr_device_map_.empty()) {
        // homogeneous execution.
        ICHECK_EQ(targets_.size(), 1U);
        const auto& it = targets_.begin();
        target = (*it).second;
      } else {
        ICHECK_GT(expr_device_map_.count(func), 0U)
            << "Found not annotated expression, please make sure "
               "context analysis has been executed";
        int dev_type = expr_device_map_[func].device_type;
        if (targets_.count(dev_type) == 0) {
          target = CreateDefaultTarget(dev_type);
        } else {
          target = targets_[expr_device_map_[func].device_type];
        }
      }
    }

    CCacheKey key(func, target);
    auto cfunc = engine_->Lower(key);

    auto op_index = -1;
    if (func->GetAttr<String>(attr::kCompiler).defined()) {
      op_index = context_->cached_funcs.size();
      context_->cached_funcs.push_back(cfunc);
    } else {
      // TODO(jroesch): support lowered funcs for multiple targets
      ICHECK_EQ(cfunc->funcs->functions.size(), 1);
      auto pfunc = Downcast<tir::PrimFunc>((*cfunc->funcs->functions.begin()).second);
      if (context_->seen_funcs.find(pfunc) == context_->seen_funcs.end()) {
        op_index = context_->cached_funcs.size();
        context_->cached_funcs.push_back(cfunc);
        context_->seen_funcs[pfunc] = op_index;
      } else {
        op_index = context_->seen_funcs[pfunc];
      }
    }

    Emit(Instruction::InvokePacked(op_index, argument_registers.size(), output_tuple->fields.size(),
                                   argument_registers));
  }

  void VisitExpr_(const CallNode* call_node) {
    Expr op = call_node->op;

    // First we handle the case in which we are using an opaque
    // operator used to define a sub-dialect, such as memory
    // allocation operations.
    if (op.as<OpNode>()) {
      OpMatch<void> matcher;
      matcher
          .Match("vm.invoke_tvm_op",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 3);
                   EmitInvokeTVMOp(Downcast<Function>(args[0]), args[1], args[2]);
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
                   // instruction.
                   auto const_shape = args[2].as<ConstantNode>();

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
                 [this, call_node](const Array<Expr>& args, const Attrs& attrs,
                                   const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 2);
                   // Compute the size of the allocation.
                   this->VisitExpr(args[0]);
                   auto size_register = last_register_;

                   ICHECK(args[1].as<ConstantNode>());
                   NDArray alignment_arr = args[1].as<ConstantNode>()->data;
                   ICHECK_EQ(alignment_arr->dtype.code, 0U)
                       << "The dtype of constant shape must be int32 or int64, but got "
                       << DLDataType2String(alignment_arr->dtype);
                   ICHECK_EQ(alignment_arr->dtype.bits, 64U);
                   Index alignment = reinterpret_cast<int64_t*>(alignment_arr->data)[0];

                   // Get the dtype hint from the attributes.
                   auto alloc_attrs = attrs.as<AllocStorageAttrs>();
                   ICHECK(alloc_attrs != nullptr) << "must be the AllocStorage attrs";
                   auto dtype = alloc_attrs->dtype;

                   Index device_type;
                   if (expr_device_map_.empty()) {
                     // TODO(zhiics) There is bug if all expressions are annotated with the device
                     // that is different the first one in the target list.
                     auto& kv = *(targets_.begin());
                     device_type = kv.first;
                   } else {
                     ICHECK_GT(expr_device_map_.count(GetRef<Call>(call_node)), 0U)
                         << " The alloc_storage node is not annotated";
                     device_type = expr_device_map_[GetRef<Call>(call_node)].device_type;
                   }

                   Emit(Instruction::AllocStorage(size_register, alignment, dtype, device_type,
                                                  NewRegister()));
                 })
          .Match("vm.shape_func",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 3);
                   auto shape_func = Downcast<Function>(args[0]);
                   auto inputs = Downcast<Tuple>(args[1]);
                   auto outputs = Downcast<Tuple>(args[2]);
                   EmitShapeFunc(shape_func, inputs->fields, outputs->fields);
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
          .Match("device_copy",
                 [this](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 1U);
                   this->VisitExpr(args[0]);
                   auto src_reg = last_register_;

                   auto device_copy_attrs = attrs.as<DeviceCopyAttrs>();
                   ICHECK(device_copy_attrs != nullptr) << "Must be the device copy attrs";
                   Index src_device_type = device_copy_attrs->src_dev_type;
                   Index dst_device_type = device_copy_attrs->dst_dev_type;
                   Emit(Instruction::DeviceCopy(src_reg, src_device_type, dst_device_type,
                                                NewRegister()));
                 })
          .Match("memory.kill",
                 [](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   LOG(FATAL) << "memory.kill is not yet supported";
                 });
      matcher(GetRef<Call>(call_node));
      return;
    }

    // In the case its not one of these specialized operators we will generate code
    // for one of the "standard" cases.
    std::vector<Index> args_registers;

    for (auto arg : call_node->args) {
      this->VisitExpr(arg);
      args_registers.push_back(last_register_);
    }

    if (auto global_node = op.as<GlobalVarNode>()) {
      // In the case we are invoking a global we need to find its
      // global ID, and then check whether it is closure invocation
      // or whether it is a standard global, and emit the correct
      // calling convention.
      auto global = GetRef<GlobalVar>(global_node);
      auto it = context_->global_map.find(global);
      ICHECK(it != context_->global_map.end());
      DLOG(INFO) << "VisitExpr_: generating invoke for " << global->name_hint
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
    } else if (auto constructor_node = op.as<ConstructorNode>()) {
      // In the constructor case, we simply need to find its tag
      // and emit a call to allocate the data structure.
      auto constructor = GetRef<Constructor>(constructor_node);
      Emit(Instruction::AllocADT(constructor->tag, call_node->args.size(), args_registers,
                                 NewRegister()));
    } else if (auto var_node = op.as<VarNode>()) {
      // If we are calling a variable, it must be the case that it is a closure so we
      // emit invoke closure here.
      VisitExpr(GetRef<Var>(var_node));
      Emit(Instruction::InvokeClosure(last_register_, args_registers, NewRegister()));
    } else if (auto inner_call_node = op.as<CallNode>()) {
      VisitExpr(GetRef<Call>(inner_call_node));
      Emit(Instruction::InvokeClosure(last_register_, args_registers, NewRegister()));
    } else {
      // Finally if there are any other cases this is a bug.
      LOG(FATAL) << "internal error: unreachable code,"
                 << "should be transformed away by previous passes"
                 << PrettyPrint(GetRef<Expr>(call_node));
    }
  }

  void VisitExpr_(const FunctionNode* func_node) {
    if (!func_node->HasNonzeroAttr(attr::kPrimitive)) {
      LOG(FATAL) << "local functions should have been removed by lambda lifting:" << std::endl
                 << "Program: " << AsText(GetRef<Function>(func_node), false) << std::endl
                 << "AST: " << GetRef<Function>(func_node);
    }
  }

  /*!
   * \brief Compile a match value
   * Generate byte code that compute the value specificed in val
   *
   * \return The register number assigned for the final value
   */
  RegName CompileMatchValue(MatchValuePtr val) {
    if (std::dynamic_pointer_cast<RegisterValue>(val)) {
      auto r = std::dynamic_pointer_cast<RegisterValue>(val);
      return r->rergister_num;
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
   * It first converts the pattern match expression into a desicision tree, the condition
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
  /*! \brief Compiler engine to lower primitive functions. */
  CompileEngine engine_;
  /*! \brief Global shared meta data */
  VMCompilerContext* context_;
  /*! \brief Target devices. */
  std::unordered_map<int, tvm::Target> targets_;
  /*! \brief Host target. */
  Target target_host_;
  /*! \brief Map from Relay expr to device type. */
  ExprDeviceMap expr_device_map_;
};

PackedFunc VMCompiler::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "lower") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 3);
      IRModule mod = args[0];
      this->Lower(mod, args[1], args[2]);
    });
  } else if (name == "codegen") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 0);
      this->Codegen();
    });
  } else if (name == "get_executable") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = runtime::Module(exec_); });
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
      ICHECK_EQ(args.num_args, 3);
      *rv = this->OptimizeModule(args[0], args[1], args[2]);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void VMCompiler::SetParam(const std::string& name, runtime::NDArray data_in) {
  params_[name] = data_in;
}

void VMCompiler::Lower(IRModule mod, const TargetsMap& targets, const tvm::Target& target_host) {
  exec_ = make_object<Executable>();
  targets_ = targets;
  target_host_ = target_host;

  // Run the optimizations necessary to target the VM.
  context_.module = OptimizeModule(mod, targets_, target_host_);

  // Populate the global map.
  //
  // This maps global variables to a global index
  // in the VMFunction table.
  PopulateGlobalMap();

  // Next we get ready by allocating space for
  // the global state.
  exec_->functions.resize(context_.module->functions.size());

  // Collect the annotated device information.
  // This indicates which device each Relay expr should be executed on.
  ExprDeviceMap expr_device_map = AnalyzeContext();

  for (auto named_func : context_.module->functions) {
    auto gvar = named_func.first;
    if (auto* n = named_func.second.as<FunctionNode>()) {
      auto func = GetRef<Function>(n);
      VMFunctionCompiler func_compiler(&context_, targets_, target_host_, expr_device_map);
      auto vm_func = func_compiler.Compile(gvar, func);

      size_t func_index = context_.global_map.at(gvar);
      ICHECK(func_index < exec_->functions.size());
      exec_->functions[func_index] = vm_func;
    }
  }

#if USE_RELAY_DEBUG
  for (auto vm_func : exec_->functions) {
    DLOG(INFO) << vm_func << "-------------";
  }
#endif  // USE_RELAY_DEBUG

  // populate constants
  for (auto data : context_.constants) {
    exec_->constants.push_back(data);
  }

  for (auto i : context_.const_device_type) {
    exec_->const_device_type.push_back(i);
  }

  // update global function map
  for (auto gv : context_.global_map) {
    exec_->global_map.insert({gv.first->name_hint, gv.second});
  }

  // update primitive function map
  size_t primitive_index = 0;
  for (const auto& cfunc : context_.cached_funcs) {
    exec_->primitive_map.insert({cfunc->func_name, primitive_index++});
  }
}

transform::Sequential MemoryOpt(tvm::Target host_target, TargetsMap targets) {
  Array<Pass> pass_seqs;
  // Manifest the allocations.
  pass_seqs.push_back(transform::ManifestAlloc(host_target, targets));

  // Compute away possibly introduced constant computation.
  pass_seqs.push_back(transform::FoldConstant());

  // Fuse the shape functions.
  pass_seqs.push_back(transform::FuseOps());

  // Manifest the allocations needed for the shape functions.
  pass_seqs.push_back(transform::ManifestAlloc(host_target, targets));

  // Fuse the shape functions.
  pass_seqs.push_back(transform::FuseOps());

  // TODO(mbrookhart, jroesch, masahi): this pass is very slow, and is
  // incomplete to provide memory resuse optimizations. Disable it until we can
  // rewrite it in C++ and complete it.
  // // Perform memory planning in order to coalesce/reduce allocations.
  // pass_seqs.push_back(transform::MemoryPlan());

  // Compute away constant computation introduced by coalescing allocations.
  pass_seqs.push_back(transform::FoldConstant());

  // Fuse the shape functions.
  pass_seqs.push_back(transform::FuseOps());

  // Create allocations for math introduced by dynamic region math.
  pass_seqs.push_back(transform::ManifestAlloc(host_target, targets));

  // Compute away possibly introduced constant computation.
  pass_seqs.push_back(transform::FoldConstant());

  // Lift constants to the top-level of the block to simplify VM code generation.
  // TODO(@icemelon9, @jroesch): Remove this pass for now because some
  //  instructions need to access to constant
  // pass_seqs.push_back(transform::LiftConstants());

  return transform::Sequential(pass_seqs);
}

IRModule VMCompiler::OptimizeModule(IRModule mod, const TargetsMap& targets,
                                    const Target& target_host) {
  if (params_.size()) {
    BaseFunc base_func = mod->Lookup("main");
    ICHECK(base_func->IsInstance<FunctionNode>())
        << "VM compiler expects to compile relay::Function";
    auto f = relay::backend::BindParamsByName(Downcast<Function>(base_func), params_);
    auto gvar = mod->GetGlobalVar("main");
    mod->Add(gvar, f);
  }

  Array<Pass> pass_seqs;
  Array<runtime::String> entry_functions{"main"};
  pass_seqs.push_back(transform::RemoveUnusedFunctions(entry_functions));
  pass_seqs.push_back(transform::ToBasicBlockNormalForm());
  // Run all dialect legalization passes.
  pass_seqs.push_back(relay::qnn::transform::Legalize());

  // Legalize pass is restricted to homogeneous execution for now.
  if (targets.size() == 1) {
    pass_seqs.push_back(transform::Legalize());
  }

  // eta expand to support constructors in argument position
  pass_seqs.push_back(transform::EtaExpand(
      /* expand_constructor */ true, /* expand_global_var */ false));

  pass_seqs.push_back(transform::SimplifyInference());
  PackedFunc fskip = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
    Expr expr = args[0];
    if (expr.as<CallNode>()) {
      auto call_node = expr.as<CallNode>();
      auto op_node = call_node->op.as<OpNode>();
      if (op_node->name == "cast") {
        auto attrs = call_node->attrs.as<CastAttrs>();
        if (attrs->dtype == DataType::Int(32)) {
          *rv = true;
        }
      }
    }
    *rv = false;
  });
  pass_seqs.push_back(transform::EliminateCommonSubexpr(fskip));
  pass_seqs.push_back(transform::SimplifyExpr());
  pass_seqs.push_back(transform::InlinePrimitives());

  pass_seqs.push_back(transform::CombineParallelConv2D(3));
  pass_seqs.push_back(transform::CombineParallelDense(3));
  pass_seqs.push_back(transform::CombineParallelBatchMatmul(3));
  pass_seqs.push_back(transform::FoldConstant());
  pass_seqs.push_back(transform::FoldScaleAxis());
  pass_seqs.push_back(transform::CanonicalizeCast());
  pass_seqs.push_back(transform::CanonicalizeOps());

  // Alter layout transformation is only applied to homogeneous execution yet.
  if (targets.size() == 1) {
    pass_seqs.push_back(transform::AlterOpLayout());
  }

  // Fast math optimizations.
  pass_seqs.push_back(transform::FastMath());
  pass_seqs.push_back(transform::FoldConstant());

  if (targets_.size() > 1) {
    // Handle heterogeneous compilation.
    int fallback_dev = GetFallbackDevice();
    pass_seqs.push_back(transform::RewriteAnnotatedOps(fallback_dev));
  }

  pass_seqs.push_back(transform::FuseOps());
  // Do layout rewrite for auto-scheduler.
  transform::PassContext pass_ctx = PassContext::Current();
  if (backend::IsAutoSchedulerEnabled() && targets.size() == 1) {
    const auto& target = (*targets.begin()).second;
    Pass major_pass = transform::AutoSchedulerLayoutRewrite();
    bool enable_layout_rewrite_targets =
        target->kind->device_type == kDLCPU || target->GetAttr<String>("device", "") == "mali";
    if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
      With<Target> tctx(target);
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
  pass_seqs.push_back(transform::InlinePrimitives());

  // Inline the functions that are lifted to the module scope. We perform this
  // pass after all other optimization passes but before the memory allocation
  // pass. This is because memory allocation pass will insert `invoke_tvm_op`
  // and we use these ops to invoke the symbols in the module generated by
  // external codegen.
  pass_seqs.push_back(transform::Inline());

  pass_seqs.push_back(MemoryOpt(target_host, targets));
  pass_seqs.push_back(transform::InferType());

  transform::Sequential seq(pass_seqs);
  tvm::With<relay::transform::PassContext> ctx(pass_ctx);
  if (targets.size() == 1) {
    const auto& it = targets.begin();
    With<Target> tctx((*it).second);
    return seq(mod);
  }
  return seq(mod);
}

void VMCompiler::PopulateGlobalMap() {
  // First we populate global map.
  size_t global_index = 0;
  for (auto named_func : context_.module->functions) {
    auto gvar = named_func.first;
    context_.global_map.insert({gvar, global_index++});
  }
}

void VMCompiler::Codegen() {
  if (!context_.module.defined()) {
    LOG(WARNING) << "Did you forget to call VMCompiler::Lower?";
    return;
  }
  auto const& cached_funcs = context_.cached_funcs;
  if (cached_funcs.size() == 0) {
    return;
  }
  std::unordered_map<std::string, IRModule> funcs;

  for (auto& cfunc : cached_funcs) {
    std::string target_str = cfunc->target->str();
    // NOTE: because module, is mutable, we need to make an
    // explicit copy of the IRModule.
    IRModule mod = cfunc->funcs;
    mod.CopyOnWrite();

    if (target_str == "ext_dev") {
      // Collect metadata in functions that are handled by external codegen.
      ICHECK(mod->ContainGlobalVar(cfunc->func_name));
      Function func = Downcast<Function>(mod->Lookup(cfunc->func_name));
      backend::UpdateConstants(func, &params_);
      continue;
    } else if (funcs.count(target_str) == 0) {
      funcs.emplace(target_str, mod);
    } else {
      funcs[target_str]->Update(mod);
    }
  }

  auto compile_engine = CompileEngine::Global();
  auto ext_mods = compile_engine->LowerExternalFunctions();
  runtime::Module lib;
  if (funcs.size() > 0) {
    Map<String, IRModule> build_funcs;
    for (const auto& i : funcs) {
      build_funcs.Set(i.first, i.second);
    }
    lib = tvm::build(build_funcs, target_host_);
  } else {
    // There is no function handled by TVM. We create a virtual main module
    // to make sure a DSO module will be also available.
    lib = codegen::CSourceModuleCreate(";", "", Array<String>{});
  }
  lib = codegen::CreateMetadataModule(params_, lib, ext_mods, target_host_);
  exec_->SetLib(lib);
}

ExprDeviceMap VMCompiler::AnalyzeContext() const {
  Device default_device;
  ExprDeviceMap expr_device_map;
  if (targets_.size() > 1) {
    int fallback_dev = GetFallbackDevice();
    default_device.device_type = static_cast<DLDeviceType>(fallback_dev);
    default_device.device_id = 0;
    expr_device_map = ContextAnalysis(context_.module, default_device);
  } else {
    const auto& tgt = targets_.begin();
    default_device.device_type = static_cast<DLDeviceType>((*tgt).first->value);
    if (default_device.device_type != kDLCPU) {
      default_device.device_id = 0;
      expr_device_map = ContextAnalysis(context_.module, default_device);
    }
  }
  return expr_device_map;
}

runtime::Module CreateVMCompiler() {
  auto exec = make_object<VMCompiler>();
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay._vm._VMCompiler").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CreateVMCompiler();
});

}  // namespace vm
}  // namespace relay
}  // namespace tvm
