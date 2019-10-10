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
 *  Copyright (c) 2019 by Contributors
 * \file src/relay/backend/vm/compiler.cc
 * \brief A compiler from relay::Module to the VM byte code.
 */

#include <tvm/operation.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/logging.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm.h>
#include <topi/tags.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../../../runtime/vm/naive_allocator.h"
#include "../../backend/compile_engine.h"
#include "../../pass/pass_util.h"
#include "compiler.h"

namespace tvm {
namespace relay {

namespace transform {

Pass LambdaLift();
Pass InlinePrimitives();

}  // namespace transform

namespace vm {

using namespace tvm::runtime;
using namespace tvm::runtime::vm;
using namespace relay::transform;

// (@jroesch): VM passes, eventually declare as passes.
bool IsClosure(const Function& func);

void InstructionPrint(std::ostream& os, const Instruction& instr);

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

  AccessField(MatchValuePtr parent, size_t index)
  : parent(parent), index(index) {}

  ~AccessField() {}
};

/*!
 * \brief Condition in a decision tree
 */
struct ConditionNode {
  virtual ~ConditionNode() {}
};

using ConditionNodePtr = std::shared_ptr<ConditionNode>;

/*!
 * \brief A var binding condition
 */
struct VarBinding : ConditionNode {
  Var var;
  MatchValuePtr val;

  VarBinding(Var var, MatchValuePtr val)
          : var(var), val(val) {}

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

  TagCompare(MatchValuePtr obj, size_t target)
          : obj(obj), target_tag(target) {
  }

  ~TagCompare() {}
};

using TreeNodePtr = typename relay::TreeNode<ConditionNodePtr>::pointer;
using TreeLeafNode = relay::TreeLeafNode<ConditionNodePtr>;
using TreeLeafFatalNode = relay::TreeLeafFatalNode<ConditionNodePtr>;
using TreeBranchNode = relay::TreeBranchNode<ConditionNodePtr>;

TreeNodePtr BuildDecisionTreeFromPattern(MatchValuePtr data,
                                         Pattern pattern,
                                         TreeNodePtr then_branch,
                                         TreeNodePtr else_branch) {
  if (pattern.as<PatternWildcardNode>()) {
    // We ignore wildcard binding since it's not producing new vars
    return then_branch;
  } else if (pattern.as<PatternVarNode>()) {
    auto pat = pattern.as<PatternVarNode>();
    auto pattern = GetRef<PatternVar>(pat);
    auto cond = std::make_shared<VarBinding>(pattern->var, data);
    return TreeBranchNode::Make(cond, then_branch, else_branch);
  } else if (auto pcn = pattern.as<PatternConstructorNode>()) {
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
    auto pt = pattern.as<PatternTupleNode>();
    CHECK(pt) << "unhandled case: " << pattern;
    size_t field_index = 0;
    for (auto& p : pt->patterns) {
      auto d = std::make_shared<AccessField>(data, field_index);
      then_branch = BuildDecisionTreeFromPattern(d, p, then_branch, else_branch);
      field_index++;
    }
    return then_branch;
  }
}

TreeNodePtr BuildDecisionTreeFromClause(MatchValuePtr data,
                                        Clause clause,
                                        TreeNodePtr else_branch) {
  return BuildDecisionTreeFromPattern(data, clause->lhs,
                                      TreeLeafNode::Make(clause->rhs), else_branch);
}

TreeNodePtr BuildDecisionTreeFromClauses(MatchValuePtr data, tvm::Array<Clause> clauses) {
  // When nothing matches, the VM throws fatal error
  TreeNodePtr else_branch = TreeLeafFatalNode::Make();
  // Start from the last clause
  for (auto it = clauses.rbegin(); it != clauses.rend(); ++it) {
    else_branch = BuildDecisionTreeFromClause(data, *it, else_branch);
  }
  return else_branch;
}

class VMFunctionCompiler : ExprFunctor<void(const Expr& expr)> {
 public:
  VMFunctionCompiler(VMCompilerContext* context, TargetsMap targets, Target target_host)
      : last_register_(0),
        registers_num_(0),
        engine_(CompileEngine::Global()),
        context_(context),
        targets_(targets),
        target_host_(target_host) {}

  VMFunction Compile(const GlobalVar& var, const Function& func) {
    size_t i = 0;
    // We then assign register num to the free variables
    for (auto param : func->params) {
      auto arg_register = NewRegister();
      CHECK_EQ(i, arg_register);
      var_register_map_.insert({param, arg_register});
      params_.push_back(param->name_hint());
      ++i;
    }

    if (IsClosure(func)) {
      Function inner_func = Downcast<Function>(func->body);
      for (auto param : inner_func->params) {
        auto arg_register = NewRegister();
        CHECK_EQ(i, arg_register);
        var_register_map_.insert({param, arg_register});
        params_.push_back(param->name_hint());
        ++i;
      }
      this->VisitExpr(inner_func->body);
    } else {
      this->VisitExpr(func->body);
    }
    instructions_.push_back(Instruction::Ret(last_register_));
    return VMFunction(var->name_hint, params_, instructions_, registers_num_);
  }

 protected:
  size_t NewRegister() { return registers_num_++; }

  inline void Emit(const Instruction& instr) {
    DLOG(INFO) << "VMCompiler::Emit: instr=" << instr;
    CHECK((int)instr.op < 100) << "Invalid opcode " << (int)instr.op;
    switch (instr.op) {
      case Opcode::AllocDatatype:
      case Opcode::AllocTensor:
      case Opcode::AllocTensorReg:
      case Opcode::GetField:
      case Opcode::GetTag:
      case Opcode::LoadConst:
      case Opcode::LoadConsti:
      case Opcode::Invoke:
      case Opcode::AllocClosure:
      case Opcode::Move:
      case Opcode::InvokeClosure:
        last_register_ = instr.dst;
        break;
      case Opcode::InvokePacked:
        last_register_ = instr.packed_args[instr.arity - 1];
        break;
      case Opcode::If:
      case Opcode::Ret:
      case Opcode::Goto:
      case Opcode::Fatal:
        break;
    }
    instructions_.push_back(instr);
  }

  void VisitExpr_(const ConstantNode* const_node) {
    size_t konst_idx = context_->constants.size();
    context_->constants.push_back(const_node->data);
    Emit(Instruction::LoadConst(konst_idx, NewRegister()));
  }

  void VisitExpr_(const VarNode* var_node) {
    auto var = GetRef<Var>(var_node);
    auto reg_it = this->var_register_map_.find(var);
    CHECK(reg_it != this->var_register_map_.end());
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
    Emit(Instruction::AllocDatatype(
      0,
      tuple->fields.size(),
      fields_registers,
      NewRegister()));
  }

  void VisitExpr_(const MatchNode* match_node) {
    auto match = GetRef<Match>(match_node);

    this->VisitExpr(match->data);
    CompileMatch(match);
  }

  void VisitExpr_(const LetNode* let_node) {
    DLOG(INFO) << AsText(let_node->value);
    this->VisitExpr(let_node->value);
    var_register_map_.insert({let_node->var, this->last_register_});
    this->VisitExpr(let_node->body);
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
    CHECK(it != context_->global_map.end());
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

    size_t true_register = last_register_;
    Emit(Instruction::Goto(0));

    // Finally store how many instructions there are in the
    // true branch.
    auto after_true = this->instructions_.size();

    this->VisitExpr(if_node->false_branch);

    size_t false_register = last_register_;

    // In else-branch, override the then-branch register
    Emit(Instruction::Move(false_register, true_register));
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

    this->last_register_ = true_register;
  }

  Index EmitGetShape(const TensorTypeNode* ttype, Index reg) {
    bool const_shape = true;
    std::vector<int64_t> shape;
    for (auto dim : ttype->shape) {
      if (auto kdim = dim.as<IntImm>()) {
        shape.push_back(kdim->value);
      } else {
        const_shape = false;
      }
    }
    if (const_shape) {
      int64_t ndim = shape.size();
      DLContext cpu_ctx;
      cpu_ctx.device_type = kDLCPU;
      cpu_ctx.device_id = 0;
      NDArray shape_tensor;
      if (ndim == 0) {
        shape_tensor = NDArray::Empty({}, Type2TVMType(Int(64)), cpu_ctx);
      } else {
        shape_tensor = NDArray::Empty({ndim}, Type2TVMType(Int(64)), cpu_ctx);
        int64_t* dims = reinterpret_cast<int64_t*>(shape_tensor->data);
        for (size_t i = 0; i < shape.size(); ++i) {
          dims[i] = shape[i];
        }
      }
      size_t konst_idx = context_->constants.size();
      context_->constants.push_back(shape_tensor);
      Emit(Instruction::LoadConst(konst_idx, NewRegister()));
      return last_register_;
    }
    // For dynamic shape, we need insert shape_of op to get its shape at runtime
    auto attrs = make_node<ShapeOfAttrs>();
    attrs->dtype = Int(64);
    static const Op& op = Op::Get("shape_of");
    auto input = VarNode::make("input", GetRef<Type>(ttype));
    auto expr = CallNode::make(op, {input}, Attrs(attrs), {});
    auto func = FunctionNode::make({input}, expr, IncompleteTypeNode::make(Kind::kType), {});
    auto mod = ModuleNode::make({}, {});
    auto main_gv = GlobalVarNode::make("main");
    mod->Add(main_gv, func);
    func = mod->Lookup(main_gv);

    // shape_of op has to be run on the host target
    // TODO(@icemelon9): handle heterogeneous target, such as cuda
    auto key = CCacheKeyNode::make(func, target_host_);
    auto cfunc = engine_->Lower(key);
    auto op_index = -1;
    if (context_->seen_funcs.find(cfunc->funcs[0]) == context_->seen_funcs.end()) {
      op_index = context_->cached_funcs.size();
      context_->cached_funcs.push_back(cfunc);
      context_->seen_funcs[cfunc->funcs[0]] = op_index;
    } else {
      op_index = context_->seen_funcs[cfunc->funcs[0]];
    }
    std::vector<Index> arg_regs{reg};
    int64_t ndim = ttype->shape.size();
    if (ndim == 0) {
      Emit(Instruction::AllocTensor({}, Int(64), NewRegister()));
    } else {
      Emit(Instruction::AllocTensor({ndim}, Int(64), NewRegister()));
    }
    Index shape_reg = last_register_;
    arg_regs.push_back(shape_reg);
    Emit(Instruction::InvokePacked(op_index, 2, 1, arg_regs));
    return shape_reg;
  }

  std::vector<Index> EmitShapeFunc(const Type& ret_type, const Function& func,
                                   const std::vector<Index>& unpacked_arg_regs) {
    // Find the mapping from params to registers
    int idx = 0;
    std::vector<std::vector<Index>> param_regs;
    std::vector<std::vector<const TensorTypeNode*>> param_types;
    for (auto param : func->params) {
      auto ty = param->checked_type();
      std::vector<Index> regs;
      std::vector<const TensorTypeNode*> types;
      if (auto ttype = ty.as<TensorTypeNode>()) {
        regs.push_back(unpacked_arg_regs[idx++]);
        types.push_back(ttype);
      } else if (const auto tuple_ty = ret_type.as<TupleTypeNode>()) {
        for (size_t j = 0; j < tuple_ty->fields.size(); ++j, ++idx) {
          regs.push_back(unpacked_arg_regs[idx]);
          auto ttype = tuple_ty->fields[j].as<TensorTypeNode>();
          CHECK(ttype);
          types.push_back(ttype);
        }
      } else {
        LOG(FATAL) << "unsupported parameter type " << ty;
      }
      param_regs.push_back(regs);
      param_types.push_back(types);
    }

    // Lower shape function
    auto key = CCacheKeyNode::make(func, target_host_);
    auto cfunc = engine_->LowerShapeFunc(key);
    int op_index = -1;
    if (context_->seen_funcs.count(cfunc->funcs[0]) == 0) {
      op_index = context_->cached_funcs.size();
      context_->cached_funcs.push_back(cfunc);
      context_->seen_funcs[cfunc->funcs[0]] = op_index;
    } else {
      op_index = context_->seen_funcs[cfunc->funcs[0]];
    }

    // Prepare input and output registers
    std::vector<Index> shape_func_args;
    std::vector<Index> shape_regs;
    for (size_t i = 0; i < func->params.size(); ++i) {
      int state = cfunc->shape_func_param_states[i]->value;
      if (state & kNeedInputData) {
        for (auto reg : param_regs[i]) {
          // TODO(@icemelon9): Need to copy data here for heterogeneous exec
          shape_func_args.push_back(reg);
        }
      }
      if (state & kNeedInputShape) {
        for (size_t j = 0; j < param_regs[i].size(); ++j) {
          shape_func_args.push_back(EmitGetShape(param_types[i][j], param_regs[i][j]));
        }
      }
    }
    for (auto t : cfunc->outputs) {
      int64_t ndim = t->shape[0].as<IntImm>()->value;
      Emit(Instruction::AllocTensor({ndim}, t->dtype, NewRegister()));
      shape_func_args.push_back(last_register_);
      shape_regs.push_back(last_register_);
    }

    int arity = shape_func_args.size();
    int ret_count = shape_regs.size();
    Emit(Instruction::InvokePacked(op_index, arity, ret_count, shape_func_args));

    // Alloc return tensors given the shape regs
    std::vector<DataType> ret_dtypes;
    if (const auto* tuple_type = ret_type.as<TupleTypeNode>()) {
      for (auto field : tuple_type->fields) {
        const TensorTypeNode* tty = field.as<TensorTypeNode>();
        CHECK(tty);
        ret_dtypes.push_back(tty->dtype);
      }
    } else {
      auto tty = ret_type.as<TensorTypeNode>();
      CHECK(tty);
      ret_dtypes.push_back(tty->dtype);
    }
    std::vector<Index> ret_regs;
    for (size_t i = 0; i < shape_regs.size(); ++i) {
      Emit(Instruction::AllocTensorReg(shape_regs[i], ret_dtypes[i], NewRegister()));
      ret_regs.push_back(last_register_);
    }
    return ret_regs;
  }

  std::vector<Index> AllocReturnType(const Type& ret_type, const Function& func,
                                     const std::vector<Index>& unpacked_arg_regs) {
    auto op = func->body.as<CallNode>()->op;
    // 1. If either func param types or ret type is dynamic, we need to insert
    // shape func to perform type checking at runtime.
    // 2. We skip the shape_of function since currently Relay doesn't support
    // dynamic rank tensor.
    if (op != Op::Get("shape_of") && IsDynamic(func->checked_type())) {
      return EmitShapeFunc(ret_type, func, unpacked_arg_regs);
    }
    std::vector<Index> ret_regs;
    auto alloc_tensor = [&](const TensorTypeNode* ttype) {
      const TensorType& tensor_type = GetRef<TensorType>(ttype);
      std::vector<int64_t> shape;
      for (auto dim : tensor_type->shape) {
        shape.push_back(Downcast<tvm::Integer>(dim)->value);
      }
      Emit(Instruction::AllocTensor(shape, Type2TVMType(tensor_type->dtype), NewRegister()));
      ret_regs.push_back(last_register_);
    };
    if (const TensorTypeNode* ttype = ret_type.as<TensorTypeNode>()) {
      alloc_tensor(ttype);
    } else if (const TupleTypeNode* ttype = ret_type.as<TupleTypeNode>()) {
      for (auto field : ttype->fields) {
        alloc_tensor(field.as<TensorTypeNode>());
      }
    } else {
      LOG(FATAL) << "Unsupported return value type";
    }
    return ret_regs;
  }

  void EmitInvokePrimitive(const Function& func,
                           const std::vector<Index>& arg_registers,
                           const Type& ret_type) {
    std::vector<Index> unpacked_arg_regs;
    std::vector<Instruction> allocs;

    // Arity calculation must flatten tuples.
    size_t arity = 0;
    CHECK_EQ(func->params.size(), arg_registers.size());
    for (size_t i = 0; i < func->params.size(); i++) {
      auto ty = func->params[i]->checked_type();
      if (ty.as<TensorTypeNode>()) {
        unpacked_arg_regs.push_back(arg_registers[i]);
        arity += 1;
      } else if (auto tuple_ty = ty.as<TupleTypeNode>()) {
        for (size_t f = 0; f < tuple_ty->fields.size(); f++) {
          const auto& field = tuple_ty->fields[f];
          CHECK(field.as<TensorTypeNode>())
            << "only supports non-nested tuples currently "
            << "found " << field;
          auto dst =  NewRegister();
          Emit(Instruction::GetField(arg_registers[i], f, dst));
          unpacked_arg_regs.push_back(dst);
        }
        arity += tuple_ty->fields.size();
      } else {
        LOG(FATAL) << "unsupported parameter type " << ty;
      }
    }

    auto ret_regs = AllocReturnType(ret_type, func, unpacked_arg_regs);
    size_t return_count = ret_regs.size();
    arity += return_count;
    for (auto reg : ret_regs) {
      unpacked_arg_regs.push_back(reg);
    }

    // Next generate the invoke instruction.
    CHECK(func->IsPrimitive());
    Target target;
    if (targets_.size() == 1) {
      // homogeneous execution.
      for (auto kv : targets_) {
        target = kv.second;
      }
    } else {
      // heterogeneous execution.
      LOG(FATAL) << "Currently VM compiler doesn't support heterogeneous compilation";
    }
    auto key = CCacheKeyNode::make(func, target);
    auto cfunc = engine_->Lower(key);
    // TODO(jroesch): support lowered funcs for multiple targets
    CHECK_EQ(cfunc->funcs.size(), 1);
    auto op_index = -1;
    if (context_->seen_funcs.find(cfunc->funcs[0]) == context_->seen_funcs.end()) {
      op_index = context_->cached_funcs.size();
      context_->cached_funcs.push_back(cfunc);
      context_->seen_funcs[cfunc->funcs[0]] = op_index;
    } else {
      op_index = context_->seen_funcs[cfunc->funcs[0]];
    }

    Emit(Instruction::InvokePacked(op_index, arity, return_count, unpacked_arg_regs));

    if (return_count > 1) {
      // return value is a tuple, we need to create a tuple
      std::vector<Index> fields_registers;
      for (size_t i = arity - return_count; i < arity; ++i) {
        fields_registers.push_back(unpacked_arg_regs[i]);
      }
      Emit(Instruction::AllocDatatype(0, return_count, fields_registers, NewRegister()));
    }
  }

  void VisitExpr_(const CallNode* call_node) {
    std::vector<Index> args_registers;

    for (auto arg : call_node->args) {
      this->VisitExpr(arg);
      args_registers.push_back(last_register_);
    }

    Expr op = call_node->op;

    if (auto func_node = op.as<FunctionNode>()) {
      CHECK(func_node->IsPrimitive());
      EmitInvokePrimitive(GetRef<Function>(func_node), args_registers, call_node->checked_type());
    } else if (auto global_node = op.as<GlobalVarNode>()) {
      auto global = GetRef<GlobalVar>(global_node);
      auto it = context_->global_map.find(global);
      CHECK(it != context_->global_map.end());
      DLOG(INFO) << "VisitExpr_: generating invoke for " << global->name_hint
                      << " with func_index=" << it->second;

      auto func = context_->module->Lookup(global);
      if (IsClosure(func)) {
        auto arity = func->params.size();
        Emit(Instruction::AllocClosure(it->second, arity, args_registers, NewRegister()));
      } else {
        Emit(Instruction::Invoke(it->second, args_registers, NewRegister()));
      }
    } else if (auto constructor_node = op.as<ConstructorNode>()) {
      auto constructor = GetRef<Constructor>(constructor_node);
      Emit(Instruction::AllocDatatype(constructor->tag, call_node->args.size(), args_registers,
                                      NewRegister()));
    } else if (auto var_node = op.as<VarNode>()) {
      VisitExpr(GetRef<Var>(var_node));
      Emit(Instruction::InvokeClosure(last_register_, args_registers, NewRegister()));
    } else {
      LOG(FATAL) << "unsupported case in vm compiler: " << op;
    }
  }

  void VisitExpr_(const FunctionNode* func_node) {
    if (!func_node->IsPrimitive()) {
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

  void CompileTreeNode(TreeNodePtr tree) {
    if (std::dynamic_pointer_cast<TreeLeafNode>(tree)) {
      auto node = std::dynamic_pointer_cast<TreeLeafNode>(tree);
      VisitExpr(node->body);
    } else if (std::dynamic_pointer_cast<TreeLeafFatalNode>(tree)) {
      Emit(Instruction::Fatal());
    } else if (std::dynamic_pointer_cast<TreeBranchNode>(tree)) {
      auto node = std::dynamic_pointer_cast<TreeBranchNode>(tree);
      if (std::dynamic_pointer_cast<TagCompare>(node->cond)) {
        // For Tag compariton, generate branches
        auto cond = std::dynamic_pointer_cast<TagCompare>(node->cond);
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
        auto cond = std::dynamic_pointer_cast<VarBinding>(node->cond);
        var_register_map_[cond->var] = CompileMatchValue(cond->val);
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
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> expr_map_;
  /*! \brief Instructions in the VMFunction. */
  std::vector<Instruction> instructions_;
  /*! \brief Parameter names of the function. */
  std::vector<std::string> params_;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, RegName, NodeHash, NodeEqual> var_register_map_;
  /*! \brief Last used register number. */
  size_t last_register_;
  /*! \brief Total number of virtual registers allocated. */
  size_t registers_num_;
  /*! \brief Compiler engine to lower primitive functions. */
  CompileEngine engine_;
  /*! \brief Global shared meta data */
  VMCompilerContext* context_;
  /*! \brief Target devices. */
  TargetsMap targets_;
  /*! \brief Host target. */
  Target target_host_;
};


PackedFunc VMCompiler::GetFunction(const std::string& name,
                                   const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "compile") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.num_args, 3);
      Module mod = args[0];
      this->Compile(mod, args[1], args[2]);
    });
  } else if (name == "get_vm") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = runtime::Module(vm_);
    });
  } else if (name == "set_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Map<std::string, Constant> params = args[0];
      for (const auto& kv : params) {
        this->SetParam(kv.first, kv.second->data);
      }
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void VMCompiler::SetParam(const std::string& name, runtime::NDArray data_in) {
  params_[name] = data_in;
}

relay::Function VMCompiler::BindParamsByName(
    relay::Function func,
    const std::unordered_map<std::string, runtime::NDArray>& params) {
  std::unordered_map<std::string, relay::Var> name_dict;
  std::unordered_set<relay::Var, NodeHash, NodeEqual> repeat_var;
  for (auto arg : func->params) {
    const auto &name = arg->name_hint();
    if (name_dict.count(name)) {
      repeat_var.insert(arg);
    } else {
      name_dict[name] = arg;
    }
  }
  std::unordered_map<relay::Var, Expr, NodeHash, NodeEqual> bind_dict;
  for (auto &kv : params) {
    if (name_dict.count(kv.first) == 0) {
      continue;
    }
    auto arg = name_dict.at(kv.first);
    if (repeat_var.count(arg)) {
      LOG(FATAL) << "Multiple args in the function have name " << kv.first;
    }
    bind_dict[arg] = ConstantNode::make(kv.second);
  }
  Expr bound_expr = relay::Bind(func, bind_dict);
  Function ret = Downcast<Function>(bound_expr);
  CHECK(ret.defined())
      << "The returning type is expected to be a Relay Function."
      << "\n";
  return ret;
}


void VMCompiler::Compile(Module mod,
                         const TargetsMap& targets,
                         const tvm::Target& target_host) {
  CHECK_EQ(targets.size(), 1)
    << "Currently VM compiler doesn't support heterogeneous compilation";
  if (params_.size()) {
    auto f = BindParamsByName(mod->Lookup("main"), params_);
    auto gvar = mod->GetGlobalVar("main");
    mod->Add(gvar, f);
  }

  InitVM();
  targets_ = targets;
  target_host_ = target_host;

  // Run some optimizations first, this code should
  // be moved to pass manager.
  context_.module = OptimizeModule(mod, targets_);

  // Populate the global map.
  //
  // This maps global variables to a global index
  // in the VMFunction table.
  PopulateGlobalMap();

  // Next we get ready by allocating space for
  // the global state.
  vm_->functions.resize(context_.module->functions.size());

  for (auto named_func : context_.module->functions) {
    auto gvar = named_func.first;
    auto func = named_func.second;
    VMFunctionCompiler func_compiler(&context_, targets_, target_host_);
    auto vm_func = func_compiler.Compile(gvar, func);

    size_t func_index = context_.global_map.at(gvar);
    CHECK(func_index < vm_->functions.size());
    vm_->functions[func_index] = vm_func;
  }

#if USE_RELAY_DEBUG
  for (auto vm_func : vm_->functions) {
    DLOG(INFO) << vm_func << "-------------";
  }
#endif  // USE_RELAY_DEBUG

  // populate constants
  for (auto data : context_.constants) {
    vm_->constants.push_back(Object::Tensor(data));
  }

  LibraryCodegen();

  for (auto gv : context_.global_map) {
    vm_->global_map.insert({gv.first->name_hint, gv.second});
  }
}

Module VMCompiler::OptimizeModule(const Module& mod, const TargetsMap& targets) {
  Array<Pass> pass_seqs;
  // Run all dialect legalization passes.
  pass_seqs.push_back(relay::qnn::transform::Legalize());

  // Legalize pass is restricted to homogeneous execution for now.
  if (targets.size() == 1) {
    pass_seqs.push_back(transform::Legalize());
  }

  pass_seqs.push_back(transform::SimplifyInference());
  PackedFunc fskip = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
    Expr expr = args[0];
    if (expr.as<CallNode>()) {
      auto call_node = expr.as<CallNode>();
      auto op_node = call_node->op.as<OpNode>();
      if (op_node->name == "cast") {
        auto attrs = call_node->attrs.as<CastAttrs>();
        if (attrs->dtype == Int(32)) {
          *rv = true;
        }
      }
    }
    *rv = false;
  });
  pass_seqs.push_back(transform::EliminateCommonSubexpr(fskip));
  pass_seqs.push_back(transform::InlinePrimitives());

  pass_seqs.push_back(transform::CombineParallelConv2D(3));
  pass_seqs.push_back(transform::CombineParallelDense(3));
  pass_seqs.push_back(transform::FoldConstant());
  pass_seqs.push_back(transform::FoldScaleAxis());
  pass_seqs.push_back(transform::CanonicalizeCast());
  pass_seqs.push_back(transform::CanonicalizeOps());

  // Alter layout transformation is only applied to homogeneous execution yet.
  if (targets.size() == 1) {
    pass_seqs.push_back(transform::AlterOpLayout());
  }

  pass_seqs.push_back(transform::FoldConstant());

  pass_seqs.push_back(transform::FuseOps());
  pass_seqs.push_back(transform::ToANormalForm());
  pass_seqs.push_back(transform::LambdaLift());
  pass_seqs.push_back(transform::InlinePrimitives());

  transform::Sequential seq(pass_seqs);
  transform::PassContext pass_ctx = PassContext::Current();
  // TODO(wweic): Support heterogenous execution
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

void VMCompiler::LibraryCodegen() {
  auto const &cached_funcs = context_.cached_funcs;
  if (cached_funcs.size() == 0) {
    return;
  }
  std::unordered_map<std::string, Array<LoweredFunc>> tgt_funcs;
  for (auto &cfunc : cached_funcs) {
    std::string target_str = cfunc->target->str();
    if (tgt_funcs.count(target_str) == 0) {
      tgt_funcs.emplace(target_str, Array<LoweredFunc>{cfunc->funcs[0]});
    } else {
      tgt_funcs[target_str].push_back(cfunc->funcs[0]);
    }
  }
  Map<Target, Array<LoweredFunc>> funcs;
  for (auto &it : tgt_funcs) {
    funcs.Set(Target::Create(it.first), it.second);
  }

  if (const auto *f = runtime::Registry::Get("relay.backend.build")) {
    // The target is just a dummy arg because funcs already contains corresponding target
    // therefore target won't be used in the build function
    runtime::Module mod = (*f)(funcs, Target(), target_host_);
    CHECK(mod.operator->());
    vm_->lib = mod;
  } else {
    LOG(FATAL) << "relay.backend.build is not registered";
  }
  size_t primitive_index = 0;
  for (auto cfunc : cached_funcs) {
    vm_->primitive_map.insert({cfunc->funcs[0]->name, primitive_index++});
  }
}

runtime::Module CreateVMCompiler() {
  std::shared_ptr<VMCompiler> exec = std::make_shared<VMCompiler>();
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay._vm._VMCompiler")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CreateVMCompiler();
});

}  // namespace vm
}  // namespace relay
}  // namespace tvm
