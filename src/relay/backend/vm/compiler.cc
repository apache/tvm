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

#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/logging.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm.h>
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

// Compute the constant pool, i.e a mapping from Constant node to constant index.
struct ConstantPool : ExprVisitor {
  std::set<GlobalVar> visited;
  Module module;
  ConstMap const_map;
  ConstTensorShapeMap const_tensor_shape_map;

  size_t index;

  explicit ConstantPool(const Module& mod) : module(mod), const_map(), index(0) {}

  void VisitExpr_(const GlobalVarNode* var_node) {
    auto gvar = GetRef<GlobalVar>(var_node);
    if (visited.find(gvar) == visited.end()) {
      visited.insert(gvar);
      this->VisitExpr(this->module->Lookup(gvar));
    }
  }

  void VisitExpr_(const ConstantNode* const_node) {
    auto konst = GetRef<Constant>(const_node);
    auto it = this->const_map.find(konst);
    if (it == this->const_map.end()) {
      this->const_map.insert({konst, index++});
    }
  }
};

std::tuple<ConstMap, ConstTensorShapeMap> LayoutConstantPool(const Module& module) {
  auto cp = ConstantPool(module);
  for (auto& func : module->functions) {
    cp.VisitExpr(func.first);
  }
  return std::make_tuple(cp.const_map, cp.const_tensor_shape_map);
}

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
  } else {
    auto pat = pattern.as<PatternConstructorNode>();
    auto pattern = GetRef<PatternConstructor>(pat);
    auto tag = pattern->constructor->tag;

    size_t field_index = 0;
    for (auto& p : pattern->patterns) {
      auto d = std::make_shared<AccessField>(data, field_index);
      then_branch = BuildDecisionTreeFromPattern(d, p, then_branch, else_branch);
      field_index++;
    }
    auto cond = std::make_shared<TagCompare>(data, tag);
    return TreeBranchNode::Make(cond, then_branch, else_branch);
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
  VMFunctionCompiler(VMCompilerContext* context, TargetsMap targets)
      : last_register_(0),
        registers_num_(0),
        engine_(CompileEngine::Global()),
        context_(context),
        targets_(targets) {}

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
    auto rconst = GetRef<Constant>(const_node);
    auto it = this->context_->const_map.find(rconst);
    CHECK(it != this->context_->const_map.end());
    Emit(Instruction::LoadConst(it->second, NewRegister()));
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
    DLOG(INFO) << let_node->value;
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

  Instruction AllocTensorFromType(const TensorTypeNode* ttype) {
    TVMType dltype = Type2TVMType(ttype->dtype);
    auto tensor_type = GetRef<TensorType>(ttype);
    std::vector<int64_t> shape;
    for (auto dim : tensor_type->shape) {
      shape.push_back(Downcast<tvm::Integer>(dim)->value);
    }
    return Instruction::AllocTensor(shape, dltype, NewRegister());
  }

  void EmitInvokePrimitive(const Function& func,
                           const std::vector<Index>& args_registers,
                           const Type& ret_type) {
    std::vector<Index> unpacked_arg_regs;
    std::vector<Instruction> allocs;

    // Arity calculation must flatten tuples.
    size_t arity = 0;
    CHECK_EQ(func->params.size(), args_registers.size());
    for (size_t i = 0; i < func->params.size(); i++) {
      auto ty = func->params[i]->checked_type();
      if (ty.as<TensorTypeNode>()) {
        unpacked_arg_regs.push_back(args_registers[i]);
        arity += 1;
      } else if (auto tuple_ty = ty.as<TupleTypeNode>()) {
        for (size_t f = 0; f < tuple_ty->fields.size(); f++) {
          const auto& field = tuple_ty->fields[f];
          CHECK(field.as<TensorTypeNode>())
            << "only supports non-nested tuples currently "
            << "found " << field;
          auto dst =  NewRegister();
          Emit(Instruction::GetField(args_registers[i], f, dst));
          unpacked_arg_regs.push_back(dst);
        }
        arity += tuple_ty->fields.size();
      } else {
        LOG(FATAL) << "unsupported parameter type " << ty;
      }
    }

    size_t return_val_count = 0;
    if (const TensorTypeNode* ttype = ret_type.as<TensorTypeNode>()) {
      // Allocate space for the return tensor.
      auto alloc = AllocTensorFromType(ttype);
      allocs.push_back(alloc);
      return_val_count = 1;
    } else if (const TupleTypeNode* ttype = ret_type.as<TupleTypeNode>()) {
      std::vector<Index> fields_registers;

      for (size_t i = 0; i < ttype->fields.size(); ++i) {
        auto f = ttype->fields[i];
        auto f_type = f.as<TensorTypeNode>();
        allocs.push_back(AllocTensorFromType(f_type));
        fields_registers.push_back(allocs.back().dst);
      }
      return_val_count = ttype->fields.size();
    } else {
      LOG(FATAL) << "Unsupported return value type";
    }

    arity += return_val_count;
    for (auto& alloc : allocs) {
      Emit(alloc);
      unpacked_arg_regs.push_back(alloc.dst);
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
      op_index = context_->lowered_funcs.size();
      context_->lowered_funcs.push_back(cfunc->funcs[0]);
      context_->seen_funcs[cfunc->funcs[0]] = op_index;
    } else {
      op_index = context_->seen_funcs[cfunc->funcs[0]];
    }

    Emit(Instruction::InvokePacked(op_index, arity, return_val_count, unpacked_arg_regs));

    if (return_val_count > 1) {
      // return value is a tuple, we need to create a tuple
      std::vector<Index> fields_registers;
      for (size_t i = arity - return_val_count; i < arity; ++i) {
        fields_registers.push_back(unpacked_arg_regs[i]);
      }
      Emit(Instruction::AllocDatatype(0, return_val_count, fields_registers, NewRegister()));
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
};


PackedFunc VMCompiler::GetFunction(const std::string& name,
                                   const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "compile") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.num_args, 3);
      this->Compile(args[0], args[1], args[2]);
    });
  } else if (name == "get_vm") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = runtime::Module(vm_);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void VMCompiler::Compile(const Module& mod_ref,
                         const TargetsMap& targets,
                         const tvm::Target& target_host) {
  CHECK_EQ(targets.size(), 1)
    << "Currently VM compiler doesn't support heterogeneous compilation";

  InitVM();
  targets_ = targets;
  target_host_ = target_host;

  // Run some optimizations first, this code should
  // be moved to pass manager.
  context_.module = OptimizeModule(mod_ref);

  // Populate the global map.
  //
  // This maps global variables to a global index
  // in the VMFunction table.
  PopulateGlobalMap();

  // Next we populate constant map.
  auto constant_analysis_result = LayoutConstantPool(context_.module);
  context_.const_map = std::get<0>(constant_analysis_result);
  context_.const_tensor_shape_map = std::get<1>(constant_analysis_result);

  // Next we get ready by allocating space for
  // the global state.
  vm_->functions.resize(context_.module->functions.size());
  vm_->constants.resize(context_.const_map.size() + context_.const_tensor_shape_map.size());

  for (auto pair : context_.const_map) {
    vm_->constants[pair.second] = Object::Tensor(pair.first->data);
  }

  for (auto pair : context_.const_tensor_shape_map) {
    vm_->constants[pair.second.first] = Object::Tensor(pair.second.second);
  }

  for (auto named_func : context_.module->functions) {
    auto gvar = named_func.first;
    auto func = named_func.second;
    VMFunctionCompiler func_compiler(&context_, targets_);
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

  LibraryCodegen();

  for (auto gv : context_.global_map) {
    vm_->global_map.insert({gv.first->name_hint, gv.second});
  }
}

Module VMCompiler::OptimizeModule(const Module& mod) {
  // TODO(@icemelon9): check number of targets and build config, add more optimization pass
  transform::Sequential seq({transform::SimplifyInference(),
                             transform::ToANormalForm(),
                             transform::InlinePrimitives(),
                             transform::LambdaLift(),
                             transform::InlinePrimitives(),
                             transform::FuseOps()});
  auto pass_ctx = transform::PassContext::Create();
  tvm::With<relay::transform::PassContext> ctx(pass_ctx);
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
  auto const& lowered_funcs = context_.lowered_funcs;
  if (lowered_funcs.size() == 0) {
    return;
  }
  // TODO(@icemelon9): support heterogeneous targets
  Target target;
  for (auto kv : targets_) {
    target = kv.second;
  }
  if (const auto* f = runtime::Registry::Get("relay.backend.build")) {
    runtime::Module mod =
        (*f)(tvm::Array<LoweredFunc>(lowered_funcs.begin(), lowered_funcs.end()), target,
             target_host_);
    CHECK(mod.operator->());
    vm_->lib = mod;
  } else {
    LOG(FATAL) << "relay.backend.build is not registered";
  }
  size_t primitive_index = 0;
  for (auto lfunc : lowered_funcs) {
    vm_->primitive_map.insert({lfunc->name, primitive_index++});
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
