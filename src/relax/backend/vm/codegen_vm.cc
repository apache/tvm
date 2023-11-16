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
 * \file src/relax/backend/vm/codegen_vm.cc
 * \brief A codegen to generate VM executable from a Relax IRModule.
 */
#include <tvm/driver/driver_api.h>
#include <tvm/relax/exec_builder.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/runtime/relax_vm/bytecode.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../../../target/metadata_module.h"
#include "../../../target/source/codegen_source_base.h"

namespace tvm {
namespace relax {
namespace relax_vm {

using tvm::Target;
using namespace relax;
using namespace tvm::runtime;
using namespace tvm::runtime::relax_vm;

// Helper function to get the function name of the registered packed function implementation of
// relax operator.
FCallPacked GetPackedFuncName(const Call& call) {
  static auto op_map = Op::GetAttrMap<FCallPacked>("FCallPacked");
  if (call->op.as<OpNode>()) {
    Op op = Downcast<Op>(call->op);
    if (op_map.count(op)) {
      return op_map[op];
    }
  }
  return {};
}

/*!
 * \brief A class to generate VM executable for Relax functions.
 */
class CodeGenVM : public ExprFunctor<Instruction::Arg(const Expr&)> {
 public:
  explicit CodeGenVM(relax::ExecBuilder builder, IRModule ctx_mod)
      : builder_(builder), ctx_mod_(ctx_mod) {}

  static IRModule Run(relax::ExecBuilder builder, IRModule mod) {
    IRModule res_mod = mod;
    res_mod.CopyOnWrite();
    CodeGenVM codegen(builder, mod);
    // Remove relax function and turn into TIR func.
    for (const auto& [gvar, f] : mod->functions) {
      if (auto* func = f.as<FunctionNode>()) {
        codegen.Codegen(GetRef<Function>(func));
        res_mod->Remove(gvar);
      }
    }
    return res_mod;
  }

 protected:
  size_t NewRegister() { return registers_num_++; }

  // Convert Arg value to a register, trigger copy if needed
  Instruction::Arg EnsureReg(Instruction::Arg arg) {
    if (arg.kind() == Instruction::ArgKind::kRegister) {
      return arg;
    } else {
      RegName dst_reg = NewRegister();
      builder_->EmitCall("vm.builtin.copy", {arg}, dst_reg);
      return Instruction::Arg::Register(dst_reg);
    }
  }

  void Codegen(const Function& func) {
    Optional<String> gsymbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(gsymbol.defined()) << "there should be no local functions in Relax VM codegen phase. "
                                 "Did you forget to apply LambdaLift or AttachGlobalSymbol Pass?";

    Array<String> param_names;
    for (Var param : func->params) {
      param_names.push_back(param->name_hint());
    }

    builder_->EmitFunction(gsymbol.value(), func->params.size(), param_names);

    for (size_t i = 0; i < func->params.size(); ++i) {
      RegName r = NewRegister();
      ICHECK_EQ(r, static_cast<RegName>(i));
      this->var_arg_map_.insert({func->params[i], Instruction::Arg::Register(r)});
    }
    Instruction::Arg ret = ExprFunctor::VisitExpr(func->body);
    builder_->EmitRet(EnsureReg(ret));
    builder_->EndFunction(gsymbol.value());
    // reset register number to be 0;
    registers_num_ = 0;
    var_arg_map_.clear();
  }

  Instruction::Arg VisitExpr_(const SeqExprNode* op) final {
    for (auto block : op->blocks) {
      for (Binding binding : block->bindings) {
        Expr expr = GetBoundValue(binding);

        Instruction::Arg value = VisitExpr(expr);
        if (expr.as<VarNode>()) {
          // For a normalized relax module, there should be one
          // register for each relax::Binding.  This makes the Relax
          // semantics of R.vm.kill_* operate the same as the Python
          // "del" operator.  These bindings may be removable by using
          // relax.transform.CanonicalizeBindings earlier in lowering.
          RegName new_reg = NewRegister();
          builder_->EmitCall("vm.builtin.copy", {value}, new_reg);
          value = Instruction::Arg::Register(new_reg);
        }

        this->var_arg_map_.insert({binding->var, value});
      }
    }

    Instruction::Arg ret_reg = this->VisitExpr(op->body);
    return ret_reg;
  }

  Instruction::Arg VisitExpr_(const CallNode* call_node) final {
    Call call = GetRef<Call>(call_node);

    if (call_node->op == null_value_op_) {
      return Instruction::Arg::Register(Instruction::kVoidRegister);
    }

    // allocate dst register.
    RegName dst_reg = HasVoidStructInfo(call) ? Instruction::kVoidRegister : NewRegister();
    if (call->op.as<OpNode>()) {
      // special case generate for the intrinsics whose attribute fields
      // cannot be represented by args in the CallNode
      FCallPacked name = GetPackedFuncName(call);
      if (!name.empty()) {
        // If the operator has a registered packed function implementation, emit call to that packed
        // function.
        EmitPackedFuncCall(call, name, dst_reg);
      } else if (call_node->op == call_builtin_with_ctx_op_) {
        // TODO(relax-team) migrate most handling of op to
        // directly map to call_builtin_with_ctx before codegen and simplify vm codegen.
        EmitCallBuiltinWithCtx(call, dst_reg);
      } else if (call_node->op == alloc_storage_op_) {
        EmitAllocStorage(call, dst_reg);
      } else if (call_node->op == alloc_tensor_op_) {
        EmitAllocTensor(call, dst_reg);
      } else if (call_node->op == kill_object_op_) {
        dst_reg = EmitKillObject(call);
      } else {
        // every "normal" operator is lowered to a global var in the IRModule. The Attrs for those
        // ops are handled in a pass when lowering them to TIR.
        LOG(FATAL) << "CodeGenVM cannot handle this intrinsic now:\n" << call_node->op;
      }
    } else {
      EmitNormalCall(call, dst_reg);
    }
    return Instruction::Arg::Register(dst_reg);
  }

  Instruction::Arg VisitExpr_(const IfNode* op) final {
    const If& ife = GetRef<If>(op);
    Instruction::Arg cond_value = this->VisitExpr(ife->cond);

    // Reserve a register for cond
    RegName cond_reg = NewRegister();
    builder_->EmitCall("vm.builtin.read_if_cond", {cond_value}, cond_reg);

    // obtain the temp exec in progress.
    vm::Executable* exec = builder_->exec();

    // Record the offset of If instruction
    size_t if_offset = exec->instr_offset.size();

    builder_->EmitIf(Instruction::Arg::Register(cond_reg), 3);
    size_t num_instr = exec->instr_offset.size();
    Instruction::Arg true_value = this->VisitExpr(ife->true_branch);
    // Reserve a register for return
    size_t merge_register = NewRegister();
    // Copy the output from true branch to merge register
    builder_->EmitCall("vm.builtin.copy", {true_value}, merge_register);

    // Record the offset of Goto instruction
    size_t goto_offset = exec->instr_offset.size();

    builder_->EmitGoto(1);

    // Calculate the false offset of If
    size_t false_offset = exec->instr_offset.size() - num_instr + 1;

    Instruction::Arg false_value = this->VisitExpr(ife->false_branch);
    // Copy the output data of false branch to merge register
    builder_->EmitCall("vm.builtin.copy", {false_value}, merge_register);

    // Update the offsets of the If instruction emitted above
    // Jump to the behind of the next goto instruction
    exec->SetInstructionData(if_offset, 2, static_cast<ExecWord>(false_offset));
    // Update the pc_offset of Goto instruction
    // Jump over the false branch
    size_t pc_offset = exec->instr_offset.size() - goto_offset;
    exec->SetInstructionData(goto_offset, 1, static_cast<ExecWord>(pc_offset));
    return Instruction::Arg::Register(merge_register);
  }

  Instruction::Arg VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = this->var_arg_map_.find(var);
    ICHECK(it != this->var_arg_map_.end()) << "Var " << var << " is not defined";
    return it->second;
  }

  Instruction::Arg VisitExpr_(const ConstantNode* op) final {
    return builder_->ConvertConstant(op->data);
  }

  Instruction::Arg VisitExpr_(const ShapeExprNode* op) final {
    std::vector<int64_t> shape;
    for (PrimExpr e : op->values) {
      if (auto* int_value = e.as<IntImmNode>()) {
        shape.push_back(int_value->value);
      } else {
        LOG(FATAL) << "Should only use constant shape after shape lowering: " << op->values;
      }
    }
    return builder_->ConvertConstant(ShapeTuple(shape));
  }

  Instruction::Arg VisitExpr_(const PrimValueNode* op) final {
    if (auto* int_imm = op->value.as<IntImmNode>()) {
      return builder_->ConvertConstant(int_imm->value);
    } else if (auto* float_imm = op->value.as<FloatImmNode>()) {
      return builder_->ConvertConstant(float_imm->value);
    } else {
      LOG(FATAL) << "PrimValue should only contain constant after  VMShapeLower, "
                 << "but received " << GetRef<Expr>(op) << " with type " << op->value->GetTypeKey();
    }
  }

  Instruction::Arg VisitExpr_(const StringImmNode* op) final {
    return builder_->ConvertConstant(op->value);
  }

  Instruction::Arg VisitExpr_(const DataTypeImmNode* op) final {
    return builder_->ConvertConstant(op->value);
  }

  Instruction::Arg VisitExpr_(const TupleNode* op) final {
    Tuple tuple = GetRef<Tuple>(op);
    std::vector<Instruction::Arg> args;
    for (Expr arg : tuple->fields) {
      args.push_back(this->VisitExpr(arg));
    }
    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.builtin.make_tuple", args, dst_register);

    return Instruction::Arg::Register(dst_register);
  }

  Instruction::Arg VisitExpr_(const TupleGetItemNode* op) final {
    TupleGetItem expr = GetRef<TupleGetItem>(op);
    std::vector<Instruction::Arg> args = {this->VisitExpr(expr->tuple)};

    args.push_back(builder_->ConvertConstant(expr->index));

    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.builtin.tuple_getitem", args, dst_register);

    return Instruction::Arg::Register(dst_register);
  }

  Instruction::Arg VisitExpr_(const GlobalVarNode* op) final {
    GlobalVar gvar = GetRef<GlobalVar>(op);
    Optional<String> symbol;
    VMFuncInfo::FuncKind kind = VMFuncInfo::FuncKind::kPackedFunc;

    // Run a look up in the env to see if it maps to an extern func.
    auto it = ctx_mod_->functions.find(gvar);
    if (it != ctx_mod_->functions.end()) {
      BaseFunc func = (*it).second;
      if (auto* efunc = func.as<ExternFuncNode>()) {
        symbol = efunc->global_symbol;
        kind = VMFuncInfo::FuncKind::kPackedFunc;
      } else if (func.as<FunctionNode>()) {
        symbol = gvar->name_hint;
        kind = VMFuncInfo::FuncKind::kVMFunc;
      }
    }
    // GlobalVar can be reference to a Relax function or a TIR primfunc
    // At this point: all global var must corresponds to the right symbol.
    // TODO(relax-team): switch everything to extern before splitting TIR/relax
    // so we do not have idle global var here.
    if (!symbol.defined()) {
      symbol = gvar->name_hint;
      kind = VMFuncInfo::FuncKind::kPackedFunc;
    }
    // declare the function to be safe.
    ICHECK(symbol.defined());
    builder_->DeclareFunction(symbol.value(), kind);
    return builder_->GetFunction(symbol.value());
  }

  Instruction::Arg VisitExpr_(const ExternFuncNode* op) final {
    static const constexpr char* kCSource = "c_source";
    static const constexpr char* kCSourceFmt = "c_source_fmt";
    if (Optional<String> opt_code = op->attrs.GetAttr<String>(kCSource)) {
      String sym = op->global_symbol;
      String fmt = op->attrs.GetAttr<String>(kCSourceFmt).value_or("c");
      String code = opt_code.value();
      Module c_source_module =
          codegen::CSourceModuleCreate(/*code=*/code, /*fmt=*/fmt, /*func_names=*/{sym},
                                       /*const_vars=*/{});
      builder_->exec()->Import(c_source_module);
    }
    builder_->DeclareFunction(op->global_symbol, VMFuncInfo::FuncKind::kPackedFunc);
    return builder_->GetFunction(op->global_symbol);
  }

  void EmitAllocStorage(const Call& call_node, RegName dst_reg) {
    ICHECK_EQ(call_node->args.size(), 4);
    // Handle args of the call
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg::Register(Instruction::kVMRegister));
    // buffer size, dtype, device index
    for (auto arg : call_node->args) {
      args.push_back(this->VisitExpr(arg));
    }
    builder_->EmitCall("vm.builtin.alloc_storage", args, dst_reg);
  }

  void EmitAllocTensor(const Call& call_node, RegName dst_reg) {
    ICHECK_EQ(call_node->args.size(), 4);
    std::vector<Instruction::Arg> args;
    args.reserve(4);
    for (Expr arg : call_node->args) {
      args.push_back(this->VisitExpr(arg));
    }
    builder_->EmitCall("vm.builtin.alloc_tensor", args, dst_reg);
  }

  RegName EmitKillObject(const Call& call_node) {
    ICHECK_EQ(call_node->args.size(), 1);
    Instruction::Arg arg = this->VisitExpr(call_node->args[0]);
    ICHECK(arg.kind() == Instruction::ArgKind::kRegister)
        << "Expected the object to be killed to be stored in a register, "
        << "but argument " << call_node->args[0] << " produced VM instruction of type "
        << arg.kind();
    RegName dst_reg = arg.value();
    builder_->EmitCall("vm.builtin.null_value", {}, dst_reg);
    return dst_reg;
  }

  void EmitCallBuiltinWithCtx(const Call& call_node, RegName dst_reg) {
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg::Register(Instruction::kVMRegister));

    auto func = this->VisitExpr(call_node->args[0]);
    auto tuple_arg = Downcast<Tuple>(call_node->args[1]);

    // Handle args of the call
    for (Expr arg : tuple_arg->fields) {
      args.push_back(this->VisitExpr(arg));
    }

    builder_->EmitCall(func, args, dst_reg);
  }

  void EmitNormalCall(const Call& call_node, RegName dst_reg) {
    Instruction::Arg func = VisitExpr(call_node->op);
    std::vector<Instruction::Arg> args = VisitArray(call_node->args);
    builder_->EmitCall(func, args, dst_reg);
  }

  // Emits call to packed function `name` with arguments copied over from `call_node` args
  void EmitPackedFuncCall(const Call& call_node, const FCallPacked& name, RegName dst_reg) {
    std::vector<Instruction::Arg> args = VisitArray(call_node->args);
    builder_->EmitCall(name, args, dst_reg);
  }

  std::vector<Instruction::Arg> VisitArray(const Array<Expr>& arr) {
    std::vector<Instruction::Arg> ret;
    for (size_t i = 0; i < arr.size(); ++i) {
      ret.push_back(this->VisitExpr(arr[i]));
    }
    return ret;
  }

  /*! \brief Internal ExecBuilder. */
  relax::ExecBuilder builder_;
  /*!
   * \brief Total number of virtual registers allocated.
   * \note The first two registers are reserved for special registers.
   */
  size_t registers_num_ = 0;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, Instruction::Arg, ObjectPtrHash, ObjectPtrEqual> var_arg_map_;
  /*! \brief the context module. */
  IRModule ctx_mod_;
  /*! \brief Cache ops that need to be frequently used later to reduce lookup overhead. */
  const Op& alloc_storage_op_ = Op::Get("relax.vm.alloc_storage");
  const Op& alloc_tensor_op_ = Op::Get("relax.vm.alloc_tensor");
  const Op& kill_object_op_ = Op::Get("relax.vm.kill_object");
  const Op& call_builtin_with_ctx_op_ = Op::Get("relax.call_builtin_with_ctx");
  const Op& null_value_op_ = Op::Get("relax.null_value");
};

/*!
 * \brief Create the Relax VM executable from all relax.Function in mod.
 *        and add them to exec_builder.
 * \param exec_builder Builder to collect executables.
 * \param mod Input module.
 * \return Left over IRModule that may contain other functions.
 */
IRModule VMCodeGen(ExecBuilder exec_builder, IRModule mod) {
  return CodeGenVM::Run(exec_builder, mod);
}

TVM_REGISTER_GLOBAL("relax.VMCodeGen").set_body_typed(VMCodeGen);

/*!
 * \brief Link the libraries together.
 */
Module VMLink(ExecBuilder builder, Target target, Optional<Module> lib, Array<Module> ext_libs,
              Map<String, runtime::NDArray> params) {
  // TODO(relax-team) Revisit the param and ext_lib options.
  ObjectPtr<Executable> executable = builder->Get();
  if (!lib.defined()) {
    lib = codegen::CSourceModuleCreate(";", "", Array<String>{});
  }
  std::unordered_map<std::string, runtime::NDArray> conv_params;
  for (const auto& [name, param] : params) {
    conv_params[name] = param;
  }
  Module combined_lib = codegen::CreateMetadataModule(
      conv_params, lib.value(), ext_libs, target,

      // TODO(@sunggg): Currently, CRT uses relay-specific executor for uTVM support.
      // Before jumping into details, only support cpp runtime for now.
      relay::Runtime::Create("cpp"),
      relay::Executor::Create("graph"),  // TODO(@sunggg): pass arbitrarily executor. CPP runtime
                                         // won't use this anyways.
      relay::backend::ExecutorCodegenMetadata());
  executable->Import(combined_lib);
  return Module(executable);
}

TVM_REGISTER_GLOBAL("relax.VMLink").set_body_typed(VMLink);

}  // namespace relax_vm
}  // namespace relax
}  // namespace tvm
