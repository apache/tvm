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
 * \file src/relax/backend/vm/exec_builder.cc
 */
#include <tvm/relax/exec_builder.h>

#include <sstream>

namespace tvm {
namespace relax {

using namespace vm;

TVM_REGISTER_NODE_TYPE(ExecBuilderNode);

ExecBuilder ExecBuilderNode::Create() {
  ExecBuilder ret(make_object<ExecBuilderNode>());
  ret->exec_ = make_object<Executable>();
  return ret;
}

Executable* ExecBuilderNode::exec() const { return exec_.get(); }

ObjectPtr<Executable> ExecBuilderNode::Get() {
  this->Formalize();
  this->CheckExecutable();
  return exec_;
}

vm::Instruction::Arg ExecBuilderNode::ConvertConstant_(TVMRetValue cvalue) {
  // emit constant immediate as immediate.
  if (cvalue.type_code() == kDLInt) {
    int64_t val = cvalue.operator int64_t();
    if (val <= vm::Instruction::kValueMaxLimit && val >= vm::Instruction::kValueMinLimit) {
      return vm::Instruction::Arg::Immediate(val);
    }
  }
  // convert string to object string
  if (cvalue.type_code() == kTVMStr) {
    cvalue = cvalue.operator String();
  }

  // run dedup for object with structural equality
  if (cvalue.IsObjectRef<ObjectRef>()) {
    ObjectRef obj = cvalue.operator ObjectRef();
    auto it = const_dedup_map_.find(obj);
    if (it != const_dedup_map_.end()) {
      return vm::Instruction::Arg::ConstIdx(it->second);
    }
    vm::Index idx = exec_->constants.size();
    exec_->constants.push_back(cvalue);
    const_dedup_map_[obj] = idx;
    return vm::Instruction::Arg::ConstIdx(idx);
  } else {
    // emit normal constant
    vm::Index idx = exec_->constants.size();
    exec_->constants.push_back(cvalue);
    return vm::Instruction::Arg::ConstIdx(idx);
  }
}

void ExecBuilderNode::DeclareFunction(const std::string& func_name, VMFuncInfo::FuncKind kind) {
  auto it = exec_->func_map.find(func_name);
  if (it != exec_->func_map.end()) {
    ICHECK(kind == exec_->func_table[it->second].kind)
        << "Function " << func_name << "already declared in a different kind";
    return;
  }
  VMFuncInfo vmfunc;
  vmfunc.kind = kind;
  vmfunc.name = func_name;
  // use num args to mark undefined.
  vmfunc.start_instr = 0;
  vmfunc.num_args = -2;
  vmfunc.register_file_size = 0;
  exec_->func_map[func_name] = exec_->func_table.size();
  exec_->func_table.push_back(vmfunc);
}

vm::Instruction::Arg ExecBuilderNode::GetFunction(const std::string& func_name) {
  auto it = exec_->func_map.find(func_name);
  ICHECK(it != exec_->func_map.end()) << "Cannot find function " << func_name;
  return vm::Instruction::Arg::FuncIdx(it->second);
}

void ExecBuilderNode::EmitFunction(const std::string& func_name, int64_t num_inputs,
                                   Optional<Array<String>> param_names,
                                   vm::VMFuncInfo::FuncKind kind, int64_t init_register_size) {
  auto it = exec_->func_map.find(func_name);
  if (it == exec_->func_map.end()) {
    this->DeclareFunction(func_name, kind);
  }
  auto& vmfunc = exec_->func_table.at(exec_->func_map.at(func_name));
  ICHECK_EQ(vmfunc.name, func_name);
  ICHECK_EQ(vmfunc.num_args, -2) << "Function " << func_name << " already defined";
  vmfunc.num_args = num_inputs;
  if (param_names.defined()) {
    std::vector<std::string> names;
    for (auto name : param_names.value()) {
      names.push_back(name);
    }
    vmfunc.param_names = names;
  }
  vmfunc.register_file_size = init_register_size;
  if (kind == vm::VMFuncInfo::FuncKind::kVMFunc) {
    vmfunc.start_instr = exec_->instr_offset.size();
  }
}

void ExecBuilderNode::EndFunction(const std::string& func_name) {
  auto it = exec_->func_map.find(func_name);
  ICHECK(it != exec_->func_map.end());
  VMFuncInfo& vmfunc = exec_->func_table.at(it->second);
  ICHECK_EQ(vmfunc.end_instr, 0) << "EndFuncton can only be called once";

  if (vmfunc.kind == vm::VMFuncInfo::FuncKind::kVMFunc) {
    vmfunc.end_instr = exec_->instr_offset.size();
  }
}

void ExecBuilderNode::EmitCall(vm::Instruction::Arg func, std::vector<vm::Instruction::Arg> args,
                               vm::RegName dst) {
  ICHECK(func.kind() == vm::Instruction::ArgKind::kFuncIdx);
  // store instruction
  exec_->instr_offset.push_back(exec_->instr_data.size());
  exec_->instr_data.push_back(static_cast<ExecWord>(Opcode::Call));
  exec_->instr_data.push_back(dst);
  exec_->instr_data.push_back(func.value());
  exec_->instr_data.push_back(args.size());
  for (Instruction::Arg arg : args) {
    exec_->instr_data.push_back(arg.data());
  }
}

void ExecBuilderNode::EmitCall(const std::string& func, std::vector<Instruction::Arg> args,
                               RegName dst) {
  auto it = exec_->func_map.find(func);
  if (it == exec_->func_map.end()) {
    this->DeclareFunction(func, VMFuncInfo::FuncKind::kPackedFunc);
  }
  Index func_idx = exec_->func_map.at(func);
  EmitCall(vm::Instruction::Arg::FuncIdx(func_idx), args, dst);
}

void ExecBuilderNode::EmitRet(vm::Instruction::Arg result) {
  ICHECK(result.kind() == vm::Instruction::ArgKind::kRegister);
  exec_->instr_offset.push_back(exec_->instr_data.size());
  exec_->instr_data.push_back(static_cast<ExecWord>(Opcode::Ret));
  exec_->instr_data.push_back(result.value());
}

void ExecBuilderNode::EmitGoto(Index pc_offset) {
  exec_->instr_offset.push_back(exec_->instr_data.size());
  exec_->instr_data.push_back(static_cast<ExecWord>(Opcode::Goto));
  exec_->instr_data.push_back(pc_offset);
}

void ExecBuilderNode::EmitIf(vm::Instruction::Arg cond, vm::Index false_offset) {
  ICHECK(cond.kind() == vm::Instruction::ArgKind::kRegister);
  exec_->instr_offset.push_back(exec_->instr_data.size());
  exec_->instr_data.push_back(static_cast<ExecWord>(Opcode::If));
  exec_->instr_data.push_back(cond.value());
  exec_->instr_data.push_back(false_offset);
}

void ExecBuilderNode::CheckExecutable() {
  for (auto it = exec_->func_table.cbegin(); it != exec_->func_table.cend(); ++it) {
    if (it->kind == VMFuncInfo::FuncKind::kPackedFunc) continue;
    if (it->kind == VMFuncInfo::FuncKind::kVMTIRFunc) {
      ICHECK_GE(it->register_file_size, it->num_args + 1)
          << "Function " << it->name << " do not meet register file constraint.";
      continue;
    }
    Index num_inputs = it->num_args;
    std::unordered_set<RegName> dst_registers;
    std::unordered_set<RegName> arg_registers;
    size_t start_instr = it->start_instr;
    size_t end_instr = it->end_instr;

    CHECK_LT(start_instr, end_instr)
        << "Function " << it->name << " EndFunction has not be been called";

    auto check_reg_defined = [&](Instruction::Arg arg) {
      if (arg.kind() != Instruction::ArgKind::kRegister) return;
      if (arg.value() >= Instruction::kBeginSpecialReg) return;
      if (arg.value() < num_inputs) return;

      if (dst_registers.find(arg.value()) == dst_registers.end()) {
        LOG(FATAL) << "register r(" << arg.value() << ") in VM function \"" << it->name
                   << "\" is used as input while it is never defined"
                   << " as a destination. Dump:\n"
                   << exec_->AsText();
      }
    };

    auto check_const_defined = [&](Instruction::Arg arg) {
      if (arg.kind() != Instruction::ArgKind::kConstIdx) return;
      CHECK_LT(arg.value(), exec_->constants.size())
          << "Constant index " << arg.value() << " exceed size of constant pool. Dump:\n"
          << exec_->AsText();
    };

    auto check_func_defined = [&](Instruction::Arg arg) {
      if (arg.kind() != Instruction::ArgKind::kFuncIdx) return;
      CHECK_LT(arg.value(), exec_->func_table.size())
          << "Func index " << arg.value() << " exceed size of fun_table. Dump:\n"
          << exec_->AsText();
    };

    for (size_t idx = start_instr; idx < end_instr; ++idx) {
      Instruction instr = exec_->GetInstruction(idx);
      switch (instr.op) {
        case Opcode::Call: {
          check_func_defined(Instruction::Arg::FuncIdx(instr.func_idx));
          for (int i = 0; i < instr.num_args; ++i) {
            check_reg_defined(instr.args[i]);
            check_const_defined(instr.args[i]);
            check_func_defined(instr.args[i]);
            arg_registers.emplace(instr.args[i].value());
          }
          if (instr.dst != Instruction::kVoidRegister) {
            dst_registers.emplace(instr.dst);
          }
          break;
        }
        case Opcode::Ret: {
          arg_registers.emplace(instr.result);
          check_reg_defined(Instruction::Arg::Register(instr.result));
          break;
        }
        case Opcode::Goto: {
          ICHECK_NE(instr.pc_offset, 0);
          break;
        }
        case Opcode::If: {
          ICHECK_GT(instr.false_offset, 1);
          check_reg_defined(Instruction::Arg::Register(instr.cond));
          arg_registers.emplace(instr.cond);
          break;
        }
        default:
          LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
          break;
      }
    }
  }
}

void ExecBuilderNode::Formalize() {
  // a pass to formalize user-specified register indexes in the order of use
  // and decide the number of registers to allocate for each VMFunction in the Executable
  for (auto it = this->exec_->func_table.begin(); it != this->exec_->func_table.end(); ++it) {
    if (it->kind == VMFuncInfo::FuncKind::kPackedFunc) continue;
    if (it->kind == VMFuncInfo::FuncKind::kVMTIRFunc) continue;

    Index num_inputs = it->num_args;
    RegName register_idx = num_inputs;
    std::unordered_map<RegName, RegName> register_map;
    size_t start_instr = it->start_instr;
    size_t end_instr = it->end_instr;

    for (size_t idx = start_instr; idx < end_instr; ++idx) {
      Instruction instr = this->exec_->GetInstruction(idx);
      switch (instr.op) {
        case Opcode::Call: {
          // rewrite args
          for (int i = 0; i < instr.num_args; ++i) {
            if (instr.args[i].kind() == Instruction::ArgKind::kRegister &&
                instr.args[i].value() >= num_inputs &&
                instr.args[i].value() < Instruction::kBeginSpecialReg &&
                register_map.find(instr.args[i].value()) != register_map.end()) {
              this->exec_->instr_data[this->exec_->instr_offset[idx] + 4 + i] =
                  register_map[instr.args[i].value()];
            }
          }
          if (instr.dst >= num_inputs && instr.dst < Instruction::kBeginSpecialReg) {
            auto it = register_map.find(instr.dst);
            if (it != register_map.end()) {
              this->exec_->instr_data[this->exec_->instr_offset[idx] + 1] = it->second;
            } else {
              this->exec_->instr_data[this->exec_->instr_offset[idx] + 1] = register_idx;
              register_map[instr.dst] = register_idx++;
            }
          }
          break;
        }
        case Opcode::Ret: {
          if (register_map.find(instr.result) != register_map.end()) {
            this->exec_->instr_data[this->exec_->instr_offset[idx] + 1] =
                register_map[instr.result];
          }
          break;
        }
        case Opcode::Goto: {
          break;
        }
        case Opcode::If: {
          if (register_map.find(instr.cond) != register_map.end()) {
            this->exec_->instr_data[this->exec_->instr_offset[idx] + 1] = register_map[instr.cond];
          }
          break;
        }
        default:
          LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
          break;
      }
    }
    it->register_file_size = register_idx;
  }
}

TVM_REGISTER_GLOBAL("relax.ExecBuilderCreate").set_body_typed(ExecBuilderNode::Create);

TVM_REGISTER_GLOBAL("relax.ExecBuilderConvertConstant")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      ExecBuilder builder = args[0];
      TVMRetValue rt;
      rt = args[1];
      *ret = builder->ConvertConstant(rt).data();
    });

TVM_REGISTER_GLOBAL("relax.ExecBuilderEmitFunction")
    .set_body_typed([](ExecBuilder builder, String func, int64_t num_inputs,
                       Optional<Array<String>> param_names) {
      builder->EmitFunction(func, num_inputs, param_names);
    });

TVM_REGISTER_GLOBAL("relax.ExecBuilderEndFunction")
    .set_body_method<ExecBuilder>(&ExecBuilderNode::EndFunction);

TVM_REGISTER_GLOBAL("relax.ExecBuilderDeclareFunction")
    .set_body_typed([](ExecBuilder builder, String name, int32_t kind) {
      builder->DeclareFunction(name, static_cast<VMFuncInfo::FuncKind>(kind));
    });

TVM_REGISTER_GLOBAL("relax.ExecBuilderEmitCall")
    .set_body_typed([](ExecBuilder builder, String name, Array<IntImm> args, int64_t dst) {
      std::vector<Instruction::Arg> args_;
      for (size_t i = 0; i < args.size(); ++i) {
        args_.push_back(Instruction::Arg::FromData(args[i]->value));
      }
      auto dst_ = Instruction::Arg::Register(dst);
      builder->EmitCall(name, args_, dst_.value());
    });

TVM_REGISTER_GLOBAL("relax.ExecBuilderEmitRet")
    .set_body_typed([](ExecBuilder builder, int64_t data) {
      builder->EmitRet(Instruction::Arg::FromData(data));
    });

TVM_REGISTER_GLOBAL("relax.ExecBuilderEmitGoto")
    .set_body_method<ExecBuilder>(&ExecBuilderNode::EmitGoto);

TVM_REGISTER_GLOBAL("relax.ExecBuilderEmitIf")
    .set_body_typed([](ExecBuilder builder, int64_t data, vm::Index false_offset) {
      builder->EmitIf(Instruction::Arg::FromData(data), false_offset);
    });

TVM_REGISTER_GLOBAL("relax.ExecBuilderR").set_body_typed([](ExecBuilder builder, int64_t value) {
  return Instruction::Arg::Register(value).data();
});

TVM_REGISTER_GLOBAL("relax.ExecBuilderImm").set_body_typed([](ExecBuilder builder, int64_t value) {
  return Instruction::Arg::Immediate(value).data();
});

TVM_REGISTER_GLOBAL("relax.ExecBuilderC").set_body_typed([](ExecBuilder builder, int64_t value) {
  return Instruction::Arg::ConstIdx(value).data();
});

TVM_REGISTER_GLOBAL("relax.ExecBuilderF").set_body_typed([](ExecBuilder builder, String value) {
  return builder->GetFunction(value).data();
});

TVM_REGISTER_GLOBAL("relax.ExecBuilderGet").set_body_typed([](ExecBuilder builder) {
  ObjectPtr<Executable> p_exec = builder->Get();
  return runtime::Module(p_exec);
});

}  // namespace relax
}  // namespace tvm
