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

#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/algorithms.h>
#include <tvm/tir/usmp/analysis.h>
#include <tvm/tir/usmp/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <stack>
#include <string>

namespace tvm {
namespace tir {
namespace usmp {

/*! \brief Creates Allocate nodes with special annotations
 * for I/O tensors in the graph to be memory planned.*/
class IOAllocateCreator : public StmtExprVisitor {
 public:
  explicit IOAllocateCreator(const IRModule& module) {
    main_func_ = Downcast<PrimFunc>(module->Lookup(::tvm::runtime::symbol::tvm_module_main));
    ICHECK(main_func_.defined()) << "main function is not in the module";
    for (const auto& gv_func : module->functions) {
      if (gv_func.second->IsInstance<PrimFuncNode>()) {
        functions_.Set(gv_func.first->name_hint, Downcast<PrimFunc>(gv_func.second));
      }
    }
    mod_ = module->ShallowCopy();
  }
  IRModule operator()();

 private:
  void VisitExpr_(const BufferLoadNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;

  /*! \brief Updates aliases that buffer vars inside the primfunc refer
   * to in terms call arguments they get bound to.*/
  void UpdateAliases(const Array<PrimExpr>& args, const PrimFunc& func);

  /*! \brief The IRModule that is being mutated */
  IRModule mod_;
  /*! \brief The main function that calls into operator subgraphs */
  PrimFunc main_func_;
  /*! \brief The input Vars of the main function */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> inputs_;
  /*! \brief The output Vars of the main function */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> outputs_;
  /*! \brief The buffer vars associated with the I/O Vars */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> io_buffer_vars_;
  /*! \brief The aliases that buffer vars inside the primfunc refer
   * to in terms call arguments */
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> aliases_;
  /*!
   * \brief The TIR main function calls by name to PrimFuncs to be able to
   * support BYOC. Therefore, this Map records functions that are present
   * in the IRModule by name/
   */
  Map<String, PrimFunc> functions_;
};

/*!
 * \brief The function obtains the matched buffer vars for
 * the params of the PrimFunc.
 */
Array<Var> static GetMatchedBuffers(const PrimFunc& func) {
  Array<Var> buffer_vars;
  for (unsigned int i = 0; i < func->params.size() - 1; i++) {
    Var param = func->params[i];
    buffer_vars.push_back(func->buffer_map[param]->data);
  }
  Var last_param = func->params.back();
  // Checks whether last var is present in the buffer map
  // because it could be the resource handle
  if (func->buffer_map.find(last_param) != func->buffer_map.end()) {
    buffer_vars.push_back(func->buffer_map[last_param]->data);
  }
  return buffer_vars;
}

/*!
 * \brief The function updates aliases that each buffer var with its
 * associated argument in the callsite.
 */
void IOAllocateCreator::UpdateAliases(const Array<PrimExpr>& args, const PrimFunc& func) {
  auto param_buffers = GetMatchedBuffers(func);
  // Last var could be a resource handle that does not have a Buffer
  ICHECK(args.size() == param_buffers.size() || args.size() - 1 == param_buffers.size());
  for (size_t i = 0; i < param_buffers.size(); i++) {
    auto arg = args[i];
    if (arg->IsInstance<VarNode>()) {
      auto param_buf = param_buffers[i];
      aliases_[param_buf] = Downcast<Var>(arg);
    }
  }
}

void IOAllocateCreator::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::call_extern()) || op->op.same_as(builtin::tvm_call_cpacked())) {
    StringImm func_name = Downcast<StringImm>(op->args[0])->value;
    if (functions_.find(func_name->value) != functions_.end()) {
      auto func = functions_.at(func_name->value);
      auto actual_args = Array<PrimExpr>(op->args.begin() + 1, op->args.end());
      this->UpdateAliases(actual_args, func);
      VisitStmt(func->body);
      return;
    }
  }
  if (op->op->IsInstance<PrimFuncNode>()) {
    auto func = Downcast<PrimFunc>(op->op);
    this->UpdateAliases(op->args, func);
    VisitStmt(func->body);
    return;
  }
  StmtExprVisitor::VisitExpr_(op);
}

void IOAllocateCreator::VisitExpr_(const BufferLoadNode* op) {
  if (aliases_.find(op->buffer->data) != aliases_.end()) {
    Var aliased_var = aliases_[op->buffer->data];
    if (io_buffer_vars_.find(aliased_var) != io_buffer_vars_.end()) {
      ICHECK(outputs_.find(aliased_var) == outputs_.end())
          << "BufferLoad nodes should not be reading from output buffer vars.";
      inputs_.insert(aliased_var);
    }
  }
  StmtExprVisitor::VisitExpr_(op);
}

void IOAllocateCreator::VisitStmt_(const BufferStoreNode* op) {
  if (aliases_.find(op->buffer->data) != aliases_.end()) {
    Var aliased_var = aliases_[op->buffer->data];
    if (io_buffer_vars_.find(aliased_var) != io_buffer_vars_.end()) {
      ICHECK(inputs_.find(aliased_var) == inputs_.end())
          << "BufferStore nodes should not be writing to input buffer vars.";
      outputs_.insert(aliased_var);
    }
  }
  StmtExprVisitor::VisitStmt_(op);
}

IRModule IOAllocateCreator::operator()() {
  Array<Var> new_main_params;
  Stmt main_body = main_func_->body;
  for (const Var& param : main_func_->params) {
    if (main_func_->buffer_map.find(param) != main_func_->buffer_map.end()) {
      Var buffer_var = main_func_->buffer_map[param]->data;
      io_buffer_vars_.insert(buffer_var);
      aliases_[buffer_var] = buffer_var;
    }
  }
  VisitStmt(main_body);
  ICHECK(io_buffer_vars_.size() == inputs_.size() + outputs_.size())
      << "Every IO Buffer var should be categorized either to be input or output";
  for (const Var& param : main_func_->params) {
    if (main_func_->buffer_map.find(param) != main_func_->buffer_map.end()) {
      Buffer param_buffer = main_func_->buffer_map[param];
      String io_annotation;
      if (inputs_.find(param_buffer->data) != inputs_.end()) {
        io_annotation = String(kInputTensorAllocate);
      } else {
        io_annotation = String(kOutputTensorAllocate);
      }
      main_body = Allocate(param_buffer->data, param_buffer->dtype, param_buffer->shape,
                           const_true(), main_body, {{io_annotation, param->name_hint}});
    } else {
      new_main_params.push_back(param);
    }
  }
  const GlobalVar& gv = mod_->GetGlobalVar(::tvm::runtime::symbol::tvm_module_main);
  mod_->Update(gv, PrimFunc(new_main_params, main_body, main_func_->ret_type,
                            main_func_->buffer_map, main_func_->attrs, main_func_->span));
  return mod_;
}

namespace transform {

tvm::transform::Pass CreateAllocatesForIO() {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return IOAllocateCreator(m)();
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.usmp.CreateAllocatesForIO", {});
}

TVM_REGISTER_GLOBAL("tir.usmp.transform.CreateAllocatesForIO").set_body_typed(CreateAllocatesForIO);

}  // namespace transform

}  // namespace usmp
}  // namespace tir
}  // namespace tvm
