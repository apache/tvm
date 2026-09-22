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

#include <tvm/ffi/cast.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/script/builder/frame.h>
#include <tvm/relax/script/builder/ir.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  RelaxFrameNode::RegisterReflection();
  SeqExprFrameNode::RegisterReflection();
  FunctionFrameNode::RegisterReflection();
  BindingBlockFrameNode::RegisterReflection();
  IfFrameNode::RegisterReflection();
  ThenFrameNode::RegisterReflection();
  ElseFrameNode::RegisterReflection();
}

void RelaxFrameNode::EnterWithScope() {
  span = IRBuilder::Current()->GetCurrentSourceSpan();
  IRBuilderFrameNode::EnterWithScope();
}

void SeqExprFrameNode::ExitWithScope() {
  // At this moment, there should be at most one BindingBlockFrame which hasn't ended. In this case,
  // call its `ExitBindingBlockFrame` and check if there is any more unended BindingBlockFrame.
  if (ffi::Optional<BindingBlockFrame> block_frame =
          IRBuilder::Current()->GetLastFrame<BindingBlockFrame>()) {
    block_frame.value()->ExitWithScope();
    TVM_FFI_CHECK(!IRBuilder::Current()->GetLastFrame<BindingBlockFrame>().has_value(), ValueError)
        << "There is some remaining BindingBlockFrame that is not properly popped out.";
  }
  RelaxFrameNode::ExitWithScope();
}

void SeqExprFrameNode::EnterWithScope() {
  RelaxFrameNode::EnterWithScope();
  BindingBlock()->EnterWithScope();
}

void FunctionFrameNode::EnterWithScope() {
  this->block_builder->BeginScope(params);
  if (declaration) {
    RelaxFrameNode::EnterWithScope();
  } else {
    SeqExprFrameNode::EnterWithScope();
  }
}

void FunctionFrameNode::ExitWithScope() {
  using ir::IRModuleFrame;
  using tvm::relax::Expr;
  IRBuilder builder = IRBuilder::Current();
  if (declaration) {
    TVM_FFI_CHECK(name.has_value(), ValueError) << "A function declaration requires a name";
    TVM_FFI_CHECK(local || builder->FindFrame<IRModuleFrame>().has_value(), ValueError)
        << "A function declaration requires an IRModule frame";
    RelaxFrameNode::ExitWithScope();
    block_builder->EndScope();
    function = tvm::relax::Function::CreateEmpty(params, ret_ty.value_or(tvm::relax::AnyType()),
                                                 is_pure.value_or(true), DictAttrs(attrs), span);
    if (local) {
      auto ty = tvm::relax::GetType(function.value());
      local_var = CheckBindingBlockFrameExistAndUnended()->is_dataflow
                      ? tvm::relax::DataflowVar(name.value(), ty, span)
                      : tvm::Var(name.value(), ty, span);
    } else {
      global_var = ir::DeclFunction(name.value(), function.value());
    }
    return;
  }
  SeqExprFrameNode::ExitWithScope();
  // Step 1: Create the function.
  TVM_FFI_CHECK(output.has_value(), ValueError)
      << "A Relax function must have a return value. Please use "
         "`return` to return an Expr";

  Expr body =
      this->block_builder->Normalize(tvm::relax::SeqExpr(binding_blocks, output.value(), span));
  // if the function is not private, add a global symbol to its attributes
  if (!is_private.value_or(false) && name.has_value() && !attrs.count(tvm::attr::kGlobalSymbol)) {
    attrs.Set(tvm::attr::kGlobalSymbol, name.value());
  }
  this->block_builder->EndScope();
  tvm::relax::Function func(/*params=*/params,
                            /*body=*/body,
                            /*ret_ty=*/ret_ty,
                            /*is_pure=*/is_pure.value_or(true),
                            /*attrs=*/DictAttrs(attrs),
                            /*span=*/span);
  function = func;
  // Step 2: Update IRModule.
  if (local) {
    TVM_FFI_CHECK(local_var.has_value(), ValueError)
        << "A local function definition requires its declared reference";
    bool recursive = false;
    for (const tvm::Var& var : tvm::relax::FreeVars(func)) {
      recursive = recursive || var.same_as(local_var.value());
    }
    // Retain the declared reference identity while publishing its inferred type,
    // just as DefFunction refines a module's global reference after definition.
    Type reference_type = tvm::relax::GetType(func);
    if (recursive) {
      // A recursive reference has its own provisional signature.  Its formal
      // primitive parameters must not bind the definition's parameter objects.
      // Keep lexical captures intact while renewing these signature binders.
      ffi::Map<tvm::Var, tvm::Expr> signature_params;
      for (const tvm::Var& param : params) {
        if (param.as<PrimVar>()) {
          signature_params.Set(param, param.CopyWithName(param->name));
        }
      }
      if (!signature_params.empty()) {
        auto signature = reference_type.as_or_throw<tvm::relax::FuncType>();
        auto bind_param = [&](const Type& ty) { return tvm::relax::Bind(ty, signature_params); };
        reference_type =
            tvm::relax::FuncType(signature->params.value().Map(bind_param),
                                 bind_param(signature->ret), signature->purity, signature->span);
      }
      if (local_var.value()->IsInstance<tvm::relax::DataflowVarNode>()) {
        // The self-reference is captured by the nested function, so it must
        // survive the dataflow region.  Region finalization rewrites every use
        // together, including recursive calls, to the same ordinary Var.
        CheckBindingBlockFrameExistAndUnended()->output_vars.push_back(local_var.value());
      }
    }
    local_var.value()->ty = reference_type;
    EmitVarBinding(tvm::relax::VarBinding(local_var.value(), func, span));
  } else if (!builder->HasConstructionFrames()) {
    // Case 0. No outer frame, return function directly
    TVM_FFI_CHECK(!builder->result.has_value(), ValueError)
        << "Builder.result has already been set";
    builder->result = func;
  } else if (ffi::Optional<IRModuleFrame> opt_frame = builder->FindFrame<IRModuleFrame>()) {
    // Case 1. A global function of an IRModule
    TVM_FFI_CHECK(name.has_value(), ValueError)
        << "The function name must be defined before exiting the "
           "function scope, if it's defined in a Module";
    const IRModuleFrame& frame = opt_frame.value();
    const ffi::String& func_name = name.value_or("");
    if (!frame->global_var_map.count(func_name)) {
      // First time visiting the function.
      ir::DeclFunction(func_name, func);
    }
    // Define the function.
    // Note we do checks to disallow redefinition of functions inside the `DefFunction`.
    ir::DefFunction(func_name, func);
    global_var = frame->global_var_map[func_name];
  } else {
    TVM_FFI_THROW(ValueError) << "Cannot find where to insert Relax.Function";
  }
}

void BindingBlockFrameNode::EnterWithScope() {
  // Step 1. If the last frame is a block frame. The start of a new block frame marks the end of the
  // last block frame.
  ffi::Optional<BindingBlockFrame> block_frame =
      IRBuilder::Current()->GetLastFrame<BindingBlockFrame>();
  if (block_frame.has_value()) {
    block_frame.value()->ExitWithScope();
    // Block frames cannot appear consecutively.
    TVM_FFI_ICHECK(!IRBuilder::Current()->GetLastFrame<BindingBlockFrame>());
  }
  // Step 2. Deal with the new block frame.
  RelaxFrameNode::EnterWithScope();
  ffi::Optional<FunctionFrame> func_frame = IRBuilder::Current()->FindFrame<FunctionFrame>();
  TVM_FFI_CHECK(func_frame.has_value(), ValueError)
      << "Cannot find FunctionFrame when creating BindingBlocks, Please ensure "
         "creating the block under Relax function scope.";
  const tvm::relax::BlockBuilder& block_builder = func_frame.value()->block_builder;
  if (is_dataflow) {
    block_builder->BeginDataflowBlock();
  } else {
    block_builder->BeginBindingBlock();
  }
}

class VarReplacer : public tvm::relax::ExprMutator {
 public:
  explicit VarReplacer(
      std::unordered_map<tvm::Var, tvm::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap) {
    var_remap_ = std::move(var_remap);
  }

  tvm::Var VisitVarDef(const tvm::Var& var) override {
    // ExprMutator only applies var_remap_ at usage sites.  This
    // applies var_remap_ at each definition site as well.
    if (auto it = var_remap_.find(var); it != var_remap_.end()) {
      return it->second;
    } else {
      return var;
    }
  }
};

void BindingBlockFrameNode::ExitWithScope() {
  // Step 1. Pop the current frame out of the frame stack.
  RelaxFrameNode::ExitWithScope();

  // Step 2. Get the constructed binding block from the block builder. The block should have at
  // lease one binding - otherwise, the block is not supposed to be created.
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  tvm::relax::BindingBlock block = block_builder->EndBlock();
  if (block->bindings.empty()) {
    return;
  }

  // Step 3. Rewrite the dataflow block.
  if (is_dataflow) {
    // Step 3.0.  Define a map to replace variables
    ffi::Array<tvm::Var> new_output_vars;
    std::unordered_map<tvm::Var, tvm::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap;
    for (const auto& output_var : output_vars) {
      if (var_remap.count(output_var)) {
        continue;
      }
      tvm::Var new_output_var(output_var->name, tvm::relax::GetType(output_var), output_var->span);
      new_output_vars.push_back(new_output_var);
      if (auto span = binding_spans.Get(output_var)) {
        binding_spans.Set(new_output_var, span.value());
      }
      var_remap[output_var] = new_output_var;
    }
    VarReplacer mutator(std::move(var_remap));

    // Step 3.1. Rewrite block binding
    block = mutator.VisitBindingBlock(block);

    // Step 3.3. Rewrite output vars
    output_vars = std::move(new_output_vars);

    // Step 3.4 Rewrite usage of output var, if any
    auto function = FindFunctionFrame("R.dataflow()");
    if (function->output.has_value()) {
      function->output = mutator.VisitExpr(function->output.value());
    }
  }

  // Variable rewriting may rebuild bindings, so attach their own source ranges last.
  block->span = span;
  for (const auto& binding : block->bindings) {
    if (auto span = binding_spans.Get(binding->var)) {
      binding->span = span.value();
    }
  }

  // Step 3. Get the last frame from the IRBuilder frame stack.
  ffi::Optional<RelaxFrame> opt_last_frame = IRBuilder::Current()->GetLastFrame<RelaxFrame>();
  TVM_FFI_ICHECK(opt_last_frame.has_value());
  RelaxFrame last_frame = opt_last_frame.value();

  // Step 4. Since we popped out any possible block frame when entering the "with" scope of the
  // current frame, the last frame cannot be a block frame.
  TVM_FFI_ICHECK(!last_frame->IsInstance<BindingBlockFrameNode>());

  // Step 5. Push the block frame into the corresponding field of the last frame.
  if (const auto* seq_frame = last_frame.as<SeqExprFrameNode>()) {
    auto frame = ffi::GetRef<SeqExprFrame>(seq_frame);
    frame->binding_blocks.push_back(block);
  } else {
    TVM_FFI_THROW(ValueError)
        << "Currently the last frame is supposed to be either a function frame "
           "or a block frame. However, the last frame is \""
        << last_frame->GetTypeKey() << "\".";
  }

  // Step 6. Start another binding block when a dataflow block ended.
  if (is_dataflow) {
    BindingBlock()->EnterWithScope();
  }
}

void IfFrameNode::EnterWithScope() {
  const ffi::Array<IRBuilderFrame>& frames = IRBuilder::Current()->frames;
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if ((*it)->IsInstance<FunctionFrameNode>()) {
      break;
    }
    const auto* block_frame = (*it).as<BindingBlockFrameNode>();
    if (block_frame && block_frame->is_dataflow) {
      TVM_FFI_THROW(ValueError) << "Cannot create an IfFrame inside a dataflow block.";
    }
  }
  RelaxFrameNode::EnterWithScope();
}

void IfFrameNode::ExitWithScope() {
  RelaxFrameNode::ExitWithScope();
  TVM_FFI_CHECK(then_expr.has_value(), ValueError)
      << "The body of then part is expected to be defined before exiting.";
  TVM_FFI_CHECK(else_expr.has_value(), ValueError)
      << "The body of else part is expected to be defined before exiting.";
  auto body = tvm::relax::If(condition, then_expr.value(), else_expr.value(), span);
  var = EmitWithSpan(body, std::nullopt, std::nullopt);
  IRBuilder::Name(var_name, var);
}

void ThenFrameNode::EnterWithScope() {
  IfFrame frame = FindIfFrame("R.Then");
  TVM_FFI_CHECK(!frame->then_expr.has_value(), ValueError)
      << "Duplicate then branch declaration, previous one is " << frame->then_expr.value();
  SeqExprFrameNode::EnterWithScope();
}

void ThenFrameNode::ExitWithScope() {
  SeqExprFrameNode::ExitWithScope();
  ffi::String var_name;
  output = GetSeqExprForBranch(ffi::GetRef<ThenFrame>(this), &var_name);
  IfFrame frame = FindIfFrame("R.Then");
  frame->then_expr = output;
  frame->var_name = var_name;
}

void ElseFrameNode::EnterWithScope() {
  IfFrame frame = FindIfFrame("R.Else");
  TVM_FFI_ICHECK(frame->then_expr.has_value()) << "The else branch should follow then branch";
  TVM_FFI_CHECK(!frame->else_expr.has_value(), ValueError)
      << "Duplicate else branch declaration, previous one is " << frame->else_expr.value();
  SeqExprFrameNode::EnterWithScope();
}

void ElseFrameNode::ExitWithScope() {
  SeqExprFrameNode::ExitWithScope();
  ffi::String var_name;
  output = GetSeqExprForBranch(ffi::GetRef<ElseFrame>(this), &var_name);
  IfFrame frame = FindIfFrame("R.Else");
  frame->else_expr = output;
  TVM_FFI_ICHECK(frame->var_name == var_name)
      << "This last binding of both branches must provide the same variable.  "
      << "However, the R.Then branch provides variable " << frame->var_name
      << ", while the R.Else branch provides variable " << var_name;
}

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
