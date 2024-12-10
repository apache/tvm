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
#include <tvm/relax/analysis.h>
#include <tvm/relax/struct_info.h>
#include <tvm/script/ir_builder/relax/ir.h>
#include <tvm/tir/op.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

///////////////////////////////// Vars //////////////////////////////////

using tvm::script::ir_builder::details::Namer;

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::relax::VarNode>([](const ObjectRef& node, String name) -> void {
      using tvm::relax::VarNode;
      using tvm::relax::IdNode;
      const VarNode* var = node.as<VarNode>();
      IdNode* vid = const_cast<IdNode*>(var->vid.get());
      vid->name_hint = name;
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::relax::DataflowVarNode>([](const ObjectRef& node, String name) -> void {
      using tvm::relax::DataflowVarNode;
      using tvm::relax::IdNode;
      const DataflowVarNode* var = node.as<DataflowVarNode>();
      IdNode* vid = const_cast<IdNode*>(var->vid.get());
      vid->name_hint = name;
    });

/////////////////////////////// Function ////////////////////////////////

FunctionFrame Function(const Bool& is_pure, const Bool& is_private) {
  ObjectPtr<FunctionFrameNode> n = make_object<FunctionFrameNode>();
  const IRBuilder& ir_builder = IRBuilder::Current();
  Optional<tvm::IRModule> mod = NullOpt;
  if (const Optional<ir::IRModuleFrame> mod_frame = ir_builder->GetLastFrame<ir::IRModuleFrame>()) {
    mod = tvm::IRModule(mod_frame.value()->functions);
  }
  n->block_builder = tvm::relax::BlockBuilder::Create(
      /*mod=*/mod, tvm::relax::BlockBuilder::DisableOperatorSpecificNormalizationForTVMScript());
  n->is_pure = is_pure;
  n->is_private = is_private;
  return FunctionFrame(n);
}

tvm::relax::Var Arg(const String& name, const tvm::relax::StructInfo& struct_info) {
  FunctionFrame frame = FindFunctionFrame("R.Arg");
  tvm::relax::Var var(name, struct_info);
  frame->params.push_back(var);
  frame->block_builder->AddDefinitionToScope(var);

  return var;
}

void FuncName(const String& name) {
  FunctionFrame frame = FindFunctionFrame("R.func_name");
  if (frame->name.defined()) {
    LOG(FATAL) << "ValueError: Duplicate function name, previous one is: \"" << frame->name.value()
               << "\"";
  }
  frame->name = name;
}

void FuncAttrs(Map<String, ObjectRef> attrs) {
  FunctionFrame frame = FindFunctionFrame("R.func_attr");
  for (const auto& [key, value] : attrs) {
    if (key == tvm::attr::kGlobalSymbol && frame->is_private.value_or(Bool(false))->value) {
      LOG(FATAL) << "ValueError: "
                 << "A private function may not have the kGlobalSymbol (\""
                 << tvm::attr::kGlobalSymbol << "\") attribute.  "
                 << "However, a private function specified the global symbol as " << value;
    }
    if (auto prev = frame->attrs.Get(key)) {
      LOG(FATAL) << "ValueError: "
                 << "Duplicate R.func_attr annotation for key = \"" << key << "\".  "
                 << "Previous value was " << prev.value() << ", with later definition as " << value;
    } else {
      frame->attrs.Set(key, value);
    }
  }
}

void FuncRetStructInfo(const tvm::relax::StructInfo& ret_sinfo) {
  FunctionFrame frame = FindFunctionFrame("R.func_ret_struct_info");
  if (frame->ret_struct_info.defined()) {
    LOG(FATAL) << "ValueError: Duplicate function return struct info, previous one is:\n "
               << frame->ret_struct_info.value();
  }
  frame->ret_struct_info = ret_sinfo;
}

void FuncRetValue(const tvm::relax::Expr& value) {
  // Step 0. Normalize the value.
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  tvm::relax::Expr normalized_value = block_builder->Normalize(value);

  IRBuilder ir_builder = IRBuilder::Current();

  // Step 1. The current Relax TVMScript syntax only allows function return appearing at the end of
  // a function body. Therefore if there is any unended block frame when dealing with function
  // return, we should end the block frame.

  if (auto opt = ir_builder->GetLastFrame<BlockFrame>()) {
    auto block_frame = opt.value();
    for (const auto& var : tvm::relax::FreeVars(normalized_value)) {
      if (var->IsInstance<tvm::relax::DataflowVarNode>()) {
        block_frame->output_vars.push_back(var);
      }
    }
  }
  // Step 2. Add the output value to the function frame.
  FunctionFrame frame = FindFunctionFrame("return");
  CHECK(!frame->output.defined())
      << "ValueError: "
      << "Relax functions do not support multiple return statement.  "
      << "However, return of " << normalized_value << " occurred after a return of "
      << frame->output << ".  "
      << "Please make sure function only has a single return statement, "
      << "which appears at the end of function.";

  frame->output = std::move(normalized_value);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Function").set_body_typed(Function);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.Arg").set_body_typed(Arg);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncName").set_body_typed(FuncName);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncAttrs").set_body_typed(FuncAttrs);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncRetStructInfo").set_body_typed(FuncRetStructInfo);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncRetValue").set_body_typed(FuncRetValue);

///////////////////////////// BindingBlock //////////////////////////////

BlockFrame Dataflow() {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->is_dataflow = true;
  n->block_ended = false;
  return BlockFrame(n);
}

BlockFrame BindingBlock() {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->is_dataflow = false;
  n->block_ended = false;
  return BlockFrame(n);
}

void DataflowBlockOutput(const Array<tvm::relax::Var>& vars) {
  // Step 1. Check that we're in a Dataflow block that is not ended.
  Optional<BlockFrame> block_frame = IRBuilder::Current()->GetLastFrame<BlockFrame>();
  CHECK(block_frame.defined() && block_frame.value()->is_dataflow)
      << "ValueError: `R.output` should appear inside a dataflow block. However, the current "
         "innermost block is not a dataflow block.";
  CHECK(!block_frame.value()->block_ended)
      << "ValueError: It is not allowed for a dataflow block to have multiple output operation.";

  // Step 2. Mark the block frame ended of construction, so that any followup binding after this
  // mark in the dataflow block will lead to an error.
  block_frame.value()->block_ended = true;

  // Step 3. All the output variables must be global variables and must be emitted by this dataflow
  // block.
  const Array<tvm::relax::Var>& emitted_vars = block_frame.value()->emitted_vars;
  for (const tvm::relax::Var& var : vars) {
    CHECK(std::find(emitted_vars.begin(), emitted_vars.end(), var) != emitted_vars.end())
        << "ValueError: An output variable is not emitted by this dataflow block. Please make sure "
           "all dataflow block output variables are emitted exactly by this block.";
    block_frame.value()->output_vars.push_back(var);
  }
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Dataflow").set_body_typed(Dataflow);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.BindingBlock").set_body_typed(BindingBlock);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.DataflowBlockOutput")
    .set_body_typed(DataflowBlockOutput);

/////////////////////////////// Bindings ///////////////////////////////

tvm::relax::Var Emit(const tvm::relax::Expr& expr,
                     const Optional<tvm::relax::StructInfo>& annotate_struct_info) {
  using tvm::relax::GetStructInfo;
  BlockFrame block_frame = CheckBlockFrameExistAndUnended();
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  if (annotate_struct_info.defined()) {
    const auto& sinfo = annotate_struct_info.value();
    if (!expr->struct_info_.defined()) {
      UpdateStructInfo(expr, sinfo);
    } else {
      CHECK(StructInfoBaseCheck(sinfo, GetStructInfo(expr)) != tvm::relax::BaseCheckResult::kFailL0)
          << "Invalid annotation. Got rhs value struct info: " << GetStructInfo(expr)
          << ", given struct info: " << sinfo;
    }
  }
  tvm::relax::Var var = block_builder->Emit(expr);
  block_frame->emitted_vars.push_back(var);
  return var;
}

tvm::relax::Var EmitMatchCast(const tvm::relax::Expr& value,
                              const tvm::relax::StructInfo& struct_info) {
  BlockFrame block_frame = CheckBlockFrameExistAndUnended();
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();

  tvm::relax::Var var = block_builder->EmitMatchCast(value, struct_info);
  block_frame->emitted_vars.push_back(var);
  return var;
}

tvm::relax::Var EmitVarBinding(const tvm::relax::VarBinding& binding) {
  BlockFrame block_frame = CheckBlockFrameExistAndUnended();
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  block_builder->EmitNormalized(binding);
  block_frame->emitted_vars.push_back(binding->var);
  return binding->var;
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Emit").set_body_typed(Emit);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.EmitMatchCast").set_body_typed(EmitMatchCast);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.EmitVarBinding").set_body_typed(EmitVarBinding);

/////////////////////////////// SeqExpr ///////////////////////////////

SeqExprFrame SeqExpr() {
  ObjectPtr<SeqExprFrameNode> n = make_object<SeqExprFrameNode>();
  return SeqExprFrame(n);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.SeqExpr").set_body_typed(SeqExpr);

///////////////////////////// If Then Else /////////////////////////////

IfFrame If(tvm::relax::Expr condition) {
  ObjectPtr<IfFrameNode> n = make_object<IfFrameNode>();
  n->condition = condition;
  n->then_expr = NullOpt;
  n->else_expr = NullOpt;
  return IfFrame(n);
}

ThenFrame Then() {
  ObjectPtr<ThenFrameNode> n = make_object<ThenFrameNode>();
  return ThenFrame(n);
}

ElseFrame Else() {
  ObjectPtr<ElseFrameNode> n = make_object<ElseFrameNode>();
  return ElseFrame(n);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.If").set_body_typed(If);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.Then").set_body_typed(Then);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.Else").set_body_typed(Else);

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
