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
#include "./utils.h"

namespace tvm {
namespace tir {

bool InstructionKindNode::IsPostproc() const {
  static InstructionKind inst_enter_postproc = InstructionKind::Get("EnterPostproc");
  return this == inst_enter_postproc.get();
}

Instruction::Instruction(InstructionKind kind, Array<ObjectRef> inputs, Array<ObjectRef> attrs,
                         Array<ObjectRef> outputs) {
  ObjectPtr<InstructionNode> n = make_object<InstructionNode>();
  n->kind = std::move(kind);
  n->inputs = std::move(inputs);
  n->attrs = std::move(attrs);
  n->outputs = std::move(outputs);
  this->data_ = std::move(n);
}

using InstructionKindRegistry = AttrRegistry<InstructionKindRegEntry, InstructionKind>;

InstructionKind InstructionKind::Get(const String& name) {
  const InstructionKindRegEntry* reg = InstructionKindRegistry::Global()->Get(name);
  ICHECK(reg != nullptr) << "AttributeError: Instruction kind " << name << " is not registered";
  return reg->inst_kind_;
}

InstructionKindRegEntry::InstructionKindRegEntry(uint32_t reg_index) {
  this->inst_kind_ = InstructionKind(make_object<InstructionKindNode>());
}

InstructionKindRegEntry& InstructionKindRegEntry::RegisterOrGet(const String& name) {
  return InstructionKindRegistry::Global()->RegisterOrGet(name);
}

/**************** Repr ****************/

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<InstructionNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* self = obj.as<InstructionNode>();
      ICHECK_NOTNULL(self);
      Array<ObjectRef> inputs;
      inputs.reserve(self->inputs.size());
      for (const ObjectRef& obj : self->inputs) {
        if (!obj.defined()) {
          inputs.push_back(String("None"));
        } else if (obj->IsInstance<BlockRVNode>() || obj->IsInstance<LoopRVNode>()) {
          inputs.push_back(String("_"));
        } else if (const auto* str_obj = obj.as<StringObj>()) {
          inputs.push_back(String('"' + std::string(str_obj->data) + '"'));
        } else if (obj->IsInstance<IntImmNode>() || obj->IsInstance<FloatImmNode>()) {
          inputs.push_back(obj);
        } else if (const auto* expr = obj.as<PrimExprNode>()) {
          PrimExpr new_expr =
              Substitute(GetRef<PrimExpr>(expr), [](const Var& var) -> Optional<PrimExpr> {
                ObjectPtr<VarNode> new_var = make_object<VarNode>(*var.get());
                new_var->name_hint = "_";
                return Var(new_var);
              });
          std::ostringstream os;
          os << new_expr;
          inputs.push_back(String(os.str()));
        } else if (obj.as<IndexMapNode>()) {
          inputs.push_back(obj);
        } else {
          LOG(FATAL) << "TypeError: Stringifying is not supported for type: " << obj->GetTypeKey();
          throw;
        }
      }
      p->stream << self->kind->f_as_python(
          /*inputs=*/inputs,
          /*attrs=*/self->attrs,
          /*decision=*/NullOpt,
          /*outputs=*/Array<String>(self->outputs.size(), String("_")));
    });

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(InstructionNode);
TVM_REGISTER_NODE_TYPE(InstructionKindNode);

TVM_REGISTER_GLOBAL("tir.schedule.InstructionKindGet").set_body_typed(InstructionKind::Get);
TVM_REGISTER_GLOBAL("tir.schedule.Instruction")
    .set_body_typed([](InstructionKind kind, Array<ObjectRef> inputs, Array<ObjectRef> attrs,
                       Array<ObjectRef> outputs) -> Instruction {
      return Instruction(kind, inputs, attrs, outputs);
    });

}  // namespace tir
}  // namespace tvm
