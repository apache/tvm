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
#include <tvm/ffi/reflection/registry.h>

#include "./utils.h"

namespace tvm {
namespace tir {

TVM_FFI_STATIC_INIT_BLOCK() {
  InstructionKindNode::RegisterReflection();
  InstructionNode::RegisterReflection();
}

bool InstructionKindNode::IsPostproc() const {
  static InstructionKind inst_enter_postproc = InstructionKind::Get("EnterPostproc");
  return this == inst_enter_postproc.get();
}

Instruction::Instruction(InstructionKind kind, ffi::Array<Any> inputs, ffi::Array<Any> attrs,
                         ffi::Array<Any> outputs) {
  ObjectPtr<InstructionNode> n = ffi::make_object<InstructionNode>();
  n->kind = std::move(kind);
  n->inputs = std::move(inputs);
  n->attrs = std::move(attrs);
  n->outputs = std::move(outputs);
  this->data_ = std::move(n);
}

using InstructionKindRegistry = AttrRegistry<InstructionKindRegEntry, InstructionKind>;

InstructionKind InstructionKind::Get(const ffi::String& name) {
  const InstructionKindRegEntry* reg = InstructionKindRegistry::Global()->Get(name);
  ICHECK(reg != nullptr) << "AttributeError: Instruction kind " << name << " is not registered";
  return reg->inst_kind_;
}

InstructionKindRegEntry::InstructionKindRegEntry(uint32_t reg_index) {
  this->inst_kind_ = InstructionKind(ffi::make_object<InstructionKindNode>());
}

InstructionKindRegEntry& InstructionKindRegEntry::RegisterOrGet(const ffi::String& name) {
  return InstructionKindRegistry::Global()->RegisterOrGet(name);
}

/**************** Repr ****************/

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<InstructionNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* self = obj.as<InstructionNode>();
      ICHECK_NOTNULL(self);
      ffi::Array<Any> inputs;
      inputs.reserve(self->inputs.size());
      for (const Any& obj : self->inputs) {
        if (obj == nullptr) {
          inputs.push_back(ffi::String("None"));
        } else if (auto opt_str = obj.as<ffi::String>()) {
          inputs.push_back(ffi::String('"' + (*opt_str).operator std::string() + '"'));
        } else if (obj.as<BlockRVNode>() || obj.as<LoopRVNode>()) {
          inputs.push_back(ffi::String("_"));
        } else if (obj.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
          inputs.push_back(obj);
        } else if (obj.as<IntImmNode>() || obj.as<FloatImmNode>()) {
          inputs.push_back(obj);
        } else if (const auto* expr = obj.as<PrimExprNode>()) {
          PrimExpr new_expr = Substitute(
              ffi::GetRef<PrimExpr>(expr), [](const Var& var) -> ffi::Optional<PrimExpr> {
                ObjectPtr<VarNode> new_var = ffi::make_object<VarNode>(*var.get());
                new_var->name_hint = "_";
                return Var(new_var);
              });
          std::ostringstream os;
          os << new_expr;
          inputs.push_back(ffi::String(os.str()));
        } else if (obj.as<IndexMapNode>()) {
          inputs.push_back(obj);
        } else {
          LOG(FATAL) << "TypeError: Stringifying is not supported for type: " << obj.GetTypeKey();
          throw;
        }
      }
      p->stream << self->kind->f_as_python(
          /*inputs=*/inputs,
          /*attrs=*/self->attrs,
          /*decision=*/Any(nullptr),
          /*outputs=*/ffi::Array<ffi::String>(self->outputs.size(), ffi::String("_")));
    });

/**************** FFI ****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tir.schedule.InstructionKindGet", InstructionKind::Get)
      .def("tir.schedule.Instruction",
           [](InstructionKind kind, ffi::Array<Any> inputs, ffi::Array<Any> attrs,
              ffi::Array<Any> outputs) -> Instruction {
             return Instruction(kind, inputs, attrs, outputs);
           });
}

}  // namespace tir
}  // namespace tvm
