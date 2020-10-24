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
 * \file src/ir/op.cc
 * \brief Primitive operators and intrinsics.
 */
#include <tvm/ir/op.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>

#include <memory>

#include "../node/attr_registry.h"

namespace tvm {

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

using OpRegistry = AttrRegistry<OpRegEntry, Op>;

// find operator by name
const Op& Op::Get(const String& name) {
  const OpRegEntry* reg = OpRegistry::Global()->Get(name);
  ICHECK(reg != nullptr) << "AttributeError: Operator " << name << " is not registered";
  return reg->op();
}

OpRegEntry::OpRegEntry(uint32_t reg_index) {
  ObjectPtr<OpNode> n = make_object<OpNode>();
  n->index_ = reg_index;
  op_ = Op(n);
}

OpRegEntry& OpRegEntry::RegisterOrGet(const String& name) {
  return OpRegistry::Global()->RegisterOrGet(name);
}

// Get attribute map by key
const AttrRegistryMapContainerMap<Op>& Op::GetAttrMapContainer(const String& attr_name) {
  return OpRegistry::Global()->GetAttrMap(attr_name);
}

// Check if a key is present in the registry.
bool Op::HasAttrMap(const String& attr_name) { return OpRegistry::Global()->HasAttrMap(attr_name); }

// Resets attr of the OpAttrMap.
void OpRegEntry::reset_attr(const std::string& attr_name) {
  OpRegistry::Global()->ResetAttr(attr_name, op_);
}

void OpRegEntry::UpdateAttr(const String& key, TVMRetValue value, int plevel) {
  OpRegistry::Global()->UpdateAttr(key, op_, value, plevel);
}

// Frontend APIs
TVM_REGISTER_GLOBAL("ir.ListOpNames").set_body_typed([]() {
  return OpRegistry::Global()->ListAllNames();
});

TVM_REGISTER_GLOBAL("ir.GetOp").set_body_typed([](String name) -> Op { return Op::Get(name); });

TVM_REGISTER_GLOBAL("ir.OpGetAttr").set_body_typed([](Op op, String attr_name) -> TVMRetValue {
  auto op_map = Op::GetAttrMap<TVMRetValue>(attr_name);
  TVMRetValue rv;
  if (op_map.count(op)) {
    rv = op_map[op];
  }
  return rv;
});

TVM_REGISTER_GLOBAL("ir.OpSetAttr")
    .set_body_typed([](Op op, String attr_name, runtime::TVMArgValue value, int plevel) {
      auto& reg = OpRegistry::Global()->RegisterOrGet(op->name).set_name();
      reg.set_attr(attr_name, value, plevel);
    });

TVM_REGISTER_GLOBAL("ir.OpResetAttr").set_body_typed([](Op op, String attr_name) {
  auto& reg = OpRegistry::Global()->RegisterOrGet(op->name);
  reg.reset_attr(attr_name);
});

TVM_REGISTER_GLOBAL("ir.RegisterOpAttr")
    .set_body_typed([](String op_name, String attr_key, runtime::TVMArgValue value, int plevel) {
      auto& reg = OpRegistry::Global()->RegisterOrGet(op_name).set_name();
      // enable resgiteration and override of certain properties
      if (attr_key == "num_inputs" && plevel > 128) {
        reg.set_num_inputs(value);
      } else if (attr_key == "attrs_type_key" && plevel > 128) {
        LOG(FATAL) << "attrs type key no longer supported";
      } else {
        // normal attr table override.
        if (value.type_code() == kTVMPackedFuncHandle) {
          // do an eager copy of the PackedFunc
          PackedFunc f = value;
          // If we get a function from frontend, avoid deleting it.
          auto* fcopy = new PackedFunc(f);
          reg.set_attr(attr_key, *fcopy, plevel);
        } else {
          reg.set_attr(attr_key, value, plevel);
        }
      }
    });

// helper to get internal dev function in objectref.
struct Op2ObjectPtr : public ObjectRef {
  static ObjectPtr<Object> Get(const Op& op) { return GetDataPtr<Object>(op); }
};

ObjectPtr<Object> CreateOp(const std::string& name) {
  // Hack use TVMRetValue as exchange
  auto op = Op::Get(name);
  ICHECK(op.defined()) << "Cannot find op \'" << name << '\'';
  return Op2ObjectPtr::Get(op);
}

TVM_REGISTER_NODE_TYPE(OpNode).set_creator(CreateOp).set_repr_bytes(
    [](const Object* n) -> std::string { return static_cast<const OpNode*>(n)->name; });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<OpNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const OpNode*>(ref.get());
      p->stream << "Op(" << node->name << ")";
    });

}  // namespace tvm
