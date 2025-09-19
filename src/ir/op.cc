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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/module.h>
#include <tvm/tir/op_attr_types.h>

#include <memory>

#include "../node/attr_registry.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() { OpNode::RegisterReflection(); }

using ffi::Any;
using ffi::Function;
using ffi::PackedArgs;
using tir::FLowerIntrinsic;

using OpRegistry = AttrRegistry<OpRegEntry, Op>;

// find operator by name
const Op& Op::Get(const ffi::String& name) {
  const OpRegEntry* reg = OpRegistry::Global()->Get(name);
  ICHECK(reg != nullptr) << "AttributeError: Operator " << name << " is not registered";
  return reg->op();
}

OpRegEntry::OpRegEntry(uint32_t reg_index) {
  ObjectPtr<OpNode> n = ffi::make_object<OpNode>();
  n->index_ = reg_index;
  op_ = Op(n);
}

OpRegEntry& OpRegEntry::RegisterOrGet(const ffi::String& name) {
  return OpRegistry::Global()->RegisterOrGet(name);
}

// Get attribute map by key
const AttrRegistryMapContainerMap<Op>& Op::GetAttrMapContainer(const ffi::String& attr_name) {
  return OpRegistry::Global()->GetAttrMap(attr_name);
}

// Check if a key is present in the registry.
bool Op::HasAttrMap(const ffi::String& attr_name) {
  return OpRegistry::Global()->HasAttrMap(attr_name);
}

// Resets attr of the OpAttrMap.
void OpRegEntry::reset_attr(const std::string& attr_name) {
  OpRegistry::Global()->ResetAttr(attr_name, op_);
}

void OpRegEntry::UpdateAttr(const ffi::String& key, ffi::Any value, int plevel) {
  OpRegistry::Global()->UpdateAttr(key, op_, value, plevel);
}

// Frontend APIs
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.ListOpNames", []() { return OpRegistry::Global()->ListAllNames(); })
      .def("ir.GetOp", [](ffi::String name) -> Op { return Op::Get(name); })
      .def("ir.OpGetAttr",
           [](Op op, ffi::String attr_name) -> ffi::Any {
             auto op_map = Op::GetAttrMap<ffi::Any>(attr_name);
             ffi::Any rv;
             if (op_map.count(op)) {
               rv = op_map[op];
             }
             return rv;
           })
      .def("ir.OpHasAttr",
           [](Op op, ffi::String attr_name) -> bool { return Op::HasAttrMap(attr_name); })
      .def("ir.OpSetAttr",
           [](Op op, ffi::String attr_name, ffi::AnyView value, int plevel) {
             auto& reg = OpRegistry::Global()->RegisterOrGet(op->name).set_name();
             reg.set_attr(attr_name, value, plevel);
           })
      .def("ir.OpResetAttr",
           [](Op op, ffi::String attr_name) {
             auto& reg = OpRegistry::Global()->RegisterOrGet(op->name);
             reg.reset_attr(attr_name);
           })
      .def("ir.RegisterOp",
           [](ffi::String op_name, ffi::String descr) {
             const OpRegEntry* reg = OpRegistry::Global()->Get(op_name);
             ICHECK(reg == nullptr)
                 << "AttributeError: Operator " << op_name << " is registered before";
             auto& op = OpRegistry::Global()->RegisterOrGet(op_name).set_name();
             op.describe(descr);
           })
      .def("ir.OpAddArgument",
           [](Op op, ffi::String name, ffi::String type, ffi::String description) {
             auto& reg = OpRegistry::Global()->RegisterOrGet(op->name).set_name();
             reg.add_argument(name, type, description);
           })
      .def("ir.OpSetSupportLevel",
           [](Op op, int level) {
             auto& reg = OpRegistry::Global()->RegisterOrGet(op->name).set_name();
             reg.set_support_level(level);
           })
      .def("ir.OpSetNumInputs",
           [](Op op, int n) {
             auto& reg = OpRegistry::Global()->RegisterOrGet(op->name).set_name();
             reg.set_num_inputs(n);
           })
      .def("ir.OpSetAttrsTypeKey",
           [](Op op, ffi::String key) {
             auto& reg = OpRegistry::Global()->RegisterOrGet(op->name).set_name();
             reg.set_attrs_type_key(key);
           })
      .def("ir.RegisterOpAttr",
           [](ffi::String op_name, ffi::String attr_key, ffi::AnyView value, int plevel) {
             auto& reg = OpRegistry::Global()->RegisterOrGet(op_name).set_name();
             // enable resgiteration and override of certain properties
             if (attr_key == "num_inputs" && plevel > 128) {
               reg.set_num_inputs(value.cast<int>());
             } else if (attr_key == "attrs_type_key" && plevel > 128) {
               LOG(FATAL) << "attrs type key no longer supported";
             } else {
               reg.set_attr(attr_key, value, plevel);
             }
           })
      .def("ir.RegisterOpLowerIntrinsic",
           [](ffi::String name, ffi::Function f, ffi::String target, int plevel) {
             tvm::OpRegEntry::RegisterOrGet(name).set_attr<FLowerIntrinsic>(
                 target + ".FLowerIntrinsic", f, plevel);
           });
  // override OpNode to use name as the repr
  refl::TypeAttrDef<OpNode>()
      .def("__data_to_json__",
           [](const OpNode* node) -> ffi::String {
             // simply save as the string
             return node->name;
           })
      .def("__data_from_json__", [](const ffi::String& name) -> Op { return Op::Get(name); });
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<OpNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const OpNode*>(ref.get());
      p->stream << "Op(" << node->name << ")";
    });

}  // namespace tvm
