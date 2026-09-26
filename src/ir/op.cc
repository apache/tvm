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
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>

namespace tvm {

// Owns canonical Ops and mutable attribute columns for the lifetime of the process.
class OpRegistry {
 public:
  static OpRegistry* Global() {
    static auto* registry = new OpRegistry();
    return registry;
  }

  Op Get(const ffi::String& name) const {
    auto op = ops_.Get(name);
    TVM_FFI_CHECK(op.has_value(), AttributeError) << "Operator " << name << " is not registered";
    return *op;
  }

  Op GetOrCreate(const ffi::String& name) {
    if (auto op = ops_.Get(name)) return *op;
    auto node = ffi::make_object<OpNode>();
    node->name = name;
    node->index_ = static_cast<uint32_t>(ops_.size());
    Op op(std::move(node));
    ops_.Set(name, op);
    return op;
  }

  bool Contains(const ffi::String& name) const { return ops_.count(name); }

  ffi::Array<ffi::String> ListNames() const {
    ffi::Array<ffi::String> names;
    for (const auto& [name, op] : ops_) names.push_back(name);
    return names;
  }

  bool HasAttrMap(const ffi::String& name) const { return attrs_.count(name); }

  ffi::List<ffi::Any> GetAttrColumn(const ffi::String& name) const {
    auto column = attrs_.Get(name);
    TVM_FFI_CHECK(column.has_value(), InternalError) << "Attribute '" << name << "' is not registered";
    return *column;
  }

  void SetAttr(const Op& op, const ffi::String& name, ffi::Any value, bool override) {
    TVM_FFI_CHECK(value != nullptr, ValueError)
        << "Registered attribute is null for " << name << " of operator " << op->name;
    auto column = attrs_.Get(name).value_or(ffi::List<ffi::Any>());
    uint32_t index = op->index_;
    TVM_FFI_CHECK(override || index >= column.size() || column[index] == nullptr, ValueError)
        << "Attribute " << name << " of " << op->name << " is already registered";
    if (index >= column.size()) column.resize(index + 1);
    column.Set(index, std::move(value));
    // Store the same List object: cached views must survive both updates and growth.
    attrs_.Set(name, column);
  }

  void ResetAttr(const Op& op, const ffi::String& name) {
    if (auto column = attrs_.Get(name)) {
      if (op->index_ < column->size()) column->Set(op->index_, ffi::Any());
    }
  }

 private:
  ffi::Dict<ffi::String, Op> ops_;
  ffi::Dict<ffi::String, ffi::List<ffi::Any>> attrs_;
};

Op Op::Get(const ffi::String& name) { return OpRegistry::Global()->Get(name); }

ffi::Array<ffi::String> Op::ListNames() { return OpRegistry::Global()->ListNames(); }

bool Op::HasAttrMap(const ffi::String& name) { return OpRegistry::Global()->HasAttrMap(name); }

ffi::List<ffi::Any> Op::GetAttrColumn(const ffi::String& name) {
  return OpRegistry::Global()->GetAttrColumn(name);
}

OpDef::OpDef(const ffi::String& name) : op_(OpRegistry::Global()->GetOrCreate(name)) {}

OpDef::OpDef(const ffi::String& name, const ffi::String& doc) : OpDef(name) { get()->doc = doc; }

OpDef& OpDef::arg(const ffi::String& name, const ffi::String& type_schema, const ffi::String& doc) {
  auto node = ffi::make_object<ArgumentInfoNode>();
  node->name = name;
  node->type_schema = type_schema;
  node->doc = doc;
  get()->args_info.push_back(ArgumentInfo(std::move(node)));
  return *this;
}

OpDef& OpDef::ty_arg(const ffi::String& name, const ffi::String& type_schema,
                     const ffi::String& doc) {
  auto node = ffi::make_object<ArgumentInfoNode>();
  node->name = name;
  node->type_schema = type_schema;
  node->doc = doc;
  get()->ty_args_info.push_back(ArgumentInfo(std::move(node)));
  return *this;
}

OpDef& OpDef::set_attrs_type_key(const ffi::String& key) {
  uint32_t index = ffi::TypeKeyToIndex(key.c_str());
  get()->attrs_type_key = key;
  get()->attrs_type_index = index;
  return *this;
}

void OpDef::UpdateAttr(const ffi::String& name, ffi::Any value, bool override) {
  OpRegistry::Global()->SetAttr(op_, name, std::move(value), override);
}

OpDef& OpDef::reset_attr(const ffi::String& name) {
  OpRegistry::Global()->ResetAttr(op_, name);
  return *this;
}

void OpNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  refl::ObjectDef<OpNode>()
      .def_ro("name", &OpNode::name)
      .def_ro("doc", &OpNode::doc, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("args_info", &OpNode::args_info, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("attrs_type_key", &OpNode::attrs_type_key, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("allow_extra_args", &OpNode::allow_extra_args, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("ty_args_info", &OpNode::ty_args_info, refl::AttachFieldFlag::SEqHashIgnore())
      .def_static("get", &Op::Get,
                  "get(op_name: str) -> Op\n\nReturn the canonical named Op; raise AttributeError "
                  "if unregistered.")
      .def_static(
          "list_op_names", &Op::ListNames,
          "list_op_names() -> list[str]\n\nReturn registered operator names in unspecified order.")
      .def(
          "get_attr",
          [](Op op, ffi::String name) {
            return Op::GetAttrMap<ffi::Any>(name).get(op, ffi::Any());
          },
          "get_attr(attr_name: str) -> object\n\nReturn this Op's attribute or None; an "
          "unregistered column raises InternalError.")
      .def(
          "has_attr", [](Op, ffi::String name) { return Op::HasAttrMap(name); },
          "has_attr(attr_name: str) -> bool\n\nReturn whether the attribute column exists in the "
          "registry.")
      .def("_set_attr", [](Op op, ffi::String name, ffi::Any value,
                           bool override) { OpDef(op->name)
                               .set_attr(name, value, override); })
      .def(
          "reset_attr", [](Op op, ffi::String name) { OpDef(op->name)
              .reset_attr(name); },
          "reset_attr(attr_name: str) -> None\n\nRemove this Op's current value; missing values "
          "are ignored and cached views observe removal.")
      .def(
          "add_argument",
          [](Op op, ffi::String name, ffi::String type_schema, ffi::String doc) {
            OpDef(op->name)
                .arg(name, type_schema, doc);
          },
          "add_argument(name: str, type_schema: str, doc: str) -> None\n\nAppend an argument's "
          "name, JSON FFI type schema, and documentation without validating operands.")
      .def(
          "set_allow_extra_args", [](Op op) { OpDef(op->name)
              .allow_extra_args(); },
          "set_allow_extra_args() -> None\n\nAllow value arguments after the required prefix.")
      .def(
          "add_type_argument",
          [](Op op, ffi::String name, ffi::String type_schema, ffi::String doc) {
            OpDef(op->name)
                .ty_arg(name, type_schema, doc);
          },
          "add_type_argument(name: str, type_schema: str, doc: str) -> None\n\nAppend a "
          "type-argument "
          "name, JSON FFI type schema, and documentation without imposing a count rule.")
      .def(
          "set_attrs_type_key",
          [](Op op, ffi::String key) { OpDef(op->name)
              .set_attrs_type_key(key); },
          "set_attrs_type_key(key: str) -> None\n\nResolve and set the attribute object type key "
          "and runtime index together; an unknown key raises before updating.");
}

namespace {

TVMFFIAny OpVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // Ops are unique registry atoms.  Avoid reflecting through their registry metadata.
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny OpMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny OpMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  ArgumentInfoNode::RegisterReflection();
  OpNode::RegisterReflection();
  refl::TypeAttrDef<OpNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&OpVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&OpMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&OpMaybeInplaceMutate))
      .def("__data_to_json__", [](const OpNode* node) { return node->name; })
      .def("__data_from_json__", &Op::Get);
  refl::GlobalDef()
      .def("ir.RegisterOp",
           [](ffi::String name, ffi::String doc) {
             TVM_FFI_CHECK(!OpRegistry::Global()->Contains(name), AttributeError)
                 << "Operator " << name << " is registered before";
             OpDef(name, doc);
           })
      .def("ir.RegisterOpAttr", [](ffi::String name, ffi::String key, ffi::Any value, bool override) {
        OpDef(name)
            .set_attr(key, value, override);
      });
}

}  // namespace tvm
