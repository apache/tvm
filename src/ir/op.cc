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
    TVM_FFI_CHECK(column.has_value(), InternalError)
        << "Attribute '" << name << "' is not registered";
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

void Op::ThrowInvalidCall(const OpNode* op) {
  TVM_FFI_THROW(TypeError) << "Op `" << op->name << "`: invalid Call";
}

bool Op::HasAttrMap(const ffi::String& name) { return OpRegistry::Global()->HasAttrMap(name); }

ffi::List<ffi::Any> Op::GetAttrColumn(const ffi::String& name) {
  return OpRegistry::Global()->GetAttrColumn(name);
}

OpDef::OpDef(const ffi::String& name) : op_(OpRegistry::Global()->GetOrCreate(name)) {}

OpDef::OpDef(const ffi::String& name, const ffi::String& doc) : OpDef(name) { get()->doc = doc; }

OpDef& OpDef::add_arg(const ffi::String& name, const ffi::String& doc) {
  auto node = ffi::make_object<ArgumentInfoNode>();
  node->name = name;
  node->doc = doc;
  get()->args_info.push_back(ArgumentInfo(std::move(node)));
  return *this;
}

OpDef& OpDef::add_ty_arg(const ffi::String& name, const ffi::String& doc) {
  auto node = ffi::make_object<ArgumentInfoNode>();
  node->name = name;
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

namespace {

ffi::Array<ArgumentInfo> MakeArgumentInfos(const ffi::Array<ffi::String>& names,
                                           const ffi::Array<ffi::String>& docs) {
  TVM_FFI_CHECK_EQ(names.size(), docs.size(), ValueError);
  ffi::Array<ArgumentInfo> infos;
  for (size_t i = 0; i < names.size(); ++i) {
    auto info = ffi::make_object<ArgumentInfoNode>();
    info->name = names[i];
    info->doc = docs[i];
    infos.push_back(ArgumentInfo(std::move(info)));
  }
  return infos;
}

ffi::Optional<ArgumentInfo> MakeTailInfo(const ffi::Array<ffi::String>& tail) {
  TVM_FFI_CHECK(tail.empty() || tail.size() == 2, ValueError)
      << "A variadic signature entry must have a name and documentation";
  if (tail.empty()) return std::nullopt;
  auto info = ffi::make_object<ArgumentInfoNode>();
  info->name = tail[0];
  info->doc = tail[1];
  return ArgumentInfo(std::move(info));
}

TVM_FFI_INLINE ffi::Expected<void> ValidateCountSignature(const CallNode* call) noexcept {
  const OpNode* op = call ? call->op.as<OpNode>() : nullptr;
  if (TVM_FFI_PREDICT_FALSE(!op)) {
    return TVM_FFI_UNEXPECTED(TypeError) << "Invalid Op Call";
  }
  if (TVM_FFI_PREDICT_FALSE(!call->args.defined() || !call->ty_args.defined())) {
    return TVM_FFI_UNEXPECTED(TypeError) << "Op `" << op->name << "`: invalid Call";
  }
  const size_t n_args = op->args_info.size();
  const bool var_args = op->var_args_info.has_value();
  if (TVM_FFI_PREDICT_FALSE(var_args ? call->args.size() < n_args : call->args.size() != n_args)) {
    return TVM_FFI_UNEXPECTED(TypeError)
           << "Op `" << op->name << "`: Call.args expected " << (var_args ? "at least " : "")
           << n_args << (n_args == 1 ? " argument" : " arguments") << ", got " << call->args.size();
  }
  const size_t n_ty_args = op->ty_args_info.size();
  const bool var_ty_args = op->var_ty_args_info.has_value();
  if (TVM_FFI_PREDICT_FALSE(var_ty_args ? call->ty_args.size() < n_ty_args
                                        : call->ty_args.size() != n_ty_args)) {
    return TVM_FFI_UNEXPECTED(TypeError)
           << "Op `" << op->name << "`: Call.ty_args expected " << (var_ty_args ? "at least " : "")
           << n_ty_args << (n_ty_args == 1 ? " type argument" : " type arguments") << ", got "
           << call->ty_args.size();
  }
  return {};
}

void SetOpSignature(Op op, const ffi::Array<ffi::String>& arg_names,
                    const ffi::Array<ffi::String>& arg_docs,
                    const ffi::Array<ffi::String>& ty_arg_names,
                    const ffi::Array<ffi::String>& ty_arg_docs,
                    const ffi::Array<ffi::String>& var_args,
                    const ffi::Array<ffi::String>& var_ty_args) {
  auto args_info = MakeArgumentInfos(arg_names, arg_docs);
  auto ty_args_info = MakeArgumentInfos(ty_arg_names, ty_arg_docs);
  auto var_args_info = MakeTailInfo(var_args);
  auto var_ty_args_info = MakeTailInfo(var_ty_args);
  auto* node = const_cast<OpNode*>(op.operator->());
  if (!node->validator_is_custom) {
    using View = ffi::reflection::NativeFunctionView<ffi::Expected<void>(const CallNode*)>;
    node->validator = ffi::reflection::NativeFunction<ffi::Expected<void>(const CallNode*)>::From(
        View::FromNative<&ValidateCountSignature>());
  }
  node->args_info = std::move(args_info);
  node->ty_args_info = std::move(ty_args_info);
  node->var_args_info = std::move(var_args_info);
  node->var_ty_args_info = std::move(var_ty_args_info);
  node->attrs_type_key = "";
  node->attrs_type_index = 0;
}

}  // namespace

void OpNode::RegisterReflection() {
  namespace refl = ffi::reflection;
  // clang-format off
  refl::ObjectDef<OpNode>()
      .def_ro("index", &OpNode::index_, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("name", &OpNode::name)
      .def_ro("doc", &OpNode::doc, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("args_info", &OpNode::args_info, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("attrs_type_key", &OpNode::attrs_type_key, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("ty_args_info", &OpNode::ty_args_info, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("var_args_info", &OpNode::var_args_info, refl::AttachFieldFlag::SEqHashIgnore())
      .def_ro("var_ty_args_info", &OpNode::var_ty_args_info, refl::AttachFieldFlag::SEqHashIgnore())
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
      .def("_set_attr", [](Op op, ffi::String name, ffi::Any value, bool override) {
        OpDef(op->name)
            .set_attr(name, value, override);
      })
      .def("_set_signature", &SetOpSignature)
      .def(
          "reset_attr", [](Op op, ffi::String name) {
            OpDef(op->name)
                .reset_attr(name);
          },
          "reset_attr(attr_name: str) -> None\n\nRemove this Op's current value; missing values "
          "are ignored and cached views observe removal.")
      .def(
          "add_arg",
          [](Op op, ffi::String name, ffi::String doc) {
            OpDef(op->name)
                .add_arg(name, doc);
          },
          "add_arg(name: str, doc: str) -> None\n\nAppend an argument's "
          "name and documentation.")
      .def(
          "add_ty_arg",
          [](Op op, ffi::String name, ffi::String doc) {
            OpDef(op->name)
                .add_ty_arg(name, doc);
          },
          "add_ty_arg(name: str, doc: str) -> None\n\nAppend a "
          "type-argument "
          "name and documentation without imposing a count rule.")
      .def(
          "set_attrs_type_key",
          [](Op op, ffi::String key) {
            OpDef(op->name)
                .set_attrs_type_key(key);
          },
          "set_attrs_type_key(key: str) -> None\n\nResolve and set the attribute object type key "
          "and runtime index together; an unknown key raises before updating.");
  // clang-format on
}

namespace {

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> OpVisit(ffi::StructuralVisitorObj*,
                                                                         ffi::AnyView) noexcept {
  // Ops are unique registry atoms.  Avoid reflecting through their registry metadata.
  return ffi::Optional<ffi::VisitInterrupt>(std::nullopt);
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> OpMutate(ffi::StructuralMutatorObj*,
                                                                  ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> OpMaybeInplaceMutate(
    ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ffi::reflection;
  ArgumentInfoNode::RegisterReflection();
  OpNode::RegisterReflection();
  refl::TypeAttrDef<OpNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&OpVisit>())
      .attr(refl::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&OpMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&OpMaybeInplaceMutate>())
      .def("__data_to_json__", [](const OpNode* node) { return node->name; })
      .def("__data_from_json__", &Op::Get);
  // clang-format off
  refl::GlobalDef()
      .def("ir.GetOp", &Op::Get)
      .def("ir.ListOpNames", &Op::ListNames)
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
  // clang-format on
}

}  // namespace tvm
