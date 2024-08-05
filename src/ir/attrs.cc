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
 * \file attrs.cc
 */
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>

#include "attr_functor.h"

namespace tvm {

void DictAttrsNode::VisitAttrs(AttrVisitor* v) { v->Visit("__dict__", &dict); }

void DictAttrsNode::VisitNonDefaultAttrs(AttrVisitor* v) { v->Visit("__dict__", &dict); }

namespace {

/* \brief  Normalize attributes from runtime types to Relax IR types
 *
 * While conversion from `tvm::runtime` types to compile-time IR
 * types usually occurs as part of FFI conversions, the attributes
 * are not converted, as they are stored in a `Map<String,
 * ObjectRef>`.  While this is required to allow attribute values to
 * contain `ObjectRef` instances that are not IR expressions, the
 * conversion should still be applied when possible.
 *
 * \param obj The IR attribute value to be normalized
 *
 * \return The normalized attribute value
 */
ObjectRef NormalizeAttr(ObjectRef obj) {
  if (auto dict_attrs = obj.as<DictAttrsNode>()) {
    auto new_dict = Downcast<Map<String, ObjectRef>>(NormalizeAttr(dict_attrs->dict));
    if (new_dict.same_as(dict_attrs->dict)) {
      return obj;
    } else {
      return DictAttrs(new_dict);
    }
  } else if (auto runtime_bool = obj.as<runtime::Bool::ContainerType>()) {
    return Bool(runtime_bool->value);
  } else if (auto runtime_int = obj.as<runtime::Int::ContainerType>()) {
    return Integer(runtime_int->value);
  } else if (auto opt_array = obj.as<Array<ObjectRef>>()) {
    return opt_array.value().Map([](const ObjectRef& inner) { return NormalizeAttr(inner); });
  } else if (auto opt_map = obj.as<Map<String, ObjectRef>>()) {
    auto map = opt_map.value();

    Map<String, ObjectRef> updates;
    for (const auto& [key, inner] : map) {
      auto new_inner = NormalizeAttr(inner);
      if (!new_inner.same_as(inner)) {
        updates.Set(key, new_inner);
      }
    }
    for (const auto& [key, new_inner] : updates) {
      map.Set(key, new_inner);
    }

    return map;

  } else {
    return obj;
  }
}
}  // namespace

DictAttrs WithAttrs(DictAttrs attrs, Map<String, ObjectRef> new_attrs) {
  if (new_attrs.empty()) {
    return attrs;
  }

  auto* write_ptr = attrs.CopyOnWrite();
  Map<String, ObjectRef> attr_dict = std::move(write_ptr->dict);

  for (const auto& [key, value] : new_attrs) {
    attr_dict.Set(key, NormalizeAttr(value));
  }

  write_ptr->dict = std::move(attr_dict);
  return attrs;
}

DictAttrs WithAttr(DictAttrs attrs, String key, ObjectRef value) {
  auto* write_ptr = attrs.CopyOnWrite();
  Map<String, ObjectRef> attr_dict = std::move(write_ptr->dict);
  attr_dict.Set(key, NormalizeAttr(value));

  write_ptr->dict = std::move(attr_dict);
  return attrs;
}

DictAttrs WithoutAttr(DictAttrs attrs, const std::string& key) {
  auto* write_ptr = attrs.CopyOnWrite();
  Map<String, ObjectRef> attr_dict = std::move(write_ptr->dict);
  attr_dict.erase(key);

  write_ptr->dict = std::move(attr_dict);
  return attrs;
}

void DictAttrsNode::InitByPackedArgs(const runtime::TVMArgs& args, bool allow_unknown) {
  for (int i = 0; i < args.size(); i += 2) {
    std::string key = args[i];
    runtime::TVMArgValue val = args[i + 1];
    if (val.IsObjectRef<ObjectRef>()) {
      dict.Set(key, val.operator ObjectRef());
    } else if (String::CanConvertFrom(val)) {
      dict.Set(key, val.operator String());
    } else {
      dict.Set(key, val.operator PrimExpr());
    }
  }

  dict = Downcast<Map<String, ObjectRef>>(NormalizeAttr(dict));
}

Array<AttrFieldInfo> DictAttrsNode::ListFieldInfo() const { return {}; }

DictAttrs::DictAttrs(Map<String, ObjectRef> dict) {
  dict = Downcast<Map<String, ObjectRef>>(NormalizeAttr(dict));

  ObjectPtr<DictAttrsNode> n = make_object<DictAttrsNode>();
  n->dict = std::move(dict);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DictAttrsNode);

TVM_REGISTER_NODE_TYPE(AttrFieldInfoNode);

TVM_REGISTER_GLOBAL("ir.DictAttrsGetDict").set_body_typed([](DictAttrs attrs) {
  return attrs->dict;
});

TVM_REGISTER_GLOBAL("ir.AttrsListFieldInfo").set_body_typed([](Attrs attrs) {
  return attrs->ListFieldInfo();
});

}  // namespace tvm
