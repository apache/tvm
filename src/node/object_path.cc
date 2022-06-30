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

#include <tvm/node/object_path.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <cstring>

using namespace tvm::runtime;

namespace tvm {

// ============== ObjectPathNode ==============

ObjectPathNode::ObjectPathNode(ObjectPathNode* parent)
    : parent_(GetRef<ObjectRef>(parent)), length_(parent == nullptr ? 1 : parent->length_ + 1) {}

// --- GetParent ---

ObjectPath ObjectPathNode::GetParent() const { return Downcast<ObjectPath>(parent_); }

TVM_REGISTER_GLOBAL("node.ObjectPathGetParent").set_body_typed([](const ObjectPath& path) {
  return path->GetParent();
});

// --- Length ---

size_t ObjectPathNode::Length() const { return length_; }

TVM_REGISTER_GLOBAL("node.ObjectPathLength").set_body_typed([](const ObjectPath& path) {
  return static_cast<int64_t>(path->Length());
});

// --- GetPrefix ---

ObjectPath ObjectPathNode::GetPrefix(size_t length) const {
  if (length > Length()) {
    throw std::out_of_range("Attempted to get a prefix longer than the path itself");
  }

  const ObjectPathNode* node = this;
  size_t suffix_len = Length() - length;
  for (size_t i = 0; i < suffix_len; ++i) {
    node = node->ParentNode();
  }

  return GetRef<ObjectPath>(node);
}

TVM_REGISTER_GLOBAL("node.ObjectPathGetPrefix")
    .set_body_typed([](const ObjectPath& path, int64_t length) {
      if (length < 0) {
        throw std::out_of_range("Prefix length can't be negative");
      }
      return path->GetPrefix(static_cast<size_t>(length));
    });

// --- IsPrefixOf ---

bool ObjectPathNode::IsPrefixOf(const ObjectPath& other) const {
  if (!other.defined()) {
    return false;
  }

  size_t this_len = Length();
  if (this_len > other->Length()) {
    return false;
  }
  return this->PathsEqual(other->GetPrefix(this_len));
}

TVM_REGISTER_GLOBAL("node.ObjectPathIsPrefixOf")
    .set_body_typed([](const ObjectPath& a, const ObjectPath& b) { return a->IsPrefixOf(b); });

// --- Attr ---

ObjectPath ObjectPathNode::Attr(const char* attr_key) {
  if (attr_key != nullptr) {
    return ObjectPath(make_object<AttributeAccessPathNode>(this, attr_key));
  } else {
    return ObjectPath(make_object<UnknownAttributeAccessPathNode>(this));
  }
}

ObjectPath ObjectPathNode::Attr(String attr_key) {
  if (attr_key.defined()) {
    return ObjectPath(make_object<AttributeAccessPathNode>(this, attr_key));
  } else {
    return ObjectPath(make_object<UnknownAttributeAccessPathNode>(this));
  }
}

TVM_REGISTER_GLOBAL("node.ObjectPathAttr")
    .set_body_typed([](const ObjectPath& path, Optional<String> attr_key) {
      return path->Attr(attr_key.defined() ? attr_key.value() : String(nullptr));
    });

// --- ArrayIndex ---

ObjectPath ObjectPathNode::ArrayIndex(size_t index) {
  return ObjectPath(make_object<ArrayIndexPathNode>(this, index));
}

TVM_REGISTER_GLOBAL("node.ObjectPathArrayIndex")
    .set_body_typed([](const ObjectPath& path, size_t index) { return path->ArrayIndex(index); });

// --- MissingArrayElement ---

ObjectPath ObjectPathNode::MissingArrayElement(size_t index) {
  return ObjectPath(make_object<MissingArrayElementPathNode>(this, index));
}

TVM_REGISTER_GLOBAL("node.ObjectPathMissingArrayElement")
    .set_body_typed([](const ObjectPath& path, size_t index) {
      return path->MissingArrayElement(index);
    });

// --- MapValue ---

ObjectPath ObjectPathNode::MapValue(ObjectRef key) {
  return ObjectPath(make_object<MapValuePathNode>(this, std::move(key)));
}

TVM_REGISTER_GLOBAL("node.ObjectPathMapValue")
    .set_body_typed([](const ObjectPath& path, const ObjectRef& key) {
      return path->MapValue(key);
    });

// --- MissingMapEntry ---

ObjectPath ObjectPathNode::MissingMapEntry() {
  return ObjectPath(make_object<MissingMapEntryPathNode>(this));
}

TVM_REGISTER_GLOBAL("node.ObjectPathMissingMapEntry").set_body_typed([](const ObjectPath& path) {
  return path->MissingMapEntry();
});

// --- PathsEqual ----

bool ObjectPathNode::PathsEqual(const ObjectPath& other) const {
  if (!other.defined() || Length() != other->Length()) {
    return false;
  }

  const ObjectPathNode* lhs = this;
  const ObjectPathNode* rhs = static_cast<const ObjectPathNode*>(other.get());

  while (lhs != nullptr && rhs != nullptr) {
    if (lhs->type_index() != rhs->type_index()) {
      return false;
    }
    if (!lhs->LastNodeEqual(rhs)) {
      return false;
    }
    lhs = lhs->ParentNode();
    rhs = rhs->ParentNode();
  }

  return lhs == nullptr && rhs == nullptr;
}

TVM_REGISTER_GLOBAL("node.ObjectPathEqual")
    .set_body_typed([](const ObjectPath& lhs, const ObjectPath& rhs) {
      return lhs->PathsEqual(rhs);
    });

// --- Repr ---

std::string GetObjectPathRepr(const ObjectPathNode* node) {
  std::string ret;
  while (node != nullptr) {
    std::string node_str = node->LastNodeString();
    ret.append(node_str.rbegin(), node_str.rend());
    node = static_cast<const ObjectPathNode*>(node->GetParent().get());
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

static void PrintObjectPathRepr(const ObjectRef& node, ReprPrinter* p) {
  p->stream << GetObjectPathRepr(static_cast<const ObjectPathNode*>(node.get()));
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable).set_dispatch<ObjectPathNode>(PrintObjectPathRepr);

// --- Private/protected methods ---

const ObjectPathNode* ObjectPathNode::ParentNode() const {
  return static_cast<const ObjectPathNode*>(parent_.get());
}

// ============== ObjectPath ==============

/* static */ ObjectPath ObjectPath::Root() { return ObjectPath(make_object<RootPathNode>()); }

TVM_REGISTER_GLOBAL("node.ObjectPathRoot").set_body_typed([]() { return ObjectPath::Root(); });

// ============== Individual path classes ==============

// ----- Root -----

RootPathNode::RootPathNode() : ObjectPathNode(nullptr) {}

bool RootPathNode::LastNodeEqual(const ObjectPathNode* other) const { return true; }

std::string RootPathNode::LastNodeString() const { return "<root>"; }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable).set_dispatch<RootPathNode>(PrintObjectPathRepr);

// ----- AttributeAccess -----

AttributeAccessPathNode::AttributeAccessPathNode(ObjectPathNode* parent, String attr_key)
    : ObjectPathNode(parent), attr_key(std::move(attr_key)) {}

bool AttributeAccessPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  const auto* otherAttrAccess = static_cast<const AttributeAccessPathNode*>(other);
  return attr_key == otherAttrAccess->attr_key;
}

std::string AttributeAccessPathNode::LastNodeString() const { return "." + attr_key; }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AttributeAccessPathNode>(PrintObjectPathRepr);

// ----- UnknownAttributeAccess -----

UnknownAttributeAccessPathNode::UnknownAttributeAccessPathNode(ObjectPathNode* parent)
    : ObjectPathNode(parent) {}

bool UnknownAttributeAccessPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  // Consider any two unknown attribute accesses unequal
  return false;
}

std::string UnknownAttributeAccessPathNode::LastNodeString() const {
  return ".<unknown attribute>";
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<UnknownAttributeAccessPathNode>(PrintObjectPathRepr);

// ----- ArrayIndexPath -----

ArrayIndexPathNode::ArrayIndexPathNode(ObjectPathNode* parent, size_t index)
    : ObjectPathNode(parent), index(index) {}

bool ArrayIndexPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  const auto* otherArrayIndex = static_cast<const ArrayIndexPathNode*>(other);
  return index == otherArrayIndex->index;
}

std::string ArrayIndexPathNode::LastNodeString() const { return "[" + std::to_string(index) + "]"; }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable).set_dispatch<ArrayIndexPathNode>(PrintObjectPathRepr);

// ----- MissingArrayElement -----

MissingArrayElementPathNode::MissingArrayElementPathNode(ObjectPathNode* parent, size_t index)
    : ObjectPathNode(parent), index(index) {}

bool MissingArrayElementPathNode::LastNodeEqual(const ObjectPathNode* other) const {
  const auto* otherMissingElement = static_cast<const MissingArrayElementPathNode*>(other);
  return index == otherMissingElement->index;
}

std::string MissingArrayElementPathNode::LastNodeString() const {
  return "[<missing element #" + std::to_string(index) + ">]";
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MissingArrayElementPathNode>(PrintObjectPathRepr);

// ----- MapValue -----

MapValuePathNode::MapValuePathNode(ObjectPathNode* parent, ObjectRef key)
    : ObjectPathNode(parent), key(std::move(key)) {}

bool MapValuePathNode::LastNodeEqual(const ObjectPathNode* other) const {
  const auto* otherMapValue = static_cast<const MapValuePathNode*>(other);
  return ObjectEqual()(key, otherMapValue->key);
}

std::string MapValuePathNode::LastNodeString() const {
  std::ostringstream s;
  s << "[" << key << "]";
  return s.str();
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable).set_dispatch<MapValuePathNode>(PrintObjectPathRepr);

// ----- MissingMapEntry -----

MissingMapEntryPathNode::MissingMapEntryPathNode(ObjectPathNode* parent) : ObjectPathNode(parent) {}

bool MissingMapEntryPathNode::LastNodeEqual(const ObjectPathNode* other) const { return true; }

std::string MissingMapEntryPathNode::LastNodeString() const { return "[<missing entry>]"; }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MissingMapEntryPathNode>(PrintObjectPathRepr);

}  // namespace tvm
