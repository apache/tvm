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
 *  Copyright (c) 2018 by Contributors
 * \file attrs.cc
 */
#include <tvm/attrs.h>
#include <tvm/api_registry.h>
#include "attr_functor.h"

namespace tvm {

void DictAttrsNode::VisitAttrs(AttrVisitor* v)  {
  v->Visit("__dict__", &dict);
}

void DictAttrsNode::VisitNonDefaultAttrs(AttrVisitor* v) {
  v->Visit("__dict__", &dict);
}

void DictAttrsNode::InitByPackedArgs(
    const runtime::TVMArgs& args, bool allow_unknown) {
  for (int i = 0; i < args.size(); i += 2) {
    std::string key = args[i];
    runtime::TVMArgValue val = args[i + 1];
    if (val.type_code() == kNodeHandle) {
      dict.Set(key, val.operator NodeRef());
    } else if (val.type_code() == kStr) {
      dict.Set(key, Expr(val.operator std::string()));
    } else {
      dict.Set(key, val.operator Expr());
    }
  }
}

Array<AttrFieldInfo> DictAttrsNode::ListFieldInfo() const {
  return {};
}

Attrs DictAttrsNode::make(Map<std::string, NodeRef> dict) {
  NodePtr<DictAttrsNode> n = make_node<DictAttrsNode>();
  n->dict = std::move(dict);
  return Attrs(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<DictAttrsNode>([](const DictAttrsNode *op, IRPrinter *p) {
    p->stream << op->dict;
});

TVM_REGISTER_NODE_TYPE(DictAttrsNode);

TVM_REGISTER_NODE_TYPE(AttrFieldInfoNode);


using namespace ir;
// Equal handler.
bool AttrsEqualHandler::Equal(const NodeRef& lhs, const NodeRef& rhs) {
  if (lhs.same_as(rhs)) return true;
  if (!lhs.defined() || !rhs.defined()) return false;
  return this->VisitAttr(lhs, rhs);
}

bool AttrsEqualHandler::VisitAttrDefault_(const Node* lhs, const NodeRef& other) {
  if (lhs->derived_from<BaseAttrsNode>()) {
    AttrsEqual equal;
    equal.handler_ = this;
    return static_cast<const BaseAttrsNode*>(lhs)->ContentEqual(
        other.get(), equal);
  }
  return lhs == other.get();
}

bool AttrsEqualHandler::VisitAttr_(const IntImm* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<IntImm>()) {
    return lhs->value == rhs->value;
  }
  return false;
}

bool AttrsEqualHandler::VisitAttr_(const UIntImm* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<UIntImm>()) {
    return lhs->value == rhs->value;
  }
  return false;
}

bool AttrsEqualHandler::VisitAttr_(const FloatImm* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<FloatImm>()) {
    return lhs->value == rhs->value;
  }
  return false;
}

bool AttrsEqualHandler::VisitAttr_(const StringImm* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<StringImm>()) {
    return lhs->value == rhs->value;
  }
  return false;
}

bool AttrsEqualHandler::VisitAttr_(const ArrayNode* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<ArrayNode>()) {
    if (rhs->data.size() != lhs->data.size()) return false;
    for (size_t  i = 0; i < lhs->data.size(); ++i) {
      if (!Equal(NodeRef(lhs->data[i]), NodeRef(rhs->data[i]))) return false;
    }
  }
  return true;
}

bool AttrsEqualHandler::VisitAttr_(const StrMapNode* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<StrMapNode>()) {
    if (rhs->data.size() != lhs->data.size()) return false;
    for (const auto& kv : lhs->data) {
      auto it = rhs->data.find(kv.first);
      if (it == rhs->data.end()) return false;
      if (!Equal(NodeRef(kv.second), NodeRef(it->second))) return false;
    }
  }
  return true;
}

#define TVM_DEFINE_ATTRS_BINOP_EQUAL(NodeName)                          \
  bool AttrsEqualHandler::VisitAttr_(const NodeName* lhs, const NodeRef& other) { \
    if (const auto* rhs = other.as<NodeName>()) {                       \
      if (!Equal(lhs->a, rhs->a)) return false;                         \
      if (!Equal(lhs->b, rhs->b)) return false;                         \
      return true;                                                      \
    } else {                                                            \
      return false;                                                     \
    }                                                                   \
  }                                                                     \

TVM_DEFINE_ATTRS_BINOP_EQUAL(Add);
TVM_DEFINE_ATTRS_BINOP_EQUAL(Sub);
TVM_DEFINE_ATTRS_BINOP_EQUAL(Mul);
TVM_DEFINE_ATTRS_BINOP_EQUAL(Div);
TVM_DEFINE_ATTRS_BINOP_EQUAL(Mod);
TVM_DEFINE_ATTRS_BINOP_EQUAL(FloorDiv);
TVM_DEFINE_ATTRS_BINOP_EQUAL(FloorMod);
TVM_DEFINE_ATTRS_BINOP_EQUAL(Max);
TVM_DEFINE_ATTRS_BINOP_EQUAL(Min);
TVM_DEFINE_ATTRS_BINOP_EQUAL(GE);
TVM_DEFINE_ATTRS_BINOP_EQUAL(GT);
TVM_DEFINE_ATTRS_BINOP_EQUAL(LE);
TVM_DEFINE_ATTRS_BINOP_EQUAL(LT);
TVM_DEFINE_ATTRS_BINOP_EQUAL(EQ);
TVM_DEFINE_ATTRS_BINOP_EQUAL(NE);
TVM_DEFINE_ATTRS_BINOP_EQUAL(And);
TVM_DEFINE_ATTRS_BINOP_EQUAL(Or);

bool AttrsEqualHandler::VisitAttr_(const Not* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<Not>()) {
    return Equal(lhs->a, rhs->a);
  } else {
    return false;
  }
}

bool AttrsEqualHandler::VisitAttr_(const Cast* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<Cast>()) {
    if (lhs->type != rhs->type) return false;
    return Equal(lhs->value, rhs->value);
  } else {
    return false;
  }
}

bool AttrsEqualHandler::VisitAttr_(const Call* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<Call>()) {
    return
        lhs->name == rhs->name &&
        lhs->type == rhs->type &&
        lhs->call_type == rhs->call_type &&
        Equal(lhs->args, rhs->args);
  } else {
    return false;
  }
}

bool AttrsEqualHandler::VisitAttr_(const Select* lhs, const NodeRef& other) {
  if (const auto* rhs = other.as<Select>()) {
    return
        Equal(lhs->condition, rhs->condition) &&
        Equal(lhs->true_value, rhs->true_value) &&
        Equal(lhs->false_value, rhs->false_value);
  } else {
    return false;
  }
}

// Hash Handler.
size_t AttrsHashHandler::VisitAttrDefault_(const Node* value) {
  if (value->derived_from<BaseAttrsNode>()) {
    AttrsHash hasher;
    hasher.handler_ = this;
    return static_cast<const BaseAttrsNode*>(value)->ContentHash(hasher);
  } else {
    return NodeHash()(GetRef<NodeRef>(value));
  }
}

size_t AttrsHashHandler::VisitAttr_(const IntImm* op) {
  return std::hash<int64_t>()(op->value);
}

size_t AttrsHashHandler::VisitAttr_(const UIntImm* op) {
  return std::hash<uint64_t>()(op->value);
}

size_t AttrsHashHandler::VisitAttr_(const FloatImm* op) {
  return std::hash<double>()(op->value);
}

size_t AttrsHashHandler::VisitAttr_(const StringImm* op) {
  return std::hash<std::string>()(op->value);
}

size_t AttrsHashHandler::VisitAttr_(const ArrayNode* op) {
  size_t result = op->data.size();
  for (size_t  i = 0; i < op->data.size(); ++i) {
    result = Combine(result, this->Hash(NodeRef(op->data[i])));
  }
  return result;
}

size_t AttrsHashHandler::VisitAttr_(const StrMapNode* lhs) {
    using Entry = std::pair<std::string, NodePtr<Node> >;
    std::vector<Entry> data(lhs->data.begin(), lhs->data.end());
    std::sort(data.begin(), data.end(), [](const Entry& a, const Entry& b) {
        return a.first < b.first;
      });
    size_t result = 0;
    for (const Entry& kv : data) {
      result = Combine(result, std::hash<std::string>()(kv.first));
      result = Combine(result, this->Hash(NodeRef(kv.second)));
    }
    return result;
}


#define TVM_DEFINE_ATTRS_BINOP_HASH(NodeName)                           \
  size_t AttrsHashHandler::VisitAttr_(const NodeName* op) {             \
    static size_t key = std::hash<std::string>()(NodeName::_type_key);  \
    return Combine(key, Combine(Hash(op->a), Hash(op->b)));             \
  }                                                                     \

TVM_DEFINE_ATTRS_BINOP_HASH(Add);
TVM_DEFINE_ATTRS_BINOP_HASH(Sub);
TVM_DEFINE_ATTRS_BINOP_HASH(Mul);
TVM_DEFINE_ATTRS_BINOP_HASH(Div);
TVM_DEFINE_ATTRS_BINOP_HASH(Mod);
TVM_DEFINE_ATTRS_BINOP_HASH(FloorDiv);
TVM_DEFINE_ATTRS_BINOP_HASH(FloorMod);
TVM_DEFINE_ATTRS_BINOP_HASH(Max);
TVM_DEFINE_ATTRS_BINOP_HASH(Min);
TVM_DEFINE_ATTRS_BINOP_HASH(GE);
TVM_DEFINE_ATTRS_BINOP_HASH(GT);
TVM_DEFINE_ATTRS_BINOP_HASH(LE);
TVM_DEFINE_ATTRS_BINOP_HASH(LT);
TVM_DEFINE_ATTRS_BINOP_HASH(EQ);
TVM_DEFINE_ATTRS_BINOP_HASH(NE);
TVM_DEFINE_ATTRS_BINOP_HASH(And);
TVM_DEFINE_ATTRS_BINOP_HASH(Or);

size_t AttrsHashHandler::VisitAttr_(const Not* op) {
  static size_t key = std::hash<std::string>()(Not::_type_key);
  return Combine(key, Hash(op->a));
}

size_t AttrsHashHandler::VisitAttr_(const Cast* op) {
  static size_t key = std::hash<std::string>()(Cast::_type_key);
  AttrsHash hasher;
  size_t res = key;
  res = Combine(res, hasher(op->type));
  res = Combine(res, Hash(op->value));
  return res;
}

size_t AttrsHashHandler::VisitAttr_(const Call* op) {
  static size_t key = std::hash<std::string>()(Call::_type_key);
  AttrsHash hasher;
  size_t res = key;
  res = Combine(res, hasher(op->name));
  res = Combine(res, hasher(op->type));
  res = Combine(res, Hash(op->args));
  return res;
}

size_t AttrsHashHandler::VisitAttr_(const Select* op) {
  static size_t key = std::hash<std::string>()(Select::_type_key);
  size_t res = key;
  res = Combine(res, Hash(op->condition));
  res = Combine(res, Hash(op->true_value));
  res = Combine(res, Hash(op->false_value));
  return res;
}


// Default case
bool AttrsEqual::operator()(const NodeRef& lhs, const NodeRef& rhs) const {
  if (lhs.same_as(rhs)) return true;
  if (handler_ == nullptr) {
    return AttrsEqualHandler().Equal(lhs, rhs);
  } else {
    return handler_->Equal(lhs, rhs);
  }
}

size_t AttrsHash::operator()(const NodeRef& node) const {
  if (!node.defined()) return 0;
  if (handler_ == nullptr) {
    return AttrsHashHandler().Hash(node);
  } else {
    return handler_->Hash(node);
  }
}

size_t DictAttrsNode::ContentHash(AttrsHash hasher) const {
  return hasher(this->dict);
}

bool DictAttrsNode::ContentEqual(const Node* other, AttrsEqual equal) const {
  if (this == other) return true;
  if (other == nullptr) return false;
  if (this->type_index() != other->type_index()) return false;
  return equal(this->dict, static_cast<const DictAttrsNode*>(other)->dict);
}

TVM_REGISTER_API("_AttrsListFieldInfo")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = args[0].operator Attrs()->ListFieldInfo();
});

}  // namespace tvm
