/*!
 *  Copyright (c) 2018 by Contributors
 * \file attrs.cc
 */
#include <tvm/attrs.h>
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

class AttrsEqualChecker :
      public AttrFunctor<bool(const NodeRef&, const NodeRef&)> {
 public:
  bool Check(const NodeRef& lhs, const NodeRef& rhs) {
    if (!equal_) return false;
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() || !rhs.defined()) return false;
    if (!this->VisitAttr(lhs, rhs)) {
      equal_ = false;
    }
    return equal_;
  }

  bool VisitAttrDefault_(const Node* lhs, const NodeRef& other) final {
    if (lhs->derived_from<BaseAttrsNode>()) {
      return static_cast<const BaseAttrsNode*>(lhs)->ContentEqual(other.get());
    }
    return lhs == other.get();
  }

  bool VisitAttr_(const IntImm* lhs, const NodeRef& other) final {
    if (const auto* rhs = other.as<IntImm>()) {
      return lhs->value == rhs->value;
    }
    return false;
  }

  bool VisitAttr_(const UIntImm* lhs, const NodeRef& other) final {
    if (const auto* rhs = other.as<UIntImm>()) {
      return lhs->value == rhs->value;
    }
    return false;
  }

  bool VisitAttr_(const FloatImm* lhs, const NodeRef& other) final {
    if (const auto* rhs = other.as<FloatImm>()) {
      return lhs->value == rhs->value;
    }
    return false;
  }

  bool VisitAttr_(const StringImm* lhs, const NodeRef& other) final {
    if (const auto* rhs = other.as<StringImm>()) {
      return lhs->value == rhs->value;
    }
    return false;
  }

  bool VisitAttr_(const ArrayNode* lhs, const NodeRef& other) final {
    if (const auto* rhs = other.as<ArrayNode>()) {
      if (rhs->data.size() != lhs->data.size()) return false;
      for (size_t  i = 0; i < lhs->data.size(); ++i) {
        if (!Check(NodeRef(lhs->data[i]), NodeRef(rhs->data[i]))) return false;
      }
    }
    return true;
  }

  bool VisitAttr_(const StrMapNode* lhs, const NodeRef& other) final {
    if (const auto* rhs = other.as<StrMapNode>()) {
      if (rhs->data.size() != lhs->data.size()) return false;
      for (const auto& kv : lhs->data) {
        auto it = rhs->data.find(kv.first);
        if (it == rhs->data.end()) return false;
        if (!Check(NodeRef(kv.second), NodeRef(it->second))) return false;
      }
    }
    return true;
  }

 private:
  bool equal_{true};
};

class AttrContentHasher :
      public AttrFunctor<void(const NodeRef&)> {
 public:
  size_t result_{0};

  void VisitAttrDefault_(const Node* value) final {
    if (value->derived_from<BaseAttrsNode>()) {
      Update(static_cast<const BaseAttrsNode*>(value)->ContentHash());
    } else {
      Update(NodeHash()(GetRef<NodeRef>(value)));
    }
  }

  void VisitAttr_(const IntImm* op) final {
    Update(std::hash<int64_t>()(op->value));
  }

  void VisitAttr_(const UIntImm* op) final {
    Update(std::hash<uint64_t>()(op->value));
  }

  void VisitAttr_(const FloatImm* op) final {
    Update(std::hash<double>()(op->value));
  }

  void VisitAttr_(const StringImm* op) final {
    Update(std::hash<std::string>()(op->value));
  }

  void VisitAttr_(const ArrayNode* op) final {
    Update(op->data.size());
    for (size_t  i = 0; i < op->data.size(); ++i) {
      this->VisitAttr(NodeRef(op->data[i]));
    }
  }

  void VisitAttr_(const StrMapNode* lhs) final {
    using Entry = std::pair<std::string, NodePtr<Node> >;
    std::vector<Entry> data(lhs->data.begin(), lhs->data.end());
    std::sort(data.begin(), data.end(), [](const Entry& a, const Entry& b) {
        return a.first < b.first;
      });
    for (const Entry& kv : data) {
      Update(std::hash<std::string>()(kv.first));
      this->VisitAttr(NodeRef(kv.second));
    }
  }

  void Update(size_t value) {
    result_ = dmlc::HashCombine(result_, value);
  }
};

bool AttrsEqual::Equal(const NodeRef& lhs, const NodeRef& rhs) {
  if (lhs.same_as(rhs)) return true;
  AttrsEqualChecker checker;
  return checker.Check(lhs, rhs);
}

size_t AttrsHash::Hash(const NodeRef& node) {
  if (!node.defined()) return 0;
  AttrContentHasher hasher;
  hasher.VisitAttr(node);
  return hasher.result_;
}

size_t DictAttrsNode::ContentHash() const {
  return AttrsHash()(this->dict);
}

bool DictAttrsNode::ContentEqual(const Node* other) const {
  if (this == other) return true;
  if (other == nullptr) return false;
  if (this->type_index() != other->type_index()) return false;
  return AttrsEqual()(this->dict, static_cast<const DictAttrsNode*>(other)->dict);
}

}  // namespace tvm
