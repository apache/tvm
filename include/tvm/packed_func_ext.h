/*!
 *  Copyright (c) 2016 by Contributors
 * \file tvm/packed_func_ext.h
 * \brief Extension package to PackedFunc
 *   This enales pass NodeRef types into/from PackedFunc.
 */
#ifndef TVM_PACKED_FUNC_EXT_H_
#define TVM_PACKED_FUNC_EXT_H_

#include <sstream>
#include <string>
#include <memory>
#include <type_traits>

#include "base.h"
#include "expr.h"
#include "tensor.h"
#include "runtime/packed_func.h"

namespace tvm {
using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

namespace runtime {
/*!
 * \brief Runtime type checker for node type.
 * \tparam T the type to be checked.
 */
template<typename T>
struct NodeTypeChecker {
  static inline bool Check(Node* sptr) {
    // This is the only place in the project where RTTI is used
    // It can be turned off, but will make non strict checking.
    // TODO(tqchen) possibly find alternative to turn of RTTI
    using ContainerType = typename T::ContainerType;
    return sptr->derived_from<ContainerType>();
  }
  static inline void PrintName(std::ostringstream& os) { // NOLINT(*)
    using ContainerType = typename T::ContainerType;
    os << ContainerType::_type_key;
  }
};

template<typename T>
struct NodeTypeChecker<Array<T> > {
  static inline bool Check(Node* sptr) {
    if (sptr == nullptr) return false;
    if (!sptr->is_type<ArrayNode>()) return false;
    ArrayNode* n = static_cast<ArrayNode*>(sptr);
    for (const auto& p : n->data) {
      if (!NodeTypeChecker<T>::Check(p.get())) return false;
    }
    return true;
  }
  static inline void PrintName(std::ostringstream& os) { // NOLINT(*)
    os << "array<";
    NodeTypeChecker<T>::PrintName(os);
    os << ">";
  }
};

template<typename V>
struct NodeTypeChecker<Map<std::string, V> > {
  static inline bool Check(Node* sptr) {
    if (sptr == nullptr) return false;
    if (!sptr->is_type<StrMapNode>()) return false;
    StrMapNode* n = static_cast<StrMapNode*>(sptr);
    for (const auto& kv : n->data) {
      if (!NodeTypeChecker<V>::Check(kv.second.get())) return false;
    }
    return true;
  }
  static inline void PrintName(std::ostringstream& os) { // NOLINT(*)
    os << "map<string";
    os << ',';
    NodeTypeChecker<V>::PrintName(os);
    os << '>';
  }
};

template<typename K, typename V>
struct NodeTypeChecker<Map<K, V> > {
  static inline bool Check(Node* sptr) {
    if (sptr == nullptr) return false;
    if (!sptr->is_type<MapNode>()) return false;
    MapNode* n = static_cast<MapNode*>(sptr);
    for (const auto& kv : n->data) {
      if (!NodeTypeChecker<K>::Check(kv.first.get())) return false;
      if (!NodeTypeChecker<V>::Check(kv.second.get())) return false;
    }
    return true;
  }
  static inline void PrintName(std::ostringstream& os) { // NOLINT(*)
    os << "map<";
    NodeTypeChecker<K>::PrintName(os);
    os << ',';
    NodeTypeChecker<V>::PrintName(os);
    os << '>';
  }
};

template<typename T>
inline std::string NodeTypeName() {
  std::ostringstream os;
  NodeTypeChecker<T>::PrintName(os);
  return os.str();
}

// extensions for tvm arg value

template<typename TNodeRef>
inline TNodeRef TVMArgValue::AsNodeRef() const {
  static_assert(
      std::is_base_of<NodeRef, TNodeRef>::value,
      "Conversion only works for NodeRef");
  if (type_code_ == kNull) return TNodeRef(NodePtr<Node>(nullptr));
  TVM_CHECK_TYPE_CODE(type_code_, kNodeHandle);
  NodePtr<Node>& sptr = *ptr<NodePtr<Node> >();
  CHECK(NodeTypeChecker<TNodeRef>::Check(sptr.get()))
      << "Expected type " << NodeTypeName<TNodeRef>()
      << " but get " << sptr->type_key();
  return TNodeRef(sptr);
}

inline TVMArgValue::operator HalideIR::Expr() const {
  if (type_code_ == kNull) return Expr();
  if (type_code_ == kDLInt) {
    return Expr(static_cast<int>(value_.v_int64));
  }
  if (type_code_ == kDLFloat) {
    return Expr(static_cast<float>(value_.v_float64));
  }
  TVM_CHECK_TYPE_CODE(type_code_, kNodeHandle);
  NodePtr<Node>& sptr = *ptr<NodePtr<Node> >();
  if (sptr->is_type<IterVarNode>()) {
    return IterVar(sptr)->var;
  }
  if (sptr->is_type<TensorNode>()) {
    return Tensor(sptr)();
  }
  CHECK(NodeTypeChecker<Expr>::Check(sptr.get()))
      << "Expected type " << NodeTypeName<Expr>()
      << " but get " << sptr->type_key();
  return Expr(sptr);
}

inline NodePtr<Node>& TVMArgValue::node_sptr() {
  TVM_CHECK_TYPE_CODE(type_code_, kNodeHandle);
  return *ptr<NodePtr<Node> >();
}


template<typename TNodeRef, typename>
inline bool TVMArgValue::IsNodeType() const {
  TVM_CHECK_TYPE_CODE(type_code_, kNodeHandle);
  NodePtr<Node>& sptr =
      *ptr<NodePtr<Node> >();
  return NodeTypeChecker<TNodeRef>::Check(sptr.get());
}

// extensions for TVMRetValue
inline TVMRetValue& TVMRetValue::operator=(
    const NodePtr<Node>& other) {
  if (other.get() == nullptr) {
    SwitchToPOD(kNull);
  } else {
    SwitchToClass<NodePtr<Node> >(kNodeHandle, other);
  }
  return *this;
}

inline TVMRetValue& TVMRetValue::operator=(const NodeRef& other) {
  if (!other.defined()) {
    SwitchToPOD(kNull);
  } else {
    SwitchToClass<NodePtr<Node> >(kNodeHandle, other.node_);
  }
  return *this;
}

template<typename TNodeRef>
inline TNodeRef TVMRetValue::AsNodeRef() const {
  static_assert(
      std::is_base_of<NodeRef, TNodeRef>::value,
      "Conversion only works for NodeRef");
  if (type_code_ == kNull) return TNodeRef();
  TVM_CHECK_TYPE_CODE(type_code_, kNodeHandle);
  NodePtr<Node>& sptr = *ptr<NodePtr<Node> >();
  CHECK(NodeTypeChecker<TNodeRef>::Check(sptr.get()))
      << "Expected type " << NodeTypeName<TNodeRef>()
      << " but get " << sptr->type_key();
  return TNodeRef(sptr);
}

inline void TVMArgsSetter::operator()(size_t i, const NodeRef& other) const {  // NOLINT(*)
  if (other.defined()) {
    values_[i].v_handle = const_cast<NodePtr<Node>*>(&(other.node_));
    type_codes_[i] = kNodeHandle;
  } else {
    type_codes_[i] = kNull;
  }
}

// type related stuffs
inline TVMRetValue& TVMRetValue::operator=(const HalideIR::Type& t) {
  return this->operator=(Type2TVMType(t));
}

inline TVMRetValue::operator HalideIR::Type() const {
  return TVMType2Type(operator TVMType());
}

inline TVMArgValue::operator HalideIR::Type() const {
  return TVMType2Type(operator TVMType());
}

inline void TVMArgsSetter::operator()(
    size_t i, const HalideIR::Type& t) const {
  this->operator()(i, Type2TVMType(t));
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_PACKED_FUNC_EXT_H_
