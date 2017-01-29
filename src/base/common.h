/*!
 *  Copyright (c) 2016 by Contributors
 * \file common.h
 * \brief Common utilities
 */
#ifndef TVM_BASE_COMMON_H_
#define TVM_BASE_COMMON_H_

#include <tvm/base.h>
#include <tvm/expr.h>
#include <string>

namespace tvm {

inline std::string Type2String(const Type& t) {
  if (t.code()  ==Type::Handle) return "handle";
  std::ostringstream os;
  os << t;
  return os.str();
}

inline Type String2Type(std::string s) {
  std::istringstream is(s);
  halide_type_code_t code = Type::Int;
  if (s.substr(0, 3) == "int") {
    code = Type::Int; s = s.substr(3);
  } else if (s.substr(0, 4) == "uint") {
    code = Type::UInt; s = s.substr(4);
  } else if (s.substr(0, 5) == "float") {
    code = Type::Float; s = s.substr(5);
  } else if (s.substr(0, 5) == "float") {
    code = Type::Float; s = s.substr(5);
  } else if (s == "handle") {
    return Handle();
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  int bits = 32, lanes = 1;
  if (sscanf(s.c_str(), "%dx%d", &bits, &lanes) == 0) {
    LOG(FATAL) << "unknown type " << s;
  }
  return Type(code, bits, lanes);
}

inline const char* TVMTypeCode2Str(int type_code) {
  switch (type_code) {
    case kInt: return "int";
    case kFloat: return "float";
    case kStr: return "str";
    case kHandle: return "Handle";
    case kNull: return "NULL";
    case kNodeHandle: return "NodeHandle";
    default: LOG(FATAL) << "unknown type_code="
                        << static_cast<int>(type_code); return "";
  }
}
template<typename T>
struct NodeTypeChecker {
  static inline bool Check(Node* sptr) {
    // This is the only place in the project where RTTI is used
    // It can be turned off, but will make non strict checking.
    // TODO(tqchen) possibly find alternative to turn of RTTI
    using ContainerType = typename T::ContainerType;
    return (dynamic_cast<ContainerType*>(sptr) != nullptr);
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

}  // namespace tvm
#endif  // TVM_BASE_COMMON_H_
