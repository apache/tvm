/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_registry.h
 * \brief Quick registry for C API.
 */
#ifndef TVM_C_API_C_API_REGISTRY_H_
#define TVM_C_API_C_API_REGISTRY_H_

#include <tvm/base.h>
#include <tvm/expr.h>
#include <tvm/c_api.h>
#include <tvm/runtime/runtime.h>
#include <memory>
#include <limits>
#include <string>
#include <vector>
#include "../base/common.h"

namespace tvm {

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

/*! \brief Variant container for API calls */
class APIVariantValue {
 public:
  /*! \brief the type id */
  int type_code{kNull};
  /*! \brief shared pointer container */
  std::shared_ptr<Node> sptr;
  /*! \brief string container */
  std::string str;
  /*! \brief the variant holder */
  TVMValue v_union;
  /*! \brief std::function */
  runtime::PackedFunc::FType func;
  // constructor
  APIVariantValue() {
  }
  // clear value
  inline void Clear() {
  }
  // assign op
  inline APIVariantValue& operator=(double value) {
    type_code = kFloat;
    v_union.v_float64 = value;
    return *this;
  }
  inline APIVariantValue& operator=(std::nullptr_t value) {
    type_code = kHandle;
    v_union.v_handle = value;
    return *this;
  }
  inline APIVariantValue& operator=(int64_t value) {
    type_code = kInt;
    v_union.v_int64 = value;
    return *this;
  }
  inline APIVariantValue& operator=(bool value) {
    type_code = kInt;
    v_union.v_int64 = value;
    return *this;
  }
  inline APIVariantValue& operator=(std::string value) {
    type_code = kStr;
    str = std::move(value);
    v_union.v_str = str.c_str();
    return *this;
  }
  inline APIVariantValue& operator=(const NodeRef& ref) {
    if (ref.node_.get() == nullptr) {
      type_code = kNull;
    } else {
      type_code = kNodeHandle;
      this->sptr = ref.node_;
    }
    return *this;
  }
  inline APIVariantValue& operator=(const runtime::PackedFunc& f) {
    type_code = kFuncHandle;
    this->func = f.body();
    return *this;
  }
  inline APIVariantValue& operator=(const Type& value) {
    return operator=(Type2String(value));
  }
  template<typename T,
           typename = typename std::enable_if<std::is_base_of<NodeRef, T>::value>::type>
  inline operator T() const {
    if (type_code == kNull) return T();
    CHECK_EQ(type_code, kNodeHandle);
    CHECK(NodeTypeChecker<T>::Check(sptr.get()))
        << "Did not get expected type " << NodeTypeName<T>();
    return T(sptr);
  }
  inline operator Expr() const {
    if (type_code == kNull) {
      return Expr();
    }
    if (type_code == kInt) return Expr(operator int());
    if (type_code == kFloat) {
      return Expr(static_cast<float>(operator double()));
    }
    CHECK_EQ(type_code, kNodeHandle);
    if (sptr->is_type<IterVarNode>()) {
      return IterVar(sptr)->var;
    } else {
      CHECK(NodeTypeChecker<Expr>::Check(sptr.get()))
          << "did not pass in Expr in a place need Expr";
      return Expr(sptr);
    }
  }
  inline operator double() const {
    CHECK_EQ(type_code, kFloat);
    return v_union.v_float64;
  }
  inline operator int64_t() const {
    CHECK_EQ(type_code, kInt);
    return v_union.v_int64;
  }
  inline operator uint64_t() const {
    CHECK_EQ(type_code, kInt);
    return v_union.v_int64;
  }
  inline operator int() const {
    CHECK_EQ(type_code, kInt);
    CHECK_LE(v_union.v_int64,
             std::numeric_limits<int>::max());
    return v_union.v_int64;
  }
  inline operator bool() const {
    CHECK_EQ(type_code, kInt)
        << "expect boolean(int) but get "
        << TVMTypeCode2Str(type_code);
    return v_union.v_int64 != 0;
  }
  inline operator std::string() const {
    CHECK_EQ(type_code, kStr)
        << "expect Str but get "
        << TVMTypeCode2Str(type_code);
    return str;
  }
  inline operator Type() const {
    return String2Type(operator std::string());
  }
  inline operator runtime::PackedFunc() const {
    CHECK_EQ(type_code, kFuncHandle);
    return runtime::PackedFunc(func);
  }
};

// common defintiion of API function.
using APIFunc = std::function<
  void(const std::vector<APIVariantValue> &args, APIVariantValue* ret)>;

/*!
 * \brief Registry entry for DataIterator factory functions.
 */
struct APIFuncReg
    : public dmlc::FunctionRegEntryBase<APIFuncReg,
                                        APIFunc> {
};

#define TVM_REGISTER_API(TypeName)                                \
  DMLC_REGISTRY_REGISTER(::tvm::APIFuncReg, APIFuncReg, TypeName) \

}  // namespace tvm

#endif  // TVM_C_API_C_API_REGISTRY_H_
