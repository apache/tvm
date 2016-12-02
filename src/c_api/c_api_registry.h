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
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace tvm {

inline std::string Type2String(const Type& t) {
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
  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  int bits = 32, lanes = 1;
  if (sscanf(s.c_str(), "%dx%d", &bits, &lanes) == 0) {
    LOG(FATAL) << "unknown type " << s;
  }
  return Type(code, bits, lanes);
}

inline const char* TypeId2Str(ArgVariantID type_id) {
  switch (type_id) {
    case kNull: return "Null";
    case kLong: return "Long";
    case kDouble: return "Double";
    case kStr: return "Str";
    case kNodeHandle: return "NodeHandle";
    default: LOG(FATAL) << "unknown type_id=" << type_id; return "";
  }
}

/*! \brief Variant container for API calls */
class APIVariantValue {
 public:
  /*! \brief the type id */
  ArgVariantID type_id{kNull};
  /*! \brief shared pointer container */
  std::shared_ptr<Node> sptr;
  /*! \brief string container */
  std::string str;
  /*! \brief the variant holder */
  ArgVariant v_union;
  // constructor
  APIVariantValue() {}
  // clear value
  inline void Clear() {
  }
  // assign op
  inline APIVariantValue& operator=(double value) {
    type_id = kDouble;
    v_union.v_double = value;
    return *this;
  }
  inline APIVariantValue& operator=(std::nullptr_t value) {
    type_id = kNull;
    return *this;
  }
  inline APIVariantValue& operator=(int64_t value) {
    type_id = kLong;
    v_union.v_long = value;
    return *this;
  }
  inline APIVariantValue& operator=(bool value) {
    type_id = kLong;
    v_union.v_long = value;
    return *this;
  }
  inline APIVariantValue& operator=(std::string value) {
    type_id = kStr;
    str = std::move(value);
    v_union.v_str = str.c_str();
    return *this;
  }
  inline APIVariantValue& operator=(const NodeRef& ref) {
    if (ref.node_.get() == nullptr) {
      type_id = kNull;
    } else {
      type_id = kNodeHandle;
      this->sptr = ref.node_;
    }
    return *this;
  }
  inline APIVariantValue& operator=(const Type& value) {
    return operator=(Type2String(value));
  }
  template<typename T,
         typename = typename std::enable_if<std::is_base_of<NodeRef, T>::value>::type>
  inline operator T() const {
    if (type_id == kNull) return T();
    CHECK_EQ(type_id, kNodeHandle);
    // use dynamic RTTI for safety
    CHECK(dynamic_cast<typename T::ContainerType*>(sptr.get()))
        << "wrong type specified";
    return T(sptr);
  }
  inline operator Expr() const {
    if (type_id == kNull) return Expr();
    if (type_id == kLong) return Expr(operator int());
    if (type_id == kDouble) {
      return Expr(static_cast<float>(operator double()));
    }
    CHECK_EQ(type_id, kNodeHandle);
    if (sptr->is_type<IterVarNode>()) {
      return IterVar(sptr)->var;
    } else {
      CHECK(dynamic_cast<typename Expr::ContainerType*>(sptr.get()))
          << "did not pass in Expr in a place need Expr";
      return Expr(sptr);
    }
  }
  inline operator double() const {
    CHECK_EQ(type_id, kDouble);
    return v_union.v_double;
  }
  inline operator int64_t() const {
    CHECK_EQ(type_id, kLong);
    return v_union.v_long;
  }
  inline operator uint64_t() const {
    CHECK_EQ(type_id, kLong);
    return v_union.v_long;
  }
  inline operator int() const {
    CHECK_EQ(type_id, kLong);
    CHECK_LE(v_union.v_long,
             std::numeric_limits<int>::max());
    return v_union.v_long;
  }
  inline operator bool() const {
    CHECK_EQ(type_id, kLong)
        << "expect boolean(int) but get " << TypeId2Str(type_id);
    return v_union.v_long != 0;
  }
  inline operator std::string() const {
    CHECK_EQ(type_id, kStr)
        << "expect Str but get " << TypeId2Str(type_id);
    return str;
  }
  inline operator Type() const {
    return String2Type(operator std::string());
  }
};

// common defintiion of API function.
using APIFunction = std::function<
  void(const std::vector<APIVariantValue> &args, APIVariantValue* ret)>;

/*!
 * \brief Registry entry for DataIterator factory functions.
 */
struct APIFunctionReg
    : public dmlc::FunctionRegEntryBase<APIFunctionReg,
                                        APIFunction> {
};

#define TVM_REGISTER_API(TypeName)                                      \
  DMLC_REGISTRY_REGISTER(::tvm::APIFunctionReg, APIFunctionReg, TypeName) \

}  // namespace tvm

#endif  // TVM_C_API_C_API_REGISTRY_H_
