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

/*! \brief Variant container for API calls */
struct APIVariantValue {
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
  inline APIVariantValue& operator=(std::string value) {
    type_id = kStr;
    str = std::move(value);
    v_union.v_str = str.c_str();
    return *this;
  }
  inline APIVariantValue& operator=(const NodeRef& ref) {
    type_id = kNodeHandle;
    this->sptr = ref.node_;
    return *this;
  }
  template<typename T,
         typename = typename std::enable_if<std::is_base_of<NodeRef, T>::value>::type>
  inline operator T() const {
    if (type_id == kNull) return T();
    CHECK_EQ(type_id, kNodeHandle);
    return T(sptr);
  }
  inline operator Expr() const {
    if (type_id == kNull) return Expr();
    if (type_id == kLong) return Expr(operator int64_t());
    if (type_id == kDouble) return Expr(operator double());
    CHECK_EQ(type_id, kNodeHandle);
    return Expr(sptr);
  }
  inline operator double() const {
    CHECK_EQ(type_id, kDouble);
    return v_union.v_double;
  }
  inline operator int64_t() const {
    CHECK_EQ(type_id, kLong);
    return v_union.v_long;
  }
  inline operator int() const {
    CHECK_EQ(type_id, kLong);
    CHECK_LE(v_union.v_long,
             std::numeric_limits<int>::max());
    return v_union.v_long;
  }
  inline operator std::string() const {
    CHECK_EQ(type_id, kStr);
    return str;
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
