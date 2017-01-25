/*!
 *  Copyright (c) 2016 by Contributors
 * \file api_registry.h
 * \brief This file defines the TVM API registry.
 *
 *  The API registry stores type-erased functions.
 *  Each registered function is automatically exposed
 *  to front-end language(e.g. python).
 *  Front-end can also pass callbacks as PackedFunc, or register
 *  then into the same global registry in C++.
 *  The goal is to mix the front-end language and the TVM back-end.
 *
 * \code
 *   // register the function as MyAPIFuncName
 *   TVM_REGISTER_API(MyAPIFuncName)
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *     // my code.
 *   });
 * \endcode
 */
#ifndef TVM_API_REGISTRY_H_
#define TVM_API_REGISTRY_H_

#include <dmlc/base.h>
#include <string>
#include "./base.h"
#include "./runtime/packed_func.h"
#include "./packed_func_ext.h"

namespace tvm {

/*! \brief Utility to register API. */
class APIRegistry {
 public:
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  APIRegistry& set_body(PackedFunc f);  // NOLINT(*)
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  APIRegistry& set_body(PackedFunc::FType f) {  // NOLINT(*)
    return set_body(PackedFunc(f));
  }
  /*!
   * \brief Register a function with given name
   * \param name The name of the function.
   */
  static APIRegistry& __REGISTER__(const std::string& name);  // NOLINT(*)

 private:
  /*! \brief name of the function */
  std::string name_;
};

/*!
 * \brief Get API function by name.
 *
 * \param name The name of the function.
 * \return the corresponding API function.
 * \note It is really PackedFunc::GetGlobal under the hood.
 */
inline PackedFunc GetAPIFunc(const std::string& name) {
  return PackedFunc::GetGlobal(name);
}

#define _TVM_REGISTER_VAR_DEF_                                          \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::APIRegistry& __make_TVMRegistry_

/*!
 * \brief Register API function globally.
 * \code
 *   TVM_REGISTER_API(MyPrint)
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *     // my code.
 *   });
 * \endcode
 */
#define TVM_REGISTER_API(OpName)                                 \
  DMLC_STR_CONCAT(_TVM_REGISTER_VAR_DEF_, __COUNTER__) =         \
      ::tvm::APIRegistry::__REGISTER__(#OpName)
}  // namespace tvm
#endif  // TVM_API_REGISTRY_H_
