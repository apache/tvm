/*!
 *  Copyright (c) 2017 by Contributors
 * \file api_registry.h
 * \brief This files include necessary headers to
 *  be used to register an global API function.
 */
#ifndef TVM_API_REGISTRY_H_
#define TVM_API_REGISTRY_H_

#include "./base.h"
#include "./packed_func_ext.h"
#include "./runtime/registry.h"

/*!
 * \brief Register an API function globally.
 * It simply redirects to TVM_REGISTER_GLOBAL
 *
 * \code
 *   TVM_REGISTER_API(MyPrint)
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *     // my code.
 *   });
 * \endcode
 */
#define TVM_REGISTER_API(OpName) TVM_REGISTER_GLOBAL(OpName)

#endif  // TVM_API_REGISTRY_H_
