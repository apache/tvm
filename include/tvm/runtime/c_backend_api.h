/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_backend_api.h
 * \brief TVM runtime backend API.
 *
 *  The functions defined in this header are intended to be
 *  used by compiled tvm operators, usually user do not need to use these
 *  function directly.
 */
#ifndef TVM_RUNTIME_C_BACKEND_API_H_
#define TVM_RUNTIME_C_BACKEND_API_H_

#include "./c_runtime_api.h"

#ifdef __cplusplus
TVM_EXTERN_C {
#endif

// Backend related functions.
/*!
 * \brief Backend function for modules to get function
 *  from its environment mod_node (its imports and global function).
 *  The user do should not call TVMFuncFree on func.
 *
 * \param mod_node The module handle.
 * \param func_name The name of the function.
 * \param out The result function.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendGetFuncFromEnv(void* mod_node,
                                     const char* func_name,
                                     TVMFunctionHandle *out);
/*!
 * \brief Backend function to register system-wide library symbol.
 *
 * \param name The name of the symbol
 * \param ptr The symbol address.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendRegisterSystemLibSymbol(const char* name, void* ptr);

/*!
 * \brief Backend function to allocate temporal workspace.
 *
 * \note The result allocate spaced is ensured to be aligned to kTempAllocaAlignment.
 *
 * \param size The size of the space requested.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \return nullptr when error is thrown, a valid ptr if success
 */
TVM_DLL void* TVMBackendAllocWorkspace(int device_type,
                                       int device_id,
                                       uint64_t size);

/*!
 * \brief Backend function to free temporal workspace.
 *
 * \param ptr The result allocated space pointer.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \return 0 when no error is thrown, -1 when failure happens
 *
 * \sa TVMBackendAllocWorkspace
 */
TVM_DLL int TVMBackendFreeWorkspace(int device_type,
                                    int device_id,
                                    void* ptr);
/*!
 * \brief Backend function for running parallel for loop.
 *
 * \param begin The start of iteration.
 * \param end The end of iteration.
 * \param lambda The lambda function to be executed.
 * \param env The environment of lambda function.
 *
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendParallelFor(
    int64_t begin,
    int64_t end,
    int (*lambda)(int64_t begin, int64_t end, void* env),
    void* env);

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // TVM_RUNTIME_C_BACKEND_API_H_
