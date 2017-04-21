/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api.h
 * \brief C API of TVM DSL
 *
 * \note The API is designed in a minimum way.
 *  Most of the API functions are registered and can be pulled out.
 *
 *  The common flow is:
 *   - Use TVMFuncListGlobalNames to get global function name
 *   - Use TVMFuncCall to call these functions.
 */
#ifndef TVM_C_API_H_
#define TVM_C_API_H_

#include "./runtime/c_runtime_api.h"

TVM_EXTERN_C {
/*! \brief handle to node */
typedef void* NodeHandle;

/*!
 * \brief Inplace translate callback argument value to return value.
 *  This is only needed for non-POD arguments.
 *
 * \param value The value to be translated.
 * \param code The type code to be translated.
 * \note This function will do a shallow copy when necessary.
 *
 * \return 0 when success, -1 when failure happens.
 */
TVM_DLL int TVMCbArgToReturn(TVMValue* value, int code);

/*!
 * \brief free the node handle
 * \param handle The node handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMNodeFree(NodeHandle handle);

/*!
 * \brief get attributes given key
 * \param handle The node handle
 * \param key The attribute name
 * \param out_value The attribute value
 * \param out_type_code The type code of the attribute.
 * \param out_success Whether get is successful.
 * \return 0 when success, -1 when failure happens
 * \note API calls always exchanges with type bits=64, lanes=1
 */
TVM_DLL int TVMNodeGetAttr(NodeHandle handle,
                           const char* key,
                           TVMValue* out_value,
                           int* out_type_code,
                           int* out_success);

/*!
 * \brief get attributes names in the node.
 * \param handle The node handle
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMNodeListAttrNames(NodeHandle handle,
                                 int *out_size,
                                 const char*** out_array);
}  // TVM_EXTERN_C
#endif  // TVM_C_API_H_
