/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api.h
 * \brief C API of TVM DSL
 */
#ifndef TVM_C_API_H_
#define TVM_C_API_H_

#ifdef __cplusplus
#define TVM_EXTERN_C extern "C"
#else
#define TVM_EXTERN_C
#endif

/*! \brief TVM_DLL prefix for windows */
#ifdef _WIN32
#ifdef TVM_EXPORTS
#define TVM_DLL TVM_EXTERN_C __declspec(dllexport)
#else
#define TVM_DLL TVM_EXTERN_C __declspec(dllimport)
#endif
#else
#define TVM_DLL TVM_EXTERN_C
#endif

/*! \brief handle to functions */
typedef void* FunctionHandle;
/*! \brief handle to node */
typedef void* NodeHandle;

/*!
 * \brief union type for returning value of attributes
 *  Attribute type can be identified by id
 */
typedef union {
  long v_long;  // NOLINT(*)
  double v_double;
  const char* v_str;
  NodeHandle v_handle;
} ArgVariant;

/*! \brief attribute types */
typedef enum {
  kNull = 0,
  kLong = 1,
  kDouble = 2,
  kStr = 3,
  kNodeHandle = 4
} ArgVariantID;

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  NNGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
TVM_DLL const char *TVMGetLastError(void);

/*!
 * \brief List all the node function name
 * \param out_size The number of functions
 * \param out_array The array of function names.
 */
TVM_DLL int TVMListFunctionNames(int *out_size,
                                 const char*** out_array);
/*!
 * \brief get function handle by name
 * \param name The name of function
 * \param handle The returning function handle
 */
TVM_DLL int TVMGetFunctionHandle(const char* name,
                                 FunctionHandle *handle);

/*!
 * \brief Get the detailed information about function.
 * \param handle The operator handle.
 * \param real_name The returned name of the function.
 *   This name is not the alias name of the atomic symbol.
 * \param description The returned description of the symbol.
 * \param num_doc_args Number of arguments that contain documents.
 * \param arg_names Name of the arguments of doc args
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \param return_type Return type of the function, if any.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMGetFunctionInfo(FunctionHandle handle,
                               const char **real_name,
                               const char **description,
                               int *num_doc_args,
                               const char ***arg_names,
                               const char ***arg_type_infos,
                               const char ***arg_descriptions,
                               const char **return_type);

/*!
 * \brief Push an argument to the function calling stack.
 * If push fails, the stack will be reset to empty
 *
 * \param arg number of attributes
 * \param type_id The typeid of attributes.
 */
TVM_DLL int TVMPushStack(ArgVariant arg,
                         int type_id);

/*!
 * \brief call a function by using arguments in the stack.
 * The stack will be cleanup to empty after this call, whether the call is successful.
 *
 * \param handle The function handle
 * \param ret_val The return value.
 * \param ret_typeid the type id of return value.
 */
TVM_DLL int TVMFunctionCall(FunctionHandle handle,
                            ArgVariant* ret_val,
                            int* ret_typeid);

/*!
 * \brief free the node handle
 * \param handle The node handle to be freed.
 */
TVM_DLL int TVMNodeFree(NodeHandle handle);

/*!
 * \brief get attributes given key
 * \param handle The node handle
 * \param key The attribute name
 * \param out_value The attribute value
 * \param out_typeid The typeif of the attribute.
 */
TVM_DLL int TVMNodeGetAttr(NodeHandle handle,
                           const char* key,
                           ArgVariant* out_value,
                           int* out_typeid);

/*!
 * \brief get attributes names in the node.
 * \param handle The node handle
 * \param out_size The number of functions
 * \param out_array The array of function names.
 */
TVM_DLL int TVMNodeListAttrNames(NodeHandle handle,
                                 int *out_size,
                                 const char*** out_array);

#endif  // TVM_C_API_H_
