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

/*! \brief handle to node creator */
typedef void* NodeCreatorHandle;
/*! \brief handle to node */
typedef void* NodeHandle;

TVM_DLL int TVMNodeCreatorGet(const char* node_type,
                              NodeCreatorHandle *handle);

TVM_DLL int TVMNodeCreate(NodeCreatorHandle handle,
                          int num_child_ref,
                          const char* child_ref_keys,
                          NodeHandle* child_node_refs,
                          int num_attrs,
                          const char* attr_keys,
                          const char* attr_vals,
                          NodeHandle* handle);

TVM_DLL int TVMNodeGetAttr(const char* key,
                           const char** value);

TVM_DLL int TVMNodeGetChildNodeRef(const char* key,
                                   NodeHandle* out);

#endif  // TVM_C_API_H_
