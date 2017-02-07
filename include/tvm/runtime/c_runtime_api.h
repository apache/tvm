/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_runtime_api.h
 * \brief TVM runtime library.
 *
 *  The philosophy of TVM project is to customize the compilation
 *  stage to generate code that can used by other projects transparently.
 *
 *  So this is a minimum runtime code gluing, and some limited
 *  memory management code to enable quick testing.
 */
#ifndef TVM_RUNTIME_C_RUNTIME_API_H_
#define TVM_RUNTIME_C_RUNTIME_API_H_

#ifdef __cplusplus
#define TVM_EXTERN_C extern "C"
#else
#define TVM_EXTERN_C
#endif

/*! \brief TVM_DLL prefix for windows */
#ifdef _WIN32
#ifdef TVM_EXPORTS
#define TVM_DLL __declspec(dllexport)
#else
#define TVM_DLL __declspec(dllimport)
#endif
#else
#define TVM_DLL
#endif

#include <stdint.h>
#include <stddef.h>


TVM_EXTERN_C {
/*! \brief type of array index. */
typedef uint32_t tvm_index_t;
/*!
 * \brief The type code in TVMType
 * \note TVMType is used in two places.
 */
typedef enum {
  kInt = 0U,
  kUInt = 1U,
  kFloat = 2U,
  kHandle = 3U,
  // The next few fields are extension types
  // that is used by TVM API calls.
  kNull = 4U,
  kArrayHandle = 5U,
  kTVMType = 6U,
  kNodeHandle = 7U,
  kFuncHandle = 8U,
  kStr = 9U,
  kBytes = 10U
} TVMTypeCode;

/*!
 * \brief The data type used in TVM Runtime.
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 *
 * \note Arguments TVM API function always takes bits=64 and lanes=1
 */
typedef struct {
  /*! \brief type code, in TVMTypeCode */
  uint8_t code;
  /*! \brief number of bits of the type */
  uint8_t bits;
  /*! \brief number of lanes, */
  uint16_t lanes;
} TVMType;

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  TVMType v_type;
} TVMValue;

/*!
 * \brief Byte array type used to pass in byte array
 *  When kBytes is used as data type.
 */
typedef struct {
  const char* data;
  size_t size;
} TVMByteArray;

/*!
 * \brief The device type
 */
typedef enum {
  /*! \brief CPU device */
  kCPU = 1,
  /*! \brief NVidia GPU device(CUDA) */
  kGPU = 2,
  /*! \brief opencl device */
  kOpenCL = 4
} TVMDeviceMask;

/*!
 * \brief The Device information, abstract away common device types.
 */
typedef struct {
  /*! \brief The device type mask */
  int dev_mask;
  /*! \brief the device id */
  int dev_id;
} TVMContext;

/*!
 * \brief Data structure representing a n-dimensional array(tensor).
 *  This is used to pass data specification into TVM.
 */
typedef struct {
  /*! \brief The data field pointer on specified device */
  void* data;
  /*! \brief The shape pointers of the array */
  const tvm_index_t* shape;
  /*!
   * \brief The stride data about each dimension of the array, can be NULL
   *  When strides is NULL, it indicates that the array is empty.
   */
  const tvm_index_t* strides;
  /*! \brief number of dimensions of the array */
  tvm_index_t ndim;
  /*! \brief The data type flag */
  TVMType dtype;
  /*! \brief The device context this array sits on */
  TVMContext ctx;
} TVMArray;

/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* TVMStreamHandle;
/*! \brief Handle to packed function handle. */
typedef void* TVMFunctionHandle;
/*! \brief Handle to hold return value. */
typedef void* TVMRetValueHandle;
/*! \brief the array handle */
typedef TVMArray* TVMArrayHandle;

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  TVMGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
TVM_DLL const char *TVMGetLastError(void);

/*!
 * \brief Initialize certain type of devices, this may
 *  not be necessary for all device types. But is needed for OpenCL.
 *
 * \param dev_mask The device mask of device type to be initialized
 * \param option_keys Additional option  keys to pass.
 * \param option_vals Additional option values to pass
 * \param num_options Number of options to be passed into it.
 * \param out_code 1: success, 0: already initialized
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMDeviceInit(int dev_mask,
                          const char** option_keys,
                          const char** option_vals,
                          int num_options,
                          int *out_code);

/*!
 * \brief Whether the specified context is enabled.
 *
 * \param ctx The context to be checked.
 * \param out_enabled whether the ctx is enabled.
 * \return Whether the function is successful.
 */
TVM_DLL int TVMContextEnabled(TVMContext ctx,
                              int* out_enabled);

/*!
 * \brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param dtype The array data type.
 * \param ctx The ctx this array sits on.
 * \param out The output handle.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMArrayAlloc(const tvm_index_t* shape,
                          tvm_index_t ndim,
                          TVMType dtype,
                          TVMContext ctx,
                          TVMArrayHandle* out);
/*!
 * \brief Free the TVM Array.
 * \param handle The array handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMArrayFree(TVMArrayHandle handle);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMArrayCopyFromTo(TVMArrayHandle from,
                               TVMArrayHandle to,
                               TVMStreamHandle stream);
/*!
 * \brief Wait until all computations on stream completes.
 * \param ctx The ctx to be synchronized.
 * \param stream The stream to be synchronized.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMSynchronize(TVMContext ctx, TVMStreamHandle stream);

/*!
 * \brief Free the function when it is no longer needed.
 * \param func The function handle
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMFuncFree(TVMFunctionHandle func);

/*!
 * \brief Call a Packed TVM Function.
 *
 * \param func node handle of the function.
 * \param arg_values The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 *
 * \param ret_val The return value.
 * \param ret_type_code the type code of return value.
 *
 * \return 0 when success, -1 when failure happens
 * \note TVM calls always exchanges with type bits=64, lanes=1
 *
 * \note API calls always exchanges with type bits=64, lanes=1
 *   If API call returns container handles (e.g. FunctionHandle)
 *   these handles should be managed by the front-end.
 *   The front-end need to call free function (e.g. TVMFuncFree)
 *   to free these handles.
 */
TVM_DLL int TVMFuncCall(TVMFunctionHandle func,
                        TVMValue* arg_values,
                        int* type_codes,
                        int num_args,
                        TVMValue* ret_val,
                        int* ret_type_code);

/*!
 * \brief Set the return value of TVMPackedCFunc.
 *
 *  This function is called by TVMPackedCFunc to set the return value.
 *  When this function is not called, the function returns null by default.
 *
 * \param ret The return value handle, pass by ret in TVMPackedCFunc
 * \param value The value to be returned.
 * \param type_code The type of the value to be returned.
 */
TVM_DLL int TVMCFuncSetReturn(TVMRetValueHandle ret,
                              TVMValue value,
                              int type_code);

/*!
 * \brief C type of packed function.
 *
 * \param args The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 * \param ret The return value handle.
 * \param resource_handle The handle additional resouce handle from fron-end.
 *
 * \sa TVMCFuncSetReturn
 */
typedef void (*TVMPackedCFunc)(
    TVMValue* args,
    int* type_codes,
    int num_args,
    TVMRetValueHandle ret,
    void* resource_handle);

/*!
 * \brief C callback to free the resource handle in C packed function.
 * \param resource_handle The handle additional resouce handle from fron-end.
 */
typedef void (*TVMPackedCFuncFinalizer)(void* resource_handle);

/*!
 * \brief Wrap a TVMPackedCFunc to become a FunctionHandle.
 *
 * The resource_handle will be managed by TVM API, until the function is no longer used.
 *
 * \param func The packed C function.
 * \param resource_handle The resource handle from front-end, can be NULL.
 * \param fin The finalizer on resource handle when the FunctionHandle get freed, can be NULL
 * \param out the result function handle.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMFuncCreateFromCFunc(TVMPackedCFunc func,
                                   void* resource_handle,
                                   TVMPackedCFuncFinalizer fin,
                                   TVMFunctionHandle *out);

/*!
 * \brief Register the function to runtime's global table.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * \param name The name of the function.
 * \param f The function to be registered.
 */
TVM_DLL int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f);

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer.
 *
 * \note The function handle of global function is managed by TVM runtime,
 *  So TVMFuncFree is should not be called when it get deleted.
 */
TVM_DLL int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out);

/*!
 * \brief List all the globally registered function name
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMFuncListGlobalNames(int *out_size,
                                   const char*** out_array);
}  // TVM_EXTERN_C

#endif  // TVM_RUNTIME_C_RUNTIME_API_H_
