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
#ifndef TVM_C_RUNTIME_API_H_
#define TVM_C_RUNTIME_API_H_

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


TVM_EXTERN_C {
/*! \brief type of array index. */
typedef uint32_t tvm_index_t;

/*!
 * \brief union type for arguments and return values
 *  in both runtime API and TVM API calls
 */
typedef union {
  long v_long;  // NOLINT(*)
  double v_double;
  const char* v_str;
  void* v_handle;
} TVMArg;

/*!
 * \brief The type index in TVM.
 */
typedef enum {
  kNull = 0,
  kLong = 1,
  kDouble = 2,
  kStr = 3,
  kNodeHandle = 4,
  kArrayHandle = 5
} TVMArgTypeID;

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

/*! \brief The type code in TVMDataType */
typedef enum {
  kInt = 0U,
  kUInt = 1U,
  kFloat = 2U
} TVMTypeCode;

/*!
 * \brief the data type
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 */
typedef struct {
  /*! \brief type code, in TVMTypeCode */
  uint8_t type_code;
  /*! \brief number of bits of the type */
  uint8_t bits;
  /*! \brief number of lanes, */
  uint16_t lanes;
} TVMDataType;

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
  TVMDataType dtype;
  /*! \brief The device context this array sits on */
  TVMContext ctx;
} TVMArray;

/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* TVMStreamHandle;
/*!
 * \brief Pointer to function handle that points to
 * a generated TVM function.
 */
typedef void* TVMFunctionHandle;
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
 * \return Whether the function is successful.
 */
TVM_DLL int TVMArrayAlloc(const tvm_index_t* shape,
                          tvm_index_t ndim,
                          TVMDataType dtype,
                          TVMContext ctx,
                          TVMArrayHandle* out);
/*!
 * \brief Free the TVM Array.
 * \param handle The array handle to be freed.
 */
TVM_DLL int TVMArrayFree(TVMArrayHandle handle);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 */
TVM_DLL int TVMArrayCopyFromTo(TVMArrayHandle from,
                               TVMArrayHandle to,
                               TVMStreamHandle stream);
/*!
 * \brief Wait until all computations on stream completes.
 * \param ctx The ctx to be synchronized.
 * \param stream The stream to be synchronized.
 */
TVM_DLL int TVMSynchronize(TVMContext ctx, TVMStreamHandle stream);

/*!
 * \brief Launch a generated TVM function
 * \param func function handle to be launched.
 * \param args The arguments
 * \param arg_type_ids The type id of the arguments
 * \param num_args Number of arguments.
 * \param stream The stream this function to be launched on.
 */
TVM_DLL int TVMLaunch(TVMFunctionHandle func,
                      TVMArg* args,
                      int* arg_type_ids,
                      int num_args,
                      TVMStreamHandle stream);
}  // TVM_EXTERN_C

#endif  // TVM_C_RUNTIME_API_H_
