/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCL_H_
#define MSCCL_H_

#include <mscclpp/gpu.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>
/* Opaque handle to communicator */
typedef struct mscclComm* mscclComm_t;
#define MSCCL_COMM_NULL NULL

#define MSCCL_UNIQUE_ID_BYTES 128
typedef struct {
  char internal[MSCCL_UNIQUE_ID_BYTES];
} mscclUniqueId;

/* Error type */
typedef enum {
  mscclSuccess = 0,
  mscclUnhandledCudaError = 1,
  mscclSystemError = 2,
  mscclInternalError = 3,
  mscclInvalidArgument = 4,
  mscclInvalidUsage = 5,
  mscclRemoteError = 6,
  mscclInProgress = 7,
  mscclNumResults = 8
} mscclResult_t;

#define MSCCL_CONFIG_UNDEF_INT INT_MIN
#define MSCCL_CONFIG_UNDEF_PTR NULL
#define MSCCL_SPLIT_NOCOLOR -1

/* Communicator configuration. Users can assign value to attributes to specify the
 * behavior of a communicator. */
typedef struct mscclConfig_v21700 {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;
  /* attributes that users are able to customize. */
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  const char* netName;
  int splitShare;
} mscclConfig_t;

/* Config initializer must be assigned to initialize config structure when it is created.
 * Not initialized config will result in MSCCL error. */
#define MSCCL_CONFIG_INITIALIZER                                                   \
  {                                                                                \
    sizeof(mscclConfig_t),                                    /* size */           \
        0xcafebeef,                                           /* magic */          \
        MSCCL_VERSION(MSCCL_MAJOR, MSCCL_MINOR, MSCCL_PATCH), /* version */        \
        MSCCL_CONFIG_UNDEF_INT,                               /* blocking */       \
        MSCCL_CONFIG_UNDEF_INT,                               /* cgaClusterSize */ \
        MSCCL_CONFIG_UNDEF_INT,                               /* minCTAs */        \
        MSCCL_CONFIG_UNDEF_INT,                               /* maxCTAs */        \
        MSCCL_CONFIG_UNDEF_PTR,                               /* netName */        \
        MSCCL_CONFIG_UNDEF_INT                                /* splitShare */     \
  }

/* Return the MSCCL_VERSION_CODE of the MSCCL library in the supplied integer.
 * This integer is coded with the MAJOR, MINOR and PATCH level of the
 * MSCCL library
 */
mscclResult_t mscclGetVersion(int* version);
mscclResult_t pmscclGetVersion(int* version);

/* Generates an Id to be used in mscclCommInitRank. mscclGetUniqueId should be
 * called once and the Id should be distributed to all ranks in the
 * communicator before calling mscclCommInitRank. */
mscclResult_t mscclGetUniqueId(mscclUniqueId* uniqueId);
mscclResult_t pmscclGetUniqueId(mscclUniqueId* uniqueId);

/* Create a new communicator (multi thread/process version) with a configuration
 * set by users. */
mscclResult_t mscclCommInitRankConfig(mscclComm_t* comm, int nranks, mscclUniqueId commId, int rank,
                                      mscclConfig_t* config);
mscclResult_t pmscclCommInitRankConfig(mscclComm_t* comm, int nranks, mscclUniqueId commId,
                                       int rank, mscclConfig_t* config);

/* Creates a new communicator (multi thread/process version).
 * rank must be between 0 and nranks-1 and unique within a communicator clique.
 * Each rank is associated to a CUDA device, which has to be set before calling
 * mscclCommInitRank.
 * mscclCommInitRank implicitly syncronizes with other ranks, so it must be
 * called by different threads/processes or use mscclGroupStart/mscclGroupEnd. */
mscclResult_t mscclCommInitRank(mscclComm_t* comm, int nranks, mscclUniqueId commId, int rank);
mscclResult_t pmscclCommInitRank(mscclComm_t* comm, int nranks, mscclUniqueId commId, int rank);

/* Creates a clique of communicators (single process version).
 * This is a convenience function to create a single-process communicator clique.
 * Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(mscclComm_t).
 * If devlist is NULL, the first ndev CUDA devices are used.
 * Order of devlist defines user-order of processors within the communicator. */
mscclResult_t mscclCommInitAll(mscclComm_t* comm, int ndev, const int* devlist);
mscclResult_t pmscclCommInitAll(mscclComm_t* comm, int ndev, const int* devlist);

/* Finalize a communicator. mscclCommFinalize flushes all issued communications,
 * and marks communicator state as mscclInProgress. The state will change to mscclSuccess
 * when the communicator is globally quiescent and related resources are freed; then,
 * calling mscclCommDestroy can locally free the rest of the resources (e.g. communicator
 * itself) without blocking. */
mscclResult_t mscclCommFinalize(mscclComm_t comm);
mscclResult_t pmscclCommFinalize(mscclComm_t comm);

/* Frees local resources associated with communicator object. */
mscclResult_t mscclCommDestroy(mscclComm_t comm);
mscclResult_t pmscclCommDestroy(mscclComm_t comm);

/* Frees resources associated with communicator object and aborts any operations
 * that might still be running on the device. */
mscclResult_t mscclCommAbort(mscclComm_t comm);
mscclResult_t pmscclCommAbort(mscclComm_t comm);

/* Creates one or more communicators from an existing one.
 * Ranks with the same color will end up in the same communicator.
 * Within the new communicator, key will be used to order ranks.
 * MSCCL_SPLIT_NOCOLOR as color will indicate the rank will not be part of any group
 * and will therefore return a NULL communicator.
 * If config is NULL, the new communicator will inherit the original communicator's
 * configuration*/
mscclResult_t mscclCommSplit(mscclComm_t comm, int color, int key, mscclComm_t* newcomm,
                             mscclConfig_t* config);
mscclResult_t pmscclCommSplit(mscclComm_t comm, int color, int key, mscclComm_t* newcomm,
                              mscclConfig_t* config);

/* Returns a string for each error code. */
const char* mscclGetErrorString(mscclResult_t result);
const char* pmscclGetErrorString(mscclResult_t result);

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
const char* mscclGetLastError(mscclComm_t comm);
const char* pmscclGetLastError(mscclComm_t comm);

/* Checks whether the comm has encountered any asynchronous errors */
mscclResult_t mscclCommGetAsyncError(mscclComm_t comm, mscclResult_t* asyncError);
mscclResult_t pmscclCommGetAsyncError(mscclComm_t comm, mscclResult_t* asyncError);

/* Gets the number of ranks in the communicator clique. */
mscclResult_t mscclCommCount(const mscclComm_t comm, int* count);
mscclResult_t pmscclCommCount(const mscclComm_t comm, int* count);

/* Returns the cuda device number associated with the communicator. */
mscclResult_t mscclCommCuDevice(const mscclComm_t comm, int* device);
mscclResult_t pmscclCommCuDevice(const mscclComm_t comm, int* device);

/* Returns the user-ordered "rank" associated with the communicator. */
mscclResult_t mscclCommUserRank(const mscclComm_t comm, int* rank);
mscclResult_t pmscclCommUserRank(const mscclComm_t comm, int* rank);

/* Reduction operation selector */
typedef enum { mscclNumOps_dummy = 5 } mscclRedOp_dummy_t;
typedef enum {
  mscclSum = 0,
  mscclProd = 1,
  mscclMax = 2,
  mscclMin = 3,
  mscclAvg = 4,
  /* mscclNumOps: The number of built-in mscclRedOp_t values. Also
   * serves as the least possible value for dynamic mscclRedOp_t's
   * as constructed by mscclRedOpCreate*** functions. */
  mscclNumOps = 5,
  /* mscclMaxRedOp: The largest valid value for mscclRedOp_t.
   * It is defined to be the largest signed value (since compilers
   * are permitted to use signed enums) that won't grow
   * sizeof(mscclRedOp_t) when compared to previous MSCCL versions to
   * maintain ABI compatibility. */
  mscclMaxRedOp = 0x7fffffff >> (32 - 8 * sizeof(mscclRedOp_dummy_t))
} mscclRedOp_t;

/* Data types */
typedef enum {
  mscclInt8 = 0,
  mscclChar = 0,
  mscclUint8 = 1,
  mscclInt32 = 2,
  mscclInt = 2,
  mscclUint32 = 3,
  mscclInt64 = 4,
  mscclUint64 = 5,
  mscclFloat16 = 6,
  mscclHalf = 6,
  mscclFloat32 = 7,
  mscclFloat = 7,
  mscclFloat64 = 8,
  mscclDouble = 8,
#if defined(__CUDA_BF16_TYPES_EXIST__) && defined(__CUDA_FP8_TYPES_EXIST__)
  mscclBfloat16 = 9,
  mscclFp8E4M3 = 10,
  mscclFp8E5M2 = 11,
  mscclNumTypes = 12
#elif defined(__CUDA_BF16_TYPES_EXIST__)
  mscclBfloat16 = 9,
  mscclNumTypes = 10
#else
  mscclNumTypes = 9
#endif
} mscclDataType_t;

/* mscclScalarResidence_t: Location and dereferencing logic for scalar arguments. */
typedef enum {
  /* mscclScalarDevice: The scalar is in device-visible memory and will be
   * dereferenced while the collective is running. */
  mscclScalarDevice = 0,

  /* mscclScalarHostImmediate: The scalar is in host-visible memory and will be
   * dereferenced before the mscclRedOpCreate***() function returns. */
  mscclScalarHostImmediate = 1
} mscclScalarResidence_t;

/*
 * mscclRedOpCreatePreMulSum
 *
 * Creates a new reduction operator which pre-multiplies input values by a given
 * scalar locally before reducing them with peer values via summation. For use
 * only with collectives launched against *comm* and *datatype*. The
 * *residence* argument indicates how/when the memory pointed to by *scalar*
 * will be dereferenced. Upon return, the newly created operator's handle
 * is stored in *op*.
 */
mscclResult_t mscclRedOpCreatePreMulSum(mscclRedOp_t* op, void* scalar, mscclDataType_t datatype,
                                        mscclScalarResidence_t residence, mscclComm_t comm);
mscclResult_t pmscclRedOpCreatePreMulSum(mscclRedOp_t* op, void* scalar, mscclDataType_t datatype,
                                         mscclScalarResidence_t residence, mscclComm_t comm);

/*
 * mscclRedOpDestroy
 *
 * Destroys the reduction operator *op*. The operator must have been created by
 * mscclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
 * destroyed as soon as the last MSCCL function which is given that operator returns.
 */
mscclResult_t mscclRedOpDestroy(mscclRedOp_t op, mscclComm_t comm);
mscclResult_t pmscclRedOpDestroy(mscclRedOp_t op, mscclComm_t comm);

/*
 * Collective communication operations
 *
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * They return when operations have been enqueued on the CUDA stream.
 *
 * Since they may perform inter-CPU synchronization, each call has to be done
 * from a different thread or process, or need to use Group Semantics (see
 * below).
 */

/*
 * Reduce
 *
 * Reduces data arrays of length count in sendbuff into recvbuff using op
 * operation.
 * recvbuff may be NULL on all calls except for root device.
 * root is the rank (not the CUDA device) where data will reside after the
 * operation is complete.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mscclResult_t mscclReduce(const void* sendbuff, void* recvbuff, size_t count,
                          mscclDataType_t datatype, mscclRedOp_t op, int root, mscclComm_t comm,
                          cudaStream_t stream);
mscclResult_t pmscclReduce(const void* sendbuff, void* recvbuff, size_t count,
                           mscclDataType_t datatype, mscclRedOp_t op, int root, mscclComm_t comm,
                           cudaStream_t stream);

/*
 * (deprecated) Broadcast (in-place)
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * This operation is implicitly in place.
 */
mscclResult_t mscclBcast(void* buff, size_t count, mscclDataType_t datatype, int root,
                         mscclComm_t comm, cudaStream_t stream);
mscclResult_t pmscclBcast(void* buff, size_t count, mscclDataType_t datatype, int root,
                          mscclComm_t comm, cudaStream_t stream);

/*
 * Broadcast
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mscclResult_t mscclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                             mscclDataType_t datatype, int root, mscclComm_t comm,
                             cudaStream_t stream);
mscclResult_t pmscclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                              mscclDataType_t datatype, int root, mscclComm_t comm,
                              cudaStream_t stream);

/*
 * All-Reduce
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mscclResult_t mscclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                             mscclDataType_t datatype, mscclRedOp_t op, mscclComm_t comm,
                             cudaStream_t stream);
mscclResult_t pmscclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                              mscclDataType_t datatype, mscclRedOp_t op, mscclComm_t comm,
                              cudaStream_t stream);

/*
 * Reduce-Scatter
 *
 * Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
mscclResult_t mscclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                 mscclDataType_t datatype, mscclRedOp_t op, mscclComm_t comm,
                                 cudaStream_t stream);
mscclResult_t pmscclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                                  mscclDataType_t datatype, mscclRedOp_t op, mscclComm_t comm,
                                  cudaStream_t stream);

/*
 * All-Gather
 *
 * Each device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
mscclResult_t mscclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                             mscclDataType_t datatype, mscclComm_t comm, cudaStream_t stream);
mscclResult_t pmscclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                              mscclDataType_t datatype, mscclComm_t comm, cudaStream_t stream);

/*
 * Send
 *
 * Send data from sendbuff to rank peer.
 *
 * Rank peer needs to call mscclRecv with the same datatype and the same count from this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple mscclSend and mscclRecv operations
 * need to progress concurrently to complete, they must be fused within a mscclGroupStart/
 * mscclGroupEnd section.
 */
mscclResult_t mscclSend(const void* sendbuff, size_t count, mscclDataType_t datatype, int peer,
                        mscclComm_t comm, cudaStream_t stream);
mscclResult_t pmscclSend(const void* sendbuff, size_t count, mscclDataType_t datatype, int peer,
                         mscclComm_t comm, cudaStream_t stream);

/*
 * Receive
 *
 * Receive data from rank peer into recvbuff.
 *
 * Rank peer needs to call mscclSend with the same datatype and the same count to this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple mscclSend and mscclRecv operations
 * need to progress concurrently to complete, they must be fused within a mscclGroupStart/
 * mscclGroupEnd section.
 */
mscclResult_t pmscclRecv(void* recvbuff, size_t count, mscclDataType_t datatype, int peer,
                         mscclComm_t comm, cudaStream_t stream);
mscclResult_t mscclRecv(void* recvbuff, size_t count, mscclDataType_t datatype, int peer,
                        mscclComm_t comm, cudaStream_t stream);

/* All-To-All
 *
 * Device (i) send (j)th block of data to device (j) and be placed as (i)th
 * block. Each block for sending/receiving has count elements, which means
 * that recvbuff and sendbuff should have a size of nranks*count elements.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
mscclResult_t mscclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                            mscclDataType_t datatype, mscclComm_t comm, cudaStream_t stream);
mscclResult_t pmscclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
                             mscclDataType_t datatype, mscclComm_t comm, cudaStream_t stream);
/*! @brief Opaque handle to MSCCL algorithm */
typedef int mscclAlgoHandle_t;

/*! @brief MSCCL Load Algorithm
 *
 * @details Load MSCCL algorithm file specified in mscclAlgoFilePath and return
 * its handle via mscclAlgoHandle. This API is expected to be called by MSCCL
 * scheduler instead of end users.
 */
mscclResult_t mscclLoadAlgo(const char* mscclAlgoFilePath, mscclAlgoHandle_t* mscclAlgoHandle,
                            int rank);
mscclResult_t pmscclLoadAlgo(const char* mscclAlgoFilePath, mscclAlgoHandle_t* mscclAlgoHandle,
                             int rank);

/*! @brief MSCCL Run Algorithm
 *
 * @details Run MSCCL algorithm specified by mscclAlgoHandle. The parameter
 * list merges all possible parameters required by different operations as this
 * is a general-purposed API. This API is expected to be called by MSCCL
 * scheduler instead of end users.
 */
mscclResult_t mscclRunAlgo(const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
                           void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
                           size_t count, mscclDataType_t dataType, int root, int peer,
                           mscclRedOp_t op, mscclAlgoHandle_t mscclAlgoHandle, mscclComm_t comm,
                           cudaStream_t stream);
mscclResult_t pmscclRunAlgo(const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
                            void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
                            size_t count, mscclDataType_t dataType, int root, int peer,
                            mscclRedOp_t op, mscclAlgoHandle_t mscclAlgoHandle, mscclComm_t comm,
                            cudaStream_t stream);

/*! @brief MSCCL Load Algorithm
 *
 * @details Unload MSCCL algorithm previous loaded using its handle. This API
 * is expected to be called by MSCCL scheduler instead of end users.
 */
mscclResult_t mscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle);
mscclResult_t pmscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle);

/*
 * Group semantics
 *
 * When managing multiple GPUs from a single thread, and since MSCCL collective
 * calls may perform inter-CPU synchronization, we need to "group" calls for
 * different ranks/devices into a single call.
 *
 * Grouping MSCCL calls as being part of the same collective operation is done
 * using mscclGroupStart and mscclGroupEnd. mscclGroupStart will enqueue all
 * collective calls until the mscclGroupEnd call, which will wait for all calls
 * to be complete. Note that for collective communication, mscclGroupEnd only
 * guarantees that the operations are enqueued on the streams, not that
 * the operation is effectively done.
 *
 * Both collective communication and mscclCommInitRank can be used in conjunction
 * of mscclGroupStart/mscclGroupEnd, but not together.
 *
 * Group semantics also allow to fuse multiple operations on the same device
 * to improve performance (for aggregated collective calls), or to permit
 * concurrent progress of multiple send/receive operations.
 */

/*
 * Group Start
 *
 * Start a group call. All calls to MSCCL until mscclGroupEnd will be fused into
 * a single MSCCL operation. Nothing will be started on the CUDA stream until
 * mscclGroupEnd.
 */
mscclResult_t mscclGroupStart();
mscclResult_t pmscclGroupStart();

/*
 * Group End
 *
 * End a group call. Start a fused MSCCL operation consisting of all calls since
 * mscclGroupStart. Operations on the CUDA stream depending on the MSCCL operations
 * need to be called after mscclGroupEnd.
 */
mscclResult_t mscclGroupEnd();
mscclResult_t pmscclGroupEnd();

#ifdef __cplusplus
}  // end extern "C"
#endif

#endif  // end include guard
