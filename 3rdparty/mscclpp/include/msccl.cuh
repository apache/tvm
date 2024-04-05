// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "msccl.h"

#define MSCCL_API extern "C" __attribute__((visibility("default")))

#define CUDACHECK(cmd)                                                                      \
  do {                                                                                      \
    cudaError_t e = cmd;                                                                    \
    if (e != cudaSuccess) {                                                                 \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

#define NUM_CHANNELS_PER_CONNECTION 64

struct channelKey {
  const void* sendbuff;
  const void* recvbuff;
  size_t bytes;
  bool operator==(const channelKey& other) const {
    return sendbuff == other.sendbuff && recvbuff == other.recvbuff && bytes == other.bytes;
  }
};

namespace std {
template <>
struct hash<channelKey> {
  std::size_t operator()(const channelKey& k) const {
    return std::hash<const void*>()(k.sendbuff) ^ std::hash<const void*>()(k.recvbuff) ^ std::hash<size_t>()(k.bytes);
  }
};
}  // namespace std

struct ChannelInfo {
  std::vector<mscclpp::SmChannel> smChannels;
  std::vector<mscclpp::SmChannel> smOutChannels;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> smOutChannelDeviceHandles;
};

struct mscclComm {
  std::shared_ptr<mscclpp::Communicator> comm;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;

  std::unordered_map<channelKey, ChannelInfo> channelInfos;
  std::shared_ptr<char> scratchBuff;
  std::vector<mscclpp::RegisteredMemory> remoteScratchRegMemories;
};

static size_t mscclTypeSize(mscclDataType_t type) {
  switch (type) {
    case mscclInt8:
    case mscclUint8:
      return 1;
    case mscclFloat16:
      return 2;
    case mscclInt32:
    case mscclUint32:
      return 4;
    case mscclInt64:
    case mscclUint64:
      return 8;
    case mscclFloat32:
      return 4;
    case mscclFloat64:
      return 8;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case mscclBfloat16:
      return 2;
#endif  // defined(__CUDA_BF16_TYPES_EXIST__)
#if defined(__CUDA_FP8_TYPES_EXIST__)
    case mscclFp8E4M3:
    case mscclFp8E5M2:
      return 1;
#endif  // defined(__CUDA_FP8_TYPES_EXIST__)
    case mscclNumTypes:
      return 0;
  }
  return 0;
}

static mscclpp::Transport getTransport(int, int) { return mscclpp::Transport::CudaIpc; }

static std::vector<mscclpp::RegisteredMemory> setupRemoteMemories(std::shared_ptr<mscclpp::Communicator> comm, int rank,
                                                                  void* buff, size_t bytes,
                                                                  mscclpp::TransportFlags transport) {
  std::vector<mscclpp::RegisteredMemory> remoteMemories;
  mscclpp::RegisteredMemory memory = comm->registerMemory(buff, bytes, transport);
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemoryFutures;
  for (int i = 0; i < comm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    remoteRegMemoryFutures.push_back(comm->recvMemoryOnSetup(i, 0));
    comm->sendMemoryOnSetup(memory, i, 0);
  }
  comm->setup();
  std::transform(remoteRegMemoryFutures.begin(), remoteRegMemoryFutures.end(), std::back_inserter(remoteMemories),
                 [](const auto& future) { return future.get(); });
  return remoteMemories;
}

static std::vector<mscclpp::SmChannel> setupSmChannels(mscclComm_t comm,
                                                       const std::vector<mscclpp::RegisteredMemory>& remoteMemories,
                                                       void* src) {
  std::vector<mscclpp::SmChannel> channels;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>>& smSemaphores = comm->smSemaphores;
  size_t nConnections = comm->connections.size();
  for (size_t idx = 0; idx < NUM_CHANNELS_PER_CONNECTION; ++idx) {
    for (size_t cid = 0; cid < nConnections; ++cid) {
      if (comm->connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        channels.emplace_back(smSemaphores[idx * nConnections + cid], remoteMemories[cid], src, nullptr);
      }
    }
  }
  return channels;
}

static std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> setupSmChannelDeviceHandles(
    const std::vector<mscclpp::SmChannel>& smChannels) {
  std::vector<mscclpp::DeviceHandle<mscclpp::SmChannel>> smChannelDeviceHandles;
  std::transform(smChannels.begin(), smChannels.end(), std::back_inserter(smChannelDeviceHandles),
                 [](const mscclpp::SmChannel& smChannel) { return mscclpp::deviceHandle(smChannel); });
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SmChannel>> ptr =
      mscclpp::allocSharedCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(smChannelDeviceHandles.size());
  mscclpp::memcpyCuda<mscclpp::DeviceHandle<mscclpp::SmChannel>>(ptr.get(), smChannelDeviceHandles.data(),
                       smChannelDeviceHandles.size(), cudaMemcpyHostToDevice);
  return ptr;
}

MSCCL_API mscclResult_t mscclGetVersion(int* version) {
  if (version == nullptr) return mscclInvalidArgument;
  *version = MSCCLPP_VERSION;
  return mscclSuccess;
}

MSCCL_API mscclResult_t mscclGetUniqueId(mscclUniqueId* uniqueId) {
  if (uniqueId == nullptr) return mscclInvalidArgument;
  if (MSCCLPP_UNIQUE_ID_BYTES != MSCCL_UNIQUE_ID_BYTES) return mscclInternalError;
  mscclpp::UniqueId id = mscclpp::TcpBootstrap::createUniqueId();
  memcpy(uniqueId, &id, sizeof(mscclUniqueId));
  return mscclSuccess;
}

MSCCL_API mscclResult_t mscclCommInitRankConfig(mscclComm_t*, int, mscclUniqueId, int,
                                                mscclConfig_t*) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclCommInitRank(mscclComm_t* comm, int nranks, mscclUniqueId commId, int rank) {
  if (comm == nullptr) return mscclInvalidArgument;
  if (nranks < 0 || rank < 0 || rank >= nranks) return mscclInvalidArgument;
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nranks);
  mscclpp::UniqueId id;
  memcpy(id.data(), &commId, sizeof(mscclUniqueId));
  bootstrap->initialize(id);
  std::shared_ptr<mscclpp::Communicator> mscclppComm = std::make_shared<mscclpp::Communicator>(bootstrap);
  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;

  for (int i = 0; i < mscclppComm->bootstrap()->getNranks(); i++) {
    if (i == rank) continue;
    mscclpp::Transport transport = getTransport(rank, i);
    connectionFutures.push_back(mscclppComm->connectOnSetup(i, 0, transport));
  }
  mscclppComm->setup();

  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
                 [](const auto& future) { return future.get(); });

  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
  for (size_t idx = 0; idx < NUM_CHANNELS_PER_CONNECTION; ++idx) {
    for (size_t cid = 0; cid < connections.size(); ++cid) {
      if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
        smSemaphores.emplace_back(
            std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(*(mscclppComm), connections[cid]));
      }
    }
  }
  mscclppComm->setup();

  mscclComm* commPtr = new mscclComm();
  commPtr->comm = mscclppComm;
  commPtr->connections = std::move(connections);
  commPtr->smSemaphores = std::move(smSemaphores);
  commPtr->scratchBuff = mscclpp::allocExtSharedCuda<char>(SCRATCH_SIZE);
  commPtr->remoteScratchRegMemories =
      setupRemoteMemories(commPtr->comm, rank, commPtr->scratchBuff.get(), SCRATCH_SIZE, mscclpp::Transport::CudaIpc);

  *comm = commPtr;
  return mscclSuccess;
}

MSCCL_API mscclResult_t mscclCommInitAll(mscclComm_t*, int, const int*) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclCommFinalize(mscclComm_t comm) {
  comm->comm->bootstrap()->barrier();
  return mscclSuccess;
}

MSCCL_API mscclResult_t mscclCommDestroy(mscclComm_t comm) {
  if (comm == nullptr) return mscclInvalidArgument;
  delete comm;
  return mscclSuccess;
}

MSCCL_API mscclResult_t mscclCommAbort(mscclComm_t) { return mscclSuccess; }

MSCCL_API mscclResult_t mscclCommSplit(mscclComm_t, int, int, mscclComm_t*, mscclConfig_t*) {
  return mscclInternalError;
}

MSCCL_API const char* mscclGetErrorString(mscclResult_t result) {
  switch (result) {
    case mscclSuccess:
      return "no error";
    case mscclUnhandledCudaError:
      return "unhandled cuda error (run with MSCCL_DEBUG=INFO for details)";
    case mscclSystemError:
      return "unhandled system error (run with MSCCL_DEBUG=INFO for details)";
    case mscclInternalError:
      return "internal error - please report this issue to the MSCCL developers";
    case mscclInvalidArgument:
      return "invalid argument (run with MSCCL_DEBUG=WARN for details)";
    case mscclInvalidUsage:
      return "invalid usage (run with MSCCL_DEBUG=WARN for details)";
    case mscclRemoteError:
      return "remote process exited or there was a network error";
    case mscclInProgress:
      return "MSCCL operation in progress";
    default:
      return "unknown result code";
  }
}

MSCCL_API const char* mscclGetLastError(mscclComm_t) { return nullptr; }

MSCCL_API mscclResult_t mscclCommGetAsyncError(mscclComm_t, mscclResult_t* asyncError) {
  if (asyncError == nullptr) return mscclInvalidArgument;
  *asyncError = mscclSuccess;
  return mscclSuccess;
}

MSCCL_API mscclResult_t mscclCommCount(const mscclComm_t comm, int* count) {
  if (comm == nullptr || count == nullptr) return mscclInvalidArgument;
  *count = comm->comm->bootstrap()->getNranks();
  return mscclSuccess;
}

MSCCL_API mscclResult_t mscclCommCuDevice(const mscclComm_t comm, int* device) {
  if (comm == nullptr || device == nullptr) return mscclInvalidArgument;
  *device = comm->comm->bootstrap()->getRank();
  return mscclSuccess;
}

MSCCL_API mscclResult_t mscclCommUserRank(const mscclComm_t comm, int* rank) {
  if (comm == nullptr || rank == nullptr) return mscclInvalidArgument;
  *rank = comm->comm->bootstrap()->getRank();
  return mscclSuccess;
}

MSCCL_API mscclResult_t mscclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                                       mscclDataType_t datatype, mscclComm_t comm,
                                       cudaStream_t stream) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclRedOpCreatePreMulSum(mscclRedOp_t*, void*, mscclDataType_t,
                                                  mscclScalarResidence_t, mscclComm_t) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclRedOpDestroy(mscclRedOp_t, mscclComm_t) { return mscclInternalError; }

MSCCL_API mscclResult_t mscclReduce(const void*, void*, size_t, mscclDataType_t, mscclRedOp_t, int,
                                    mscclComm_t, cudaStream_t) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclBcast(void*, size_t, mscclDataType_t, int, mscclComm_t, cudaStream_t) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclBroadcast(const void*, void*, size_t, mscclDataType_t, int,
                                       mscclComm_t, cudaStream_t) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclReduceScatter(const void*, void*, size_t, mscclDataType_t,
                                           mscclRedOp_t, mscclComm_t, cudaStream_t) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclSend(const void*, size_t, mscclDataType_t, int, mscclComm_t,
                                  cudaStream_t) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclRecv(void*, size_t, mscclDataType_t, int, mscclComm_t, cudaStream_t) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclAllToAll(const void*, void*, size_t, mscclDataType_t, mscclComm_t,
                                      cudaStream_t) {
  return mscclInternalError;
}

MSCCL_API mscclResult_t mscclGroupStart() { return mscclSuccess; }

MSCCL_API mscclResult_t mscclGroupEnd() { return mscclSuccess; }
