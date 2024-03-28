/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "msccl.cuh"

namespace tvm {
namespace runtime {

template <typename T>
cudaError_t allreduce(const T* buff, T* scratch, T* resultBuff,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels, int rank,
                      int nRanksPerNode, int worldSize, size_t nelems, cudaStream_t stream);

MSCCL_API mscclResult_t mscclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                                       mscclDataType_t datatype, mscclRedOp_t op, mscclComm_t comm,
                                       cudaStream_t stream) {
  size_t bytes = count * mscclTypeSize(datatype);
  if (sendbuff == nullptr || recvbuff == nullptr || bytes == 0 || comm == nullptr ||
      op != mscclSum || bytes > (1 << 24)) {
    return mscclInvalidArgument;
  }

  int rank = comm->comm->bootstrap()->getRank();
  channelKey key{sendbuff, recvbuff, bytes};
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels = nullptr;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels = nullptr;

  auto it = comm->channelInfos.find(key);
  if (it == comm->channelInfos.end()) {
    // setup smChannels (src: sendbuff, dst: remote scratch buff)
    std::vector<mscclpp::SmChannel> channels =
        setupSmChannels(comm, comm->remoteScratchRegMemories, const_cast<void*>(sendbuff));
    ChannelInfo channelInfo{channels, {}, setupSmChannelDeviceHandles(channels), nullptr};
    it = comm->channelInfos.emplace(key, channelInfo).first;

    // TODO(csullivan): Consider supporting allreduce for larger transfers
    // setup smOutChannels (src: recvbuff, dst: remote recvbuff)
    // if (bytes > (1 << 24)) {
    //   std::vector<mscclpp::RegisteredMemory> remoteMemories =
    //       setupRemoteMemories(comm->comm, rank, recvbuff, bytes, mscclpp::Transport::CudaIpc);
    //   std::vector<mscclpp::SmChannel> outChannels = setupSmChannels(comm, remoteMemories,
    //   recvbuff); it->second.smOutChannels = outChannels; it->second.smOutChannelDeviceHandles =
    //   setupSmChannelDeviceHandles(outChannels);
    // }
  }

  smChannels = it->second.smChannelDeviceHandles.get();
  smOutChannels = it->second.smOutChannelDeviceHandles.get();

  switch (datatype) {
    case mscclFloat16:
      CUDACHECK(allreduce(reinterpret_cast<const half*>(sendbuff),
                          reinterpret_cast<half*>(comm->scratchBuff.get()),
                          reinterpret_cast<half*>(recvbuff), smChannels, smOutChannels, rank,
                          NRANKS_PER_NODE, comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    case mscclFloat32:
      CUDACHECK(allreduce(reinterpret_cast<const float*>(sendbuff),
                          reinterpret_cast<float*>(comm->scratchBuff.get()),
                          reinterpret_cast<float*>(recvbuff), smChannels, smOutChannels,
                          comm->comm->bootstrap()->getRank(), NRANKS_PER_NODE,
                          comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    case mscclInt32:
    case mscclUint32:
      CUDACHECK(allreduce(reinterpret_cast<const int*>(sendbuff),
                          reinterpret_cast<int*>(comm->scratchBuff.get()),
                          reinterpret_cast<int*>(recvbuff), smChannels, smOutChannels,
                          comm->comm->bootstrap()->getRank(), NRANKS_PER_NODE,
                          comm->comm->bootstrap()->getNranks(), count, stream));
      break;
    default:
      return mscclInvalidArgument;
  }
  return mscclSuccess;
}

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce_simple(mscclpp::SmChannelDeviceHandle* smChans, const T* buff, T* scratch,
                     void* resultBuff, int rank, int worldSize, size_t nelems,
                     const uint32_t flag) {
  nelems = nelems / (sizeof(int) / sizeof(T));

  const int nPeers = worldSize - 1;
  const size_t nPkts = nelems / 2;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank / 2;
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  mscclpp::SmChannelDeviceHandle smChan = smChans[peerIdx];
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;

  size_t scratchOffset = rank * nPktsPerRank * sizeof(mscclpp::LLPacket);
  size_t resultOffset = 2 * nPkts * sizeof(mscclpp::LLPacket);
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int);
  const uint2* src = reinterpret_cast<const uint2*>(reinterpret_cast<const char*>(buff) +
                                                    rank * nelemsPerRank * sizeof(int));
  uint2* dst = reinterpret_cast<uint2*>(reinterpret_cast<char*>(resultBuff) +
                                        rank * nelemsPerRank * sizeof(int));

  // Step 1. Write to scratch buffer which exposes memory to peers via cuda IPC memory
  smChan.putPackets(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid,
                    blockDim.x * nBlocksPerPeer, flag);

  // Step 2. Get data from scratch buffer, reduce data, and write result back to peer scratch
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank;
       idx += blockDim.x * gridDim.x) {
    uint2 data = make_uint2(0, 0);
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LLPacket* dstPkt =
          reinterpret_cast<mscclpp::LLPacket*>(scratch) + remoteRank * nPktsPerRank;
      uint2 val = dstPkt[idx].read(flag);
      data = add_vectors<T>(val, data);
    }
    data = add_vectors<T>(data, src[idx]);
    dst[idx] = data;

    mscclpp::LLPacket packet;
    packet.data1 = data.x;
    packet.flag1 = flag;
    packet.data2 = data.y;
    packet.flag2 = flag;
    size_t offset = resultOffset / sizeof(mscclpp::LLPacket) + (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      smChans[index].write(offset, packet);
    }
  }

  // Step 3. Update local GPU's final result from peer scratch buffers
  mscclpp::LLPacket* dstPkt =
      reinterpret_cast<mscclpp::LLPacket*>(reinterpret_cast<char*>(scratch) + resultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint2* result = reinterpret_cast<uint2*>(reinterpret_cast<char*>(resultBuff) +
                                           remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank;
       idx += blockDim.x * nBlocksPerPeer) {
    uint2 data = dstPkt[idx + dstOffset].read(flag);
    result[idx].x = data.x;
    result[idx].y = data.y;
  }
}

template <typename T>
cudaError_t allreduce(const T* buff, T* scratch, T* resultBuff,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels, int rank,
                      int nRanksPerNode, int worldSize, size_t nelems, cudaStream_t stream) {
  static uint32_t flag = 1;
  size_t num_bytes = sizeof(T) * nelems;
  ICHECK(num_bytes <= (1 << 24)) << "mscclpp allreduce expects bytes transfered < " << (1 << 24)
                                 << " but got num_bytes = " << num_bytes << " bytes";
  allreduce_simple<<<105, 1024, 0, stream>>>(smChannels, buff, scratch, resultBuff, rank, worldSize,
                                             nelems, flag++);

  return cudaGetLastError();
}

}  // namespace runtime
}  // namespace tvm
