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

/*!
 * \file src/runtime/relax_vm/hexagon/builtin.cc
 * \brief The hexagon graph related builtin functions for Relax virtual machine.
 */

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/vm.h>

#include "../../hexagon/hexagon_device_api.h"
namespace tvm {
namespace runtime {
namespace relax_vm {

TVM_REGISTER_GLOBAL("vm.builtin.hexagon.dma_copy")
    .set_body_typed([](TVMArgValue vm_ptr, NDArray src_arr, NDArray dst_arr, int queue_id,
                       bool bypass_cache) {
      const DLTensor* dptr = dst_arr.operator->();
      const DLTensor* sptr = src_arr.operator->();
      void* dst = dptr->data;
      void* src = sptr->data;
      uint32_t size = 1;
      int ret = DMA_RETRY;
      for (int i = 0; i < dptr->ndim; i++) {
        size = size * dptr->shape[i];
      }
      size = size * sizeof(dptr->dtype);
      ICHECK(size > 0);
      do {
        ret = tvm::runtime::hexagon::HexagonDeviceAPI::Global()->UserDMA()->Copy(
            queue_id, dst, src, size, bypass_cache);
      } while (ret == DMA_RETRY);
      CHECK(ret == DMA_SUCCESS);
    });

TVM_REGISTER_GLOBAL("vm.builtin.hexagon.dma_wait")
    .set_body_typed([](TVMArgValue vm_ptr, int queue_id, int inflight_dma) {
      ICHECK(inflight_dma >= 0);
      tvm::runtime::hexagon::HexagonDeviceAPI::Global()->UserDMA()->Wait(queue_id, inflight_dma);
    });

NDArray AllocNDArrayFromOffsets(TVMArgValue vm_ptr, Storage storage, uint64_t offset,
                                Storage data_storage, ShapeTuple data_storage_offsets,
                                ShapeTuple logical_shape, ShapeTuple shape_2d, DLDataType dtype) {
  auto* storage_obj = storage.operator->();

  auto* container = storage_obj->CreateNDArrayContainer(offset, logical_shape, dtype);

  size_t needed_size = sizeof(void*) * shape_2d[0];
  auto offset_ptr =
      reinterpret_cast<uint8_t*>(storage_obj->buffer.data) + static_cast<size_t>(offset);
  auto cast_offset_ptr = reinterpret_cast<uint8_t**>(offset_ptr);
  uint8_t* data_base = reinterpret_cast<uint8_t*>(data_storage->buffer.data);
  size_t indx = 0;
  for (auto elem : data_storage_offsets) {
    cast_offset_ptr[indx] = &(data_base[static_cast<size_t>(elem)]);
    indx++;
  }

  container->dl_tensor.data = reinterpret_cast<void*>(offset_ptr);
  container->dl_tensor.byte_offset = 0;

  NDArray ret(GetObjectPtr<Object>(container));
  // RAII in effect, now run the check.

  ICHECK((offset + sizeof(void*) * shape_2d[0]) <= storage_obj->buffer.size)
      << "storage allocation failure, attempted to allocate " << needed_size << " at offset "
      << offset << " in region that is " << storage_obj->buffer.size << "bytes";

  tvm::runtime::hexagon::HexagonDeviceAPI::Global()->SetPhysicalShape(
      ret.operator->(), 2, const_cast<int64_t*>(shape_2d.data()));

  return ret;
}

TVM_REGISTER_GLOBAL("vm.builtin.hexagon.alloc_discontiguous_tensor")
    .set_body_typed(AllocNDArrayFromOffsets);

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
