/*!
 *  Copyright (c) 2018 by Contributors
 * \file data_buffer.cc
 * \brief Buffer related API for VTA.
 * \note Buffer API remains stable across VTA designes.
 */
#include "./data_buffer.h"

void* VTABufferAlloc(size_t size) {
  return vta::DataBuffer::Alloc(size);
}

void VTABufferFree(void* buffer) {
  vta::DataBuffer::Free(vta::DataBuffer::FromHandle(buffer));
}

void VTABufferCopy(const void* from,
                   size_t from_offset,
                   void* to,
                   size_t to_offset,
                   size_t size,
                   int kind_mask) {
  vta::DataBuffer* from_buffer = nullptr;
  vta::DataBuffer* to_buffer = nullptr;

  if (kind_mask & 2) {
    from_buffer = vta::DataBuffer::FromHandle(from);
    from = from_buffer->virt_addr();
  }
  if (kind_mask & 1) {
    to_buffer = vta::DataBuffer::FromHandle(to);
    to = to_buffer->virt_addr();
  }
  if (from_buffer) {
    from_buffer->InvalidateCache(from_offset, size);
  }

  memcpy(static_cast<char*>(to) + to_offset,
         static_cast<const char*>(from) + from_offset,
         size);
  if (to_buffer) {
    to_buffer->FlushCache(to_offset, size);
  }
}
