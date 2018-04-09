/*!
 *  Copyright (c) 2018 by Contributors
 * \file data_buffer.h
 * \brief VTA runtime internal data buffer structure.
 */
#ifndef VTA_DATA_BUFFER_H_
#define VTA_DATA_BUFFER_H_

#include <vta/driver.h>
#include <vta/runtime.h>
#include <cassert>
#include <cstring>

namespace vta {

/*! \brief Enable coherent access between VTA and CPU. */
static const bool kBufferCoherent = true;

/*!
 * \brief Data buffer represents data on CMA.
 */
struct DataBuffer {
  /*! \return Virtual address of the data. */
  void* virt_addr() const {
    return data_;
  }
  /*! \return Physical address of the data. */
  uint32_t phy_addr() const {
    return phy_addr_;
  }
  /*!
   * \brief Invalidate the cache of given location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void InvalidateCache(size_t offset, size_t size) {
    if (!kBufferCoherent) {
      VTAInvalidateCache(reinterpret_cast<void*>(phy_addr_ + offset), size);
    }
  }
  /*!
   * \brief Invalidate the cache of certain location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void FlushCache(size_t offset, size_t size) {
    if (!kBufferCoherent) {
      VTAFlushCache(reinterpret_cast<void*>(phy_addr_ + offset), size);
    }
  }
  /*!
   * \brief Allocate a buffer of a given size.
   * \param size The size of the buffer.
   */
  static DataBuffer* Alloc(size_t size) {
    void* data = VTAMemAlloc(size, 1);
    assert(data != nullptr);
    DataBuffer* buffer = new DataBuffer();
    buffer->data_ = data;
    buffer->phy_addr_ = VTAGetMemPhysAddr(data);
    return buffer;
  }
  /*!
   * \brief Free the data buffer.
   * \param buffer The buffer to be freed.
   */
  static void Free(DataBuffer* buffer) {
    VTAMemFree(buffer->data_);
    delete buffer;
  }
  /*!
   * \brief Create data buffer header from buffer ptr.
   * \param buffer The buffer pointer.
   * \return The corresponding data buffer header.
   */
  static DataBuffer* FromHandle(const void* buffer) {
    return const_cast<DataBuffer*>(
        reinterpret_cast<const DataBuffer*>(buffer));
  }

 private:
  /*! \brief The internal data. */
  void* data_;
  /*! \brief The physical address of the buffer, excluding header. */
  uint32_t phy_addr_;
};

}  // namespace vta

#endif  // VTA_DATA_BUFFER_H_
