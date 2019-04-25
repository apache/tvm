/*!
 *  Copyright (c) 2019 by Contributors
 * \file target_data_layout_encoder.h
 * \brief uTVM data layout encoder
 */
#ifndef TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_
#define TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_

#include <dmlc/memory_io.h>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include "device/utvm_runtime.h"

namespace tvm {
namespace runtime {

/*!
 * \brief helper class for writing into `TargetDataLayoutEncoder`
 */
class Slot {
 public:
  /*!
   * \brief constructor
   * \param buf shared pointer to parent backing buffer
   * \param start_offs start byte offset of the slot in the backing buffer
   * \param size size (in bytes) of the memory region allocated for this slot
   * \param dev_start_addr start address of the slot in the device's memory
   */
  Slot(std::shared_ptr<std::vector<uint8_t>> buf, size_t start_offs, size_t size,
       void* dev_start_addr)
      : buf_(buf),
        start_offs_(start_offs),
        curr_offs_(0),
        size_(size),
        dev_start_addr_(dev_start_addr) {}

  /*!
   * \brief writes `sizeof(T)` bytes of data from `src_ptr`
   * \param src_ptr address of the buffer to be read from
   */
  template <typename T>
  void Write(const T* src_ptr) {
    Write(src_ptr, sizeof(T));
  }

  /*!
   * \brief writes `sizeof(T) * length` bytes of data from `src_ptr`
   * \param src_ptr address of the buffer to be read from
   * \param length address of the buffer to be read from
   */
  template <typename T>
  void WriteArray(const T* src_ptr, size_t length) {
    Write(src_ptr, sizeof(T) * length);
  }

  /*!
   * \brief fills this slot with data from `src_ptr`
   * \param src_ptr address of the buffer to be read from
   * \param length address of the buffer to be read from
   */
  template <typename T>
  void WriteEntire(const T* src_ptr) {
    CHECK(curr_offs_ == 0) << "slot has already been written to";
    Write(src_ptr, size_);
  }

  /*!
   * \brief writes `size` bytes of data from `src_ptr` into the backing buffer
   * \param src_ptr address of the buffer to be read from
   * \param size number of bytes to be written
   */
  void Write(const void* src_ptr, size_t size) {
    if (size == 0) return;
    CHECK(curr_offs_ + size <= size_) << "not enough space in slot";
    uint8_t* curr_ptr = &(*buf_)[start_offs_ + curr_offs_];
    std::memcpy(curr_ptr, src_ptr, size);
    curr_offs_ += size;
  }

  /*!
   * \brief returns start address of the slot in device memory
   * \return device start address
   */
  void* dev_start_addr() {
    return dev_start_addr_;
  }

  /*!
   * \brief returns number of bytes allocated for this slot
   * \return size of this slot
   */
  size_t size() {
    return size_;
  }

 private:
  // We store a pointer to the backing buffer and a byte offset, instead of just
  // a pointer at the offset into the buffer, in order to prevent stale
  // references on vector resize.

  /*! \brief shared pointer to parent backing buffer */
  std::shared_ptr<std::vector<uint8_t>> buf_;
  /*! \brief start offset of the slot in the backing buffer */
  size_t start_offs_;
  /*! \brief current offset relative to the start offset of this slot */
  size_t curr_offs_;
  /*! \brief size (in bytes) of the memory region allocated for this slot */
  size_t size_;
  /*! \brief start address of the slot in the device's memory */
  void* dev_start_addr_;
};

/*!
 * \brief data encoder for uTVM that builds a host-side buffer
 */
class TargetDataLayoutEncoder {
 public:
  /*!
   * \brief constructor
   * \param dev_start_addr start address of the encoder in device memory
   * \param dev_base_addr base address of the device
   */
  explicit TargetDataLayoutEncoder(void* dev_start_addr, const void* dev_base_addr)
      : buf_(std::make_shared<std::vector<uint8_t>>()),
        curr_offs_(0),
        dev_start_addr_(dev_start_addr),
        dev_base_addr_(dev_base_addr) {}

  /*!
   * \brief allocates a slot for `sizeof(T)` bytes of data
   * \return slot of size `sizeof(T)` bytes
   */
  template <typename T>
  Slot Alloc() {
    return Alloc(sizeof(T));
  }

  /*!
   * \brief allocates a slot for `sizeof(T) * length` bytes of data
   * \param length number of elements in the array being allocated for
   * \return slot of size `sizeof(T) * length` bytes
   */
  template <typename T>
  Slot AllocArray(size_t length) {
    return Alloc(sizeof(T) * length);
  }

  /*!
   * \brief allocates a slot for `size` bytes of data
   * \param size number of bytes to allocate
   * \return slot of size `size` bytes
   */
  Slot Alloc(size_t size) {
    if (curr_offs_ + size > buf_->size()) {
      buf_->resize(curr_offs_ + size);
    }
    size_t slot_start_offs = curr_offs_;
    curr_offs_ += size;
    return Slot(buf_, slot_start_offs, size, GetDevAddr(slot_start_offs));
  }

  /*!
   * \brief writes arguments to the host-side buffer
   * \param args pointer to the args to be written
   * \return device address of the allocated args
   */
  void* Write(UTVMArgs* args) {
    Slot utvm_args_slot = Alloc<UTVMArgs>();

    const int* type_codes = args->type_codes;
    int num_args = args->num_args;

    Slot tvm_vals_slot = AllocArray<TVMValue*>(num_args);
    Slot type_codes_slot = AllocArray<const int>(num_args);

    for (int i = 0; i < num_args; i++) {
      switch (type_codes[i]) {
        case kNDArrayContainer: {
          void* val_addr = Write(reinterpret_cast<TVMArray*>(args->values[i].v_handle));
          tvm_vals_slot.Write(&val_addr);
          break;
        }
        // TODO(mutinifni): implement other cases if needed
        default:
          LOG(FATAL) << "Unsupported type code for writing args: " << type_codes[i];
          break;
      }
    }
    type_codes_slot.WriteEntire(type_codes);

    UTVMArgs dev_args = {
      .values = reinterpret_cast<TVMValue*>(tvm_vals_slot.dev_start_addr()),
      .type_codes = reinterpret_cast<int*>(type_codes_slot.dev_start_addr()),
      .num_args = num_args,
    };
    utvm_args_slot.Write(&dev_args);
    return utvm_args_slot.dev_start_addr();
  }

  /*!
   * \brief writes a `TVMArray` to the host-side buffer
   * \param arr pointer to the TVMArray to be written
   * \param dev_base_addr base address of the device
   * \return device address of the allocated `TVMArray`
   */
  void* Write(TVMArray* arr) {
    Slot tvm_arr_slot = Alloc<TVMArray>();
    Slot shape_slot = AllocArray<int64_t>(arr->ndim);

    // `shape` and `strides` are stored on the host, so we need to write them to
    // the device first. The `data` field is already allocated on the device and
    // is a device pointer, so we don't need to write it.
    shape_slot.WriteEntire(arr->shape);
    void* shape_addr = shape_slot.dev_start_addr();
    void* strides_addr = nullptr;
    if (arr->strides != nullptr) {
      Slot stride_slot = AllocArray<int64_t>(arr->ndim);
      stride_slot.WriteEntire(arr->strides);
      strides_addr = stride_slot.dev_start_addr();
    }

    // Copy `arr`, update the copy's pointers to be device pointers, then
    // write the copy to `tvm_arr_slot`.
    TVMArray dev_arr = *arr;
    // Add the base address of the device to the array's data's device offset to
    // get a device address.
    dev_arr.data = reinterpret_cast<uint8_t*>(const_cast<void*>(dev_base_addr_)) +
                   reinterpret_cast<std::uintptr_t>(arr->data);
    dev_arr.shape = static_cast<int64_t*>(shape_addr);
    dev_arr.strides = static_cast<int64_t*>(strides_addr);
    tvm_arr_slot.Write(&dev_arr);
    return tvm_arr_slot.dev_start_addr();
  }

  /*!
   * \brief returns the corresponding device address for the offset `offset`
   * \param offset byte offset from the beginning of the backing buffer
   * \return device address
   */
  void* GetDevAddr(size_t offset) {
    return reinterpret_cast<uint8_t*>(dev_start_addr_) + offset;
  }

  /*!
   * \brief returns the array backing the encoder's buffer
   * \return array backing the encoder's buffer
   */
  const uint8_t* data() {
    return buf_->data();
  }

  /*!
   * \brief returns current size of the encoder's buffer
   * \return buffer size
   */
  size_t buf_size() {
    return buf_->size();
  }

 private:
  /*! \brief in-memory backing buffer */
  std::shared_ptr<std::vector<uint8_t>> buf_;
  /*! \brief current offset */
  size_t curr_offs_;
  /*! \brief start address of the encoder in device memory */
  void* dev_start_addr_;
  /*! \brief base address of the device */
  const void* dev_base_addr_;
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_
