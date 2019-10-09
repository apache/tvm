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
 * \file tvm/runtime/ndarray.h
 * \brief A device-independent managed NDArray abstraction.
 */
#ifndef TVM_RUNTIME_NDARRAY_H_
#define TVM_RUNTIME_NDARRAY_H_

#include <atomic>
#include <vector>
#include <utility>
#include "c_runtime_api.h"
#include "serializer.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Managed NDArray.
 *  The array is backed by reference counted blocks.
 */
class NDArray {
 public:
  // internal container type
  class Container;
  /*! \brief default constructor */
  NDArray() {}
  /*!
   * \brief cosntruct a NDArray that refers to data
   * \param data The data this NDArray refers to
   */
  explicit inline NDArray(Container* data);
  /*!
   * \brief copy constructor.
   *
   * It does not make a copy, but the reference count of the input NDArray is incremented
   *
   * \param other NDArray that shares internal data with the input NDArray.
   */
  inline NDArray(const NDArray& other);  // NOLINT(*)
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  NDArray(NDArray&& other) // NOLINT(*)
      : data_(other.data_) {
    other.data_ = nullptr;
  }
  /*! \brief destructor */
  ~NDArray() {
    this->reset();
  }
  /*!
   * \brief Swap this array with another NDArray
   * \param other The other NDArray
   */
  void swap(NDArray& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }
  /*!
   * \brief copy assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  NDArray& operator=(const NDArray& other) {  // NOLINT(*)
    // copy-and-swap idiom
    NDArray(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief move assignmemt
   * \param other The value to be assigned.
   * \return reference to self.
   */
  NDArray& operator=(NDArray&& other) {  // NOLINT(*)
    // copy-and-swap idiom
    NDArray(std::move(other)).swap(*this); // NOLINT(*)
    return *this;
  }
  /*! \return If NDArray is defined */
  bool defined() const {
    return data_ != nullptr;
  }
  /*! \return If both NDArray reference the same container */
  bool same_as(const NDArray& other) const {
    return data_ == other.data_;
  }
  /*! \brief reset the content of NDArray to be nullptr */
  inline void reset();
  /*!
   * \return the reference counter
   * \note this number is approximate in multi-threaded setting.
   */
  inline int use_count() const;
  /*! \return Pointer to content of DLTensor */
  inline const DLTensor* operator->() const;
  /*!
   * \brief Copy data content from another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchrously if it involves a GPU context.
   *       TVMSynchronize is necessary.
   */
  inline void CopyFrom(DLTensor* other);
  inline void CopyFrom(const NDArray& other);
  /*!
   * \brief Copy data content into another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchrously if it involves a GPU context.
   *       TVMSynchronize is necessary.
   */
  inline void CopyTo(DLTensor* other) const;
  inline void CopyTo(const NDArray& other) const;
  /*!
   * \brief Copy the data to another context.
   * \param ctx The target context.
   * \return The array under another context.
   */
  inline NDArray CopyTo(const DLContext& ctx) const;
  /*!
   * \brief Load NDArray from stream
   * \param stream The input data stream
   * \return Whether load is successful
   */
  inline bool Load(dmlc::Stream* stream);
  /*!
   * \brief Save NDArray to stream
   * \param stream The output data stream
   */
  inline void Save(dmlc::Stream* stream) const;
  /*!
   * \brief Create a NDArray that shares the data memory with the current one.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \note The memory size of new array must be smaller than the current one.
   */
  TVM_DLL NDArray CreateView(
      std::vector<int64_t> shape, DLDataType dtype);
  /*!
   * \brief Create a reference view of NDArray that
   *  represents as DLManagedTensor.
   * \return A DLManagedTensor
   */
  TVM_DLL DLManagedTensor* ToDLPack() const;
  /*!
   * \brief Create an empty NDArray.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \param ctx The context of the Array.
   * \return The created Array
   */
  TVM_DLL static NDArray Empty(std::vector<int64_t> shape,
                               DLDataType dtype,
                               DLContext ctx);
  /*!
   * \brief Create a NDArray backed by a dlpack tensor.
   *
   * This allows us to create a NDArray using the memory
   * allocated by an external deep learning framework
   * that is DLPack compatible.
   *
   * The memory is retained until the NDArray went out of scope.
   * \param tensor The DLPack tensor to copy from.
   * \return The created NDArray view.
   */
  TVM_DLL static NDArray FromDLPack(DLManagedTensor* tensor);
  /*!
   * \brief Function to copy data from one array to another.
   * \param from The source array.
   * \param to The target array.
   * \param stream The stream used in copy.
   */
  TVM_DLL static void CopyFromTo(
      DLTensor* from, DLTensor* to, TVMStreamHandle stream = nullptr);

  TVM_DLL std::vector<int64_t> Shape() const;

  // internal namespace
  struct Internal;
 protected:
  /*! \brief Internal Data content */
  Container* data_{nullptr};
  // enable internal functions
  friend struct Internal;
  friend class TVMPODValue_;
  friend class TVMArgValue;
  friend class TVMRetValue;
  friend class TVMArgsSetter;
};

/*!
 * \brief The type trait indicates subclass of TVM's NDArray.
 *  For irrelavant classes, code = -1.
 *  For TVM NDArray itself, code = 0.
 *  All subclasses of NDArray should override code > 0.
 */
template<typename T>
struct array_type_info {
  /*! \brief the value of the traits */
  static const int code = -1;
};

// Overrides the type trait for tvm's NDArray.
template<>
struct array_type_info<NDArray> {
  static const int code = 0;
};

/*!
 * \brief Save a DLTensor to stream
 * \param strm The outpu stream
 * \param tensor The tensor to be saved.
 */
inline bool SaveDLTensor(dmlc::Stream* strm, const DLTensor* tensor);

/*!
 * \brief Reference counted Container object used to back NDArray.
 *
 *  This object is DLTensor compatible:
 *    the pointer to the NDArrayContainer can be directly
 *    interpreted as a DLTensor*
 *
 * \note do not use this function directly, use NDArray.
 */
class NDArray::Container {
 public:
  // NOTE: the first part of this structure is the same as
  // DLManagedTensor, note that, however, the deleter
  // is only called when the reference counter goes to 0
  /*!
   * \brief The corresponding dl_tensor field.
   * \note it is important that the first field is DLTensor
   *  So that this data structure is DLTensor compatible.
   *  The head ptr of this struct can be viewed as DLTensor*.
   */
  DLTensor dl_tensor;

  /*!
   * \brief addtional context, reserved for recycling
   * \note We can attach additional content here
   *  which the current container depend on
   *  (e.g. reference to original memory when creating views).
   */
  void* manager_ctx{nullptr};
  /*!
   * \brief Customized deleter
   *
   * \note The customized deleter is helpful to enable
   *  different ways of memory allocator that are not
   *  currently defined by the system.
   */
  void (*deleter)(Container* self) = nullptr;

 protected:
  friend class NDArray;
  friend class TVMPODValue_;
  friend class TVMArgValue;
  friend class TVMRetValue;
  friend class RPCWrappedFunc;
  /*!
   * \brief Type flag used to indicate subclass.
   *  Default value 0 means normal NDArray::Conatainer.
   *
   *  We can extend a more specialized NDArray::Container
   *  and use the array_type_code_ to indicate
   *  the specific array subclass.
   */
  int32_t array_type_code_{0};
  /*! \brief The internal reference counter */
  std::atomic<int> ref_counter_{0};

  /*!
   * \brief The shape container,
   *  can be used used for shape data.
   */
  std::vector<int64_t> shape_;

 public:
  /*! \brief default constructor */
  Container() {
    dl_tensor.data = nullptr;
    dl_tensor.ndim = 0;
    dl_tensor.shape = nullptr;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
  }

  Container(void* data,
            std::vector<int64_t> shape,
            DLDataType dtype,
            DLContext ctx) {
    dl_tensor.data = data;
    shape_ = std::move(shape);
    dl_tensor.ndim = static_cast<int>(shape_.size());
    dl_tensor.shape = dmlc::BeginPtr(shape_);
    dl_tensor.dtype = dtype;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
    dl_tensor.ctx = ctx;
  }

  /*! \brief developer function, increases reference counter */
  void IncRef() {
    ref_counter_.fetch_add(1, std::memory_order_relaxed);
  }
  /*! \brief developer function, decrease reference counter */
  void DecRef() {
    if (ref_counter_.fetch_sub(1, std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      if (this->deleter != nullptr) {
        (*this->deleter)(this);
      }
    }
  }
};

// implementations of inline functions
// the usages of functions are documented in place.
inline NDArray::NDArray(Container* data)
  : data_(data) {
  if (data != nullptr) {
    data_->IncRef();
  }
}

inline NDArray::NDArray(const NDArray& other)
  : data_(other.data_) {
  if (data_ != nullptr) {
    data_->IncRef();
  }
}

inline void NDArray::reset() {
  if (data_ != nullptr) {
    data_->DecRef();
    data_ = nullptr;
  }
}

/*! \brief return the size of data the DLTensor hold, in term of number of bytes
 *
 *  \param arr the input DLTensor
 *
 *  \return number of  bytes of data in the DLTensor.
 */
inline size_t GetDataSize(const DLTensor& arr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < arr.ndim; ++i) {
    size *= static_cast<size_t>(arr.shape[i]);
  }
  size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
  return size;
}

inline void NDArray::CopyFrom(DLTensor* other) {
  CHECK(data_ != nullptr);
  CopyFromTo(other, &(data_->dl_tensor));
}

inline void NDArray::CopyFrom(const NDArray& other) {
  CHECK(data_ != nullptr);
  CHECK(other.data_ != nullptr);
  CopyFromTo(&(other.data_->dl_tensor), &(data_->dl_tensor));
}

inline void NDArray::CopyTo(DLTensor* other) const {
  CHECK(data_ != nullptr);
  CopyFromTo(&(data_->dl_tensor), other);
}

inline void NDArray::CopyTo(const NDArray& other) const {
  CHECK(data_ != nullptr);
  CHECK(other.data_ != nullptr);
  CopyFromTo(&(data_->dl_tensor), &(other.data_->dl_tensor));
}

inline NDArray NDArray::CopyTo(const DLContext& ctx) const {
  CHECK(data_ != nullptr);
  const DLTensor* dptr = operator->();
  NDArray ret = Empty(std::vector<int64_t>(dptr->shape, dptr->shape + dptr->ndim),
                      dptr->dtype, ctx);
  this->CopyTo(ret);
  return ret;
}

inline int NDArray::use_count() const {
  if (data_ == nullptr) return 0;
  return data_->ref_counter_.load(std::memory_order_relaxed);
}

inline const DLTensor* NDArray::operator->() const {
  return &(data_->dl_tensor);
}

/*! \brief Magic number for NDArray file */
constexpr uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

inline bool SaveDLTensor(dmlc::Stream* strm,
                         DLTensor* tensor) {
  uint64_t header = kTVMNDArrayMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  // Always save data as CPU context
  //
  // Parameters that get serialized should be in CPU by default.
  // So even the array's context is GPU, it will be stored as CPU array.
  // This is used to prevent case when another user loads the parameters
  // back on machine that do not have GPU or related context.
  //
  // We can always do array.CopyTo(target_ctx) to get a corresponding
  // array in the target context.
  DLContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  strm->Write(cpu_ctx);
  strm->Write(tensor->ndim);
  strm->Write(tensor->dtype);
  int ndim = tensor->ndim;
  strm->WriteArray(tensor->shape, ndim);
  int type_bytes = tensor->dtype.bits / 8;
  int64_t num_elems = 1;
  for (int i = 0; i < ndim; ++i) {
    num_elems *= tensor->shape[i];
  }
  int64_t data_byte_size = type_bytes * num_elems;
  strm->Write(data_byte_size);

  if (DMLC_IO_NO_ENDIAN_SWAP &&
      tensor->ctx.device_type == kDLCPU &&
      tensor->strides == nullptr &&
      tensor->byte_offset == 0) {
    // quick path
    strm->Write(tensor->data, data_byte_size);
  } else {
    std::vector<uint8_t> bytes(data_byte_size);
    CHECK_EQ(TVMArrayCopyToBytes(
        tensor, dmlc::BeginPtr(bytes), data_byte_size), 0)
        << TVMGetLastError();
    if (!DMLC_IO_NO_ENDIAN_SWAP) {
      dmlc::ByteSwap(dmlc::BeginPtr(bytes), type_bytes, num_elems);
    }
    strm->Write(dmlc::BeginPtr(bytes), data_byte_size);
  }
  return true;
}

inline void NDArray::Save(dmlc::Stream* strm) const {
  SaveDLTensor(strm, const_cast<DLTensor*>(operator->()));
}

inline bool NDArray::Load(dmlc::Stream* strm) {
  uint64_t header, reserved;
  CHECK(strm->Read(&header))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&reserved))
      << "Invalid DLTensor file format";
  CHECK(header == kTVMNDArrayMagic)
      << "Invalid DLTensor file format";
  DLContext ctx;
  int ndim;
  DLDataType dtype;
  CHECK(strm->Read(&ctx))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&ndim))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&dtype))
      << "Invalid DLTensor file format";
  CHECK_EQ(ctx.device_type, kDLCPU)
      << "Invalid DLTensor context: can only save as CPU tensor";
  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    CHECK(strm->ReadArray(&shape[0], ndim))
        << "Invalid DLTensor file format";
  }
  NDArray ret = NDArray::Empty(shape, dtype, ctx);
  int64_t num_elems = 1;
  int elem_bytes = (ret->dtype.bits + 7) / 8;
  for (int i = 0; i < ret->ndim; ++i) {
    num_elems *= ret->shape[i];
  }
  int64_t data_byte_size;
  CHECK(strm->Read(&data_byte_size))
      << "Invalid DLTensor file format";
  CHECK(data_byte_size == num_elems * elem_bytes)
      << "Invalid DLTensor file format";
  CHECK(strm->Read(ret->data, data_byte_size))
      << "Invalid DLTensor file format";
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    dmlc::ByteSwap(ret->data, elem_bytes, num_elems);
  }
  *this = ret;
  return true;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_NDARRAY_H_
