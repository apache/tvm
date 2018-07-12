/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm/runtime/ndarray.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RUNTIME_NDARRAY_H_
#define TVM_RUNTIME_NDARRAY_H_

#include <atomic>
#include <vector>
#include <utility>
#include "./c_runtime_api.h"

namespace tvm {
namespace runtime {
/*!
 * \brief Managed NDArray.
 *  The array is backed by reference counted blocks.
 */
class NDArray {
 public:
  // internal container type
  struct Container;
  /*! \brief default constructor */
  NDArray() {}
  /*!
   * \brief cosntruct a NDArray that refers to data
   * \param data The data this NDArray refers to
   */
  explicit inline NDArray(Container* data);
  /*!
   * \brief copy constructor
   * \param other The value to be copied
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
  inline void CopyTo(DLTensor* other);
  inline void CopyTo(const NDArray& other);
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
   *
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

  // internal namespace
  struct Internal;
 private:
  /*! \brief Internal Data content */
  Container* data_{nullptr};
  // enable internal functions
  friend struct Internal;
  friend class TVMRetValue;
  friend class TVMArgsSetter;
};

/*!
 * \brief Reference counted Container object used to back NDArray.
 *
 *  This object is DLTensor compatible:
 *    the pointer to the NDArrayContainer can be directly
 *    interpreted as a DLTensor*
 *
 * \note: do not use this function directly, use NDArray.
 */
struct NDArray::Container {
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
  /*! \brief default constructor */
  Container() {
    dl_tensor.data = nullptr;
    dl_tensor.ndim = 0;
    dl_tensor.shape = nullptr;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
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

 private:
  friend class NDArray;
  /*!
   * \brief The shape container,
   *  can be used used for shape data.
   */
  std::vector<int64_t> shape_;
  /*! \brief The internal array object */
  std::atomic<int> ref_counter_{0};
};

// implementations of inline functions
// the usages of functions are documented in place.
inline NDArray::NDArray(Container* data)
  : data_(data) {
  data_->IncRef();
}

inline NDArray::NDArray(const NDArray& other)
  : data_(other.data_) {
  data_->IncRef();
}

inline void NDArray::reset() {
  if (data_ != nullptr) {
    data_->DecRef();
    data_ = nullptr;
  }
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

inline void NDArray::CopyTo(DLTensor* other) {
  CHECK(data_ != nullptr);
  CopyFromTo(&(data_->dl_tensor), other);
}

inline void NDArray::CopyTo(const NDArray& other) {
  CHECK(data_ != nullptr);
  CHECK(other.data_ != nullptr);
  CopyFromTo(&(data_->dl_tensor), &(other.data_->dl_tensor));
}

inline int NDArray::use_count() const {
  if (data_ == nullptr) return 0;
  return data_->ref_counter_.load(std::memory_order_relaxed);
}

inline const DLTensor* NDArray::operator->() const {
  return &(data_->dl_tensor);
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_NDARRAY_H_
