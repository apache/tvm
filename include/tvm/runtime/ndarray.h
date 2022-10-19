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

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/serializer.h>

#include <atomic>
#include <functional>
#include <utility>
#include <vector>

namespace tvm {

// alias DLDevice
using Device = DLDevice;

// A 'null' device type, does not correspond to any DLDeviceType enum.
// TODO(mbs): This is to help us as we transition away from representing the 'homogenous' case
// as a singleton target map indexed by the invalid DLDeviceType '0'.
constexpr DLDeviceType kNullDeviceType = static_cast<DLDeviceType>(0);

// An 'invalid' device type, does not correspond to any DLDeviceType enum.
constexpr DLDeviceType kInvalidDeviceType = static_cast<DLDeviceType>(-1);

namespace runtime {

/*!
 * \brief Managed NDArray.
 *  The array is backed by reference counted blocks.
 */
class NDArray : public ObjectRef {
 public:
  /*! \brief ContainerBase used to back the TVMArrayHandle */
  class ContainerBase;
  /*! \brief NDArray internal container type */
  class Container;
  /*! \brief Container type for Object system. */
  using ContainerType = Container;
  /*! \brief default constructor */
  NDArray() {}
  /*!
   * \brief constructor.
   * \param data ObjectPtr to the data container.
   */
  explicit NDArray(ObjectPtr<Object> data) : ObjectRef(data) {}

  /*! \brief reset the content of NDArray to be nullptr */
  inline void reset();
  /*!
   * \return the reference counter
   * \note this number is approximate in multi-threaded setting.
   */
  inline int use_count() const;
  /*! \return Pointer to content of DLTensor */
  inline const DLTensor* operator->() const;
  /*! \return Whether the tensor is contiguous */
  inline bool IsContiguous() const;
  /*!
   * \brief Copy data content from another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchronously if it involves a GPU context.
   *       TVMSynchronize is necessary.
   */
  inline void CopyFrom(const DLTensor* other);
  inline void CopyFrom(const NDArray& other);
  /*!
   * \brief Copy data content from a byte buffer.
   * \param data The source bytes to be copied from.
   * \param nbytes The size of the buffer in bytes
   *        Must be equal to the size of the NDArray.
   * \note The copy always triggers a TVMSynchronize.
   */
  TVM_DLL void CopyFromBytes(const void* data, size_t nbytes);
  /*!
   * \brief Copy data content into another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchronously if it involves a GPU context.
   *       TVMSynchronize is necessary.
   */
  inline void CopyTo(DLTensor* other) const;
  inline void CopyTo(const NDArray& other) const;
  /*!
   * \brief Copy data content into another array.
   * \param data The source bytes to be copied from.
   * \param nbytes The size of the data buffer.
   *        Must be equal to the size of the NDArray.
   * \note The copy always triggers a TVMSynchronize.
   */
  TVM_DLL void CopyToBytes(void* data, size_t nbytes) const;
  /*!
   * \brief Copy the data to another device.
   * \param dev The target device.
   * \return The array under another device.
   */
  inline NDArray CopyTo(const Device& dev) const;
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
  TVM_DLL NDArray CreateView(ShapeTuple shape, DLDataType dtype);
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
   * \param dev The device of the array.
   * \param mem_scope The memory scope of the array.
   * \return The created Array
   */
  TVM_DLL static NDArray Empty(ShapeTuple shape, DLDataType dtype, Device dev,
                               Optional<String> mem_scope = NullOpt);
  /*!
   * \brief Create a NDArray backed by an external DLTensor without memory copying.
   *
   * If DLTensor is not contiguous or has bad aligned data, It fails.
   * This allows us to create a NDArray using the memory
   * allocated by an external source. Responsibility for memory
   * retaining lies with the external source.
   * \param dl_tensor The DLTensor for NDArray base.
   * \return The created NDArray view.
   */
  TVM_DLL static NDArray FromExternalDLTensor(const DLTensor& dl_tensor);
  /*!
   * \brief Create new NDArray, data is copied from DLTensor.
   *
   * \param dl_tensor The DLTensor to copy from.
   * \param dev device location of the created NDArray.
   * \return The created NDArray view.
   */
  TVM_DLL static NDArray NewFromDLTensor(DLTensor* dl_tensor, const Device& dev);
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
  TVM_DLL static void CopyFromTo(const DLTensor* from, DLTensor* to,
                                 TVMStreamHandle stream = nullptr);

  TVM_DLL ShapeTuple Shape() const;
  TVM_DLL runtime::DataType DataType() const;
  /*!
   * \brief Check conditions for construction NDArray over DLTensor without copying.
   * There are three conditions to check:
   * 1. Destination device is the same as DLTensor device
   * 2. Destination device id is the same as DLTensor device id
   * 3. Memory in DLTensor is aligned as expected for NDArray
   * \param tensor the DLTensor.
   * \param dev destination device.
   * \return true if all conditions are satisfied.
   */
  TVM_DLL static bool AbilityOfZeroCopyForDLTensor(DLTensor* tensor, const Device& dev);
  // internal namespace
  struct Internal;

 private:
  TVM_DLL static bool IsAligned(const DLTensor& tensor);

 protected:
  friend class TVMPODValue_;
  friend class TVMRetValue;
  friend class TVMArgsSetter;
  /*!
   * \brief Get mutable internal container pointer.
   * \return a mutable container pointer.
   */
  inline Container* get_mutable() const;
  // Helper functions for FFI handling.
  /*!
   * \brief Construct NDArray's Data field from array handle in FFI.
   * \param handle The array handle.
   * \return The corresponding ObjectPtr to the constructed container object.
   *
   * \note We keep a special calling convention for NDArray by passing
   *       ContainerBase pointer in FFI.
   *       As a result, the argument is compatible to DLTensor*.
   */
  inline static ObjectPtr<Object> FFIDataFromHandle(TVMArrayHandle handle);
  /*!
   * \brief DecRef resource managed by an FFI array handle.
   * \param handle The array handle.
   */
  inline static void FFIDecRef(TVMArrayHandle handle);
  /*!
   * \brief Get FFI Array handle from ndarray.
   * \param nd The object with ndarray type.
   * \return The result array handle.
   */
  inline static TVMArrayHandle FFIGetHandle(const ObjectRef& nd);
};

/*!
 * \brief Save a DLTensor to stream
 * \param strm The output stream
 * \param tensor The tensor to be saved.
 */
inline bool SaveDLTensor(dmlc::Stream* strm, const DLTensor* tensor);

/*!
 * \brief The container base structure
 *        contains all the fields except for the Object header.
 *
 * \note We explicitly declare this structure in order to pass
 *       PackedFunc argument using ContainerBase*.
 */
class NDArray::ContainerBase {
 public:
  /*!
   * \brief The corresponding dl_tensor field.
   * \note it is important that the first field is DLTensor
   *  So that this data structure is DLTensor compatible.
   *  The head ptr of this struct can be viewed as DLTensor*.
   */
  DLTensor dl_tensor;

  /*!
   * \brief additional context, reserved for recycling
   * \note We can attach additional content here
   *  which the current container depend on
   *  (e.g. reference to original memory when creating views).
   */
  void* manager_ctx{nullptr};

 protected:
  /*!
   * \brief The shape container,
   *  can be used used for shape data.
   */
  ShapeTuple shape_;
};

/*!
 * \brief Object container class that backs NDArray.
 * \note do not use this function directly, use NDArray.
 */
class NDArray::Container : public Object, public NDArray::ContainerBase {
 public:
  /*! \brief default constructor */
  Container() {
    // Initialize the type index.
    type_index_ = Container::RuntimeTypeIndex();
    dl_tensor.data = nullptr;
    dl_tensor.ndim = 0;
    dl_tensor.shape = nullptr;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
  }

  Container(void* data, ShapeTuple shape, DLDataType dtype, Device dev) {
    // Initialize the type index.
    type_index_ = Container::RuntimeTypeIndex();
    dl_tensor.data = data;
    shape_ = std::move(shape);
    dl_tensor.ndim = static_cast<int>(shape_.size());
    dl_tensor.shape = const_cast<ShapeTuple::index_type*>(shape_.data());
    dl_tensor.dtype = dtype;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
    dl_tensor.device = dev;
  }
  /*!
   * \brief Set the deleter field.
   * \param deleter The deleter.
   */
  void SetDeleter(FDeleter deleter) { deleter_ = deleter; }

  // Expose DecRef and IncRef as public function
  // NOTE: they are only for developer purposes only.
  using Object::DecRef;
  using Object::IncRef;

  // Information for object protocol.
  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeNDArray;
  static constexpr const uint32_t _type_child_slots = 0;
  static constexpr const uint32_t _type_child_slots_can_overflow = true;
  static constexpr const char* _type_key = "runtime.NDArray";
  TVM_DECLARE_BASE_OBJECT_INFO(NDArray::Container, Object);

 protected:
  friend class RPCWrappedFunc;
  friend class NDArray;
};

// implementations of inline functions
/*!
 * \brief return the size of data the DLTensor hold, in term of number of bytes
 *
 *  \param arr the input DLTensor
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

/*!
 * \brief check if a DLTensor is contiguous.
 * \param arr The input DLTensor.
 * \return The check result.
 */
static inline bool IsContiguous(const DLTensor& arr) {
  if (arr.strides == nullptr) return true;
  int64_t expected_stride = 1;
  for (int32_t i = arr.ndim; i != 0; --i) {
    int32_t k = i - 1;
    if (arr.shape[k] == 1) {
      // Skip stride check if shape[k] is 1, where the dimension is contiguous
      // regardless of the value of stride.
      //
      // For example, PyTorch will normalize stride to 1 if shape is 1 when exporting
      // to DLPack.
      // More context: https://github.com/pytorch/pytorch/pull/83158
      continue;
    }
    if (arr.strides[k] != expected_stride) return false;
    expected_stride *= arr.shape[k];
  }
  return true;
}

inline bool NDArray::IsContiguous() const {
  return ::tvm::runtime::IsContiguous(get_mutable()->dl_tensor);
}

inline void NDArray::CopyFrom(const DLTensor* other) {
  ICHECK(data_ != nullptr);
  CopyFromTo(other, &(get_mutable()->dl_tensor));
}

inline void NDArray::CopyFrom(const NDArray& other) {
  ICHECK(data_ != nullptr);
  ICHECK(other.data_ != nullptr);
  CopyFromTo(&(other.get_mutable()->dl_tensor), &(get_mutable()->dl_tensor));
}

inline void NDArray::CopyTo(DLTensor* other) const {
  ICHECK(data_ != nullptr);
  CopyFromTo(&(get_mutable()->dl_tensor), other);
}

inline void NDArray::CopyTo(const NDArray& other) const {
  ICHECK(data_ != nullptr);
  ICHECK(other.data_ != nullptr);
  CopyFromTo(&(get_mutable()->dl_tensor), &(other.get_mutable()->dl_tensor));
}

inline NDArray NDArray::CopyTo(const Device& dev) const {
  ICHECK(data_ != nullptr);
  const DLTensor* dptr = operator->();
  NDArray ret = Empty(ShapeTuple(dptr->shape, dptr->shape + dptr->ndim), dptr->dtype, dev);
  this->CopyTo(ret);
  return ret;
}

inline int NDArray::use_count() const { return data_.use_count(); }

inline const DLTensor* NDArray::operator->() const { return &(get_mutable()->dl_tensor); }

inline NDArray::Container* NDArray::get_mutable() const {
  return static_cast<NDArray::Container*>(data_.get());
}

inline ObjectPtr<Object> NDArray::FFIDataFromHandle(TVMArrayHandle handle) {
  return GetObjectPtr<Object>(
      static_cast<NDArray::Container*>(reinterpret_cast<NDArray::ContainerBase*>(handle)));
}

inline TVMArrayHandle NDArray::FFIGetHandle(const ObjectRef& nd) {
  // NOTE: it is necessary to cast to container then to base
  //       so that the FFI handle uses the ContainerBase address.
  auto ptr = reinterpret_cast<TVMArrayHandle>(static_cast<NDArray::ContainerBase*>(
      static_cast<NDArray::Container*>(const_cast<Object*>(nd.get()))));
  return ptr;
}

inline void NDArray::FFIDecRef(TVMArrayHandle handle) {
  static_cast<NDArray::Container*>(reinterpret_cast<NDArray::ContainerBase*>(handle))->DecRef();
}

inline Object* TVMArrayHandleToObjectHandle(TVMArrayHandle handle) {
  return static_cast<NDArray::Container*>(reinterpret_cast<NDArray::ContainerBase*>(handle));
}

/*! \brief Magic number for NDArray file */
constexpr uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

inline bool SaveDLTensor(dmlc::Stream* strm, const DLTensor* tensor) {
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
  // We can always do array.CopyTo(target_dev) to get a corresponding
  // array in the target context.
  Device cpu_dev;
  cpu_dev.device_type = kDLCPU;
  cpu_dev.device_id = 0;
  strm->Write(cpu_dev);
  strm->Write(tensor->ndim);
  strm->Write(tensor->dtype);
  int ndim = tensor->ndim;
  strm->WriteArray(tensor->shape, ndim);
  int type_bytes = (tensor->dtype.bits + 7) / 8;
  int64_t num_elems = 1;
  for (int i = 0; i < ndim; ++i) {
    num_elems *= tensor->shape[i];
  }
  int64_t data_byte_size = type_bytes * num_elems;
  strm->Write(data_byte_size);

  if (DMLC_IO_NO_ENDIAN_SWAP && tensor->device.device_type == kDLCPU &&
      tensor->strides == nullptr && tensor->byte_offset == 0) {
    // quick path
    strm->Write(tensor->data, data_byte_size);
  } else {
    std::vector<uint8_t> bytes(data_byte_size);
    ICHECK_EQ(
        TVMArrayCopyToBytes(const_cast<DLTensor*>(tensor), dmlc::BeginPtr(bytes), data_byte_size),
        0)
        << TVMGetLastError();
    if (!DMLC_IO_NO_ENDIAN_SWAP) {
      dmlc::ByteSwap(dmlc::BeginPtr(bytes), type_bytes, num_elems);
    }
    strm->Write(dmlc::BeginPtr(bytes), data_byte_size);
  }
  return true;
}

inline void NDArray::Save(dmlc::Stream* strm) const { SaveDLTensor(strm, operator->()); }

inline bool NDArray::Load(dmlc::Stream* strm) {
  uint64_t header, reserved;
  ICHECK(strm->Read(&header)) << "Invalid DLTensor file format";
  ICHECK(strm->Read(&reserved)) << "Invalid DLTensor file format";
  ICHECK(header == kTVMNDArrayMagic) << "Invalid DLTensor file format";
  Device dev;
  int ndim;
  DLDataType dtype;
  ICHECK(strm->Read(&dev)) << "Invalid DLTensor file format";
  ICHECK(strm->Read(&ndim)) << "Invalid DLTensor file format";
  ICHECK(strm->Read(&dtype)) << "Invalid DLTensor file format";
  ICHECK_EQ(dev.device_type, kDLCPU) << "Invalid DLTensor device: can only save as CPU tensor";
  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    ICHECK(strm->ReadArray(&shape[0], ndim)) << "Invalid DLTensor file format";
  }
  NDArray ret = NDArray::Empty(ShapeTuple(shape), dtype, dev);
  int64_t num_elems = 1;
  int elem_bytes = (ret->dtype.bits + 7) / 8;
  for (int i = 0; i < ret->ndim; ++i) {
    num_elems *= ret->shape[i];
  }
  int64_t data_byte_size;
  ICHECK(strm->Read(&data_byte_size)) << "Invalid DLTensor file format";
  ICHECK(data_byte_size == num_elems * elem_bytes) << "Invalid DLTensor file format";
  auto read_ret = strm->Read(ret->data, data_byte_size);
  // Only check non-empty data
  if (ndim > 0 && shape[0] != 0) {
    ICHECK(read_ret) << "Invalid DLTensor file format";
  }
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    dmlc::ByteSwap(ret->data, elem_bytes, num_elems);
  }
  *this = ret;
  return true;
}

}  // namespace runtime
}  // namespace tvm

namespace std {
template <>
struct hash<tvm::Device> {
  std::size_t operator()(const tvm::Device& dev) const {
    return ((dev.device_id << 8) | dev.device_type);
  }
};

template <>
struct equal_to<tvm::Device> {
  bool operator()(const tvm::Device& lhs, const tvm::Device& rhs) const {
    return (lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id);
  }
};
}  // namespace std

#endif  // TVM_RUNTIME_NDARRAY_H_
