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
 * \file tvm/runtime/object.h
 * \brief A managed object in the TVM runtime.
 */
#ifndef TVM_RUNTIME_OBJECT_H_
#define TVM_RUNTIME_OBJECT_H_

#include <tvm/ffi/object.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection.h>
#include <tvm/runtime/c_runtime_api.h>

namespace tvm {
namespace runtime {

template<typename T>
class Optional;
using namespace tvm::ffi;

/*!
 * \brief Namespace for the list of type index.
 * \note Use struct so that we have to use TypeIndex::ENumName to refer to
 *       the constant, but still able to use enum.
 */
enum TypeIndex : int32_t {
  // Standard static index assignments,
  // Frontends can take benefit of these constants.
  kRuntimeString = TVMFFITypeIndex::kTVMFFIStr,
  kRuntimeMap = TVMFFITypeIndex::kTVMFFIMap,
  kRuntimeArray = TVMFFITypeIndex::kTVMFFIArray,
  /*! \brief runtime::Module. */
  kRuntimeModule = TVMFFITypeIndex::kTVMFFIRuntimeModule,
  /*! \brief runtime::NDArray. */
  kRuntimeNDArray = TVMFFITypeIndex::kTVMFFINDArray,
  /*! \brief runtime::ShapeTuple. */
  kRuntimeShapeTuple = TVMFFITypeIndex::kTVMFFIShapeTuple,
  // Extra builtin static index here
  kCustomStaticIndex = TVMFFITypeIndex::kTVMFFIStaticObjectEnd,
  /*! \brief runtime::PackedFunc. */
  kRuntimePackedFunc = kCustomStaticIndex + 1,
  /*! \brief runtime::DRef for disco distributed runtime */
  kRuntimeDiscoDRef = kCustomStaticIndex + 2,
  /*! \brief runtime::RPCObjectRef */
  kRuntimeRPCObjectRef = kCustomStaticIndex + 3,
  // static assignments that may subject to change.
  kStaticIndexEnd,
};

class ObjectRef : public tvm::ffi::ObjectRef {
 public:
  /*! \brief default constructor */
  ObjectRef() = default;
  /*! \brief Constructor from existing object ptr */
  explicit ObjectRef(ObjectPtr<Object> data) : tvm::ffi::ObjectRef(data) {}

  using tvm::ffi::ObjectRef::as;

  /*!
   * \brief Try to downcast the ObjectRef to a
   *    Optional<T> of the requested type.
   *
   *  The function will return a NullOpt if the cast failed.
   *
   *      if (Optional<Add> opt = node_ref.as<Add>()) {
   *        // This is an add node
   *      }
   *
   * \note While this method is declared in <tvm/runtime/object.h>,
   * the implementation is in <tvm/runtime/container/optional.h> to
   * prevent circular includes.  This additional include file is only
   * required in compilation units that uses this method.
   *
   * \tparam ObjectRefType the target type, must be a subtype of ObjectRef
   */
  template <typename ObjectRefType,
            typename = std::enable_if_t<std::is_base_of_v<ObjectRef, ObjectRefType>>>
  inline Optional<ObjectRefType> as() const;

 protected:
  /*! \return return a mutable internal ptr, can be used by sub-classes. */
  Object* get_mutable() const { return data_.get(); }
  /*!
   * \brief Internal helper function downcast a ref without check.
   * \note Only used for internal dev purposes.
   * \tparam T The target reference type.
   * \return The casted result.
   */
  template <typename T>
  static T DowncastNoCheck(ObjectRef ref) {
    return T(std::move(ref.data_));
  }
  /*!
   * \brief Clear the object ref data field without DecRef
   *        after we successfully moved the field.
   * \param ref The reference data.
   */
  static void FFIClearAfterMove(ObjectRef* ref) {
    details::ObjectUnsafe::LegacyClearObjectPtrAfterMove(ref);
  }
  /*!
   * \brief Internal helper function get data_ as ObjectPtr of ObjectType.
   * \note only used for internal dev purpose.
   * \tparam ObjectType The corresponding object type.
   * \return the corresponding type.
   */
  template <typename ObjectType>
  static ObjectPtr<ObjectType> GetDataPtr(const ObjectRef& ref) {
    return ObjectPtr<ObjectType>(ref.data_.data_);
  }

  // friend classes.
  friend struct ObjectPtrHash;
  friend class TVMRetValue;
  friend class TVMArgsSetter;
  friend class ObjectInternal;
};

/*
 * \brief Define the default copy/move constructor and assign operator
 * \param TypeName The class typename.
 */
#define TVM_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName) \
  TypeName(const TypeName& other) = default;              \
  TypeName(TypeName&& other) = default;                   \
  TypeName& operator=(const TypeName& other) = default;   \
  TypeName& operator=(TypeName&& other) = default;


#define TVM_DECLARE_BASE_OBJECT_INFO TVM_FFI_DECLARE_BASE_OBJECT_INFO
#define TVM_DECLARE_FINAL_OBJECT_INFO TVM_FFI_DECLARE_FINAL_OBJECT_INFO
#define TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS TVM_FFI_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS

#define TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS TVM_DEFINE_MUTABLE_NULLABLE_OBJECT_REF_METHODS
#define TVM_DEFINE_OBJECT_REF_METHODS TVM_FFI_DEFINE_NULLABLE_OBJECT_REF_METHODS
#define TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS TVM_FFI_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS

#define TVM_STR_CONCAT_(__x, __y) __x##__y
#define TVM_STR_CONCAT(__x, __y) TVM_STR_CONCAT_(__x, __y)

// Object register type is now a nop
#define TVM_REGISTER_OBJECT_TYPE(x)

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OBJECT_H_
