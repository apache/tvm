#ifndef TVM_FFI_TRAITS_H_
#define TVM_FFI_TRAITS_H_

#include "./utils.h"
#include <sstream>
#include <string>

namespace tvm {
namespace ffi {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702)
#endif

/*!
 * \brief TypeTraits<T> is a template class that provides a set of static
 * methods that are associated with a specific type `T` for compile-time
 * dispatching.
 *
 * [Trait 0] Type2Str<T>: () -> std::string
 * Returns the string representation of the type `T`
 *
 * [Trait 1] CopyFromTVMFFIAnyToType<T>: (const TVMFFIAny *) -> T
 * Converts an `AnyView` or `Any` to `T`. It could incur copy when
 * inevitable, for example, copy the content of a Str obj if it's converted
 * to `std::string`.
 *
 * It is used in the following cases in the codebase:
 * 1) `AnyView::operator Type()`
 * 2) `Any::operator Type()`
 *
 * [Trait 2] CopyFromTypeToTVMFFIAny<T>: (T, TVMFFIAny *) -> void
 * Converts a value of type T to `AnyView`.
 *
 * It is used in the following case in the codebase:
 * 1) `AnyView::AnyView(const Type& src)`
 *
 * [Trait 3] CopyFromTVMFFIAnyToRef<T>: (const TVMFFIAny *) -> T*
 * Converts an `AnyView` or `Any` to `TVMFFIAny*`, which is subsequently used
 * to initialize `Ref<T>::data_`. Note that this method does not increment
 * the reference counter, which is supposed to be later handled automatically by
 * `Ref<T>`.
 *
 * It is used in the following cases in the codebase:
 * 1) `Ref<Type>::Ref(const AnyView& src)`
 * 2) `Ref<Type>::Ref(const Any& src)`
 *
 * [Trait 4] MoveFromTVMFFIAnyToRef<T>: (TVMFFIAny *) -> T*
 * Moves an `AnyView` or `Any` to `TVMFFIAny*`, which is subsequently used
 * to initialize `Ref<T>::data_`. Note that the reference counter will not
 * increment throughout the process because it corresponds to move semantics.
 *
 * It is used in the following cases in the codebase:
 * 1) `Ref<Type>::Ref(AnyView&& src)`
 * 2) `Ref<Type>::Ref(Any&& src)`
 *
 * [Trait 5] (Optional) CopyFromTVMFFIAnyToTypeWithStorage<T>:
 *   (const TVMFFIAny *, Any *) -> T
 * It does similar thing as [Trait 1] that converts an `AnyView` or `Any` to
 * `T`, but is onyl used by TVM FFI's calling convention, where an additional
 * storage `Any*` is provided to retain ownership when unpacking `AnyView[]`
 * before calling into a C++ function.
 *
 * Example. When converting an `AnyView (kTVMFFIRawStr)` to `Str*`,
 * which consists of two intermediate steps `AnyView (kTVMFFIRawStr)` to
 * `Ref<Str>`, and `Ref<Str>` to `Str*`, this method will store the by-product
 * `Ref<Str>` into the given storage to lifespan expiration.
 *
 * It is used in the following case(s) in the codebase:
 * 1) `AnyView::CastWithStorage<Type>(Any *)`
 */
template <typename, typename = void>
struct TypeTraits {
  constexpr static bool enabled = false;
};
template <typename T>
using TypeTraitsNoCR = TypeTraits<details::RemoveCR<T>>;
template <typename T>
using HasTypeTraits = std::enable_if_t<TypeTraitsNoCR<T>::enabled>;

template <enum TVMFFITypeIndex type_index>
struct TypeIndexTraits;

#define TVM_FFI_DEF_TYPE_INDEX_TRAITS(TypeIndex_, TypeKey)                                         \
  template <>                                                                                      \
  struct TypeIndexTraits<TypeIndex_> {                                                             \
    static constexpr const char *type_key = TypeKey;                                               \
  }
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFINone, "None");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIInt, "int");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIFloat, "float");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIPtr, "Ptr");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIDataType, "dtype");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIDevice, "Device");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIRawStr, "const char *");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIObject, "object.Object");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIList, "object.List");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIDict, "object.Dict");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIError, "object.Error");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIFunc, "object.Func");
TVM_FFI_DEF_TYPE_INDEX_TRAITS(TVMFFITypeIndex::kTVMFFIStr, "object.Str");
#undef TVM_FFI_DEF_TYPE_INDEX_TRAITS

TVM_FFI_INLINE const char *TypeIndex2TypeKey(int32_t type_index) {
#define TVM_FFI_TYPE_INDEX_SWITCH_CASE(TypeIndex_)                                                 \
  case TypeIndex_:                                                                                 \
    return TypeIndexTraits<TypeIndex_>::type_key;
  switch (static_cast<TVMFFITypeIndex>(type_index)) {
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFINone);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIInt);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIFloat);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIPtr);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIDataType);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIDevice);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIRawStr);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIObject);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIList);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIDict);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIError);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIFunc);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIStr);
  default:
#if TVM_FFI_ALLOW_DYN_TYPE
  {
    TVMFFITypeInfoHandle type_info;
    TVMFFIDynTypeIndex2Info(nullptr, type_index, &type_info);
    return type_info ? type_info->type_key : "(undefined)";
  }
#else
    return "Unknown";
#endif
  }
#undef TVM_FFI_TYPE_INDEX_SWITCH_CASE
  TVM_FFI_UNREACHABLE();
}

namespace details {
struct DummyRoot {
  static constexpr bool _type_is_static = true;
  static constexpr int32_t _type_depth = -1;
  static constexpr int32_t _type_index = -1;
  static constexpr std::array<int32_t, 0> _type_ancestors = {};
};
template <typename F, typename TypeTableType>
TVM_FFI_INLINE void InitTypeTable(F f, TypeTableType *self) {
  {
    f.template RegisterType<TVMFFITypeIndex::kTVMFFINone, DummyRoot>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIInt, DummyRoot>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIFloat, DummyRoot>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIPtr, DummyRoot>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIDevice, DummyRoot>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIDataType, DummyRoot>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIRawStr, DummyRoot>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIObject, Object>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIList, Object>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIDict, Object>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIError, Error>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIStr, Str>(self);
    f.template RegisterType<TVMFFITypeIndex::kTVMFFIFunc, Func>(self);
  }
  {
    f.template RegisterStr<TVMFFITypeIndex::kTVMFFINone, void *>(self);
    f.template RegisterStr<TVMFFITypeIndex::kTVMFFIInt, int64_t>(self);
    f.template RegisterStr<TVMFFITypeIndex::kTVMFFIFloat, double>(self);
    f.template RegisterStr<TVMFFITypeIndex::kTVMFFIPtr, void *>(self);
    f.template RegisterStr<TVMFFITypeIndex::kTVMFFIDevice, DLDevice>(self);
    f.template RegisterStr<TVMFFITypeIndex::kTVMFFIDataType, DLDataType>(self);
    f.template RegisterStr<TVMFFITypeIndex::kTVMFFIRawStr, const char *>(self);
    f.template RegisterStr<TVMFFITypeIndex::kTVMFFIStr, const char *>(self);
  }
}

template <typename T>
using Identity = T;

template <template <typename> typename F = Identity>
TVM_FFI_INLINE void StaticAnyView2Str(std::ostream &os, const TVMFFIAny *v) {
#define TVM_FFI_TYPE_INDEX_SWITCH_CASE(TypeIndex_, Type)                                           \
  case TypeIndex_: {                                                                               \
    using Traits = F<TypeTraits<Type>>;                                                            \
    os << Traits::__str__(Traits::CopyFromTVMFFIAnyToType(v));                                     \
    break;                                                                                         \
  }
  switch (static_cast<TVMFFITypeIndex>(v->type_index)) {
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFINone, void *);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIInt, int64_t);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIFloat, double);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIPtr, void *);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIDevice, DLDevice);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIDataType, DLDataType);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIRawStr, const char *);
    TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIStr, const char *);
  default: {
    os << TypeIndex2TypeKey(v->type_index) << '@' << v->v_ptr;
    break;
  }
  }
#undef TVM_FFI_TYPE_INDEX_SWITCH_CASE
}

template <typename DerivedType>
struct StaticIsInstanceOf {
  static TVM_FFI_INLINE bool Check(int32_t type_index) {
#define TVM_FFI_TYPE_INDEX_SWITCH_CASE(TypeIndex_, Type)                                           \
  case TypeIndex_: {                                                                               \
    return StaticIsInstanceOf<DerivedType>::CheckImpl<Type>();                                     \
  }
    switch (static_cast<TVMFFITypeIndex>(type_index)) {
      TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIObject, Object);
      TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIList, List);
      TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIDict, Dict);
      TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIError, Error);
      TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIFunc, Func);
      TVM_FFI_TYPE_INDEX_SWITCH_CASE(TVMFFITypeIndex::kTVMFFIStr, Str);
    default:
      return false;
    }
#undef TVM_FFI_TYPE_INDEX_SWITCH_CASE
    TVM_FFI_UNREACHABLE();
  }

private:
  template <typename SelfType>
  static constexpr bool CheckImpl() {
    constexpr auto ancestors = SelfType::_type_ancestors;
    constexpr int32_t ancestors_depth = static_cast<int32_t>(ancestors.size());
    constexpr int32_t type_depth = DerivedType::_type_depth;
    return ancestors_depth > type_depth && ancestors[type_depth] == DerivedType::_type_index;
  }
};

template <typename DerivedType, typename SelfType = Object>
TVM_FFI_INLINE bool IsInstanceOf(int32_t type_index) {
  // TODO: Support non-Object types
  if constexpr (std::is_same_v<DerivedType, Object> ||
                std::is_base_of_v</*base=*/DerivedType, /*derived=*/SelfType>) {
    return true;
  }
  // Special case: `DerivedType` is exactly the underlying type of `type_index`
  if (type_index == DerivedType::_type_index) {
    return true;
  }
  // Given an index `i = DerivedType::_type_index`,
  // and the underlying type of `T = *this`, we wanted to check if
  // `T::_type_ancestors[i] == DerivedType::_type_index`.
  //
  // There are 3 ways to reflect `T` out of `this->type_index`:
  // (Case 1) Use `SelfType` as a surrogate type if `SelfType::_type_ancestors`
  // is long enough, whose length is reflected by `SelfType::_type_depth`.
  if constexpr (SelfType::_type_depth > DerivedType::_type_depth) {
    return SelfType::_type_ancestors[DerivedType::_type_depth] == DerivedType::_type_index;
  }
  if constexpr (SelfType::_type_depth == DerivedType::_type_depth) {
    return SelfType::_type_index == DerivedType::_type_index;
  }
  // (Case 2) If `type_index` falls in static object section, use switch case to
  // enumerate all possibilities. It could be more efficient because checks can
  // potentially be simplified by the compiler.
  if (details::IsTypeIndexPOD(type_index)) {
    return ::tvm::ffi::details::StaticIsInstanceOf<DerivedType>::Check(type_index);
  }
  // (Case 3) Look up the type table for type hierarchy via `type_index`.
#if TVM_FFI_ALLOW_DYN_TYPE
  TVMFFITypeInfoHandle info;
  TVMFFIDynTypeIndex2Info(nullptr, type_index, &info);
  if (info == nullptr) {
    TVM_FFI_THROW(InternalError) << "Undefined type index: " << type_index;
  }
  return info->type_depth > DerivedType::_type_depth &&
         info->type_ancestors[DerivedType::_type_depth] == DerivedType::_type_index;
#else
  if (type_index > static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDynObjectBegin)) {
    TVM_FFI_THROW(BuildError) << "Dynamic type is not supported. Please set "
                                 "`TVM_FFI_ALLOW_DYN_TYPE=ON` when building TVM FFI";
  }
  return false;
#endif
}

} // namespace details

TVM_FFI_INLINE const char *DLDeviceType2Str(DLDeviceType type) {
  switch (type) {
  case kDLCPU:
    return "cpu";
  case kDLCUDA:
    return "cuda";
  case kDLCUDAHost:
    return "cuda_host";
  case kDLOpenCL:
    return "opencl";
  case kDLVulkan:
    return "vulkan";
  case kDLMetal:
    return "mps";
  case kDLVPI:
    return "vpi";
  case kDLROCM:
    return "rocm";
  case kDLROCMHost:
    return "rocm_host";
  case kDLExtDev:
    return "ext_dev";
  case kDLCUDAManaged:
    return "cuda_managed";
  case kDLOneAPI:
    return "oneapi";
  case kDLWebGPU:
    return "webgpu";
  case kDLHexagon:
    return "hexagon";
  case kDLMAIA:
    return "maia";
  }
  return "unknown";
}

TVM_FFI_INLINE const char *DLDataTypeCode2Str(DLDataTypeCode type_code) {
  switch (type_code) {
  case kDLInt:
    return "int";
  case kDLUInt:
    return "uint";
  case kDLFloat:
    return "float";
  case kDLOpaqueHandle:
    return "ptr";
  case kDLBfloat:
    return "bfloat";
  case kDLComplex:
    return "complex";
  case kDLBool:
    return "bool";
  }
  return "unknown";
}

/********** TypeTraits: Object *********/

template <typename ObjectType>
struct TypeTraitsDefaultForObject {
  static constexpr bool enabled = true;

  TVM_FFI_INLINE static void CopyFromTypeToTVMFFIAny(ObjectType *src, TVMFFIAny *ret) {
    if (src == nullptr) {
      ret->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone);
      ret->v_obj = nullptr;
    } else {
      ret->type_index = src->type_index;
      ret->v_obj = const_cast<TVMFFIAny *>(reinterpret_cast<const TVMFFIAny *>(src));
    }
  }

  TVM_FFI_INLINE static ObjectType *CopyFromTVMFFIAnyToType(const TVMFFIAny *v) {
    if (details::IsTypeIndexNone(v->type_index)) {
      return nullptr;
    }
    if (!details::IsTypeIndexPOD(v->type_index) &&
        details::IsInstanceOf<ObjectType>(v->type_index)) {
      return reinterpret_cast<ObjectType *>(v->v_obj);
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static ObjectType *CopyFromTVMFFIAnyToRef(const TVMFFIAny *v) {
    return CopyFromTVMFFIAnyToType(v);
  }

  TVM_FFI_INLINE static ObjectType *MoveFromTVMFFIAnyToRef(TVMFFIAny *v) {
    if (details::IsTypeIndexNone(v->type_index)) {
      return nullptr;
    }
    if (!details::IsTypeIndexPOD(v->type_index) &&
        details::IsInstanceOf<ObjectType>(v->type_index)) {
      ObjectType *ret = reinterpret_cast<ObjectType *>(v->v_obj);
      v->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone);
      v->v_obj = nullptr;
      return ret;
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static std::string Type2Str() { return ObjectType::_type_key + std::string(" *"); }
};

template <typename ObjectType>
struct TypeTraits<ObjectType *, std::enable_if_t<IsObject<ObjectType>>>
    : public TypeTraitsDefaultForObject<ObjectType> {};

/********** TypeTraits: Integer *********/

template <typename Int>
struct TypeTraits<Int, std::enable_if_t<std::is_integral_v<Int>>> {
  static constexpr bool enabled = true;
  TVM_FFI_INLINE static void CopyFromTypeToTVMFFIAny(Int src, TVMFFIAny *ret) {
    ret->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIInt);
    ret->v_int64 = static_cast<int64_t>(src);
  }

  TVM_FFI_INLINE static Int CopyFromTVMFFIAnyToType(const TVMFFIAny *v) {
    TVMFFITypeIndex type_index = static_cast<TVMFFITypeIndex>(v->type_index);
    if (type_index == TVMFFITypeIndex::kTVMFFIInt) {
      return static_cast<Int>(v->v_int64);
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static std::string Type2Str() {
    return TypeIndexTraits<TVMFFITypeIndex::kTVMFFIInt>::type_key;
  }

  TVM_FFI_INLINE static std::string __str__(int64_t src) { return std::to_string(src); }
};

/********** TypeTraits: Float *********/

template <typename Float>
struct TypeTraits<Float, std::enable_if_t<std::is_floating_point_v<Float>>> {
  static constexpr bool enabled = true;
  TVM_FFI_INLINE static void CopyFromTypeToTVMFFIAny(Float src, TVMFFIAny *ret) {
    ret->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIFloat);
    ret->v_float64 = src;
  }

  TVM_FFI_INLINE static Float CopyFromTVMFFIAnyToType(const TVMFFIAny *v) {
    TVMFFITypeIndex type_index = static_cast<TVMFFITypeIndex>(v->type_index);
    if (type_index == TVMFFITypeIndex::kTVMFFIFloat) {
      return v->v_float64;
    } else if (type_index == TVMFFITypeIndex::kTVMFFIInt) {
      return static_cast<Float>(v->v_int64);
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static std::string Type2Str() {
    return TypeIndexTraits<TVMFFITypeIndex::kTVMFFIFloat>::type_key;
  }

  TVM_FFI_INLINE static std::string __str__(double src) { return std::to_string(src); }
};

/********** TypeTraits: Opaque Pointer *********/

template <>
struct TypeTraits<void *> {
  static constexpr bool enabled = true;
  using Ptr = void *;

  TVM_FFI_INLINE static void CopyFromTypeToTVMFFIAny(Ptr src, TVMFFIAny *ret) {
    ret->type_index = (src == nullptr) ? static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone)
                                       : static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIPtr);
    ret->v_ptr = src;
  }

  TVM_FFI_INLINE static Ptr CopyFromTVMFFIAnyToType(const TVMFFIAny *v) {
    TVMFFITypeIndex type_index = static_cast<TVMFFITypeIndex>(v->type_index);
    if (type_index == TVMFFITypeIndex::kTVMFFIPtr || type_index == TVMFFITypeIndex::kTVMFFIRawStr ||
        type_index == TVMFFITypeIndex::kTVMFFINone) {
      return v->v_ptr;
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static std::string Type2Str() {
    return TypeIndexTraits<TVMFFITypeIndex::kTVMFFIPtr>::type_key;
  }

  TVM_FFI_INLINE static std::string __str__(Ptr src) {
    if (src == nullptr) {
      return "None";
    } else {
      std::ostringstream oss;
      oss << src;
      return oss.str();
    }
  }
};

template <>
struct TypeTraits<std::nullptr_t> : public TypeTraits<void *> {};

/********** TypeTraits: DLDevice *********/

template <>
struct TypeTraits<DLDevice> {
  static constexpr bool enabled = true;
  TVM_FFI_INLINE static void CopyFromTypeToTVMFFIAny(DLDevice src, TVMFFIAny *ret) {
    ret->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDevice);
    ret->v_device = src;
  }

  TVM_FFI_INLINE static DLDevice CopyFromTVMFFIAnyToType(const TVMFFIAny *v) {
    TVMFFITypeIndex type_index = static_cast<TVMFFITypeIndex>(v->type_index);
    if (type_index == TVMFFITypeIndex::kTVMFFIDevice) {
      return v->v_device;
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static std::string Type2Str() {
    return TypeIndexTraits<TVMFFITypeIndex::kTVMFFIDevice>::type_key;
  }

  TVM_FFI_INLINE static std::string __str__(DLDevice device) {
    std::ostringstream os;
    os << DLDeviceType2Str(static_cast<DLDeviceType>(device.device_type)) << ":"
       << device.device_id;
    return os.str();
  }
};

/********** TypeTraits: DLDataType *********/

template <>
struct TypeTraits<DLDataType> {
  static constexpr bool enabled = true;
  TVM_FFI_INLINE static void CopyFromTypeToTVMFFIAny(DLDataType src, TVMFFIAny *ret) {
    ret->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIDataType);
    ret->v_dtype = src;
  }

  TVM_FFI_INLINE static DLDataType CopyFromTVMFFIAnyToType(const TVMFFIAny *v) {
    TVMFFITypeIndex type_index = static_cast<TVMFFITypeIndex>(v->type_index);
    if (type_index == TVMFFITypeIndex::kTVMFFIDataType) {
      return v->v_dtype;
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static std::string Type2Str() {
    return TypeIndexTraits<TVMFFITypeIndex::kTVMFFIDataType>::type_key;
  }

  TVM_FFI_INLINE static std::string __str__(DLDataType dtype) {
    DLDataTypeCode code = static_cast<DLDataTypeCode>(dtype.code);
    int32_t bits = dtype.bits;
    int32_t lanes = dtype.lanes;
    if (code == kDLUInt && bits == 1 && lanes == 1) {
      return "bool";
    }
    if (code == kDLOpaqueHandle && bits == 0 && lanes == 0) {
      return "void";
    }
    std::ostringstream os;
    os << DLDataTypeCode2Str(code) << bits;
    if (lanes != 1) {
      os << "x" << lanes;
    }
    return os.str();
  }
};

/********** TypeTraits: String *********/

template <>
struct TypeTraits<const char *> {
  static constexpr bool enabled = true;

  TVM_FFI_INLINE static void CopyFromTypeToTVMFFIAny(const char *src, TVMFFIAny *ret) {
    ret->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr);
    ret->v_str = src;
  }

  TVM_FFI_INLINE static const char *CopyFromTVMFFIAnyToType(const TVMFFIAny *v) {
    TVMFFITypeIndex type_index = static_cast<TVMFFITypeIndex>(v->type_index);
    if (type_index == TVMFFITypeIndex::kTVMFFIRawStr) {
      return v->v_str;
    }
    if (type_index == TVMFFITypeIndex::kTVMFFIStr) {
      return reinterpret_cast<TVMFFIStr *>(v->v_obj)->data;
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static std::string Type2Str() {
    return TypeIndexTraits<TVMFFITypeIndex::kTVMFFIRawStr>::type_key;
  }

  TVM_FFI_INLINE static std::string __str__(const char *src) {
    return '"' + std::string(src) + '"';
  }
};

template <size_t N>
struct TypeTraits<char[N]> : public TypeTraits<const char *> {};

template <>
struct TypeTraits<std::string> {
  static constexpr bool enabled = true;

  TVM_FFI_INLINE static void CopyFromTypeToTVMFFIAny(const std::string &src, TVMFFIAny *ret) {
    ret->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr);
    ret->v_str = src.data();
  }

  TVM_FFI_INLINE static std::string CopyFromTVMFFIAnyToType(const TVMFFIAny *v) {
    return TypeTraits<const char *>::CopyFromTVMFFIAnyToType(v);
  }

  TVM_FFI_INLINE static std::string Type2Str() { return "str"; }

  TVM_FFI_INLINE static std::string __str__(const char *src) {
    return '"' + std::string(src) + '"';
  }
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace ffi
} // namespace tvm

#endif // TVM_FFI_TRAITS_H_
