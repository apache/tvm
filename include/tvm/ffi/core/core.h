#ifndef TVM_FFI_CORE_H_
#define TVM_FFI_CORE_H_

#include "./c_ffi_abi.h"
#include "./traits.h"
#include "./utils.h"
#include <cstring>
#include <type_traits>

namespace tvm {
namespace ffi {

#define TVM_FFI_DEF_TYPE_FRIENDS()                                                                 \
  template <typename>                                                                              \
  friend struct ::tvm::ffi::IsAnyWithExtra;                                                        \
  template <typename>                                                                              \
  friend struct ::tvm::ffi::Ref;                                                                   \
  template <typename>                                                                              \
  friend struct ::tvm::ffi::RefBase;                                                               \
  template <typename, typename>                                                                    \
  friend struct ::tvm::ffi::TypeTraits;                                                            \
  template <typename>                                                                              \
  friend struct ::tvm::ffi::TypeTraitsDefaultForObject;                                            \
  template <typename>                                                                              \
  friend struct ::tvm::ffi::DefaultObjectAllocator

/********** Section 1. Object *********/

#define TVM_FFI_DEF_STATIC_TYPE(SelfType, ParentType, TypeIndex)                                   \
public:                                                                                            \
  TVM_FFI_DEF_TYPE_FRIENDS();                                                                      \
  friend struct ::tvm::ffi::Any;                                                                   \
  friend struct ::tvm::ffi::AnyView;                                                               \
  [[maybe_unused]] static constexpr int32_t _type_index = static_cast<int32_t>(TypeIndex);         \
  using _type_parent [[maybe_unused]] = ParentType;                                                \
  [[maybe_unused]] static constexpr int32_t _type_depth = ParentType::_type_depth + 1;             \
  [[maybe_unused]] static inline constexpr std::array<int32_t, _type_depth> _type_ancestors =      \
      ::tvm::ffi::details::ObjectAncestorsConstExpr<ParentType, _type_depth>();                    \
  template <typename DerivedType>                                                                  \
  TVM_FFI_INLINE bool IsInstance() const {                                                         \
    return ::tvm::ffi::details::IsInstanceOf<DerivedType, SelfType>(this->type_index);             \
  }                                                                                                \
  TVM_FFI_INLINE const char *GetTypeKey() const { return TypeIndex2TypeKey(this->type_index); }    \
  [[maybe_unused]] static constexpr const char *_type_key = TypeIndexTraits<TypeIndex>::type_key

#define TVM_FFI_DEF_DYN_TYPE(SelfType, ParentType, TypeKey)                                        \
public:                                                                                            \
  TVM_FFI_DEF_TYPE_FRIENDS();                                                                      \
  friend struct ::tvm::ffi::Any;                                                                   \
  friend struct ::tvm::ffi::AnyView;                                                               \
  using _type_parent [[maybe_unused]] = ParentType;                                                \
  [[maybe_unused]] static constexpr int32_t _type_depth = ParentType::_type_depth + 1;             \
  [[maybe_unused]] static inline const std::array<int32_t, _type_depth> _type_ancestors =          \
      ::tvm::ffi::details::ObjectAncestors<ParentType, _type_depth>();                             \
  [[maybe_unused]] static inline int32_t _type_index = []() -> int32_t {                           \
    int32_t ret;                                                                                   \
    TVMFFIDynTypeDef(nullptr, TypeKey, _type_depth, _type_ancestors.data(), &ret);                 \
    return ret;                                                                                    \
  }();                                                                                             \
  template <typename DerivedType>                                                                  \
  TVM_FFI_INLINE bool IsInstance() const {                                                         \
    return ::tvm::ffi::details::IsInstanceOf<DerivedType, SelfType>(this->type_index);             \
  }                                                                                                \
  TVM_FFI_INLINE const char *GetTypeKey() const { return TypeIndex2TypeKey(this->type_index); }    \
  [[maybe_unused]] static constexpr const char *_type_key = TypeKey

struct Object : protected TVMFFIAny {
  TVM_FFI_DEF_STATIC_TYPE(Object, details::DummyRoot, TVMFFITypeIndex::kTVMFFIObject);

  TVM_FFI_INLINE Object() : TVMFFIAny() {}
  TVM_FFI_INLINE Object(const Object &) : TVMFFIAny() {}
  TVM_FFI_INLINE Object(Object &&) {}
  TVM_FFI_INLINE Object &operator=(const Object &) { return *this; }
  TVM_FFI_INLINE Object &operator=(Object &&) { return *this; }
  Ref<Str> str() const;
  TVM_FFI_INLINE friend std::ostream &operator<<(std::ostream &os, const Object &src) {
    TVMFFIAny v{};
    v.type_index = src.type_index;
    v.v_obj = const_cast<Object *>(&src);
    details::AnyView2Str(os, &v);
    return os;
  }
};

/********** Section 2. Ref<ObjectType> *********/

#define TVM_FFI_REF_DEF_DELEGATE_CONSTRUCTORS(SelfType, BaseType)                                  \
  template <typename Other, typename = std::is_constructible<BaseType, Other &&>>                  \
  TVM_FFI_INLINE Ref(Other &&src) : BaseType(std::forward<Other>(src)) {}                          \
  template <typename Other, typename = std::is_constructible<BaseType, Other &&>>                  \
  TVM_FFI_DEF_ASSIGN(SelfType, const Other &)

template <typename Type>
struct RefBase {
private:
  using TSelf = RefBase<Type>;
  using TSub = Ref<Type>;
  TVM_FFI_DEF_TYPE_FRIENDS();
  friend struct Any;
  friend struct AnyView;
  template <typename T>
  using EnableAnyOrViewOrRef = std::enable_if_t<details::IsAnyOrView<T> || details::IsRef<T>>;
  template <typename Base, typename Derived>
  using EnableDerivedObj =
      typename std::enable_if_t<std::is_base_of_v<Base, Derived> ||
                                (std::is_same_v<Base, Object> && IsObject<Derived>)>;

public:
  /***** Factory: the `new` operator *****/
  template <typename... Args>
  TVM_FFI_INLINE static TSub New(Args &&...args) {
    return TSub(Allocator::New(std::forward<Args>(args)...));
  }
  /***** Accessors *****/
  TVM_FFI_INLINE const Type *get() const { return reinterpret_cast<const Type *>(this->data_); }
  TVM_FFI_INLINE Type *get() { return reinterpret_cast<Type *>(data_); }
  TVM_FFI_INLINE const Type *operator->() const { return get(); }
  TVM_FFI_INLINE const Type &operator*() const { return *get(); }
  TVM_FFI_INLINE Type *operator->() { return get(); }
  TVM_FFI_INLINE Type &operator*() { return *get(); }
  /***** Misc *****/
  Ref<Str> str() const;
  TVM_FFI_INLINE friend std::ostream &operator<<(std::ostream &os, const TSelf &src) {
    TVMFFIAny v = src.AsTVMFFIAny();
    details::AnyView2Str(os, &v);
    return os;
  }
  template <typename BaseType = Type>
  BaseType *GetRawObjPtr() const {
    static_assert(std::is_same_v<BaseType, Object> || std::is_base_of_v<BaseType, Type>,
                  "Only downcasting is allowed");
    return reinterpret_cast<BaseType *>(this->data_);
  }
  template <typename BaseType = Type>
  BaseType *MoveToRawObjPtr() {
    static_assert(std::is_same_v<BaseType, Object> || std::is_base_of_v<BaseType, Type>,
                  "Only downcasting is allowed");
    BaseType *ret = reinterpret_cast<BaseType *>(this->data_);
    this->data_ = nullptr;
    return ret;
  }

protected:
  using Allocator = typename GetAllocator<Type>::Type;
  /***** Destructor *****/
  TVM_FFI_INLINE ~RefBase() { this->DecRef(); }
  /***** Constructor 0: default *****/
  TVM_FFI_INLINE RefBase() : data_(nullptr) {}
  /***** Constructor 1: From raw pointers  *****/
  TVM_FFI_INLINE RefBase(TVMFFIAny *data) : data_(data) { this->IncRef(); }
  template <typename Derived, typename = EnableDerivedObj<Type, Derived>>
  TVM_FFI_INLINE RefBase(Derived *data) : TSelf(reinterpret_cast<TVMFFIAny *>(data)) {}
  TVM_FFI_DEF_ASSIGN(TSelf, TVMFFIAny *)
  template <typename Derived, typename = EnableDerivedObj<Type, Derived>>
  TVM_FFI_DEF_ASSIGN(TSelf, Derived *)
  /***** Constructor 2: from RefBase<Type> *****/
  TVM_FFI_INLINE RefBase(const TSelf &other) : data_(other.data_) {
    this->IncRef();
  }
  TVM_FFI_INLINE RefBase(TSelf &&other) : data_(other.data_) { other.data_ = nullptr; }
  TVM_FFI_DEF_ASSIGN(TSelf, const TSelf &)
  TVM_FFI_DEF_ASSIGN(TSelf, TSelf &&)
  /***** Constructor 3: from Ref<SubType>, Any and AnyView *****/
  template <typename Other, typename = EnableAnyOrViewOrRef<Other>>
  TVM_FFI_DEF_ASSIGN(TSub, Other &&)
  template <typename Other, typename = EnableAnyOrViewOrRef<Other>>
  TVM_FFI_DEF_ASSIGN(TSub, const Other &)
  template <typename Other, typename = EnableAnyOrViewOrRef<Other>>
  TVM_FFI_INLINE RefBase(const Other &src) : RefBase(src.template GetRawObjPtr<Type>()) {}
  template <typename Other, typename = EnableAnyOrViewOrRef<Other>>
  TVM_FFI_INLINE RefBase(Other &&src)
      : data_(reinterpret_cast<TVMFFIAny *>(src.template MoveToRawObjPtr<Type>())) {
    if constexpr (std::is_same_v<Other, AnyView>) {
      this->IncRef();
    }
  }
  TVM_FFI_INLINE void IncRef() { details::IncRef(this->data_); }
  TVM_FFI_INLINE void DecRef() { details::DecRef(this->data_); }
  TVM_FFI_INLINE void Swap(TSelf &other) { std::swap(this->data_, other.data_); }
  TVM_FFI_INLINE TVMFFIAny AsTVMFFIAny() const {
    if (data_ == nullptr) {
      return TVMFFIAny();
    }
    TVMFFIAny ret{};
    ret.type_index = data_->type_index;
    ret.v_obj = data_;
    return ret;
  }

  TVMFFIAny *data_;
};

template <typename Type>
struct Ref : public RefBase<Type> {
  TVM_FFI_DEF_TYPE_FRIENDS();
  TVM_FFI_INLINE Ref() : RefBase<Type>() {}
  TVM_FFI_REF_DEF_DELEGATE_CONSTRUCTORS(Ref<Type>, RefBase<Type>)
};

/********** Section 3. AnyView *********/

struct AnyView : public TVMFFIAny {
  TVM_FFI_DEF_TYPE_FRIENDS();
  friend struct Any;
  /***** Destructor *****/
  TVM_FFI_INLINE ~AnyView() = default;
  TVM_FFI_INLINE void Reset() { *(static_cast<TVMFFIAny *>(this)) = TVMFFIAny(); }
  /***** Constructor 0: default *****/
  TVM_FFI_INLINE AnyView() : TVMFFIAny() {}
  /***** Constructor 1: from AnyView *****/
  TVM_FFI_INLINE AnyView(const AnyView &src) = default;
  TVM_FFI_INLINE AnyView &operator=(const AnyView &src) = default;
  TVM_FFI_INLINE AnyView(AnyView &&src) : TVMFFIAny(*&src) { src.Reset(); }
  TVM_FFI_DEF_ASSIGN(AnyView, AnyView &&)
  /***** Constructor 2: from Any *****/
  TVM_FFI_INLINE AnyView(const Any &src);
  TVM_FFI_DEF_ASSIGN(AnyView, const Any &)
  AnyView(Any &&src) = delete;
  AnyView &operator=(Any &&src) = delete;
  /***** Constructor 3: from Ref<T> *****/
  template <typename T>
  TVM_FFI_INLINE AnyView(const Ref<T> &src) : TVMFFIAny(src.AsTVMFFIAny()) {}
  template <typename T>
  TVM_FFI_DEF_ASSIGN(AnyView, const Ref<T> &)
  template <typename T>
  AnyView(Ref<T> &&src) = delete;
  template <typename T>
  AnyView &operator=(Ref<T> &&src) = delete;
  /***** Constructors 4: use TypeTraits<T> *****/
  template <typename Type, typename = HasTypeTraits<Type>>
  TVM_FFI_INLINE AnyView(const Type &src) : TVMFFIAny() {
    TypeTraitsNoCR<Type>::CopyFromTypeToTVMFFIAny(src, this);
  }
  template <typename Type, typename = HasTypeTraits<Type>>
  TVM_FFI_DEF_ASSIGN(AnyView, const Type &)
  /*** Converter 0: use TypeTraits<T> ***/
  template <typename Type, typename = HasTypeTraits<Type>>
  operator Type() const;
  template <typename Type, typename = HasTypeTraits<Type>>
  Type CastWithStorage(Any *storage) const;
  /***** Misc *****/
  Ref<Str> str() const;
  TVM_FFI_INLINE friend std::ostream &operator<<(std::ostream &os, const AnyView &src) {
    details::AnyView2Str(os, &src);
    return os;
  }
  template <typename Type>
  TVM_FFI_INLINE Type *GetRawObjPtr() const;
  template <typename Type>
  TVM_FFI_INLINE Type *MoveToRawObjPtr();

protected:
  TVM_FFI_INLINE void Swap(TVMFFIAny &src) {
    TVMFFIAny tmp = *this;
    *static_cast<TVMFFIAny *>(this) = src;
    src = tmp;
  }
};

/********** Section 4. Any *********/

struct Any : public TVMFFIAny {
  TVM_FFI_DEF_TYPE_FRIENDS();
  friend struct AnyView;
  /***** Destructor *****/
  TVM_FFI_INLINE ~Any() { this->Reset(); }
  TVM_FFI_INLINE void Reset() {
    this->DecRef();
    *(static_cast<TVMFFIAny *>(this)) = TVMFFIAny();
  }
  /***** Constructor 0: default *****/
  TVM_FFI_INLINE Any() : TVMFFIAny() {}
  /***** Constructor 1: from AnyView *****/
  TVM_FFI_INLINE Any(const AnyView &src) : TVMFFIAny(*static_cast<const TVMFFIAny *>(&src)) {
    if (this->type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr)) {
      // Special case: handle the case where `Any` needs to own a raw string.
      this->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr);
      this->v_obj = details::StrCopyFromCharArray(this->v_str, std::strlen(this->v_str));
    }
    this->IncRef();
  }
  TVM_FFI_INLINE Any(AnyView &&src) : Any(static_cast<const AnyView &>(src)) { // TODO: add test
    src.Reset();
  }
  TVM_FFI_DEF_ASSIGN(Any, const AnyView &)
  TVM_FFI_DEF_ASSIGN(Any, AnyView &&)
  /***** Constructor 2: from Any *****/
  TVM_FFI_INLINE Any(const Any &src) : TVMFFIAny(*static_cast<const TVMFFIAny *>(&src)) {
    this->IncRef();
  }
  TVM_FFI_INLINE Any(Any &&src) : TVMFFIAny(*static_cast<const TVMFFIAny *>(&src)) {
    *static_cast<TVMFFIAny *>(&src) = TVMFFIAny();
  }
  TVM_FFI_DEF_ASSIGN(Any, const Any &)
  TVM_FFI_DEF_ASSIGN(Any, Any &&)
  /***** Constructor 3: from Ref<T> *****/
  template <typename T>
  TVM_FFI_INLINE Any(const Ref<T> &src) : TVMFFIAny(src.AsTVMFFIAny()) {
    this->IncRef();
  }
  template <typename T>
  TVM_FFI_INLINE Any(Ref<T> &&src) : TVMFFIAny(src.AsTVMFFIAny()) {
    src.data_ = nullptr;
  }
  template <typename T>
  TVM_FFI_DEF_ASSIGN(Any, const Ref<T> &)
  template <typename T>
  TVM_FFI_DEF_ASSIGN(Any, Ref<T> &&)
  /***** Constructors 4: use TypeTraits<T> *****/
  template <typename Type, typename = HasTypeTraits<Type>>
  TVM_FFI_INLINE Any(const Type &src) : Any(AnyView(src)) {}
  template <typename Type, typename = HasTypeTraits<Type>>
  TVM_FFI_DEF_ASSIGN(Any, const Type &)
  /***** Constructors 5: Special handling for strings *****/
  TVM_FFI_INLINE Any(const std::string &s) : Any(AnyView(s)) {}
  TVM_FFI_INLINE Any(const char *s) : Any(AnyView(s)) {}
  TVM_FFI_INLINE Any(std::string &&s) : TVMFFIAny() {
    this->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr);
    this->v_obj = details::StrMoveFromStdString(std::move(s));
    this->IncRef();
  }
  template <size_t N>
  TVM_FFI_INLINE Any(const CharArray<N> &s) : TVMFFIAny() {
    this->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStr);
    this->v_obj = details::StrCopyFromCharArray(s, N - 1);
    this->IncRef();
  }
  TVM_FFI_DEF_ASSIGN(Any, const std::string &)
  TVM_FFI_DEF_ASSIGN(Any, const char *&)
  TVM_FFI_DEF_ASSIGN(Any, std::string &&)
  template <size_t N>
  TVM_FFI_DEF_ASSIGN(Any, const CharArray<N> &)
  /*** Converter 0: use TypeTraits<T> ***/
  template <typename Type, typename = HasTypeTraits<Type>>
  operator Type() const;
  /***** Misc *****/
  Ref<Str> str() const;
  TVM_FFI_INLINE friend std::ostream &operator<<(std::ostream &os, const Any &src) {
    details::AnyView2Str(os, &src);
    return os;
  }
  template <typename Type>
  TVM_FFI_INLINE Type *GetRawObjPtr() const {
    return reinterpret_cast<const AnyView *>(this)->GetRawObjPtr<Type>();
  }
  template <typename Type>
  TVM_FFI_INLINE Type *MoveToRawObjPtr() {
    return reinterpret_cast<AnyView *>(this)->MoveToRawObjPtr<Type>();
  }

protected:
  TVM_FFI_INLINE void Swap(TVMFFIAny &src) {
    TVMFFIAny tmp = *this;
    *static_cast<TVMFFIAny *>(this) = src;
    src = tmp;
  }
  TVM_FFI_INLINE void IncRef() {
    if (!details::IsTypeIndexPOD(this->type_index)) {
      details::IncRef(this->v_obj);
    }
  }
  TVM_FFI_INLINE void DecRef() {
    if (!details::IsTypeIndexPOD(this->type_index)) {
      details::DecRef(this->v_obj);
    }
  }
};

/********** Section 5. Type Conversion and Type Table *********/

TVM_FFI_INLINE AnyView::AnyView(const Any &src) : TVMFFIAny(*&src) {}

#define TVM_FFI_TRY_CONVERT(Expr, TypeStr)                                                         \
  try {                                                                                            \
    return Expr;                                                                                   \
  } catch (const TemporaryTypeError &) {                                                           \
    TVM_FFI_THROW(TypeError) << "Cannot convert from type `"                                       \
                             << TypeIndex2TypeKey(this->type_index) << "` to `" << TypeStr << "`"; \
  }                                                                                                \
  TVM_FFI_UNREACHABLE();
template <typename Type, typename>
inline AnyView::operator Type() const {
  TVM_FFI_TRY_CONVERT(TypeTraitsNoCR<Type>::CopyFromTVMFFIAnyToType(this),
                      TypeTraitsNoCR<Type>::Type2Str());
}
template <typename Type, typename>
inline Type AnyView::CastWithStorage(Any *storage) const {
  TVM_FFI_TRY_CONVERT(TypeTraitsNoCR<Type>::CopyFromTVMFFIAnyToTypeWithStorage(this, storage),
                      TypeTraitsNoCR<Type>::Type2Str());
}
template <typename Type, typename>
inline Any::operator Type() const {
  TVM_FFI_TRY_CONVERT(TypeTraitsNoCR<Type>::CopyFromTVMFFIAnyToType(this),
                      TypeTraitsNoCR<Type>::Type2Str());
}
template <typename Type>
inline Type *AnyView::GetRawObjPtr() const {
  TVM_FFI_TRY_CONVERT(TypeTraitsNoCR<Type *>::CopyFromTVMFFIAnyToRef(this), Type::_type_key);
}
template <typename Type>
inline Type *AnyView::MoveToRawObjPtr() {
  TVM_FFI_TRY_CONVERT(TypeTraitsNoCR<Type *>::MoveFromTVMFFIAnyToRef(this), Type::_type_key);
}
#undef TVM_FFI_TRY_CONVERT

#if TVM_FFI_ALLOW_DYN_TYPE
TVM_FFI_INLINE void TypeSetAttr(int32_t type_index, const char *attr_key, AnyView attr_value) {
  TVMFFIDynTypeSetAttr(nullptr, type_index, attr_key, &attr_value);
}

TVM_FFI_INLINE AnyView TypeGetAttr(int32_t type_index, const char *attr_key) {
  TVMFFIAnyHandle attr;
  TVMFFIDynTypeGetAttr(nullptr, type_index, attr_key, &attr);
  return AnyView(*(static_cast<AnyView *>(attr)));
}
#endif

} // namespace ffi
} // namespace tvm

#endif // TVM_FFI_CORE_H_
