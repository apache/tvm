#ifndef TVM_FFI_UTILS_H_
#define TVM_FFI_UTILS_H_
#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanReverse64)
#pragma intrinsic(_InterlockedIncrement)
#pragma intrinsic(_InterlockedDecrement)
#endif
#if __cplusplus >= 202002L
#include <bit>
#endif
#include "./c_ffi_abi.h"
#include <array>
#include <cstddef>
#include <sstream>
#include <type_traits>

namespace tvm {
namespace ffi {

/********** Section 1. Macros *********/

#if defined(_MSC_VER)
#define TVM_FFI_INLINE __forceinline
#else
#define TVM_FFI_INLINE inline __attribute__((always_inline))
#endif

#if defined(_MSC_VER)
#define TVM_FFI_UNREACHABLE() __assume(false)
#else
#define TVM_FFI_UNREACHABLE() __builtin_unreachable()
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TVM_FFI_SYMBOL_HIDE __attribute__((visibility("hidden")))
#else
#define TVM_FFI_SYMBOL_HIDE
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TVM_FFI_FUNC_SIG __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define TVM_FFI_FUNC_SIG __FUNCSIG__
#else
#define TVM_FFI_FUNC_SIG __func__
#endif

#define TVM_FFI_DEF_ASSIGN(SelfType, SourceType)                                                   \
  TVM_FFI_INLINE SelfType &operator=(SourceType other) {                                           \
    SelfType(std::move(other)).Swap(*this);                                                        \
    return *this;                                                                                  \
  }

#define TVM_FFI_THROW(ErrorKind)                                                                   \
  ::tvm::ffi::details::ErrorBuilder(#ErrorKind, __FILE__, TVM_FFI_FUNC_SIG, __LINE__).Get()

/********** Section 2. Any/Object: Extensions and inheritance *********/

template <typename TExtra>
struct AnyWithExtra {
  int32_t type_index;
  int32_t ref_cnt;
  void (*deleter)(void *);
  TExtra _extra;
  TVM_FFI_INLINE AnyWithExtra() = default;
  // TODO: add tests for copy/move constructors
  TVM_FFI_INLINE AnyWithExtra(const AnyWithExtra &other) : _extra(other._extra) {}
  TVM_FFI_INLINE AnyWithExtra(AnyWithExtra &&other) : _extra(std::move(other._extra)) {}
  TVM_FFI_INLINE AnyWithExtra &operator=(const AnyWithExtra &other) {
    this->_extra = other._extra;
    return *this;
  }
  TVM_FFI_INLINE AnyWithExtra &operator=(AnyWithExtra &&other) {
    this->_extra = std::move(other._extra);
    return *this;
  }
};

struct AnyView;
struct Any;
struct Object;
struct Dict;
struct Error;
struct Str;
struct Func;
template <typename>
struct Ref;
template <typename>
struct RefBase;
struct List;

template <typename T>
struct IsAnyWithExtra {
  template <typename U>
  static auto test(AnyWithExtra<U> *) -> std::true_type;
  static auto test(...) -> std::false_type;
  using type = decltype(test(std::declval<T *>()));
};
template <typename T>
constexpr static bool IsObject = std::is_base_of_v<Object, T> || IsAnyWithExtra<T>::type::value;

template <typename ObjectType>
struct DefaultObjectAllocator {
  using Storage = typename std::aligned_storage<sizeof(ObjectType), alignof(ObjectType)>::type;

  template <typename... Args>
  inline static ObjectType *NewImpl(size_t num_storages, Args &&...args) {
    Storage *data = new Storage[num_storages];
    try {
      new (data) ObjectType(std::forward<Args>(args)...);
    } catch (...) {
      delete[] data;
      throw;
    }
    ObjectType *ret = reinterpret_cast<ObjectType *>(data);
    ret->type_index = ObjectType::_type_index;
    ret->ref_cnt = 0;
    ret->deleter = DefaultObjectAllocator::Deleter;
    return ret;
  }

  template <typename... Args>
  TVM_FFI_INLINE static ObjectType *New(Args &&...args) {
    return NewImpl(1, std::forward<Args>(args)...);
  }

  template <typename PadType, typename... Args>
  TVM_FFI_INLINE static ObjectType *NewWithPad(size_t pad_size, Args &&...args) {
    return NewImpl((sizeof(ObjectType) + pad_size * sizeof(PadType) + sizeof(Storage) - 1) /
                       sizeof(Storage),
                   std::forward<Args>(args)...);
  }

  static void Deleter(void *objptr) {
    ObjectType *tptr = static_cast<ObjectType *>(objptr);
    tptr->ObjectType::~ObjectType();
    delete[] reinterpret_cast<Storage *>(tptr);
  }
};

template <typename T, typename = void>
struct GetAllocator {
  using Type = DefaultObjectAllocator<T>;
};
template <typename T>
struct GetAllocator<T, std::void_t<typename T::Allocator>> {
  using Type = typename T::Allocator;
};

template <size_t N>
using CharArray = const char[N];

/********** Section 3. Atomic Operations *********/

namespace details {
uint64_t TVMFFIStrHash(const TVMFFIStr *str);
int32_t TVMFFIStrCmp(const TVMFFIStr *a, const TVMFFIStr *b);
TVM_FFI_INLINE int32_t AtomicIncrementRelaxed(int32_t *ptr) {
#ifdef _MSC_VER
  return _InterlockedIncrement(reinterpret_cast<volatile long *>(ptr)) - 1;
#else
  return __atomic_fetch_add(ptr, 1, __ATOMIC_RELAXED);
#endif
}
TVM_FFI_INLINE int32_t AtomicDecrementRelAcq(int32_t *ptr) {
#ifdef _MSC_VER
  return _InterlockedDecrement(reinterpret_cast<volatile long *>(ptr)) + 1;
#else
  return __atomic_fetch_sub(ptr, 1, __ATOMIC_ACQ_REL);
#endif
}

TVM_FFI_INLINE void IncRef(TVMFFIObject *obj) {
  if (obj != nullptr) {
    AtomicIncrementRelaxed(&obj->ref_cnt);
  }
}

TVM_FFI_INLINE void DecRef(TVMFFIObject *obj) {
  if (obj != nullptr) {
    if (AtomicDecrementRelAcq(&obj->ref_cnt) == 1) {
      if (obj->deleter) {
        obj->deleter(obj);
      }
    }
  }
}

TVM_FFI_INLINE bool IsTypeIndexNone(int32_t type_index) {
  return type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone);
}

TVM_FFI_INLINE bool IsTypeIndexPOD(int32_t type_index) {
  return type_index < static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIStaticObjectBegin);
}

} // namespace details

/********** Section 4. Errors *********/

struct TemporaryTypeError : public std::exception {};

namespace details {

template <typename T>
using RemoveCR = std::remove_const_t<std::remove_reference_t<T>>;
template <typename T>
static constexpr bool IsAnyOrView = std::is_same_v<T, Any> || std::is_same_v<T, AnyView>;

template <typename>
struct IsRefImpl {
  static constexpr bool value = false;
};

template <typename T>
struct IsRefImpl<Ref<T>> {
  static constexpr bool value = true;
};
template <typename T>
constexpr bool IsRef = IsRefImpl<T>::value;

void AnyView2Str(std::ostream &os, const TVMFFIAny *v);
TVMFFIObject *StrMoveFromStdString(std::string &&source);
TVMFFIObject *StrCopyFromCharArray(const char *source, size_t length);
[[noreturn]] void TVMErrorFromBuilder(std::string &&kind, std::string &&lineno,
                                      std::string &&message) noexcept(false);

template <typename ParentType, size_t size>
inline constexpr std::array<int32_t, size> ObjectAncestorsConstExpr() {
  std::array<int32_t, size> ret{};
  for (size_t i = 0; i < size; ++i) {
    ret[i] = (i + 1 == size) ? (ParentType::_type_index) : (ParentType::_type_ancestors[i]);
  }
  return ret;
}
template <typename ParentType, size_t size>
inline std::array<int32_t, size> ObjectAncestors() {
  std::array<int32_t, size> ret{};
  for (size_t i = 0; i < size; ++i) {
    ret[i] = (i + 1 == size) ? (ParentType::_type_index) : (ParentType::_type_ancestors[i]);
  }
  return ret;
}

// Disable warning about throwing in the destructor
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4722) // throw in the destructor, which is expected behavior
#endif

struct ErrorBuilder {
  const char *kind;
  TVMFFIStackFrame frame;
  std::ostringstream oss;
  std::string line_info;

  explicit ErrorBuilder(const char *kind, const char *filename, const char *func, int32_t lineno)
      : kind(kind), frame{filename, func, lineno, ErrorBuilder::StackFrameDeleter} {
    this->line_info = std::string(filename) + ":" + std::to_string(lineno);
  }

  [[noreturn]] ~ErrorBuilder() noexcept(false) {
    details::TVMErrorFromBuilder(this->kind, std::move(this->line_info), oss.str());
    throw;
  }

  std::ostringstream &Get() { return this->oss; }

protected:
  static void StackFrameDeleter(void *self) { delete static_cast<TVMFFIStackFrame *>(self); }
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

} // namespace details
} // namespace ffi
} // namespace tvm
#endif // TVM_FFI_UTILS_H_
