#ifndef TVM_FFI_STR_H_
#define TVM_FFI_STR_H_
#include <cstring>
#include <sstream>
#include <tvm/ffi/core/core.h>

namespace tvm {
namespace ffi {
namespace details {
struct StrHeader {
  int64_t length;
  const char *data;
};
} // namespace details

struct Str : private AnyWithExtra<details::StrHeader> {
  struct Allocator;
  using Header = AnyWithExtra<details::StrHeader>;
  TVM_FFI_DEF_STATIC_TYPE(Str, Object, TVMFFITypeIndex::kTVMFFIStr);

  TVM_FFI_INLINE const char *c_str() const { return this->_extra.data; }
  TVM_FFI_INLINE const char *data() const { return this->_extra.data; }
  TVM_FFI_INLINE int64_t length() const { return this->_extra.length; }
  TVM_FFI_INLINE int64_t size() const { return this->_extra.length; }
  inline uint64_t Hash() const {
    return details::TVMFFIStrHash(reinterpret_cast<const TVMFFIStr *>(this));
  }

protected:
  using Header::_extra;
  TVM_FFI_INLINE Str() : Header() {}
};

static_assert(sizeof(Str) == sizeof(TVMFFIStr));
static_assert(offsetof(Str::Header, _extra.length) == offsetof(TVMFFIStr, length));
static_assert(offsetof(Str::Header, _extra.data) == offsetof(TVMFFIStr, data));

namespace details {

struct StrStd : public Str {
  using Allocator = ::tvm::ffi::DefaultObjectAllocator<StrStd>;
  template <typename>
  friend struct ::tvm::ffi::DefaultObjectAllocator;

protected:
  TVM_FFI_INLINE StrStd(std::string &&str) : Str(), container(std::move(str)) {
    this->_extra.length = static_cast<int64_t>(container.length());
    this->_extra.data = container.data();
  }

  std::string container;
};

struct StrPad : public Str {
  using Allocator = ::tvm::ffi::DefaultObjectAllocator<StrPad>;
  template <typename>
  friend struct ::tvm::ffi::DefaultObjectAllocator;

protected:
  TVM_FFI_INLINE StrPad(const char *str, size_t N) : Str() {
    char *str_copy = reinterpret_cast<char *>(this) + sizeof(Str);
    std::memcpy(str_copy, str, N);
    str_copy[N - 1] = '\0';

    this->_extra.length = static_cast<int64_t>(N) - 1;
    this->_extra.data = str_copy;
  }
};
} // namespace details

struct Str::Allocator {
  TVM_FFI_INLINE static Str *New(std::string &&str) {
    return details::StrStd::Allocator::New(std::move(str));
  }
  TVM_FFI_INLINE static Str *New(const std::string &str) {
    int64_t N = static_cast<int64_t>(str.length()) + 1;
    return details::StrPad::Allocator::NewWithPad<char>(N, str.data(), N);
  }
  TVM_FFI_INLINE static Str *New(const char *str) {
    int64_t N = static_cast<int64_t>(std::strlen(str)) + 1;
    return details::StrPad::Allocator::NewWithPad<char>(N, str, N);
  }
  template <size_t N>
  TVM_FFI_INLINE static Str *New(const CharArray<N> &str) {
    return details::StrPad::Allocator::NewWithPad<char>(N, str, N);
  }
  TVM_FFI_INLINE static Str *New(const char *str, size_t N) {
    return details::StrPad::Allocator::NewWithPad<char>(N, str, N);
  }
};

template <>
struct Ref<Str> : public RefBase<Str> {
  TVM_FFI_DEF_TYPE_FRIENDS();
  TVM_FFI_INLINE Ref() : RefBase<Str>() {}
  TVM_FFI_REF_DEF_DELEGATE_CONSTRUCTORS(Ref<Str>, RefBase<Str>)

  using RefBase<Str>::New;
  template <size_t N>
  TVM_FFI_INLINE static TSub New(const CharArray<N> &arg) {
    return TSub(Str::Allocator::template New<N>(arg));
  }
};

TVM_FFI_INLINE Ref<Str> Object::str() const {
  std::ostringstream os;
  os << *this;
  return Ref<Str>::New(os.str());
}

template <typename ObjectType>
TVM_FFI_INLINE Ref<Str> RefBase<ObjectType>::str() const {
  std::ostringstream os;
  os << *this;
  return Ref<Str>::New(os.str());
}

TVM_FFI_INLINE Ref<Str> AnyView::str() const {
  std::ostringstream os;
  os << *this;
  return Ref<Str>::New(os.str());
}

TVM_FFI_INLINE Ref<Str> Any::str() const {
  std::ostringstream os;
  os << *this;
  return Ref<Str>::New(os.str());
}

/********** TypeTraits *********/

template <>
struct TypeTraits<Str *> : public TypeTraitsDefaultForObject<Str> {
private:
  using TypeTraitsDefaultForObject<Str>::MoveFromTVMFFIAnyToRef;
  using TypeTraitsDefaultForObject<Str>::CopyFromTVMFFIAnyToRef;

public:
  TVM_FFI_INLINE static Str *CopyFromTVMFFIAnyToRef(const TVMFFIAny *v) {
    if (details::IsTypeIndexNone(v->type_index)) {
      return nullptr;
    }
    if (v->type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr)) {
      return Str::Allocator::New(v->v_str, std::strlen(v->v_str) + 1);
    }
    if (!details::IsTypeIndexPOD(v->type_index) && details::IsInstanceOf<Str>(v->type_index)) {
      return reinterpret_cast<Str *>(v->v_obj);
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static Str *CopyFromTVMFFIAnyToTypeWithStorage(const TVMFFIAny *v, Any *storage) {
    if (details::IsTypeIndexNone(v->type_index)) {
      return nullptr;
    }
    if (v->type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIRawStr)) {
      Str *ret = Str::Allocator::New(v->v_str, std::strlen(v->v_str) + 1);
      *storage = Ref<Str>(ret);
      return ret;
    }
    if (!details::IsTypeIndexPOD(v->type_index) && details::IsInstanceOf<Str>(v->type_index)) {
      return reinterpret_cast<Str *>(v->v_obj);
    }
    throw TemporaryTypeError();
  }

  TVM_FFI_INLINE static Str *MoveFromTVMFFIAnyToRef(TVMFFIAny *v) {
    Str *ret = CopyFromTVMFFIAnyToRef(v);
    v->type_index = static_cast<int32_t>(TVMFFITypeIndex::kTVMFFINone);
    v->v_obj = nullptr;
    return ret;
  }
};

namespace details {

Any FuncInvoke(const void *func, int32_t num_args, const TVMFFIAny *args);

inline int32_t TVMFFIStrCmp(const TVMFFIStr *a, const TVMFFIStr *b) {
  if (a->length != b->length) {
    return a->length - b->length;
  }
  return std::strncmp(a->data, b->data, a->length);
}

inline uint64_t TVMFFIStrHash(const TVMFFIStr *str) {
  const constexpr uint64_t kMultiplier = 1099511628211ULL;
  const constexpr uint64_t kMod = 2147483647ULL;
  const char *it = str->data;
  const char *end = it + str->length;
  uint64_t result = 0;
  for (; it + 8 <= end; it += 8) {
    uint64_t b = (static_cast<uint64_t>(it[0]) << 56) | (static_cast<uint64_t>(it[1]) << 48) |
                 (static_cast<uint64_t>(it[2]) << 40) | (static_cast<uint64_t>(it[3]) << 32) |
                 (static_cast<uint64_t>(it[4]) << 24) | (static_cast<uint64_t>(it[5]) << 16) |
                 (static_cast<uint64_t>(it[6]) << 8) | static_cast<uint64_t>(it[7]);
    result = (result * kMultiplier + b) % kMod;
  }
  if (it < end) {
    uint64_t b = 0;
    if (it + 4 <= end) {
      b = (static_cast<uint64_t>(it[0]) << 24) | (static_cast<uint64_t>(it[1]) << 16) |
          (static_cast<uint64_t>(it[2]) << 8) | static_cast<uint64_t>(it[3]);
      it += 4;
    }
    if (it + 2 <= end) {
      b = (b << 16) | (static_cast<uint64_t>(it[0]) << 8) | static_cast<uint64_t>(it[1]);
      it += 2;
    }
    if (it + 1 <= end) {
      b = (b << 8) | static_cast<uint64_t>(it[0]);
      it += 1;
    }
    result = (result * kMultiplier + b) % kMod;
  }
  return result;
}

TVM_FFI_INLINE void AnyView2Str(std::ostream &os, const TVMFFIAny *v) {
#if TVM_FFI_ALLOW_DYN_TYPE
  AnyView attr = TypeGetAttr(v->type_index, "__str__");
  if (details::IsTypeIndexNone(attr.type_index)) {
    os << TypeIndex2TypeKey(v->type_index) << '@' << v->v_ptr;
  } else {
    os << Ref<Str>(FuncInvoke(attr.v_obj, 1, v))->c_str();
  }
#else
  return StaticAnyView2Str(os, v);
#endif
}

TVM_FFI_INLINE TVMFFIObject *StrMoveFromStdString(std::string &&source) {
  return reinterpret_cast<TVMFFIObject *>(Str::Allocator::New(std::move(source)));
}

TVM_FFI_INLINE TVMFFIObject *StrCopyFromCharArray(const char *source, size_t length) {
  return reinterpret_cast<TVMFFIObject *>(Str::Allocator::New(source, length + 1));
}

} // namespace details

} // namespace ffi
} // namespace tvm

#endif // TVM_FFI_STR_H_
