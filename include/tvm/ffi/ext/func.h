#ifndef TVM_FFI_FUNC_H_
#define TVM_FFI_FUNC_H_

#include "./error.h"
#include "./str.h"
#include <functional>
#include <sstream>
#include <tvm/ffi/core/core.h>
#include <type_traits>

namespace tvm {
namespace ffi {
namespace details {
struct FuncInfo {
  using Call = void(const Func *, int32_t, const AnyView *, Any *);
  using SafeCall = int32_t(const Func *, int32_t, const AnyView *, Any *);
  Call *call;
  SafeCall *safe_call;
  static TVM_FFI_SYMBOL_HIDE int32_t SafeCallImpl(const Func *, int32_t, const AnyView *, Any *);
};
} // namespace details

struct Func : private AnyWithExtra<details::FuncInfo> {
  friend struct details::FuncInfo;
  using Header = AnyWithExtra<details::FuncInfo>;
  struct Allocator;
  TVM_FFI_DEF_STATIC_TYPE(Func, Object, TVMFFITypeIndex::kTVMFFIFunc);

  template <typename... Args>
  Any operator()(Args &&...args) const;

protected:
  TVM_FFI_INLINE Func(details::FuncInfo::Call *f) : AnyWithExtra<details::FuncInfo>() {
    this->_extra.call = f;
    this->_extra.safe_call = details::FuncInfo::SafeCallImpl;
  }
};
static_assert(sizeof(Func) == sizeof(TVMFFIFunc));
static_assert(offsetof(Func::Header, _extra.call) == offsetof(TVMFFIFunc, call));
static_assert(offsetof(Func::Header, _extra.safe_call) == offsetof(TVMFFIFunc, safe_call));

template <>
struct Ref<Func> : public RefBase<Func> {
  TVM_FFI_DEF_TYPE_FRIENDS();
  TVM_FFI_INLINE Ref() : RefBase<Func>() {}
  TVM_FFI_REF_DEF_DELEGATE_CONSTRUCTORS(Ref<Func>, RefBase<Func>)

  template <typename... Args>
  TVM_FFI_INLINE auto operator()(Args &&...args) const {
    return (*get())(std::forward<Args>(args)...);
  }
  template <typename... Args>
  TVM_FFI_INLINE auto operator()(Args &&...args) {
    return (*get())(std::forward<Args>(args)...);
  }
};

template <std::size_t N>
struct AnyViewArray {
  template <typename... Args>
  void Fill(Args &&...args);
  AnyView v[N];
};

template <>
struct AnyViewArray<0> {
  template <typename... Args>
  void Fill(Args &&...args);
  AnyView *v = nullptr;
};

template <typename... Args>
TVM_FFI_INLINE Any Func::operator()(Args &&...args) const {
  constexpr size_t N = sizeof...(Args);
  AnyViewArray<N> stack_args;
  stack_args.Fill(std::forward<Args>(args)...);
  return ::tvm::ffi::details::FuncInvoke(this, N, stack_args.v);
}

namespace details {

template <typename, typename>
struct UnpackCall;
template <typename T>
struct FuncFunctor;

TVM_FFI_INLINE int32_t FuncInfo::SafeCallImpl(const Func *self, int32_t num_args,
                                              const AnyView *args, Any *ret) {
  try {
    self->_extra.call(self, num_args, args, ret);
    return 0;
  } catch (TVMError &err) {
    err.MoveToAny(ret);
    return 1;
  } catch (std::runtime_error &err) {
    *ret = Ref<Str>::New(err.what());
    return 1;
  }
  TVM_FFI_UNREACHABLE();
}

TVM_FFI_INLINE Any FuncInvoke(const void *self, int32_t num_args, const TVMFFIAny *args) {
  const TVMFFIFunc *func = static_cast<const TVMFFIFunc *>(self);
  FuncInfo::SafeCall *local_safe_call = FuncInfo::SafeCallImpl;
  Any ret;
  if (reinterpret_cast<void *>(func->safe_call) == reinterpret_cast<void *>(local_safe_call)) {
    func->call(func, num_args, args, &ret);
    return ret;
  }
  int32_t err_code = func->safe_call(func, num_args, args, &ret);
  if (err_code == 0) {
    return ret;
  }
  if (ret.type_index == static_cast<int32_t>(TVMFFITypeIndex::kTVMFFIError)) {
    Ref<Error> err = Ref<Error>(std::move(ret));
    throw TVMError(err);
  }
  TVM_FFI_THROW(InternalError) << "Error code: " << err_code;
  TVM_FFI_UNREACHABLE();
}

template <typename FuncType>
struct FuncImpl : public Func {
  using TSelf = FuncImpl<FuncType>;
  using Allocator = DefaultObjectAllocator<TSelf>;

  TVM_FFI_INLINE static void CallPacked(const Func *obj, int32_t num_args, const AnyView *args,
                                        Any *ret) {
    static_cast<const TSelf *>(obj)->func_(num_args, args, ret);
  }

  TVM_FFI_INLINE static void CallUnpacked(const Func *obj, int32_t num_args, const AnyView *args,
                                          Any *ret) {
    using ArgType = typename FuncFunctor<FuncType>::ArgType;
    constexpr int32_t N = std::tuple_size_v<ArgType>;
    if (num_args != N) {
      TVM_FFI_THROW(TypeError) << "Mismatched number of arguments when calling: `"
                               << FuncFunctor<FuncType>::Sig() << "`. Expected " << N << " but got "
                               << num_args << " arguments";
    }
    using IdxSeq = std::make_index_sequence<N>;
    using RetType = typename FuncFunctor<FuncType>::RetType;
    UnpackCall<RetType, ArgType>::template Run<FuncType>(&static_cast<const TSelf *>(obj)->func_,
                                                         args, ret, IdxSeq{});
  }

  TVM_FFI_INLINE explicit FuncImpl(FuncType func, details::FuncInfo::Call *f)
      : Func(f), func_(std::forward<FuncType>(func)) {}

  TVM_FFI_INLINE static Func *FromPacked(FuncType func) {
    return Allocator::New(std::forward<FuncType>(func), TSelf::CallPacked);
  }

  TVM_FFI_INLINE static Func *FromUnpacked(FuncType func) {
    return Allocator::New(std::forward<FuncType>(func), TSelf::CallUnpacked);
  }

  mutable std::decay_t<FuncType> func_;
};

template <typename Type>
struct Type2Str {
  [[maybe_unused]] static inline std::string type_str = TypeTraitsNoCR<Type>::Type2Str();
};
template <>
struct Type2Str<Any> {
  static constexpr const char *type_str = "Any";
};
template <>
struct Type2Str<AnyView> {
  static constexpr const char *type_str = "AnyView";
};
template <>
struct Type2Str<void> {
  static constexpr const char *type_str = "void";
};

template <typename ArgType>
struct Arg2Str {
  template <size_t i>
  static TVM_FFI_INLINE void Apply(std::ostream &os) {
    using Arg = std::tuple_element_t<i, ArgType>;
    if constexpr (i != 0) {
      os << ", ";
    }
    os << i << ": " << Type2Str<Arg>::type_str;
  }
  template <size_t... I>
  static TVM_FFI_INLINE void Run(std::ostream &os, std::index_sequence<I...>) {
    using TExpander = int[];
    (void)TExpander{0, (Apply<I>(os), 0)...};
  }
};

template <typename T>
static constexpr bool ArgSupported = IsAnyOrView<RemoveCR<T>> || TypeTraitsNoCR<T>::enabled;
template <typename T>
static constexpr bool RetSupported =
    IsAnyOrView<RemoveCR<T>> || std::is_void_v<T> || TypeTraitsNoCR<T>::enabled;

template <typename R, typename... Args>
struct FuncFunctorImpl {
  using FType = R(Args...);
  using ArgType = std::tuple<Args...>;
  using RetType = R;
  static constexpr bool packed =
      std::is_convertible_v<FType, std::function<void(int32_t, const AnyView *, Any *)>>;
  static constexpr bool unpacked = (ArgSupported<Args> && ...) && (RetSupported<R>);
  static TVM_FFI_INLINE std::string Sig() {
    using IdxSeq = std::make_index_sequence<sizeof...(Args)>;
    std::ostringstream ss;
    ss << "(";
    Arg2Str<std::tuple<Args...>>::Run(ss, IdxSeq{});
    ss << ") -> " << Type2Str<R>::type_str;
    return ss.str();
  }
};

template <typename R, typename... Args>
struct FuncFunctor<R(Args...)> : FuncFunctorImpl<R, Args...> {};
template <typename R, typename... Args>
struct FuncFunctor<R (*)(Args...)> : FuncFunctorImpl<R, Args...> {};

template <int32_t i, typename>
struct PrependIntegerSeq;
template <int32_t i, int32_t... Is>
struct PrependIntegerSeq<i, std::integer_sequence<int32_t, Is...>> {
  using type = std::integer_sequence<int32_t, i, Is...>;
};
template <int32_t I, typename>
struct IndexIntSeq;
template <int32_t I, int32_t Head, int32_t... Tail>
struct IndexIntSeq<I, std::integer_sequence<int32_t, Head, Tail...>>
    : IndexIntSeq<I - 1, std::integer_sequence<int32_t, Tail...>> {};
template <int32_t Head, int32_t... Tail>
struct IndexIntSeq<0, std::integer_sequence<int32_t, Head, Tail...>> {
  static constexpr int32_t value = Head;
};

template <typename Function, typename StorageInfo>
struct UnpackCallArgConverter {
  template <typename _Type, size_t i>
  struct AsType {
    inline static auto Run(const AnyView &v, Any *storage) {
      using Type = std::decay_t<_Type>;
      constexpr int32_t storage_index = IndexIntSeq<i, typename StorageInfo::sum>::value;
      try {
        if constexpr (storage_index == -1) {
          return v.operator Type();
        } else {
          return v.CastWithStorage<Type>(storage + storage_index);
        }
      } catch (const TVMError &) {
        TVM_FFI_THROW(TypeError) << "Mismatched type on argument #" << i << " when calling: `"
                                 << FuncFunctor<Function>::Sig() << "`. Expected `"
                                 << Type2Str<Type>::type_str << "` but got `"
                                 << TypeIndex2TypeKey(v.type_index) << "`";
      }
      TVM_FFI_UNREACHABLE();
    }
  };

  template <size_t i>
  struct AsType<AnyView, i> {
    TVM_FFI_INLINE static AnyView Run(const AnyView &v, Any *) { return v; }
  };

  template <size_t i>
  struct AsType<Any, i> {
    TVM_FFI_INLINE static Any Run(const AnyView &v, Any *) { return v; }
  };
};

template <typename T, typename = void>
struct RequiresFFIStorage {
  static constexpr int32_t value = 0;
};
template <typename T>
struct RequiresFFIStorage<
    T, std::void_t<decltype(TypeTraitsNoCR<T>::CopyFromTVMFFIAnyToTypeWithStorage)>> {
  static constexpr int32_t value = 1;
};

template <typename... Args>
struct FFIStorageInfo;
template <>
struct FFIStorageInfo<> {
  constexpr static int32_t total = 0;
  using sum = std::integer_sequence<int32_t>;
};
template <typename I, typename... Is>
struct FFIStorageInfo<I, Is...> {
private:
  using Prev = FFIStorageInfo<Is...>;
  constexpr static int32_t need_ffi_storage = RequiresFFIStorage<I>::value;

public:
  constexpr static int32_t total = Prev::total + need_ffi_storage;
  using sum =
      typename PrependIntegerSeq<(need_ffi_storage ? total : 0) - 1, typename Prev::sum>::type;
};

template <typename RetType, typename... Args>
struct UnpackCall<RetType, std::tuple<Args...>> {
  template <typename Function, typename FuncType, size_t... I>
  TVM_FFI_INLINE static void Run(FuncType *func, const AnyView *args, Any *ret,
                                 std::index_sequence<I...>) {
    using Storage = FFIStorageInfo<Args...>;
    using CVT = UnpackCallArgConverter<Function, Storage>;
    if constexpr (Storage::total > 0 && std::is_void_v<RetType>) {
      Any storage[Storage::total];
      ret->Reset();
      (*func)(CVT::template AsType<Args, I>::Run(args[I], storage)...);
    } else if constexpr (Storage::total > 0 && !std::is_void_v<RetType>) {
      Any storage[Storage::total];
      *ret = (*func)(CVT::template AsType<Args, I>::Run(args[I], storage)...);
    } else if constexpr (Storage::total == 0 && std::is_void_v<RetType>) {
      ret->Reset();
      (*func)(CVT::template AsType<Args, I>::Run(args[I], nullptr)...);
    } else if constexpr (Storage::total == 0 && !std::is_void_v<RetType>) {
      *ret = (*func)(CVT::template AsType<Args, I>::Run(args[I], nullptr)...);
    }
  }
};

template <typename... Args>
struct AnyViewArrayFill {
  template <std::size_t i, typename Arg>
  TVM_FFI_INLINE static void Apply(AnyView *v, Arg &&arg) {
    v[i] = AnyView(std::forward<Arg>(arg));
  }
  template <std::size_t... I>
  TVM_FFI_INLINE static void Run(AnyView *v, Args &&...args, std::index_sequence<I...>) {
    using TExpander = int[];
    (void)TExpander{0, (Apply<I>(v, std::forward<Args>(args)), 0)...};
  }
};

template <typename... Args>
TVM_FFI_INLINE void FillAnyView(AnyView *v, Args &&...args) {
  constexpr std::size_t N = sizeof...(args);
  if constexpr (N > 0) {
    using IdxSeq = std::make_index_sequence<N>;
    ::tvm::ffi::details::AnyViewArrayFill<Args...>::Run(v, std::forward<Args>(args)..., IdxSeq{});
  }
}

} // namespace details

struct Func::Allocator {
public:
  template <typename FuncType, typename = void>
  struct Impl {};

  template <typename FuncType>
  struct Impl<FuncType, std::enable_if_t<details::FuncFunctor<FuncType>::packed>> {
    TVM_FFI_INLINE static Func *Run(FuncType func) {
      return details::FuncImpl<FuncType>::FromPacked(std::forward<FuncType>(func));
    }
  };

  template <typename FuncType>
  struct Impl<FuncType, std::enable_if_t<details::FuncFunctor<FuncType>::unpacked>> {
    TVM_FFI_INLINE static Func *Run(FuncType func) {
      return details::FuncImpl<FuncType>::FromUnpacked(std::forward<FuncType>(func));
    }
  };

  template <typename FuncType>
  TVM_FFI_INLINE static Func *New(FuncType func) {
    return Impl<FuncType>::Run(std::forward<FuncType>(func));
  }
};

template <std::size_t N>
template <typename... Args>
TVM_FFI_INLINE void AnyViewArray<N>::Fill(Args &&...args) {
  static_assert(sizeof...(args) == N, "Invalid number of arguments");
  ::tvm::ffi::details::FillAnyView(v, std::forward<Args>(args)...);
}

template <typename... Args>
TVM_FFI_INLINE void AnyViewArray<0>::Fill(Args &&...args) {
  static_assert(sizeof...(args) == 0, "Invalid number of arguments");
}

} // namespace ffi
} // namespace tvm

#endif // TVM_FFI_FUNC_H_
