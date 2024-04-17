#ifndef TVM_FFI_ERROR_H_
#define TVM_FFI_ERROR_H_

#include <string>
#include <tvm/ffi/core/core.h>
#include <vector>

namespace tvm {
namespace ffi {
namespace details {
struct _ErrorHeader {
  const char *kind;
  int32_t num_frames;
  const char **linenos;
  const char *message;
};
using ErrorHeader = AnyWithExtra<_ErrorHeader>;
static_assert(sizeof(ErrorHeader) == sizeof(TVMFFIError));
static_assert(offsetof(ErrorHeader, _extra.kind) == offsetof(TVMFFIError, kind));
static_assert(offsetof(ErrorHeader, _extra.num_frames) == offsetof(TVMFFIError, num_frames));
static_assert(offsetof(ErrorHeader, _extra.linenos) == offsetof(TVMFFIError, linenos));
static_assert(offsetof(ErrorHeader, _extra.message) == offsetof(TVMFFIError, message));
} // namespace details

struct Error : private details::ErrorHeader {
  TVM_FFI_DEF_STATIC_TYPE(Error, Object, TVMFFITypeIndex::kTVMFFIError);

  std::string kind;
  std::vector<std::string> linenos;
  std::string message;
  std::vector<const char *> lineno_holder;

  Error(std::string kind, std::string lineno, std::string message)
      : kind(kind), linenos({lineno}), message(message), lineno_holder() {
    this->_extra.kind = this->kind.data();
    this->_extra.num_frames = static_cast<int32_t>(this->linenos.size());
    this->_extra.message = this->message.data();
    this->RefreshLinenoHolder();
  }

  void Append(std::string lineno) {
    linenos.push_back(lineno);
    this->RefreshLinenoHolder();
  }

protected:
  void RefreshLinenoHolder() {
    this->lineno_holder.resize(linenos.size());
    for (size_t i = 0; i < linenos.size(); ++i) {
      this->lineno_holder[i] = linenos[i].data();
    }
    this->_extra.linenos = this->lineno_holder.data();
  }
};

struct TVMError : public std::exception {
  Ref<Error> data_;

  TVMError(Ref<Error> data) : data_(data) {}
  TVMError(const TVMError &other) : data_(other.data_) {}
  TVMError(TVMError &&other) : data_(std::move(other.data_)) {}
  TVMError &operator=(const TVMError &) = delete;
  TVMError &operator=(TVMError &&) = delete;

  const char *what() const noexcept(true) override {
    if (data_.get() == nullptr) {
      return "TVMError: Unspecified";
    }
    return data_->message.c_str();
  }

  void MoveToAny(Any *v) {
    *v = data_;
    data_ = Ref<Error>(nullptr);
  }
};

namespace details {
[[noreturn]] TVM_FFI_INLINE void TVMErrorFromBuilder(std::string &&kind, std::string &&lineno,
                                                     std::string &&message) noexcept(false) {
  Ref<Error> ret = Ref<Error>::New(std::move(kind), std::move(lineno), std::move(message));
  throw TVMError(ret);
}
} // namespace details

} // namespace ffi
} // namespace tvm

#endif // TVM_FFI_ERROR_H_
