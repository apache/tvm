#ifndef TVM_FFI_C_FFI_ABI_H_
#define TVM_FFI_C_FFI_ABI_H_

#include <dlpack/dlpack.h>
#include <stdint.h>

#if !defined(TVM_FFI_API) && defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define TVM_FFI_API EMSCRIPTEN_KEEPALIVE
#endif
#if !defined(TVM_FFI_API) && defined(_MSC_VER)
#ifdef TVM_FFI_EXPORTS
#define TVM_FFI_API __declspec(dllexport)
#else
#define TVM_FFI_API __declspec(dllimport)
#endif
#endif
#ifndef TVM_FFI_API
#define TVM_FFI_API __attribute__((visibility("default")))
#endif

#ifndef TVM_FFI_ALLOW_DYN_TYPE
#define TVM_FFI_ALLOW_DYN_TYPE 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
enum class TVMFFITypeIndex : int32_t {
#else
typedef enum {
#endif
  // [Section] On-stack POD Types: [0, kTVMFFIStaticObjectBegin)
  // N.B. `kTVMFFIRawStr` is a string backed by a `\0`-terminated char array,
  // which is not owned by TVMFFIAny. It is required that the following
  // invariant holds:
  // - `Any::type_index` is never `kTVMFFIRawStr`
  // - `AnyView::type_index` can be `kTVMFFIRawStr`
  kTVMFFINone = 0,
  kTVMFFIInt = 1,
  kTVMFFIFloat = 2,
  kTVMFFIPtr = 3,
  kTVMFFIDataType = 4,
  kTVMFFIDevice = 5,
  kTVMFFIRawStr = 6,
  // [Section] Static Boxed: [kTVMFFIStaticObjectBegin, kTVMFFIDynObjectBegin)
  kTVMFFIStaticObjectBegin = 64,
  kTVMFFIObject = 64,
  kTVMFFIList = 65,
  kTVMFFIDict = 66,
  kTVMFFIError = 67,
  kTVMFFIFunc = 68,
  kTVMFFIStr = 69,
  // [Section] Dynamic Boxed: [kTVMFFIDynObjectBegin, +oo)
  kTVMFFIDynObjectBegin = 128,
#ifdef __cplusplus
};
#else
} TypeIndex;
#endif

struct TVMFFIAny;
typedef TVMFFIAny TVMFFIObject;
typedef TVMFFIObject *TVMFFIObjectHandle;
typedef TVMFFIAny *TVMFFIAnyHandle;

typedef struct TVMFFIAny {
  int32_t type_index;
  union {              // 4 bytes
    int32_t ref_cnt;   // reference counter for heap object
    int32_t small_len; // length for on-stack object
  };
  union {                     // 8 bytes
    int64_t v_int64;          // integers
    double v_float64;         // floating-point numbers
    DLDataType v_dtype;       // data type
    DLDevice v_device;        // device
    void *v_ptr;              // typeless pointers
    const char *v_str;        // raw string
    TVMFFIObjectHandle v_obj; // ref counted objects
    void (*deleter)(void *);  // Deleter of the object
    char v_bytes[8];          // small string
    char32_t v_char32[2];     // UCS4 string and Unicode
  };
} TVMFFIAny;

typedef struct {
  int32_t type_index;
  const char *type_key;
  int32_t type_depth;
  int32_t *type_ancestors;
} TVMFFITypeInfo;

typedef TVMFFITypeInfo *TVMFFITypeInfoHandle;

typedef struct {
  const char *filename;
  const char *func;
  int32_t lineno;
  void (*deleter)(void *);
} TVMFFIStackFrame;

typedef struct {
  int32_t type_index;
  int32_t ref_cnt;
  void (*deleter)(void *);
  const char *kind;
  int32_t num_frames;
  const char **linenos;
  const char *message;
} TVMFFIError;

typedef struct {
  int32_t type_index;
  int32_t ref_cnt;
  void (*deleter)(void *);
  int64_t length;
  char *data;
} TVMFFIStr;

typedef struct {
  int32_t type_index;
  int32_t ref_cnt;
  void (*deleter)(void *);
  void (*call)(const void *self, int32_t num_args, const TVMFFIAny *args, TVMFFIAny *ret);
  int32_t (*safe_call)(const void *self, int32_t num_args, const TVMFFIAny *args, TVMFFIAny *ret);
} TVMFFIFunc;

typedef struct {
  int32_t type_index;
  int32_t ref_cnt;
  void (*deleter)(void *);
  int64_t list_capacity;
  int64_t list_length;
  int64_t pool_capacity;
  int64_t pool_length;
} TVMFFIList;

typedef struct {
  int32_t type_index;
  int32_t ref_cnt;
  void (*deleter)(void *);
  int64_t capacity;
  int64_t size;
} TVMFFIDict;

#if TVM_FFI_ALLOW_DYN_TYPE
typedef void *TVMFFITypeTableHandle;
TVM_FFI_API void TVMFFIDynTypeIndex2Info(TVMFFITypeTableHandle self, int32_t type_index,
                                         TVMFFITypeInfoHandle *out_type_info);
TVM_FFI_API void TVMFFIDynTypeDef(TVMFFITypeTableHandle self, const char *type_key,
                                  int32_t type_depth, const int32_t *type_ancestors,
                                  int32_t *out_type_index);
TVM_FFI_API void TVMFFIDynTypeSetAttr(TVMFFITypeTableHandle self, int32_t type_index,
                                      const char *key, TVMFFIAnyHandle value);
TVM_FFI_API void TVMFFIDynTypeGetAttr(TVMFFITypeTableHandle self, int32_t type_index,
                                      const char *key, TVMFFIAnyHandle *out_type_attr);
#endif // TVM_FFI_ALLOW_DYN_TYPE

#ifdef __cplusplus
} // TVM_FFI_EXTERN_C
#endif

#endif // TVM_FFI_C_FFI_ABI_H_
