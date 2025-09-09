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

#ifndef TVM_FFI_TESTING_OBJECT_H_
#define TVM_FFI_TESTING_OBJECT_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace ffi {
namespace testing {

// We deliberately pad extra
// in the header to test cases
// where the object subclass address
// do not align with the base object address
// not handling properly will cause buffer overflow
class BasePad {
 public:
  int64_t extra[4];
};

class TNumberObj : public BasePad, public Object {
 public:
  // declare as one slot, with float as overflow
  static constexpr uint32_t _type_child_slots = 1;
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("test.Number", TNumberObj, Object);
};

class TNumber : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TNumber, ObjectRef, TNumberObj);
};

class TIntObj : public TNumberObj {
 public:
  int64_t value;

  TIntObj(int64_t value) : value(value) {}
  explicit TIntObj(UnsafeInit) {}

  int64_t GetValue() const { return value; }

  inline static void RegisterReflection();

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.Int", TIntObj, TNumberObj);
};

class TInt : public TNumber {
 public:
  explicit TInt(int64_t value) { data_ = make_object<TIntObj>(value); }

  static TInt StaticAdd(TInt lhs, TInt rhs) { return TInt(lhs->value + rhs->value); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TInt, TNumber, TIntObj);
};

inline void TIntObj::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TIntObj>()
      .def_ro("value", &TIntObj::value)
      .def_static("static_add", &TInt::StaticAdd, "static add method");
  // define extra type attributes
  refl::TypeAttrDef<TIntObj>()
      .def("test.GetValue", &TIntObj::GetValue)
      .attr("test.size", sizeof(TIntObj));
  // custom json serialization
  refl::TypeAttrDef<TIntObj>()
      .def("__data_to_json__",
           [](const TIntObj* self) -> Map<String, Any> {
             return Map<String, Any>{{"value", self->value}};
           })
      .def("__data_from_json__", [](Map<String, Any> json_obj) -> TInt {
        return TInt(json_obj["value"].cast<int64_t>());
      });
}

class TFloatObj : public TNumberObj {
 public:
  double value;

  TFloatObj(double value) : value(value) {}

  double Add(double other) const { return value + other; }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TFloatObj>()
        .def_ro("value", &TFloatObj::value, "float value field", refl::DefaultValue(10.0))
        .def("sub",
             [](const TFloatObj* self, double other) -> double { return self->value - other; })
        .def("add", &TFloatObj::Add, "add method");
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.Float", TFloatObj, TNumberObj);
};

class TFloat : public TNumber {
 public:
  explicit TFloat(double value) { data_ = make_object<TFloatObj>(value); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TFloat, TNumber, TFloatObj);
};

class TPrimExprObj : public Object {
 public:
  std::string dtype;
  double value;

  TPrimExprObj(std::string dtype, double value) : dtype(dtype), value(value) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TPrimExprObj>()
        .def_rw("dtype", &TPrimExprObj::dtype, "dtype field", refl::DefaultValue("float"))
        .def_ro("value", &TPrimExprObj::value, "value field", refl::DefaultValue(0))
        .def("sub", [](TPrimExprObj* self, double other) -> double {
          // this is ok because TPrimExprObj is declared asmutable
          return self->value - other;
        });
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.PrimExpr", TPrimExprObj, Object);
};

class TPrimExpr : public ObjectRef {
 public:
  explicit TPrimExpr(std::string dtype, double value) {
    data_ = make_object<TPrimExprObj>(dtype, value);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TPrimExpr, ObjectRef, TPrimExprObj);
};

class TVarObj : public Object {
 public:
  std::string name;

  TVarObj(std::string name) : name(name) {}
  // need unsafe init constructor for json serialization
  explicit TVarObj(UnsafeInit) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TVarObj>().def_ro("name", &TVarObj::name,
                                      refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.Var", TVarObj, Object);
};

class TVar : public ObjectRef {
 public:
  explicit TVar(std::string name) { data_ = make_object<TVarObj>(name); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TVar, ObjectRef, TVarObj);
};

class TFuncObj : public Object {
 public:
  Array<TVar> params;
  Array<ObjectRef> body;
  Optional<String> comment;

  // need unsafe init constructor or default constructor for json serialization
  explicit TFuncObj(UnsafeInit) {}
  TFuncObj(Array<TVar> params, Array<ObjectRef> body, Optional<String> comment)
      : params(params), body(body), comment(comment) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TFuncObj>()
        .def_ro("params", &TFuncObj::params, refl::AttachFieldFlag::SEqHashDef())
        .def_ro("body", &TFuncObj::body)
        .def_ro("comment", &TFuncObj::comment, refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.Func", TFuncObj, Object);
};

class TFunc : public ObjectRef {
 public:
  explicit TFunc(Array<TVar> params, Array<ObjectRef> body, Optional<String> comment) {
    data_ = make_object<TFuncObj>(params, body, comment);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TFunc, ObjectRef, TFuncObj);
};

class TCustomFuncObj : public Object {
 public:
  Array<TVar> params;
  Array<ObjectRef> body;
  String comment;

  TCustomFuncObj(Array<TVar> params, Array<ObjectRef> body, String comment)
      : params(params), body(body), comment(comment) {}

  bool SEqual(const TCustomFuncObj* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> cmp) const {
    if (!cmp(params, other->params, true, "params")) {
      return false;
    }
    if (!cmp(body, other->body, false, "body")) {
      return false;
    }
    return true;
  }

  uint64_t SHash(uint64_t init_hash,
                 ffi::TypedFunction<uint64_t(AnyView, uint64_t, bool)> hash) const {
    uint64_t hash_value = init_hash;
    hash_value = hash(params, hash_value, true);
    hash_value = hash(body, hash_value, false);
    return hash_value;
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TCustomFuncObj>()
        .def_ro("params", &TCustomFuncObj::params)
        .def_ro("body", &TCustomFuncObj::body)
        .def_ro("comment", &TCustomFuncObj::comment);
    refl::TypeAttrDef<TCustomFuncObj>()
        .def("__s_equal__", &TCustomFuncObj::SEqual)
        .def("__s_hash__", &TCustomFuncObj::SHash);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.CustomFunc", TCustomFuncObj, Object);
};

class TCustomFunc : public ObjectRef {
 public:
  explicit TCustomFunc(Array<TVar> params, Array<ObjectRef> body, String comment) {
    data_ = make_object<TCustomFuncObj>(params, body, comment);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TCustomFunc, ObjectRef, TCustomFuncObj);
};

}  // namespace testing

template <>
inline constexpr bool use_default_type_traits_v<testing::TPrimExpr> = true;

template <>
struct TypeTraits<testing::TPrimExpr>
    : public ObjectRefWithFallbackTraitsBase<testing::TPrimExpr, StrictBool, int64_t, double,
                                             String> {
  TVM_FFI_INLINE static testing::TPrimExpr ConvertFallbackValue(StrictBool value) {
    return testing::TPrimExpr("bool", static_cast<double>(value));
  }

  TVM_FFI_INLINE static testing::TPrimExpr ConvertFallbackValue(int64_t value) {
    return testing::TPrimExpr("int64", static_cast<double>(value));
  }

  TVM_FFI_INLINE static testing::TPrimExpr ConvertFallbackValue(double value) {
    return testing::TPrimExpr("float32", static_cast<double>(value));
  }
  // hack into the dtype to store string
  TVM_FFI_INLINE static testing::TPrimExpr ConvertFallbackValue(String value) {
    return testing::TPrimExpr(value, 0);
  }
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_TESTING_OBJECT_H_
