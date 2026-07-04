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
 * \file src/tirx/ir/function.cc
 * \brief The function data structure.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  PrimFuncNode::RegisterReflection();
  TensorIntrinNode::RegisterReflection();
}

namespace {
tvm::Type InferType(const PrimFunc& prim_func) {
  ffi::Array<tvm::Type> params;
  for (const auto& param : prim_func->params) {
    tvm::Type param_ty = [&]() -> tvm::Type {
      if (auto opt_buf = prim_func->buffer_map.Get(param)) {
        auto buf = opt_buf.value();
        relax::ShapeExpr shape(
            buf->shape.Map([](PrimExpr dim) { return cast(PrimType::Int(64), dim); }));
        return relax::TensorType(shape, buf->dtype);
      }

      if (auto prim_type = param->ty.as<PrimTypeNode>()) {
        const DLDataType& dtype = prim_type->dtype;
        if (dtype.code == kDLOpaqueHandle && (dtype.bits != 0 || dtype.lanes != 0)) {
          return relax::AnyType();
        }
      }

      return param->ty;
    }();
    params.push_back(param_ty);
  }

  tvm::Type ret = [&]() -> tvm::Type {
    if (const auto* prim = prim_func->ret_type.as<PrimTypeNode>()) {
      return tvm::PrimType(prim->dtype);
    } else if (IsVoidType(prim_func->ret_type)) {
      return relax::TupleType(ffi::Array<tvm::Type>{});
    } else {
      return relax::AnyType();
    }
  }();

  bool purity = prim_func->body.defined() ? s_tir::IsPureFunction(prim_func) : false;

  return relax::FuncType(params, ret, purity);
}
}  // namespace

// Get the function type of a PrimFunc
PrimFunc::PrimFunc(ffi::Array<tirx::Var> params, Stmt body, Type ret_type,
                   ffi::Map<tirx::Var, Buffer> buffer_map, DictAttrs attrs, Span span) {
  if (ret_type.IsMissing()) {
    ret_type = VoidType();
  }

  auto n = ffi::make_object<PrimFuncNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->buffer_map = std::move(buffer_map);
  n->attrs = std::move(attrs);
  n->ty = relax::FuncType::OpaqueFunc();
  n->span = std::move(span);
  data_ = std::move(n);

  (*this)->ty = InferType(*this);
}

FuncType PrimFuncNode::func_type_annotation() const {
  ffi::Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(param->ty);
  }
  return FuncType(param_types, ret_type);
}

class TensorIntrinManager {
 public:
  ffi::Map<ffi::String, tirx::TensorIntrin> reg;

  static TensorIntrinManager* Global() {
    static TensorIntrinManager* inst = new TensorIntrinManager();
    return inst;
  }
};

TensorIntrin::TensorIntrin(PrimFunc desc, PrimFunc impl) {
  // Check the number of func var is equal
  TVM_FFI_CHECK_EQ(desc->params.size(), impl->params.size(), ValueError)
      << "The number of parameters of the description and the implementation of the "
         "tensor intrinsic doesn't match.";
  auto is_handle = [](const Var& param) {
    auto prim_type = param->ty.as<PrimType>();
    return param->ty.as<PointerTypeNode>() || (prim_type && prim_type.value().IsHandle());
  };
  for (size_t i = 0; i < desc->params.size(); i++) {
    TVM_FFI_CHECK(is_handle(desc->params[i]), ValueError)
        << "Parameters of the description of the "
           "tensor intrinsic should be handle only.";
    TVM_FFI_CHECK(is_handle(impl->params[i]), ValueError)
        << "Parameters of the implementation of "
           "the tensor intrinsic should be handle only.";
  }
  TVM_FFI_ICHECK_EQ(desc->buffer_map.size(), impl->buffer_map.size());

  ffi::ObjectPtr<TensorIntrinNode> n = ffi::make_object<TensorIntrinNode>();
  n->desc = std::move(desc);
  n->impl = std::move(impl);
  data_ = std::move(n);
}

void TensorIntrin::Register(ffi::String name, TensorIntrin intrin, bool override) {
  TensorIntrinManager* manager = TensorIntrinManager::Global();
  if (!override) {
    TVM_FFI_CHECK_EQ(manager->reg.count(name), 0, ValueError)
        << "TensorIntrin '" << name << "' has already been registered";
  }
  manager->reg.Set(name, intrin);
}

ffi::Optional<TensorIntrin> TensorIntrin::Get(ffi::String name, bool allow_missing) {
  const TensorIntrinManager* manager = TensorIntrinManager::Global();
  auto it = manager->reg.find(name);
  if (it == manager->reg.end()) {
    if (allow_missing) {
      return std::nullopt;
    } else {
      TVM_FFI_THROW(ValueError) << "TensorIntrin '" << name << "' is not registered";
    }
  }
  return (*it).second;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.PrimFunc",
           [](ffi::Array<tirx::Var> params, Stmt body, Type ret_type,
              ffi::Map<tirx::Var, Buffer> buffer_map, DictAttrs attrs,
              Span span) { return PrimFunc(params, body, ret_type, buffer_map, attrs, span); })
      .def("tirx.TensorIntrin",
           [](PrimFunc desc_func, PrimFunc intrin_func) {
             return TensorIntrin(desc_func, intrin_func);
           })
      .def("tirx.TensorIntrinRegister", TensorIntrin::Register)
      .def("tirx.TensorIntrinGet", TensorIntrin::Get);
}

}  // namespace tirx
}  // namespace tvm
