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
#include <tvm/backend/cuda/attr.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/script/ir_builder/ir.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/tile/exec_scope.h>

#include "./utils.h"

namespace tvm {
namespace script {
using namespace tvm::prim;

namespace ir_builder {
namespace tirx {

using tvm::tirx::Layout;

namespace {

tvm::tirx::TensorType TensorTypeDecl(ffi::Array<PrimExpr> shape, PrimType dtype,
                                     runtime::MemoryScope storage_scope,
                                     ffi::Optional<ffi::Array<PrimExpr>> strides,
                                     ffi::Optional<PrimExpr> elem_offset, int align,
                                     int offset_factor, ffi::Optional<Layout> layout,
                                     ffi::Optional<Expr> data) {
  if (data.has_value()) {
    storage_scope = data.value()->ty.as_or_throw<PtrType>()->scope;
  }
  return tvm::tirx::TensorType(shape, dtype, storage_scope,
                               strides.value_or(ffi::Array<PrimExpr>()), elem_offset, align,
                               offset_factor, layout);
}

}  // namespace

FunctionFrame Function(bool is_private, bool persistent) {
  ffi::ObjectPtr<FunctionFrameObj> n = ffi::make_object<FunctionFrameObj>();
  n->name = std::nullopt;
  n->is_private = is_private;
  n->args.clear();
  n->ret_type = std::nullopt;
  n->attrs = {};
  n->persistent = persistent;
  return FunctionFrame(n);
}

FunctionFrame DeclFunction(bool is_private, bool persistent) {
  FunctionFrame frame = Function(is_private, persistent);
  frame->is_declaration = true;
  return frame;
}

Var Arg(ffi::String name, Var var) {
  FunctionFrame frame = FindFunctionFrame("T.Arg");
  details::Namer::Name(var, name);
  frame->args.push_back(var);
  return var;
}

TensorVar Arg(ffi::String name, TensorVar tensor) {
  Arg(std::move(name), tensor);
  return tensor;
}

void FuncName(ffi::String name) {
  FunctionFrame frame = FindFunctionFrame("T.func_name");
  if (frame->name.has_value()) {
    TVM_FFI_THROW(InternalError) << "ValueError: Duplicate function name, previous one is "
                                 << frame->name.value();
  }
  frame->name = name;
}

void FuncAttrs(ffi::Map<ffi::String, ffi::Any> new_attrs) {
  using namespace tvm::tirx;
  FunctionFrame frame = FindFunctionFrame("T.func_attr");
  for (const auto& [key, value] : new_attrs) {
    if (key == tvm::attr::kGlobalSymbol && frame->is_private) {
      TVM_FFI_THROW(InternalError)
          << "ValueError: "
          << "A private function may not have the kGlobalSymbol (\"" << tvm::attr::kGlobalSymbol
          << "\") attribute.  "
          << "However, a private function specified the global symbol as " << value;
    }

    if (auto prev = frame->attrs.Get(key)) {
      TVM_FFI_THROW(InternalError)
          << "ValueError: "
          << "Duplicate function annotation for key = \"" << key << "\".  "
          << "Previous value was " << prev.value() << ", with later definition as " << value;
    } else {
      frame->attrs.Set(key, value);
    }
  }
}

tvm::Type FuncRet(tvm::Type ret_type) {
  FunctionFrame frame = FindFunctionFrame("T.ret_type");
  if (frame->ret_type.has_value()) {
    TVM_FFI_THROW(InternalError) << "ValueError: Duplicate function return type, previous one is "
                                 << frame->ret_type.value();
  }
  frame->ret_type = ret_type;
  return ret_type;
}

tvm::Stmt TensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value) {
  auto tensor_type = dest->ty.as_or_throw<tvm::tirx::TensorType>();
  PrimType tensor_dtype = tensor_type->dtype;
  PrimType index_ty = indices.empty() ? PrimType::Int(32) : indices.back().ty();
  bool is_index_scalable = !indices.empty() && index_ty.IsScalableVector();
  bool is_tensor_dtype_scalable = tensor_dtype.IsScalableVector();

  TVM_FFI_ICHECK(!(is_index_scalable && is_tensor_dtype_scalable))
      << "Index dtype and tensor dtype can't both be scalable.";

  int index_lanes;
  if (indices.empty()) {
    index_lanes = 1;
  } else if (is_index_scalable) {
    index_lanes = index_ty.VScaleFactor();
  } else {
    index_lanes = index_ty.lanes();
  }

  int tensor_lanes = is_tensor_dtype_scalable ? tensor_dtype.VScaleFactor() : tensor_dtype.lanes();

  PrimType lhs_dtype = tensor_dtype;
  if (is_tensor_dtype_scalable || is_index_scalable) {
    lhs_dtype = PrimType::ScalableVector(tensor_dtype.code(), tensor_dtype.bits(),
                                         tensor_lanes * index_lanes);
  } else {
    lhs_dtype = tensor_dtype.WithLanes(tensor_dtype.lanes() * index_lanes);
  }

  PrimType rhs_dtype = value.ty();

  if (lhs_dtype != rhs_dtype) {
    TVM_FFI_ICHECK(lhs_dtype.IsScalableVector() == rhs_dtype.IsScalableVector())
        << "Can't mix scalable and fixed length vectors in a statement";

    bool lanes_match = false;
    if (lhs_dtype.IsScalableVector()) {
      lanes_match = lhs_dtype.VScaleFactor() == rhs_dtype.VScaleFactor();
    } else {
      lanes_match = lhs_dtype.lanes() == rhs_dtype.lanes();
    }

    if (!lanes_match) {
      TVM_FFI_THROW(InternalError) << "TypeError: Incompatible types in TensorStore"
                                   << ": LHS is `" << lhs_dtype << "`, RHS is `" << rhs_dtype
                                   << "`, indexing lanes: " << index_lanes;
    }
    value = tvm::prim::cast(lhs_dtype, value);
  }
  tvm::Stmt store = tvm::TensorStore(dest, indices, value);
  if (lhs_dtype != rhs_dtype) {
    if (lhs_dtype.code() != rhs_dtype.code()) {
      if ((lhs_dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) &&
          (rhs_dtype.code() == DLDataTypeCode::kDLFloat ||
           rhs_dtype.code() == DLDataTypeCode::kDLBfloat)) {
        ffi::String kernel_name = "<unknown>";
        if (ffi::Optional<FunctionFrame> frame = IRBuilder::Current()->FindFrame<FunctionFrame>()) {
          kernel_name = frame.value()->name.value_or("<anonymous>");
        }
        LOG(WARNING) << "Casting in TensorStore may lose precision"
                     << ": LHS is `" << lhs_dtype << "`, RHS is `" << rhs_dtype
                     << "`, indexing lanes: " << index_lanes << ", kernel: `" << kernel_name << "`"
                     << "\nTensorStore:\n"
                     << store;
      }
    }
  }
  AddToParent(store);
  return store;
}

Var Ptr(PrimType dtype, runtime::MemoryScope storage_scope = {}) {
  PtrType type_annotation(dtype, storage_scope);
  return tvm::Var("", type_annotation);
}

using tvm::script::ir_builder::details::Namer;

TVM_FFI_STATIC_INIT_BLOCK() {
  Namer::vtable().SetDispatch<TensorLoadNode>(
      [](const ffi::ObjectRef& node, ffi::String name) -> void {
        using namespace tvm::tirx;
        TensorLoadNode* tensor = const_cast<TensorLoadNode*>(node.as<TensorLoadNode>());
        Namer::Name(tensor->source.as_or_throw<tvm::tirx::TensorVar>(), name);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  Namer::vtable().SetDispatch<tvm::tirx::TileLayoutNode>(
      [](const ffi::ObjectRef& node, ffi::String name) -> void {

      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.CudaSharedMemoryRequirement",
           [](int64_t bytes) {
             TVM_FFI_CHECK_GE(bytes, 0, ValueError);
             const auto& frames = IRBuilder::Current()->frames;
             for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
               if (auto region = (*it).as<RegionFrame>();
                   region && region.value()->op->name == "tirx.device_entry") {
                 auto attrs = region.value()->attrs->dict;
                 int64_t previous = attrs.Get(tvm::backend::cuda::attr::kSmemRequired)
                                        .value_or(int64_t{0})
                                        .cast<int64_t>();
                 attrs.Set(tvm::backend::cuda::attr::kSmemRequired, std::max(previous, bytes));
                 region.value()->attrs = DictAttrs(attrs);
                 return;
               }
             }
             TVM_FFI_THROW(ValueError) << "SMEMPool.commit() requires an enclosing device_entry";
           })
      .def("script.ir_builder.tirx.TensorType", TensorTypeDecl)
      .def("script.ir_builder.tirx.Function", Function)
      .def("script.ir_builder.tirx.DeclFunction", DeclFunction)
      .def("script.ir_builder.tirx.Arg",
           [](ffi::String name, ffi::ObjectRef obj) -> ffi::ObjectRef {
             using namespace tvm::tirx;
             if (auto var = obj.as<Var>()) {
               return Arg(name, var.value());
             }
             TVM_FFI_THROW(InternalError)
                 << "ValueError: Unexpected type for TIR Arg: " << obj->GetTypeKey();
             throw;
           })
      .def("script.ir_builder.tirx.FuncName", FuncName)
      .def("script.ir_builder.tirx.FuncAttrs", FuncAttrs)
      .def("script.ir_builder.tirx.FuncRet", FuncRet)
      .def("script.ir_builder.tirx.TensorStore", TensorStore)
      .def("script.ir_builder.tirx.Ptr", Ptr);
}

#define TVM_TMP_STR(x) #x

#define TVM_FFI_REFL_DEF_GLOBAL_SIZE(Prefix, DType) \
  def(Prefix TVM_TMP_STR(8), DType##8)              \
      .def(Prefix TVM_TMP_STR(16), DType##16)       \
      .def(Prefix TVM_TMP_STR(32), DType##32)       \
      .def(Prefix TVM_TMP_STR(64), DType##64)

#define TVM_FFI_REFL_DEF_GLOBAL_LANES(Prefix, Func) \
  def(Prefix TVM_TMP_STR(x4), Func##x4)             \
      .def(Prefix TVM_TMP_STR(x8), Func##x8)        \
      .def(Prefix TVM_TMP_STR(x16), Func##x16)      \
      .def(Prefix TVM_TMP_STR(x32), Func##x32)      \
      .def(Prefix TVM_TMP_STR(x64), Func##x64)

#define TVM_FFI_REFL_DEF_GLOBAL_SIZES_LANES(Prefix, DType)              \
  TVM_FFI_REFL_DEF_GLOBAL_LANES(Prefix TVM_TMP_STR(8), DType##8)        \
      .TVM_FFI_REFL_DEF_GLOBAL_LANES(Prefix TVM_TMP_STR(16), DType##16) \
      .TVM_FFI_REFL_DEF_GLOBAL_LANES(Prefix TVM_TMP_STR(32), DType##32) \
      .TVM_FFI_REFL_DEF_GLOBAL_LANES(Prefix TVM_TMP_STR(64), DType##64)

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.BFloat16", BFloat16)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZE("script.ir_builder.tirx.Float", Float)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZE("script.ir_builder.tirx.UInt", UInt)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZE("script.ir_builder.tirx.Int", Int)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZES_LANES("script.ir_builder.tirx.Float", Float)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZES_LANES("script.ir_builder.tirx.UInt", UInt)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZES_LANES("script.ir_builder.tirx.Int", Int)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.BFloat16", BFloat16);
}

// Float8 variants
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float8E3M4", Float8E3M4)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float8E3M4", Float8E3M4);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float8E4M3", Float8E4M3)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float8E4M3", Float8E4M3);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float8E4M3B11FNUZ", Float8E4M3B11FNUZ)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float8E4M3B11FNUZ", Float8E4M3B11FNUZ);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float8E4M3FN", Float8E4M3FN)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float8E4M3FN", Float8E4M3FN);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float8E4M3FNUZ", Float8E4M3FNUZ)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float8E4M3FNUZ", Float8E4M3FNUZ);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float8E5M2", Float8E5M2)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float8E5M2", Float8E5M2);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float8E5M2FNUZ", Float8E5M2FNUZ)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float8E5M2FNUZ", Float8E5M2FNUZ);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float8E8M0FNU", Float8E8M0FNU)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float8E8M0FNU", Float8E8M0FNU);
}

// Float6 variants
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float6E2M3FN", Float6E2M3FN)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float6E2M3FN", Float6E2M3FN);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float6E3M2FN", Float6E3M2FN)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float6E3M2FN", Float6E3M2FN);
}

// Float4 variant
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Float4E2M1FN", Float4E2M1FN)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tirx.Float4E2M1FN", Float4E2M1FN);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Boolean", Boolean)
      .def("script.ir_builder.tirx.Handle",
           static_cast<Var (*)(ffi::Optional<PrimType>, runtime::MemoryScope)>(Handle))
      .def("script.ir_builder.tirx.TensorMap", TensorMap)
      .def("script.ir_builder.tirx.Void", Void)
      .def("script.ir_builder.tirx.min",
           [](PrimExpr a, PrimExpr b) -> PrimExpr { return tvm::min(a, b); })
      .def("script.ir_builder.tirx.max",
           [](PrimExpr a, PrimExpr b) -> PrimExpr { return tvm::max(a, b); });
}

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
