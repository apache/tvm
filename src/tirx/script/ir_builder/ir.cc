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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/script/ir_builder/ir.h>
#include <tvm/tirx/stmt.h>

#include "./utils.h"

namespace tvm {
namespace script {
using namespace tvm::prim;

namespace ir_builder {
namespace tirx {

using tvm::tirx::Layout;

namespace {

tvm::tirx::TensorType TensorTypeDecl(ffi::Array<PrimExpr> shape, PrimType dtype,
                                     ffi::Optional<Expr> data,
                                     ffi::Optional<ffi::Array<PrimExpr>> strides,
                                     ffi::Optional<PrimExpr> elem_offset, ffi::String storage_scope,
                                     int align, int offset_factor, ffi::Optional<Layout> layout) {
  if (data.has_value()) {
    storage_scope = data.value()->ty.as_or_throw<PointerType>()->storage_scope;
  }
  if (!elem_offset.has_value() && offset_factor) {
    PrimType shape_dtype = shape.empty() ? PrimType::Int(32) : shape[0].ty();
    elem_offset = tvm::PrimVar("elem_offset", shape_dtype);
  }
  return tvm::tirx::TensorType(storage_scope, dtype, shape,
                               strides.value_or(ffi::Array<PrimExpr>()), elem_offset, align,
                               offset_factor, layout);
}

}  // namespace

TensorVar TensorDecl(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::String buffer_name,
                     ffi::Optional<Expr> data, ffi::Optional<ffi::Array<PrimExpr>> strides,
                     ffi::Optional<PrimExpr> elem_offset, ffi::String storage_scope, int align,
                     int offset_factor, ffi::Optional<Layout> layout) {
  return TensorVar(buffer_name, TensorTypeDecl(shape, dtype, data, strides, elem_offset,
                                               storage_scope, align, offset_factor, layout));
}

FunctionFrame Function(bool is_private, bool persistent) {
  ffi::ObjectPtr<FunctionFrameNode> n = ffi::make_object<FunctionFrameNode>();
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

TensorVar Arg(ffi::String name, TensorVar buffer) {
  Arg(std::move(name), buffer.var());
  return buffer;
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

namespace {
ForFrame WithThreadBindingValidation(ForFrame frame) {
  auto make_loop = frame->f_make_for_loop;
  frame->f_make_for_loop = [make_loop](ffi::Array<Var> vars, ffi::Array<Range> doms,
                                       ffi::Array<ffi::Optional<PrimExpr>> steps, tvm::SeqStmt body,
                                       Span span) {
    auto loop = make_loop(vars, doms, steps, body, span).as_or_throw<tvm::For>();
    tvm::tirx::GetThreadBinding(loop);
    return loop;
  };
  return frame;
}
}  // namespace

ForFrame Serial(PrimExpr start, PrimExpr stop,
                ffi::Optional<ffi::Map<ffi::String, Any>> annotations, ffi::Optional<PrimExpr> step,
                ffi::Optional<PrimType> dtype) {
  return WithThreadBindingValidation(ir::Serial(start, stop, annotations, step, dtype));
}

ForFrame Parallel(PrimExpr start, PrimExpr stop,
                  ffi::Optional<ffi::Map<ffi::String, Any>> annotations,
                  ffi::Optional<PrimExpr> step, ffi::Optional<PrimType> dtype) {
  return WithThreadBindingValidation(ir::Parallel(start, stop, annotations, step, dtype));
}

ForFrame Vectorized(PrimExpr start, PrimExpr stop,
                    ffi::Optional<ffi::Map<ffi::String, Any>> annotations,
                    ffi::Optional<PrimExpr> step, ffi::Optional<PrimType> dtype) {
  return WithThreadBindingValidation(ir::Vectorized(start, stop, annotations, step, dtype));
}

ForFrame Unroll(PrimExpr start, PrimExpr stop,
                ffi::Optional<ffi::Map<ffi::String, Any>> annotations, ffi::Optional<PrimExpr> step,
                ffi::Optional<PrimType> dtype) {
  return WithThreadBindingValidation(ir::Unroll(start, stop, annotations, step, dtype));
}

ForFrame ThreadBinding(PrimExpr start, PrimExpr stop, ffi::String thread,
                       ffi::Optional<ffi::Map<ffi::String, Any>> annotations) {
  using namespace tvm::tirx;
  PrimExpr min = start;
  PrimExpr extent = sym::Analyzer()->Simplify(stop - start);
  ffi::ObjectPtr<ForFrameNode> n = ffi::make_object<ForFrameNode>();
  PrimType min_ty = min.ty();
  PrimType extent_ty = extent.ty();
  int bits = std::max(min_ty.bits(), extent_ty.bits());
  PrimType dtype = min_ty.WithBits(bits).WithLanes(1);
  n->vars = {Var("v", dtype)};
  n->doms = {Range::FromMinExtent(min, extent)};
  n->steps = {std::nullopt};
  n->f_make_for_loop = [annotations, thread, dtype](ffi::Array<Var> vars, ffi::Array<Range> doms,
                                                    ffi::Array<ffi::Optional<PrimExpr>> steps,
                                                    SeqStmt body, Span span) -> For {
    TVM_FFI_ICHECK_EQ(vars.size(), 1);
    TVM_FFI_ICHECK_EQ(doms.size(), 1);
    TVM_FFI_ICHECK(steps.size() == 1 && (!steps[0].has_value() || IsOne(*steps[0])));
    auto loop_annotations = annotations.value_or(ffi::Map<ffi::String, ffi::Any>());
    if (auto existing = loop_annotations.Get(tvm::tirx::attr::kThreadBinding)) {
      TVM_FFI_CHECK(existing->cast<ffi::String>() == thread, ValueError)
          << "Conflicting thread_binding annotation and thread argument";
    }
    loop_annotations.Set(tvm::tirx::attr::kThreadBinding, thread);
    return For(vars[0].as_or_throw<tvm::PrimVar>(), doms[0]->min, doms[0]->extent,
               ForKind::kParallel, body, std::move(loop_annotations), std::nullopt, span);
  };
  return ForFrame(n);
}

tvm::Stmt TensorStore(Expr dest, ffi::Array<PrimExpr> indices, PrimExpr value) {
  auto tensor_type = dest->ty.as_or_throw<tvm::tirx::TensorType>();
  PrimType buffer_dtype = tensor_type->dtype;
  PrimType index_ty = indices.empty() ? PrimType::Int(32) : indices.back().ty();
  bool is_index_scalable = !indices.empty() && index_ty.IsScalableVector();
  bool is_buffer_dtype_scalable = buffer_dtype.IsScalableVector();

  TVM_FFI_ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
      << "Index dtype and buffer dtype can't both be scalable.";

  int index_lanes;
  if (indices.empty()) {
    index_lanes = 1;
  } else if (is_index_scalable) {
    index_lanes = index_ty.VScaleFactor();
  } else {
    index_lanes = index_ty.lanes();
  }

  int buffer_lanes = is_buffer_dtype_scalable ? buffer_dtype.VScaleFactor() : buffer_dtype.lanes();

  PrimType lhs_dtype = buffer_dtype;
  if (is_buffer_dtype_scalable || is_index_scalable) {
    lhs_dtype = PrimType::ScalableVector(buffer_dtype.code(), buffer_dtype.bits(),
                                         buffer_lanes * index_lanes);
  } else {
    lhs_dtype = buffer_dtype.WithLanes(buffer_dtype.lanes() * index_lanes);
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

TensorVar DeclTensor(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::String buffer_name,
                     ffi::Optional<Expr> data, ffi::Optional<ffi::Array<PrimExpr>> strides,
                     ffi::Optional<PrimExpr> elem_offset, ffi::String storage_scope, int align,
                     int offset_factor, ffi::Optional<Layout> layout,
                     ffi::Optional<PrimExpr> allocated_addr) {
  std::string scope = static_cast<std::string>(storage_scope);
  if (scope.empty()) {
    scope = "global";
  }

  TensorVar buffer = TensorDecl(shape, dtype, buffer_name, data, strides, elem_offset,
                                storage_scope, align, offset_factor, layout);
  if (scope == "tmem" && allocated_addr.has_value()) {
    TVM_FFI_CHECK(!data.has_value(), ValueError)
        << "A TMEM declaration cannot have both data and an address";
    Span span = IRBuilder::Current()->GetCurrentSourceSpan();
    AddToParent(tvm::Bind(buffer.var(),
                          Call(std::nullopt, Op::Get("tirx.cuda.decl_tmem"),
                               {allocated_addr.value()}, {}, {buffer.type()}, span),
                          span));
    return buffer;
  }
  TVM_FFI_CHECK(!allocated_addr.has_value() || !data.has_value(), ValueError)
      << "Placement addresses apply to allocations, not pointer-backed declarations";
  TVM_FFI_CHECK(!allocated_addr.has_value() || (scope != "global" && scope != "shared" &&
                                                scope != "shared.dyn" && scope != "local"),
                ValueError)
      << "This storage scope does not support allocation placement";
  Span span = IRBuilder::Current()->GetCurrentSourceSpan();
  if (data.has_value()) {
    AddToParent(tvm::Bind(buffer.var(),
                          Call(buffer.type(), tvm::tirx::decl_tensor_op(),
                               {data.value(), tvm::Tuple(buffer->shape),
                                DataTypeImm(buffer->dtype->dtype), StringImm(buffer.scope())},
                               {}, {}, span),
                          span));
  } else {
    // Without a backing pointer, declare and allocate the tensor together.
    ffi::Array<Expr> args{tvm::Tuple(buffer->shape), DataTypeImm(buffer->dtype->dtype),
                          StringImm(buffer.scope())};
    if (allocated_addr.has_value()) args.push_back(tvm::Tuple({allocated_addr.value()}));
    AddToParent(tvm::Bind(
        buffer.var(),
        Call(buffer.type(), tvm::tirx::alloc_tensor_op(), args, DictAttrs(), {}, span), span));
  }
  return buffer;
}

TensorVar AllocTensor(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::String storage_scope,
                      ffi::Optional<ffi::Map<ffi::String, ffi::Any>> annotations) {
  TensorVar buffer = TensorDecl(shape, dtype, "", std::nullopt, std::nullopt, std::nullopt,
                                storage_scope, 0, 0, std::nullopt);
  ffi::Array<Expr> args{tvm::Tuple(buffer->shape), DataTypeImm(buffer->dtype->dtype),
                        StringImm(buffer.scope())};
  auto attrs = annotations.value_or(ffi::Map<ffi::String, ffi::Any>());
  if (auto placement = attrs.Get("buffer_allocated_addr")) {
    args.push_back(tvm::Tuple(placement.value().as_or_throw<ffi::Array<PrimExpr>>()));
    attrs.erase("buffer_allocated_addr");
  }
  AddToParent(tvm::Bind(buffer.var(), Call(buffer.type(), tvm::tirx::alloc_tensor_op(), args,
                                           DictAttrs(std::move(attrs)))));
  return buffer;
}

Var Ptr(PrimType dtype, ffi::String storage_scope = "global") {
  PointerType type_annotation(dtype, storage_scope);
  return tvm::Var("", type_annotation);
}

using tvm::script::ir_builder::details::Namer;

TVM_FFI_STATIC_INIT_BLOCK() {
  Namer::vtable().SetDispatch<TensorLoadNode>(
      [](const ffi::ObjectRef& node, ffi::String name) -> void {
        using namespace tvm::tirx;
        TensorLoadNode* buffer = const_cast<TensorLoadNode*>(node.as<TensorLoadNode>());
        Namer::Name(buffer->source.as_or_throw<tvm::tirx::TensorVar>(), name);
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
                 int64_t previous =
                     attrs.Get("cuda.smem_required").value_or(int64_t{0}).cast<int64_t>();
                 attrs.Set("cuda.smem_required", std::max(previous, bytes));
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
      .def("script.ir_builder.tirx.AllocTensor", AllocTensor)
      .def("script.ir_builder.tirx.Serial", Serial)
      .def("script.ir_builder.tirx.Parallel", Parallel)
      .def("script.ir_builder.tirx.Vectorized", Vectorized)
      .def("script.ir_builder.tirx.Unroll", Unroll)
      .def("script.ir_builder.tirx.ThreadBinding", ThreadBinding)
      .def("script.ir_builder.tirx.DeclTensor", DeclTensor)
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
      .def("script.ir_builder.tirx.Handle", Handle)
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
