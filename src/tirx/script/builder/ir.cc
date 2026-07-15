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
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/script/builder/ir.h>
#include <tvm/tirx/tirx_op.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace tirx {

using tvm::tirx::IterVar;
using tvm::tirx::Layout;

Buffer BufferDecl(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::String buffer_name,
                  ffi::Optional<Var> data, ffi::Optional<ffi::Array<PrimExpr>> strides,
                  ffi::Optional<PrimExpr> elem_offset, ffi::String storage_scope, int align,
                  int offset_factor, ffi::String buffer_type,
                  ffi::Optional<ffi::Array<IntImm>> axis_separators, ffi::Optional<Layout> layout,
                  ffi::Array<PrimExpr> allocated_addr) {
  TVM_FFI_CHECK(buffer_type == "auto" || buffer_type == "default" || buffer_type.empty(),
                ValueError)
      << "ValueError: `buffer_type` must be `auto` or `default` or empty";
  if (!allocated_addr.empty()) {
    TVM_FFI_ICHECK(!data.has_value() && !elem_offset.has_value() && !offset_factor)
        << "ValueError: `allocated_addr` can only be used with `data`, `elem_offset`, and "
           "`offset_factor` undefined";
  }
  Var buffer_data;
  if (!data.has_value()) {
    DLDataType storage_dtype = dtype->dtype;
    if (storage_dtype == DLDataType{kDLBool, 8, 1}) {
      storage_dtype = DLDataType{kDLInt, 8, 1};
    }
    buffer_data = tvm::tirx::Var(buffer_name, PointerType(PrimType(storage_dtype), storage_scope));
  } else {
    buffer_data = data.value();
  }
  if (!elem_offset.has_value() && offset_factor) {
    PrimType shape_dtype = shape.empty() ? PrimType::Int(32) : shape[0].ty();
    elem_offset = tvm::tirx::PrimVar("elem_offset", shape_dtype);
  }
  return Buffer(buffer_data, dtype, shape, strides.value_or(ffi::Array<PrimExpr>()),
                elem_offset.value_or(PrimExpr()), buffer_name, align, offset_factor,
                (buffer_type == "auto" ? tvm::tirx::kAutoBroadcast : tvm::tirx::kDefault),
                axis_separators.value_or(ffi::Array<IntImm>()), Span(), layout, allocated_addr);
}

PrimFuncFrame PrimFunc(bool is_private, bool s_tir, bool persistent) {
  ffi::ObjectPtr<PrimFuncFrameNode> n = ffi::make_object<PrimFuncFrameNode>();
  n->name = std::nullopt;
  n->is_private = is_private;
  n->args.clear();
  n->ret_type = std::nullopt;
  n->buffer_map.clear();
  n->attrs = {};
  n->env_threads.clear();
  n->root_alloc_buffers.clear();
  n->s_tir = s_tir;
  n->persistent = persistent;
  return PrimFuncFrame(n);
}

Var Arg(ffi::String name, Var var) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.Arg");
  details::Namer::Name(var, name);
  frame->args.push_back(var);
  return var;
}

Buffer Arg(ffi::String name, Buffer buffer) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.Arg");
  details::Namer::Name(buffer, name);
  // A Buffer parameter is an opaque ABI handle.  The Buffer's data Var
  // carries the exact pointee type used within the function body.
  Var handle(buffer->name + "_handle", PointerType::VoidPointerTy());
  frame->args.push_back(handle);
  frame->buffer_map.Set(handle, buffer);
  return buffer;
}

void FuncName(ffi::String name) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_name");
  if (frame->name.has_value()) {
    TVM_FFI_THROW(InternalError) << "ValueError: Duplicate prim func name, previous one is "
                                 << frame->name.value();
  }
  frame->name = name;
}

void FuncAttrs(ffi::Map<ffi::String, ffi::Any> new_attrs) {
  using namespace tvm::tirx;
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_attr");
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
          << "Duplicate prim func annotation for key = \"" << key << "\".  "
          << "Previous value was " << prev.value() << ", with later definition as " << value;
    } else {
      frame->attrs.Set(key, value);
    }
  }
}

tvm::Type FuncRet(tvm::Type ret_type) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.ret_type");
  if (frame->ret_type.has_value()) {
    TVM_FFI_THROW(InternalError) << "ValueError: Duplicate prim func return type, previous one is "
                                 << frame->ret_type.value();
  }
  frame->ret_type = ret_type;
  return ret_type;
}

Buffer MatchBuffer(ffi::ObjectRef param, ffi::Array<PrimExpr> shape, PrimType dtype,
                   ffi::Optional<Var> data, ffi::Array<PrimExpr> strides, PrimExpr elem_offset,
                   ffi::String storage_scope, int align, int offset_factor,
                   ffi::String buffer_type_str, ffi::Optional<ffi::Array<IntImm>> axis_separators,
                   ffi::Optional<Layout> layout) {
  Buffer buffer = BufferDecl(shape, dtype, "", data, strides, elem_offset, storage_scope, align,
                             offset_factor, buffer_type_str, axis_separators, layout, {});
  if (auto var = param.as<tvm::tirx::Var>()) {
    PrimFuncFrame frame = FindPrimFuncFrame("T.match_buffer");
    Var v = var.value();
    for (auto const& arg : frame->args) {
      if (arg.same_as(v)) {
        frame->buffer_map.Set(v, buffer);
        return buffer;
      }
    }
    TVM_FFI_THROW(InternalError) << "ValueError: Can not bind non-input param to buffer.";
  } else if (const auto* buffer_load = param.as<tvm::tirx::BufferLoadNode>()) {
    SBlockFrame frame = FindSBlockFrame("T.match_buffer");
    frame->match_buffers.push_back(tvm::tirx::MatchBufferRegion(
        buffer, BufferRegionFromLoad(ffi::GetRef<tvm::tirx::BufferLoad>(buffer_load))));
  } else if (const auto* buffer_region = param.as<tvm::tirx::BufferRegionNode>()) {
    SBlockFrame frame = FindSBlockFrame("T.match_buffer");
    frame->match_buffers.push_back(
        tvm::tirx::MatchBufferRegion(buffer, ffi::GetRef<tvm::tirx::BufferRegion>(buffer_region)));
  } else {
    TVM_FFI_THROW(InternalError) << "ValueError: Unexpected type for TIR MatchBuffer.";
  }
  return buffer;
}

SBlockFrame Block(ffi::String name, bool no_realize, ffi::String exec_scope) {
  ffi::ObjectPtr<SBlockFrameNode> n = ffi::make_object<SBlockFrameNode>();
  n->name = name;
  n->iter_vars.clear();
  n->reads = std::nullopt;
  n->writes = std::nullopt;
  n->init = std::nullopt;
  n->alloc_buffers.clear();
  n->match_buffers.clear();
  n->annotations = std::nullopt;
  n->iter_values.clear();
  n->predicate = std::nullopt;
  n->no_realize = no_realize;
  return SBlockFrame(n);
}

void TilePrimitiveCall(tvm::tirx::TilePrimitiveCall op_call) { AddToParent(op_call); }

ffi::Array<tvm::tirx::Var> ScopeId(ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent,
                                   ffi::String name, ffi::String cur) {
  // Determine the number of Vars to introduce. Deferred form (extents=None)
  // is always 1-axis; the verifier closure fills the extent at LowerTIRx.
  size_t n_vars = extents.has_value() ? extents.value().size() : 1;
  if (cur == "warp" || cur == "warpgroup") {
    TVM_FFI_ICHECK_EQ(n_vars, 1) << "ValueError: " << cur << " scope only supports 1D extents, got "
                                 << n_vars << "D";
  }
  ffi::Array<tvm::tirx::Var> scope_ids;
  for (size_t i = 0; i < n_vars; ++i) {
    scope_ids.push_back(tvm::tirx::PrimVar(""));
  }
  // Emit a standalone ScopeIdDefStmt to the current TIRFrame's stmts list.
  // The def is visible to all subsequent stmts within the same enclosing
  // scope (PrimFunc body, AttrStmt body, ExecScope body, etc.).
  tvm::tirx::ScopeIdDef def(
      scope_ids.Map([](tvm::tirx::Var var) { return var.as_or_throw<tvm::tirx::PrimVar>(); }),
      extents, tvm::tirx::StringPairToScopeBinding(parent, cur));
  AddToParent(tvm::tirx::ScopeIdDefStmt(def));
  return scope_ids;
}

ffi::Array<tvm::tirx::Var> ClusterId(ffi::Optional<ffi::Array<PrimExpr>> extents,
                                     ffi::String parent) {
  return ScopeId(extents, parent, "T.cluster_id", "cluster");
}

ffi::Array<tvm::tirx::Var> CtaId(ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent,
                                 ffi::Optional<ffi::Array<PrimExpr>> preferred) {
  if (preferred.has_value()) {
    TVM_FFI_ICHECK(parent == "cluster")
        << "ValueError: preferred is only valid when parent=\"cluster\", got parent=\"" << parent
        << "\"";
    TVM_FFI_ICHECK(extents.has_value())
        << "ValueError: preferred=... requires explicit extents (deferred form is incompatible)";
    ffi::Array<tvm::tirx::Var> scope_ids;
    for (size_t i = 0; i < extents.value().size(); ++i) {
      scope_ids.push_back(tvm::tirx::PrimVar(""));
    }
    tvm::tirx::ScopeIdDef def(
        scope_ids.Map([](tvm::tirx::Var var) { return var.as_or_throw<tvm::tirx::PrimVar>(); }),
        extents, tvm::tirx::StringPairToScopeBinding(parent, "cta"), preferred);
    AddToParent(tvm::tirx::ScopeIdDefStmt(def));
    return scope_ids;
  }
  return ScopeId(extents, parent, "T.cta_id", "cta");
}

ffi::Array<tvm::tirx::Var> CtaIdInPair() {
  ffi::Array<tvm::tirx::Var> scope_ids{tvm::tirx::PrimVar("")};
  tvm::tirx::ScopeIdDef def(
      scope_ids.Map([](tvm::tirx::Var var) { return var.as_or_throw<tvm::tirx::PrimVar>(); }),
      ffi::Array<PrimExpr>{IntImm::Int32(2)}, tvm::tirx::ScopeBinding::kClusterCtaPair);
  AddToParent(tvm::tirx::ScopeIdDefStmt(def));
  return scope_ids;
}

ffi::Array<tvm::tirx::Var> WarpgroupId(ffi::Optional<ffi::Array<PrimExpr>> extents,
                                       ffi::String parent) {
  return ScopeId(extents, parent, "T.warpgroup_id", "warpgroup");
}

ffi::Array<tvm::tirx::Var> WarpId(ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent) {
  return ScopeId(extents, parent, "T.warp_id", "warp");
}

ffi::Array<tvm::tirx::Var> ThreadId(ffi::Optional<ffi::Array<PrimExpr>> extents,
                                    ffi::String parent) {
  return ScopeId(extents, parent, "T.thread_id", "thread");
}

BlockInitFrame Init() { return BlockInitFrame(ffi::make_object<BlockInitFrameNode>()); }

void Where(PrimExpr predicate) {
  SBlockFrame frame = FindSBlockFrame("T.where");
  if (frame->predicate.has_value()) {
    TVM_FFI_THROW(InternalError)
        << "ValueError: Duplicate block predicate declaration, previous one is "
        << frame->predicate;
  }
  frame->predicate = predicate;
}

void Reads(ffi::Array<ffi::ObjectRef> buffer_slices) {
  using namespace tvm::tirx;
  SBlockFrame frame = FindSBlockFrame("T.reads");
  if (frame->reads.has_value()) {
    TVM_FFI_THROW(InternalError)
        << "ValueError: Duplicate read region declaration, previous one is " << frame->reads;
  }
  ffi::Array<BufferRegion> reads;
  for (const ffi::ObjectRef& obj : buffer_slices) {
    if (auto buffer_region = obj.as<BufferRegion>()) {
      reads.push_back(buffer_region.value());
    } else if (auto buffer_load = obj.as<BufferLoad>()) {
      reads.push_back(BufferRegionFromLoad(buffer_load.value()));
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid type for buffer reads.";
    }
  }
  frame->reads = reads;
}

void Writes(ffi::Array<ffi::ObjectRef> buffer_slices) {
  using namespace tvm::tirx;
  SBlockFrame frame = FindSBlockFrame("T.writes");
  if (frame->writes.has_value()) {
    TVM_FFI_THROW(InternalError)
        << "ValueError: Duplicate write region declaration, previous one is " << frame->writes;
  }
  ffi::Array<BufferRegion> writes;
  for (const ffi::ObjectRef& obj : buffer_slices) {
    if (auto buffer_region = obj.as<BufferRegion>()) {
      writes.push_back(buffer_region.value());
    } else if (auto buffer_load = obj.as<BufferLoad>()) {
      writes.push_back(BufferRegionFromLoad(buffer_load.value()));
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid type for buffer writes.";
    }
  }
  frame->writes = writes;
}

/*! \brief Recursively merge two annotations, the new attrs will override the old ones */
ffi::Map<ffi::String, Any> MergeAnnotations(const ffi::Map<ffi::String, Any>& new_attrs,
                                            const ffi::Map<ffi::String, Any>& old_attrs) {
  ffi::Map<ffi::String, Any> result = old_attrs;
  for (const auto& [key, value] : new_attrs) {
    auto old_value = old_attrs.Get(key);
    // Case 1: the key is not in the old annotations, set the key to the new value
    if (!old_value) {
      result.Set(key, value);
      continue;
    }

    // Case 2: the key is in the old annotations
    // Case 2.1: both are dicts
    auto old_dict = old_value->try_cast<ffi::Map<ffi::String, Any>>();
    auto new_dict = value.try_cast<ffi::Map<ffi::String, Any>>();
    if (old_dict && new_dict) {
      // Recursively merge the two dicts
      auto merged_dict = MergeAnnotations(*old_dict, *new_dict);
      result.Set(key, merged_dict);
      continue;
    }
    // Case 2.2: the values are not both dicts, check if the keys are the same
    if (!ffi::AnyEqual()(old_value.value(), value)) {
      TVM_FFI_THROW(InternalError)
          << "ValueError: Try to merge two annotations with different values for key `" << key
          << "`, previous one is " << old_value.value() << ", new one is " << value;
    }
  }
  return result;
}

void BlockAttrs(ffi::Map<ffi::String, Any> attrs) {
  // First try to find an SBlockFrame
  ffi::Optional<SBlockFrame> sblock_frame = IRBuilder::Current()->FindFrame<SBlockFrame>();
  if (sblock_frame.has_value()) {
    if (!sblock_frame.value()->annotations.has_value()) {
      sblock_frame.value()->annotations = attrs;
    } else {
      sblock_frame.value()->annotations =
          MergeAnnotations(attrs, sblock_frame.value()->annotations.value());
    }
    return;
  }
  TVM_FFI_THROW(InternalError)
      << "ValueError: T.sblock_attr must be called at the top of a T.sblock() "
      << "frame, but T.sblock_attr occurred outside of any such frame";
}

ffi::Variant<Buffer, AllocBufferFrame> SBlockAllocBuffer(
    ffi::Array<PrimExpr> shape, PrimType dtype, ffi::Optional<Var> data,
    ffi::Array<PrimExpr> strides, PrimExpr elem_offset, ffi::String storage_scope, int align,
    int offset_factor, ffi::String buffer_type_str,
    ffi::Optional<ffi::Array<IntImm>> axis_separators, ffi::Optional<Layout> layout,
    ffi::Array<PrimExpr> allocated_addr) {
  std::string scope = static_cast<std::string>(storage_scope);
  if (scope.empty()) {
    scope = "global";
  }
  if (scope == "global" || scope == "shared" || scope == "shared.dyn" || scope == "local") {
    TVM_FFI_ICHECK(allocated_addr.empty())
        << "ValueError: For `" << scope
        << "` scope, T.alloc_buffer does not accept `allocated_addr`";
  }
  ffi::Optional<PrimExpr> opt_elem_offset =
      elem_offset.defined() ? ffi::Optional<PrimExpr>(elem_offset) : std::nullopt;
  Buffer buffer =
      BufferDecl(shape, dtype, "", std::nullopt, strides, opt_elem_offset, storage_scope, align,
                 offset_factor, buffer_type_str, axis_separators, layout, allocated_addr);
  IRBuilder builder = IRBuilder::Current();
  auto opt_func_frame = builder->FindFrame<PrimFuncFrame>();
  if (opt_func_frame.has_value()) {
    TVM_FFI_CHECK(opt_func_frame.value()->s_tir, ValueError)
        << "ValueError: `T.sblock_alloc_buffer()` is only for s_tir PrimFuncs. "
           "Use `T.alloc_buffer()` inside default (tirx) PrimFuncs.";
  }

  // Walk up the frame stack: attach to the innermost enclosing SBlock (lifting
  // the allocation past any intermediate For/If/While frames). Fall back to the
  // PrimFunc root when no sblock is in scope. When neither is present (raw
  // IRBuilder construction used by tests), just return the buffer.
  if (ffi::Optional<SBlockFrame> block_frame = builder->FindFrame<SBlockFrame>()) {
    block_frame.value()->alloc_buffers.push_back(buffer);
  } else if (opt_func_frame.has_value()) {
    opt_func_frame.value()->root_alloc_buffers.push_back(buffer);
  }
  return buffer;
}
namespace axis {

IterVar PushBlockVar(IterVar iter_var, PrimExpr binding) {
  if (ffi::Optional<SBlockFrame> opt_frame = IRBuilder::Current()->GetLastFrame<SBlockFrame>()) {
    SBlockFrame frame = opt_frame.value();
    frame->iter_vars.push_back(iter_var);
    frame->iter_values.push_back(binding);
  } else {
    TVM_FFI_THROW(InternalError) << "TypeError: The last frame is not SBlockFrame";
  }
  return iter_var;
}

#define TVM_TIRX_IR_BUILDER_AXIS(Method, Kind, Name)                                 \
  Var Method(Range dom, PrimExpr binding, PrimType dtype) {                          \
    TVM_FFI_ICHECK(dom.defined()) << Name << " axis must have a domain";             \
    PrimType min_ty = dom->min.ty();                                                 \
    PrimType extent_ty = dom->extent.ty();                                           \
    int bits = std::max({min_ty.bits(), extent_ty.bits(), dtype.bits()});            \
    PrimType var_ty = dtype.WithBits(bits);                                          \
    return PushBlockVar(IterVar(/*dom=*/dom, /*var=*/tvm::tirx::PrimVar("", var_ty), \
                                /*iter_type=*/Kind, /*thread_tag=*/""),              \
                        binding)                                                     \
        ->var;                                                                       \
  }
TVM_TIRX_IR_BUILDER_AXIS(Spatial, tvm::tirx::IterVarType::kDataPar, "Spatial");
TVM_TIRX_IR_BUILDER_AXIS(Reduce, tvm::tirx::IterVarType::kCommReduce, "Reduction");
TVM_TIRX_IR_BUILDER_AXIS(Scan, tvm::tirx::IterVarType::kOrdered, "Scan");
TVM_TIRX_IR_BUILDER_AXIS(Opaque, tvm::tirx::IterVarType::kOpaque, "Opaque");
#undef TVM_TIRX_IR_BUILDER_AXIS

ffi::Array<Var> Remap(ffi::String kinds, ffi::Array<PrimExpr> bindings, PrimType dtype) {
  using namespace tvm::tirx;
  ffi::Array<Var> results;
  TVM_FFI_ICHECK_EQ(kinds.size(), bindings.size());
  int n = bindings.size();
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    char c = kinds.c_str()[i];
    PrimExpr e = bindings[i];
    auto v = e.as<PrimVar>();
    TVM_FFI_ICHECK(v) << "TypeError: Only Var is supported in T.axis.remap";
    Range dom{nullptr};
    for (const auto& frame : IRBuilder::Current()->frames) {
      if (const auto* for_frame = frame.as<ForFrameNode>()) {
        TVM_FFI_ICHECK_EQ(for_frame->doms.size(), for_frame->vars.size());
        int n = for_frame->doms.size();
        for (int i = 0; i < n; ++i) {
          if (for_frame->vars[i].same_as(v.value())) {
            dom = for_frame->doms[i];
            break;
          }
        }
        if (dom.defined()) {
          break;
        }
      }
    }
    TVM_FFI_ICHECK(dom.defined()) << "TypeError: Variable is not in the loop: " << v.value();
    PrimType dtype = v.value().ty();
    if (c == 'S') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/tvm::tirx::PrimVar("", dtype),
                                             /*iter_type=*/IterVarType::kDataPar,
                                             /*thread_tag=*/""),
                                     e)
                            ->var);
    } else if (c == 'R') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/tvm::tirx::PrimVar("", dtype),
                                             /*iter_type=*/IterVarType::kCommReduce,
                                             /*thread_tag=*/""),
                                     e)
                            ->var);
    } else {
      TVM_FFI_THROW(InternalError) << "Unknown axis kind: " << c;
    }
  }
  return results;
}

}  // namespace axis

#define TVM_TIRX_IR_BUILDER_FOR_FRAME(Method, Kind)                                        \
  ForFrame Method(PrimExpr start, PrimExpr stop,                                           \
                  ffi::Optional<ffi::Map<ffi::String, Any>> annotations,                   \
                  ffi::Optional<PrimExpr> step) {                                          \
    PrimExpr min = start;                                                                  \
    PrimExpr extent = arith::Analyzer()->Simplify(stop - start);                           \
    ffi::ObjectPtr<ForFrameNode> n = ffi::make_object<ForFrameNode>();                     \
    PrimType min_ty = min.ty();                                                            \
    PrimType extent_ty = extent.ty();                                                      \
    int bits = std::max(min_ty.bits(), extent_ty.bits());                                  \
    n->vars = {Var("v", min_ty.WithBits(bits).WithLanes(1))};                              \
    n->doms = {Range::FromMinExtent(min, extent)};                                         \
    n->steps = {step};                                                                     \
    n->f_make_for_loop = [annotations](ffi::Array<Var> vars, ffi::Array<Range> doms,       \
                                       ffi::Array<ffi::Optional<PrimExpr>> steps,          \
                                       tvm::tirx::Stmt body) {                             \
      TVM_FFI_ICHECK_EQ(vars.size(), 1);                                                   \
      TVM_FFI_ICHECK_EQ(doms.size(), 1);                                                   \
      TVM_FFI_ICHECK_EQ(steps.size(), 1);                                                  \
      return tvm::tirx::For(vars[0].as_or_throw<tvm::tirx::PrimVar>(), doms[0]->min,       \
                            doms[0]->extent, Kind, body, std::nullopt,                     \
                            annotations.value_or(ffi::Map<ffi::String, Any>()), steps[0]); \
    };                                                                                     \
    return ForFrame(n);                                                                    \
  }

TVM_TIRX_IR_BUILDER_FOR_FRAME(Serial, tvm::tirx::ForKind::kSerial);
TVM_TIRX_IR_BUILDER_FOR_FRAME(Parallel, tvm::tirx::ForKind::kParallel);
TVM_TIRX_IR_BUILDER_FOR_FRAME(Vectorized, tvm::tirx::ForKind::kVectorized);
TVM_TIRX_IR_BUILDER_FOR_FRAME(Unroll, tvm::tirx::ForKind::kUnrolled);

#undef TVM_TIRX_IR_BUILDER_FOR_FRAME

ForFrame ThreadBinding(PrimExpr start, PrimExpr stop, ffi::String thread,
                       ffi::Optional<ffi::Map<ffi::String, Any>> annotations) {
  using namespace tvm::tirx;
  PrimExpr min = start;
  PrimExpr extent = arith::Analyzer()->Simplify(stop - start);
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
                                                    Stmt body) -> For {
    TVM_FFI_ICHECK_EQ(vars.size(), 1);
    TVM_FFI_ICHECK_EQ(doms.size(), 1);
    TVM_FFI_ICHECK(steps.size() == 1 && (!steps[0].has_value() || is_one(*steps[0])));
    IterVar iter_var(Range(nullptr), tvm::tirx::PrimVar("iter", dtype), IterVarType::kThreadIndex,
                     thread);
    return For(vars[0].as_or_throw<tvm::tirx::PrimVar>(), doms[0]->min, doms[0]->extent,
               ForKind::kThreadBinding, body, iter_var,
               annotations.value_or(ffi::Map<ffi::String, ffi::Any>()), std::nullopt);
  };
  return ForFrame(n);
}

ForFrame Grid(ffi::Array<ffi::Variant<PrimExpr, ffi::Tuple<PrimExpr, PrimExpr>>> extents) {
  using namespace tvm::tirx;
  ffi::ObjectPtr<ForFrameNode> n = ffi::make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  n->steps.resize(extents.size());
  for (const auto& extent : extents) {
    if (auto prim_expr = extent.as<PrimExpr>()) {
      // extent is a single PrimExpr
      PrimType dtype = prim_expr.value().ty();
      n->vars.push_back(Var("v", dtype));
      n->doms.push_back(Range(tvm::IntImm(dtype, 0), prim_expr.value()));
    } else if (auto tuple = extent.as<ffi::Tuple<PrimExpr, PrimExpr>>()) {
      // extent is a tuple of two PrimExpr (start, extent)
      PrimType dtype = tuple.value().get<0>().ty();
      n->vars.push_back(Var("v", dtype));
      n->doms.push_back(Range::FromMinExtent(tuple.value().get<0>(), tuple.value().get<1>()));
    } else {
      TVM_FFI_THROW(InternalError) << "TypeError: Invalid type for grid extent";
    }
  }
  n->f_make_for_loop = [](ffi::Array<Var> vars, ffi::Array<Range> doms,
                          ffi::Array<ffi::Optional<PrimExpr>> steps, Stmt body) -> Stmt {
    TVM_FFI_ICHECK_EQ(vars.size(), doms.size());
    TVM_FFI_ICHECK_EQ(vars.size(), steps.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      body = For(var.as_or_throw<tvm::tirx::PrimVar>(), dom->min, dom->extent, ForKind::kSerial,
                 std::move(body),
                 /*thread_binding=*/std::nullopt, /*annotations=*/{}, /*step=*/steps[i]);
    }
    return body;
  };
  return ForFrame(n);
}

AssertFrame Assert(PrimExpr condition, ffi::String error_kind,
                   ffi::Array<ffi::String> message_parts) {
  ffi::ObjectPtr<AssertFrameNode> n = ffi::make_object<AssertFrameNode>();
  n->condition = condition;
  n->error_kind = tvm::tirx::StringImm(error_kind);
  ffi::Array<tvm::tirx::StringImm> parts;
  for (const auto& p : message_parts) {
    parts.push_back(tvm::tirx::StringImm(p));
  }
  n->message_parts = parts;
  return AssertFrame(n);
}

Var Bind(Expr value, ffi::Optional<Type> type_annotation, ffi::Optional<Var> var) {
  Expr value_expr = value;
  Var bind_var = [&]() {
    if (var.has_value()) {
      return var.value();
    } else if (type_annotation.has_value()) {
      return Var("v", type_annotation.value());
    } else {
      return Var("v", value_expr->ty);
    }
  }();
  AddToParent(tvm::tirx::Bind(bind_var, value_expr));
  return bind_var;
}

LaunchThreadFrame LaunchThread(Var var, PrimExpr extent) {
  IterVar iter_var{nullptr};

  if (ffi::Optional<PrimFuncFrame> opt_frame = IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    if (ffi::Optional<IterVar> opt_iter_var = opt_frame.value()->env_threads.Get(var)) {
      iter_var = opt_iter_var.value();
    } else {
      TVM_FFI_THROW(InternalError) << "ValueError: " << var->name_hint
                                   << " is not an env_thread created using T.env_thread.";
    }
  } else {
    TVM_FFI_THROW(InternalError) << "LaunchThread can only be used inside a PrimFunc";
  }
  ffi::ObjectPtr<LaunchThreadFrameNode> n = ffi::make_object<LaunchThreadFrameNode>();
  if (!iter_var->dom.defined()) {
    const_cast<tvm::tirx::IterVarNode*>(iter_var.get())->dom =
        Range(tvm::IntImm(extent.ty(), 0), extent);
  } else if (!arith::Analyzer()->CanProveEqual(iter_var->dom->extent, extent)) {
    TVM_FFI_THROW(InternalError) << "ValueError: Inconsistent extents of environment thread. "
                                 << iter_var->dom->extent << " vs " << extent;
  }
  n->iter_var = iter_var;
  n->extent = extent;
  n->attr_key = iter_var->thread_tag == "vthread" ? "virtual_thread" : "thread_extent";
  return LaunchThreadFrame(n);
}

LaunchThreadFrame LaunchThread(ffi::String thread_tag, PrimExpr extent) {
  return LaunchThread(EnvThread(thread_tag, extent.ty()), extent);
}

AttrFrame Attr(ffi::Any node, ffi::String attr_key, PrimExpr value) {
  // convert POD value to PrimExpr
  if (node.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
    node = node.cast<PrimExpr>();
  }
  ffi::ObjectPtr<AttrFrameNode> n = ffi::make_object<AttrFrameNode>();
  n->node = std::move(node);
  n->attr_key = attr_key;
  n->value = value;
  return AttrFrame(n);
}

AttrFrame DeviceEntry() {
  // Flat marker: open an AttrFrame keyed ``tirx.device_entry`` with
  // ``Bool(true)`` value. Subsequent stmts within the enclosing PrimFunc
  // body accumulate into this frame's body. The Python wrapper auto-calls
  // ``__enter__`` so users write a flat ``Tx.device_entry()`` (no ``with``).
  // To close the AttrFrame at function end, register a callback on the
  // enclosing PrimFuncFrame: ``IRBuilderFrameNode::ExitWithScope`` runs
  // callbacks before popping itself, so the AttrFrame is closed and its
  // emitted ``AttrStmt`` lands in the PrimFunc's body sequence.
  AttrFrame frame =
      Attr(IntImm::Int32(0), ffi::String(tvm::tirx::attr::kDeviceEntry), IntImm::Bool(true));
  IRBuilder builder = IRBuilder::Current();
  ffi::Optional<PrimFuncFrame> pf_frame = builder->FindFrame<PrimFuncFrame>();
  TVM_FFI_ICHECK(pf_frame.has_value())
      << "T.device_entry() must be called inside a @T.prim_func body";
  // Capture the AttrFrame by ObjectRef value so the lambda holds a strong
  // reference while the callback runs. Without this, the only reference is
  // the IRBuilder frame stack; ``ExitWithScope`` pops itself first and the
  // AttrFrameNode would be destroyed mid-method (before the body-wrapping
  // AddToParent runs).
  AttrFrame frame_ref = frame;
  pf_frame.value()->callbacks.push_back([frame_ref]() {
    const_cast<IRBuilderFrameNode*>(static_cast<const IRBuilderFrameNode*>(frame_ref.get()))
        ->ExitWithScope();
  });
  return frame;
}

WhileFrame While(PrimExpr condition) {
  ffi::ObjectPtr<WhileFrameNode> n = ffi::make_object<WhileFrameNode>();
  n->condition = condition;
  return WhileFrame(n);
}

void Break() { AddToParent(tvm::tirx::Break(Span())); }

void Continue() { AddToParent(tvm::tirx::Continue(Span())); }

IfFrame If(PrimExpr condition) {
  ffi::ObjectPtr<IfFrameNode> n = ffi::make_object<IfFrameNode>();
  n->condition = condition;
  n->then_stmts = std::nullopt;
  n->else_stmts = std::nullopt;
  return IfFrame(n);
}

ThenFrame Then() {
  ffi::ObjectPtr<ThenFrameNode> n = ffi::make_object<ThenFrameNode>();
  return ThenFrame(n);
}

ElseFrame Else() {
  ffi::ObjectPtr<ElseFrameNode> n = ffi::make_object<ElseFrameNode>();
  return ElseFrame(n);
}

HintFrame Hint(ffi::String message, ffi::Map<ffi::String, ffi::Any> attrs) {
  ffi::ObjectPtr<HintFrameNode> n = ffi::make_object<HintFrameNode>();
  n->message = message;
  n->attrs = attrs;
  return HintFrame(n);
}

ComposeOpFrame ComposeOp(ffi::Map<ffi::String, Buffer> workspace,
                         ffi::Map<ffi::String, ffi::Any> config,
                         ffi::Optional<ffi::String> dispatch) {
  ffi::ObjectPtr<ComposeOpFrameNode> n = ffi::make_object<ComposeOpFrameNode>();
  n->workspace = workspace;
  n->config = config;
  n->dispatch = dispatch;
  return ComposeOpFrame(n);
}

Var EnvThread(ffi::String thread_tag, PrimType dtype) {
  IterVar iter_var(Range{nullptr}, tvm::tirx::PrimVar("", dtype),
                   tvm::tirx::IterVarType::kThreadIndex, thread_tag);
  Var var = iter_var->var;
  if (ffi::Optional<PrimFuncFrame> opt_frame = IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    opt_frame.value()->env_threads.Set(var, iter_var);
  } else {
    TVM_FFI_THROW(InternalError) << "EnvThread can only be used inside a PrimFunc";
  }
  return var;
}

void BufferStore(Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                 ffi::Optional<PrimExpr> predicate = std::nullopt) {
  PrimType buffer_dtype = buffer->dtype;
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
      TVM_FFI_THROW(InternalError) << "TypeError: Incompatible types in BufferStore"
                                   << ": LHS is `" << lhs_dtype << "`, RHS is `" << rhs_dtype
                                   << "`, indexing lanes: " << index_lanes;
    }
    if (lhs_dtype.code() != rhs_dtype.code()) {
      if ((lhs_dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) &&
          (rhs_dtype.code() == DLDataTypeCode::kDLFloat ||
           rhs_dtype.code() == DLDataTypeCode::kDLBfloat)) {
        LOG(WARNING) << "Casting in BufferStore may lose precision"
                     << ": LHS is `" << lhs_dtype << "`, RHS is `" << rhs_dtype
                     << "`, indexing lanes: " << index_lanes;
      }
    }
    value = tvm::cast(lhs_dtype, value);
  }
  AddToParent(tvm::tirx::BufferStore(buffer, value, indices, predicate));
}

DeclBufferFrame DeclBuffer(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::String buffer_name,
                           ffi::Optional<Var> data, ffi::Optional<ffi::Array<PrimExpr>> strides,
                           ffi::Optional<PrimExpr> elem_offset, ffi::String storage_scope,
                           int align, int offset_factor, ffi::String buffer_type,
                           ffi::Optional<ffi::Array<IntImm>> axis_separators,
                           ffi::Optional<Layout> layout, ffi::Optional<PrimExpr> allocated_addr) {
  std::string scope = static_cast<std::string>(storage_scope);
  if (scope.empty()) {
    scope = "global";
  }

  // Enforce rules for T.decl_buffer based on storage scope
  ffi::Array<PrimExpr> allocated_addr_arr;
  if (scope == "tmem") {
    TVM_FFI_ICHECK(!data.has_value())
        << "ValueError: For `tmem` scope, T.decl_buffer accepts only `allocated_addr`";
    TVM_FFI_ICHECK(allocated_addr.has_value())
        << "ValueError: For `tmem` scope, T.decl_buffer requires `allocated_addr` (PrimExpr)";
    allocated_addr_arr = ffi::Array<PrimExpr>({allocated_addr.value()});
  } else if (scope == "global" || scope == "shared" || scope == "shared.dyn" || scope == "local") {
    TVM_FFI_ICHECK(!allocated_addr.has_value())
        << "ValueError: For `" << scope
        << "` scope, T.decl_buffer does not accept `allocated_addr`";
    allocated_addr_arr = ffi::Array<PrimExpr>();
  } else {
    // Other scopes: fall back to provided value if any
    if (allocated_addr.has_value()) {
      allocated_addr_arr = ffi::Array<PrimExpr>({allocated_addr.value()});
    } else {
      allocated_addr_arr = ffi::Array<PrimExpr>();
    }
  }

  ffi::ObjectPtr<DeclBufferFrameNode> n = ffi::make_object<DeclBufferFrameNode>();
  n->buffer =
      BufferDecl(shape, dtype, buffer_name, data, strides, elem_offset, storage_scope, align,
                 offset_factor, buffer_type, axis_separators, layout, allocated_addr_arr);
  // For tmem, even without `data`, we should not emit an Allocate node.
  n->allocated = (scope == "tmem") || data.has_value();
  return DeclBufferFrame(n);
}

Buffer AllocBuffer(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::String storage_scope,
                   ffi::Optional<ffi::Map<ffi::String, ffi::Any>> annotations) {
  Buffer buffer = BufferDecl(shape, dtype, "", std::nullopt, std::nullopt, std::nullopt,
                             storage_scope, 0, 0, "", std::nullopt);
  AddToParent(
      tvm::tirx::AllocBuffer(buffer, annotations.value_or(ffi::Map<ffi::String, ffi::Any>())));
  return buffer;
}

void Evaluate(Expr value) { AddToParent(tvm::tirx::Evaluate(value)); }

Var Ptr(PrimType dtype, ffi::String storage_scope = "global") {
  PointerType type_annotation(dtype, storage_scope);
  return tvm::tirx::Var("", type_annotation);
}

using tvm::script::ir_builder::details::Namer;

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tirx::BufferNode>([](const ffi::ObjectRef& node, ffi::String name) -> void {
      tvm::tirx::BufferNode* buffer =
          const_cast<tvm::tirx::BufferNode*>(node.as<tvm::tirx::BufferNode>());
      if (!buffer->name.empty() && buffer->name != std::string(name)) {
        TVM_FFI_THROW(InternalError)
            << "Buffer name conflict: buffer was created with name \"" << buffer->name
            << "\", but the parser is trying to rename it to \"" << name
            << "\". Remove the explicit `name=` argument and let the parser "
            << "auto-name the buffer from the LHS variable.";
      }
      buffer->name = name;
      Namer::Name(buffer->data, name + "_ptr");
      int n = buffer->strides.size();
      for (int i = 0; i < n; ++i) {
        PrimExpr e = buffer->strides[i];
        if (auto v = e.as<tvm::tirx::PrimVar>()) {
          Namer::Name(v.value(), name + "_s" + std::to_string(i));
        }
      }
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tirx::BufferLoadNode>([](const ffi::ObjectRef& node,
                                                ffi::String name) -> void {
      using namespace tvm::tirx;
      BufferLoadNode* buffer = const_cast<BufferLoadNode*>(node.as<BufferLoadNode>());
      Namer::Name(buffer->buffer, name);
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tirx::TileLayoutNode>([](const ffi::ObjectRef& node,
                                                ffi::String name) -> void {

    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tirx::IterVarNode>([](const ffi::ObjectRef& node, ffi::String name) -> void {
      using namespace tvm::tirx;
      IterVarNode* var = const_cast<IterVarNode*>(node.as<IterVarNode>());
      Namer::Name(var->var, name);
    });

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tirx.Buffer",
           static_cast<Buffer (*)(ffi::Array<PrimExpr>, PrimType, ffi::String, ffi::Optional<Var>,
                                  ffi::Optional<ffi::Array<PrimExpr>>, ffi::Optional<PrimExpr>,
                                  ffi::String, int, int, ffi::String,
                                  ffi::Optional<ffi::Array<IntImm>>, ffi::Optional<Layout>,
                                  ffi::Array<PrimExpr>)>(BufferDecl))
      .def("script.ir_builder.tirx.PrimFunc", PrimFunc)
      .def("script.ir_builder.tirx.Arg",
           [](ffi::String name, ffi::ObjectRef obj) -> ffi::ObjectRef {
             using namespace tvm::tirx;
             if (auto var = obj.as<Var>()) {
               return Arg(name, var.value());
             }
             if (auto buffer = obj.as<Buffer>()) {
               return Arg(name, buffer.value());
             }
             TVM_FFI_THROW(InternalError)
                 << "ValueError: Unexpected type for TIR Arg: " << obj->GetTypeKey();
             throw;
           })
      .def("script.ir_builder.tirx.FuncName", FuncName)
      .def("script.ir_builder.tirx.FuncAttrs", FuncAttrs)
      .def("script.ir_builder.tirx.FuncRet", FuncRet)
      .def("script.ir_builder.tirx.MatchBuffer", MatchBuffer)
      .def("script.ir_builder.tirx.Block", Block)
      .def("script.ir_builder.tirx.TilePrimitiveCall", TilePrimitiveCall)
      .def("script.ir_builder.tirx.ClusterId",
           [](ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent) {
             return ClusterId(extents, parent);
           })
      .def("script.ir_builder.tirx.CtaId",
           [](ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent,
              ffi::Optional<ffi::Array<PrimExpr>> preferred) {
             return CtaId(extents, parent, preferred);
           })
      .def("script.ir_builder.tirx.CtaIdInPair", CtaIdInPair)
      .def("script.ir_builder.tirx.WarpgroupId",
           [](ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent) {
             return WarpgroupId(extents, parent);
           })
      .def("script.ir_builder.tirx.WarpId",
           [](ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent) {
             return WarpId(extents, parent);
           })
      .def("script.ir_builder.tirx.ThreadId",
           [](ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent) {
             return ThreadId(extents, parent);
           })
      .def("script.ir_builder.tirx.ScopeId",
           [](ffi::Optional<ffi::Array<PrimExpr>> extents, ffi::String parent, ffi::String name,
              ffi::String cur) { return ScopeId(extents, parent, name, cur); })
      .def("script.ir_builder.tirx.Init", Init)
      .def("script.ir_builder.tirx.Where", Where)
      .def("script.ir_builder.tirx.Reads", Reads)
      .def("script.ir_builder.tirx.Writes", Writes)
      .def("script.ir_builder.tirx.BlockAttrs", BlockAttrs)
      .def("script.ir_builder.tirx.SBlockAllocBuffer", SBlockAllocBuffer)
      .def("script.ir_builder.tirx.AllocBuffer", AllocBuffer)
      .def("script.ir_builder.tirx.AxisSpatial", axis::Spatial)
      .def("script.ir_builder.tirx.AxisReduce", axis::Reduce)
      .def("script.ir_builder.tirx.AxisScan", axis::Scan)
      .def("script.ir_builder.tirx.AxisOpaque", axis::Opaque)
      .def("script.ir_builder.tirx.AxisRemap", axis::Remap)
      .def("script.ir_builder.tirx.Serial", Serial)
      .def("script.ir_builder.tirx.Parallel", Parallel)
      .def("script.ir_builder.tirx.Vectorized", Vectorized)
      .def("script.ir_builder.tirx.Unroll", Unroll)
      .def("script.ir_builder.tirx.ThreadBinding", ThreadBinding)
      .def("script.ir_builder.tirx.Grid", Grid)
      .def("script.ir_builder.tirx.Assert", Assert)
      .def("script.ir_builder.tirx.Bind", Bind)
      .def("script.ir_builder.tirx.Attr", Attr)
      .def("script.ir_builder.tirx.DeviceEntry", DeviceEntry)
      .def("script.ir_builder.tirx.While", While)
      .def("script.ir_builder.tirx.Break", Break)
      .def("script.ir_builder.tirx.Continue", Continue)
      .def("script.ir_builder.tirx.If", If)
      .def("script.ir_builder.tirx.Then", Then)
      .def("script.ir_builder.tirx.Else", Else)
      .def("script.ir_builder.tirx.DeclBuffer", DeclBuffer)
      .def("script.ir_builder.tirx.LaunchThread",
           [](ffi::Variant<tvm::tirx::Var, ffi::String> thread_tag_or_var, PrimExpr extent) {
             if (auto var = thread_tag_or_var.as<tvm::tirx::Var>()) {
               return LaunchThread(var.value(), extent);
             } else if (auto str = thread_tag_or_var.as<ffi::String>()) {
               return LaunchThread(str.value(), extent);
             } else {
               TVM_FFI_THROW(InternalError) << "ValueError: Unexpected type for TIR LaunchThread: "
                                            << thread_tag_or_var.GetTypeKey();
               throw;
             }
           })
      .def("script.ir_builder.tirx.EnvThread", EnvThread)
      .def("script.ir_builder.tirx.Hint", Hint)
      .def("script.ir_builder.tirx.ComposeOp", ComposeOp)
      .def("script.ir_builder.tirx.BufferStore", BufferStore)
      .def("script.ir_builder.tirx.Evaluate", Evaluate)
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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.ir_builder.tirx.AddToParent", AddToParent);
}

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
