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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/script/builder/ir.h>
#include <tvm/sym/analyzer.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace s_tir {

using tirx::BufferDecl;
using tirx::ForFrameNode;
using tvm::tirx::IterVar;
using tvm::tirx::IterVarType;

PrimFuncFrame PrimFunc(bool is_private, bool persistent) {
  auto n = ffi::make_object<PrimFuncFrameNode>();
  n->is_private = is_private;
  n->persistent = persistent;
  n->attrs = {};
  return PrimFuncFrame(std::move(n));
}

PrimFuncFrame DeclFunction(bool is_private, bool persistent) {
  PrimFuncFrame frame = PrimFunc(is_private, persistent);
  frame->is_declaration = true;
  return frame;
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

BlockInitFrame Init() { return BlockInitFrame(ffi::make_object<BlockInitFrameNode>()); }

void Where(PrimExpr predicate) {
  SBlockFrame frame = FindSBlockFrame("Ts.where");
  if (frame->predicate.has_value()) {
    TVM_FFI_THROW(InternalError)
        << "ValueError: Duplicate block predicate declaration, previous one is "
        << frame->predicate;
  }
  frame->predicate = predicate;
}

void Reads(ffi::Array<ffi::ObjectRef> buffer_slices) {
  using namespace tvm::tirx;
  SBlockFrame frame = FindSBlockFrame("Ts.reads");
  if (frame->reads.has_value()) {
    TVM_FFI_THROW(InternalError)
        << "ValueError: Duplicate read region declaration, previous one is " << frame->reads;
  }
  ffi::Array<TensorRegion> reads;
  for (const ffi::ObjectRef& obj : buffer_slices) {
    if (auto buffer_region = obj.as<TensorRegion>()) {
      reads.push_back(buffer_region.value());
    } else if (auto buffer_load = obj.as<TensorLoad>()) {
      reads.push_back(BufferRegionFromLoad(buffer_load.value()));
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid type for buffer reads.";
    }
  }
  frame->reads = reads;
}

void Writes(ffi::Array<ffi::ObjectRef> buffer_slices) {
  using namespace tvm::tirx;
  SBlockFrame frame = FindSBlockFrame("Ts.writes");
  if (frame->writes.has_value()) {
    TVM_FFI_THROW(InternalError)
        << "ValueError: Duplicate write region declaration, previous one is " << frame->writes;
  }
  ffi::Array<TensorRegion> writes;
  for (const ffi::ObjectRef& obj : buffer_slices) {
    if (auto buffer_region = obj.as<TensorRegion>()) {
      writes.push_back(buffer_region.value());
    } else if (auto buffer_load = obj.as<TensorLoad>()) {
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
      << "ValueError: Ts.sblock_attr must be called at the top of a Ts.sblock() "
      << "frame, but Ts.sblock_attr occurred outside of any such frame";
}

BufferVar SBlockAllocBuffer(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::Optional<Expr> data,
                            ffi::Array<PrimExpr> strides, PrimExpr elem_offset,
                            ffi::String storage_scope, int align, int offset_factor,
                            ffi::Optional<Layout> layout, ffi::Array<PrimExpr> allocated_addr) {
  std::string scope = static_cast<std::string>(storage_scope);
  if (scope.empty()) {
    scope = "global";
  }
  if (scope == "global" || scope == "shared" || scope == "shared.dyn" || scope == "local") {
    TVM_FFI_ICHECK(allocated_addr.empty())
        << "ValueError: For `" << scope
        << "` scope, Ts.alloc_buffer does not accept `allocated_addr`";
  }
  ffi::Optional<PrimExpr> opt_elem_offset =
      elem_offset.defined() ? ffi::Optional<PrimExpr>(elem_offset) : std::nullopt;
  BufferVar buffer = BufferDecl(shape, dtype, "", std::nullopt, strides, opt_elem_offset,
                                storage_scope, align, offset_factor, layout, allocated_addr);
  IRBuilder builder = IRBuilder::Current();
  auto opt_func_frame = builder->FindFrame<tirx::PrimFuncFrame>();
  if (opt_func_frame.has_value()) {
    TVM_FFI_CHECK(opt_func_frame.value().as<PrimFuncFrameNode>() != nullptr, ValueError)
        << "ValueError: `Ts.alloc_buffer()` is only for s_tir PrimFuncs. "
           "Use `T.alloc_buffer()` inside default (tirx) PrimFuncs.";
  }

  // Walk up the frame stack: attach to the innermost enclosing s_tir::SBlock (lifting
  // the allocation past any intermediate For/If/While frames). Fall back to the
  // PrimFunc root when no sblock is in scope. When neither is present (raw
  // IRBuilder construction used by tests), just return the buffer.
  if (ffi::Optional<SBlockFrame> block_frame = builder->FindFrame<SBlockFrame>()) {
    block_frame.value()->alloc_buffers.push_back(buffer);
  } else if (opt_func_frame.has_value()) {
    ffi::GetRef<PrimFuncFrame>(opt_func_frame.value().as<PrimFuncFrameNode>())
        ->root_alloc_buffers.push_back(buffer);
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

#define TVM_S_TIR_IR_BUILDER_AXIS(Method, Kind, Name)                                              \
  Var Method(Range dom, PrimExpr binding, PrimType dtype) {                                        \
    TVM_FFI_ICHECK(dom.defined()) << Name << " axis must have a domain";                           \
    PrimType min_ty = dom->min.ty();                                                               \
    PrimType extent_ty = dom->extent.ty();                                                         \
    int bits = std::max({min_ty.bits(), extent_ty.bits(), dtype.bits()});                          \
    PrimType var_ty = dtype.WithBits(bits);                                                        \
    return PushBlockVar(IterVar(/*dom=*/dom, /*var=*/tvm::PrimVar("", var_ty), /*iter_type=*/Kind, \
                                /*thread_tag=*/""),                                                \
                        binding)                                                                   \
        ->var;                                                                                     \
  }
TVM_S_TIR_IR_BUILDER_AXIS(Spatial, tvm::tirx::IterVarType::kDataPar, "Spatial");
TVM_S_TIR_IR_BUILDER_AXIS(Reduce, tvm::tirx::IterVarType::kCommReduce, "Reduction");
TVM_S_TIR_IR_BUILDER_AXIS(Scan, tvm::tirx::IterVarType::kOrdered, "Scan");
TVM_S_TIR_IR_BUILDER_AXIS(Opaque, tvm::tirx::IterVarType::kOpaque, "Opaque");
#undef TVM_S_TIR_IR_BUILDER_AXIS

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
                                             /*var=*/tvm::PrimVar("", dtype),
                                             /*iter_type=*/IterVarType::kDataPar,
                                             /*thread_tag=*/""),
                                     e)
                            ->var);
    } else if (c == 'R') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/tvm::PrimVar("", dtype),
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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.s_tir.PrimFunc", PrimFunc)
      .def("script.ir_builder.s_tir.DeclFunction", DeclFunction)
      .def("script.ir_builder.s_tir.Block", Block)
      .def("script.ir_builder.s_tir.Init", Init)
      .def("script.ir_builder.s_tir.Where", Where)
      .def("script.ir_builder.s_tir.Reads", Reads)
      .def("script.ir_builder.s_tir.Writes", Writes)
      .def("script.ir_builder.s_tir.BlockAttrs", BlockAttrs)
      .def("script.ir_builder.s_tir.SBlockAllocBuffer", SBlockAllocBuffer)
      .def("script.ir_builder.s_tir.AxisSpatial", axis::Spatial)
      .def("script.ir_builder.s_tir.AxisReduce", axis::Reduce)
      .def("script.ir_builder.s_tir.AxisScan", axis::Scan)
      .def("script.ir_builder.s_tir.AxisOpaque", axis::Opaque)
      .def("script.ir_builder.s_tir.AxisRemap", axis::Remap);
}

}  // namespace s_tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
