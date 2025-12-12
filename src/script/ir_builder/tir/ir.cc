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
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/script/ir_builder/tir/ir.h>

#include "./utils.h"
#include "tvm/ffi/string.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

using tvm::tir::IterVar;

Buffer BufferDecl(ffi::Array<PrimExpr> shape, DataType dtype, ffi::String buffer_name,
                  ffi::Optional<Var> data, ffi::Optional<ffi::Array<PrimExpr>> strides,
                  ffi::Optional<PrimExpr> elem_offset, ffi::String storage_scope, int align,
                  int offset_factor, ffi::String buffer_type,
                  ffi::Optional<ffi::Array<IntImm>> axis_separators) {
  CHECK(buffer_type == "auto" || buffer_type == "default" || buffer_type.empty())
      << "ValueError: `buffer_type` must be `auto` or `default` or empty";
  Var buffer_data;
  if (!data.defined()) {
    DataType storage_dtype = dtype;
    if (storage_dtype == DataType::Bool()) {
      storage_dtype = DataType::Int(8);
    }
    buffer_data = tvm::tir::Var(buffer_name, PointerType(PrimType(storage_dtype), storage_scope));
  } else {
    buffer_data = data.value();
  }
  if (!elem_offset.defined() && offset_factor) {
    DataType shape_dtype = shape.empty() ? DataType::Int(32) : shape[0]->dtype;
    elem_offset = tvm::tir::Var("elem_offset", shape_dtype);
  }
  return Buffer(buffer_data, dtype, shape, strides.value_or(ffi::Array<PrimExpr>()),
                elem_offset.value_or(PrimExpr()), buffer_name, align, offset_factor,
                (buffer_type == "auto" ? tvm::tir::kAutoBroadcast : tvm::tir::kDefault),
                axis_separators.value_or(ffi::Array<IntImm>()));
}

PrimFuncFrame PrimFunc(bool is_private) {
  ObjectPtr<PrimFuncFrameNode> n = ffi::make_object<PrimFuncFrameNode>();
  n->name = std::nullopt;
  n->is_private = is_private;
  n->args.clear();
  n->ret_type = std::nullopt;
  n->buffer_map.clear();
  n->attrs = {};
  n->env_threads.clear();
  n->root_alloc_buffers.clear();
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
  Var handle(buffer->name + "_handle", DataType::Handle());
  frame->args.push_back(handle);
  frame->buffer_map.Set(handle, buffer);
  return buffer;
}

void FuncName(ffi::String name) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_name");
  if (frame->name.has_value()) {
    LOG(FATAL) << "ValueError: Duplicate prim func name, previous one is " << frame->name.value();
  }
  frame->name = name;
}

void FuncAttrs(ffi::Map<ffi::String, ffi::Any> new_attrs) {
  using namespace tvm::tir;
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_attr");
  for (const auto& [key, value] : new_attrs) {
    if (key == tvm::attr::kGlobalSymbol && frame->is_private) {
      LOG(FATAL) << "ValueError: "
                 << "A private function may not have the kGlobalSymbol (\""
                 << tvm::attr::kGlobalSymbol << "\") attribute.  "
                 << "However, a private function specified the global symbol as " << value;
    }

    if (auto prev = frame->attrs.Get(key)) {
      LOG(FATAL) << "ValueError: "
                 << "Duplicate prim func annotation for key = \"" << key << "\".  "
                 << "Previous value was " << prev.value() << ", with later definition as " << value;
    } else {
      frame->attrs.Set(key, value);
    }
  }
}

tvm::Type FuncRet(tvm::Type ret_type) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.ret_type");
  if (frame->ret_type.defined()) {
    LOG(FATAL) << "ValueError: Duplicate prim func return type, previous one is "
               << frame->ret_type.value();
  }
  frame->ret_type = ret_type;
  return ret_type;
}

Buffer MatchBuffer(ObjectRef param, ffi::Array<PrimExpr> shape, DataType dtype,
                   ffi::Optional<Var> data, ffi::Array<PrimExpr> strides, PrimExpr elem_offset,
                   ffi::String storage_scope, int align, int offset_factor,
                   ffi::String buffer_type_str, ffi::Optional<ffi::Array<IntImm>> axis_separators) {
  Buffer buffer = BufferDecl(shape, dtype, "", data, strides, elem_offset, storage_scope, align,
                             offset_factor, buffer_type_str, axis_separators);
  if (const auto* var = param.as<tvm::tir::VarNode>()) {
    PrimFuncFrame frame = FindPrimFuncFrameRelaxed("T.match_buffer");
    Var v = ffi::GetRef<Var>(var);
    for (auto const& arg : frame->args) {
      if (arg.same_as(v)) {
        frame->buffer_map.Set(v, buffer);
        return buffer;
      }
    }
    LOG(FATAL) << "ValueError: Can not bind non-input param to buffer.";
  } else if (const auto* buffer_load = param.as<tvm::tir::BufferLoadNode>()) {
    BlockFrame frame = FindBlockFrame("T.match_buffer");
    frame->match_buffers.push_back(tvm::tir::MatchBufferRegion(
        buffer, BufferRegionFromLoad(ffi::GetRef<tvm::tir::BufferLoad>(buffer_load))));
  } else if (const auto* buffer_region = param.as<tvm::tir::BufferRegionNode>()) {
    BlockFrame frame = FindBlockFrame("T.match_buffer");
    frame->match_buffers.push_back(
        tvm::tir::MatchBufferRegion(buffer, ffi::GetRef<tvm::tir::BufferRegion>(buffer_region)));
  } else {
    LOG(FATAL) << "ValueError: Unexpected type for TIR MatchBuffer.";
  }
  return buffer;
}

BlockFrame Block(ffi::String name, bool no_realize) {
  ObjectPtr<BlockFrameNode> n = ffi::make_object<BlockFrameNode>();
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
  return BlockFrame(n);
}

BlockInitFrame Init() { return BlockInitFrame(ffi::make_object<BlockInitFrameNode>()); }

void Where(PrimExpr predicate) {
  BlockFrame frame = FindBlockFrame("T.where");
  if (frame->predicate.defined()) {
    LOG(FATAL) << "ValueError: Duplicate block predicate declaration, previous one is "
               << frame->predicate;
  }
  frame->predicate = predicate;
}

void Reads(ffi::Array<ObjectRef> buffer_slices) {
  using namespace tvm::tir;
  BlockFrame frame = FindBlockFrame("T.reads");
  if (frame->reads.defined()) {
    LOG(FATAL) << "ValueError: Duplicate read region declaration, previous one is " << frame->reads;
  }
  ffi::Array<BufferRegion> reads;
  for (const ObjectRef& obj : buffer_slices) {
    if (auto buffer_region = obj.as<BufferRegion>()) {
      reads.push_back(buffer_region.value());
    } else if (auto buffer_load = obj.as<BufferLoad>()) {
      reads.push_back(BufferRegionFromLoad(buffer_load.value()));
    } else {
      LOG(FATAL) << "Invalid type for buffer reads.";
    }
  }
  frame->reads = reads;
}

void Writes(ffi::Array<ObjectRef> buffer_slices) {
  using namespace tvm::tir;
  BlockFrame frame = FindBlockFrame("T.writes");
  if (frame->writes.defined()) {
    LOG(FATAL) << "ValueError: Duplicate write region declaration, previous one is "
               << frame->writes;
  }
  ffi::Array<BufferRegion> writes;
  for (const ObjectRef& obj : buffer_slices) {
    if (auto buffer_region = obj.as<BufferRegion>()) {
      writes.push_back(buffer_region.value());
    } else if (auto buffer_load = obj.as<BufferLoad>()) {
      writes.push_back(BufferRegionFromLoad(buffer_load.value()));
    } else {
      LOG(FATAL) << "Invalid type for buffer writes.";
    }
  }
  frame->writes = writes;
}

/*! \brief Recursively merge two annotations, the new attrs will override the old ones */
ffi::Map<Any, Any> MergeAnnotations(const ffi::Map<Any, Any>& new_attrs,
                                  const ffi::Map<Any, Any>& old_attrs) {
  ffi::Map<Any, Any> result = old_attrs;
  for (const auto& [key, value] : new_attrs) {
    auto old_value = old_attrs.Get(key);
    // Case 1: the key is not in the old annotations, set the key to the new value
    if (!old_value) {
      result.Set(key, value);
      continue;
    }

    // Case 2: the key is in the old annotations
    // Case 2.1: both are dicts
    auto old_dict = old_value->try_cast<ffi::Map<Any, Any>>();
    auto new_dict = value.try_cast<ffi::Map<Any, Any>>();
    if (old_dict && new_dict) {
      // Recursively merge the two dicts
      auto merged_dict = MergeAnnotations(*old_dict, *new_dict);
      result.Set(key, merged_dict);
      continue;
    }
    // Case 2.3: the values are not both dicts, check if the keys are the same
    if (!ffi::AnyEqual()(old_value.value(), value)) {
      LOG(FATAL) << "ValueError: Try to merge two annotations with different values for key `"
                 << key << "`, previous one is " << old_value.value() << ", new one is " << value;
    }
  }
  return result;
}

void BlockAttrs(ffi::Map<ffi::String, ffi::Any> attrs) {
  BlockFrame frame = FindBlockFrame("T.block_attr");
  // Case 1: the block has no annotations, set the new annotations
  if (!frame->annotations.defined()) {
    frame->annotations = attrs;
  } else {
    // Case 2: the block has annotations, merge the new annotations with the old ones
    frame->annotations = Downcast<ffi::Map<ffi::String, ffi::Any>>(MergeAnnotations(Downcast<ffi::Map<ffi::Any, ffi::Any>>(attrs), Downcast<ffi::Map<ffi::Any, ffi::Any>>(frame->annotations.value())));
  }
}

Buffer AllocBuffer(ffi::Array<PrimExpr> shape, DataType dtype, ffi::Optional<Var> data,
                   ffi::Array<PrimExpr> strides, PrimExpr elem_offset, ffi::String storage_scope,
                   int align, int offset_factor, ffi::String buffer_type_str,
                   ffi::Optional<ffi::Array<IntImm>> axis_separators) {
  Buffer buffer = BufferDecl(shape, dtype, "", data, strides, elem_offset, storage_scope, align,
                             offset_factor, buffer_type_str, axis_separators);
  IRBuilder builder = IRBuilder::Current();
  if (ffi::Optional<BlockFrame> frame = builder->GetLastFrame<BlockFrame>()) {
    frame.value()->alloc_buffers.push_back(buffer);
  } else if (ffi::Optional<BlockFrame> frame = builder->FindFrame<BlockFrame>()) {
    frame.value()->alloc_buffers.push_back(buffer);
  } else if (ffi::Optional<PrimFuncFrame> frame = builder->GetLastFrame<PrimFuncFrame>()) {
    frame.value()->root_alloc_buffers.push_back(buffer);
  } else if (ffi::Optional<PrimFuncFrame> frame = builder->FindFrame<PrimFuncFrame>()) {
    frame.value()->root_alloc_buffers.push_back(buffer);
  } else {
    LOG(FATAL) << "ValueError: Block frame or PrimFunc frame not find. Please ensure "
                  "'T.alloc_buffer' is called under T.block() or T.prim_func()";
  }
  return buffer;
}
namespace axis {

IterVar PushBlockVar(IterVar iter_var, PrimExpr binding) {
  if (ffi::Optional<BlockFrame> opt_frame = IRBuilder::Current()->GetLastFrame<BlockFrame>()) {
    BlockFrame frame = opt_frame.value();
    frame->iter_vars.push_back(iter_var);
    frame->iter_values.push_back(binding);
  } else {
    LOG(FATAL) << "TypeError: The last frame is not BlockFrame";
  }
  return iter_var;
}

#define TVM_TIR_IR_BUILDER_AXIS(Method, Kind, Name)                                           \
  Var Method(Range dom, PrimExpr binding, DataType dtype) {                                   \
    ICHECK(dom.defined()) << Name << " axis must have a domain";                              \
    int bits = std::max({dom->min.dtype().bits(), dom->extent.dtype().bits(), dtype.bits()}); \
    return PushBlockVar(IterVar(/*dom=*/dom, /*var=*/Var("", dtype.with_bits(bits)),          \
                                /*iter_type=*/Kind, /*thread_tag=*/""),                       \
                        binding)                                                              \
        ->var;                                                                                \
  }
TVM_TIR_IR_BUILDER_AXIS(Spatial, tvm::tir::IterVarType::kDataPar, "Spatial");
TVM_TIR_IR_BUILDER_AXIS(Reduce, tvm::tir::IterVarType::kCommReduce, "Reduction");
TVM_TIR_IR_BUILDER_AXIS(Scan, tvm::tir::IterVarType::kOrdered, "Scan");
TVM_TIR_IR_BUILDER_AXIS(Opaque, tvm::tir::IterVarType::kOpaque, "Opaque");
#undef TVM_TIR_IR_BUILDER_AXIS

ffi::Array<Var> Remap(ffi::String kinds, ffi::Array<PrimExpr> bindings, DataType dtype) {
  using namespace tvm::tir;
  ffi::Array<Var> results;
  ICHECK_EQ(kinds.size(), bindings.size());
  int n = bindings.size();
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    char c = kinds.c_str()[i];
    PrimExpr e = bindings[i];
    const VarNode* v = e.as<VarNode>();
    ICHECK(v) << "TypeError: Only Var is supported in T.axis.remap";
    Range dom{nullptr};
    for (const auto& frame : IRBuilder::Current()->frames) {
      if (const auto* for_frame = frame.as<ForFrameNode>()) {
        ICHECK_EQ(for_frame->doms.size(), for_frame->vars.size());
        int n = for_frame->doms.size();
        for (int i = 0; i < n; ++i) {
          if (for_frame->vars[i].get() == v) {
            dom = for_frame->doms[i];
            break;
          }
        }
        if (dom.defined()) {
          break;
        }
      }
    }
    ICHECK(dom.defined()) << "TypeError: Variable is not in the loop: " << ffi::GetRef<Var>(v);
    DataType dtype = v->dtype;
    if (c == 'S') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/Var("", dtype),
                                             /*iter_type=*/IterVarType::kDataPar,
                                             /*thread_tag=*/""),
                                     e)
                            ->var);
    } else if (c == 'R') {
      results.push_back(PushBlockVar(IterVar(/*dom=*/dom,
                                             /*var=*/Var("", dtype),
                                             /*iter_type=*/IterVarType::kCommReduce,
                                             /*thread_tag=*/""),
                                     e)
                            ->var);
    } else {
      LOG(FATAL) << "Unknown axis kind: " << c;
    }
  }
  return results;
}

}  // namespace axis

#define TVM_TIR_IR_BUILDER_FOR_FRAME(Method, Kind)                                           \
  ForFrame Method(PrimExpr start, PrimExpr stop,                                             \
                  ffi::Optional<ffi::Map<ffi::String, Any>> annotations,                     \
                  ffi::Optional<PrimExpr> step) {                                            \
    PrimExpr min = start;                                                                    \
    PrimExpr extent = arith::Analyzer().Simplify(stop - start);                              \
    ObjectPtr<ForFrameNode> n = ffi::make_object<ForFrameNode>();                            \
    int bits = std::max(min.dtype().bits(), extent.dtype().bits());                          \
    n->vars = {Var("v", DataType(min.dtype().code(), bits, 1))};                             \
    n->doms = {Range::FromMinExtent(min, extent)};                                           \
    n->steps = {step};                                                                       \
    n->f_make_for_loop = [annotations](ffi::Array<Var> vars, ffi::Array<Range> doms,         \
                                       ffi::Array<ffi::Optional<PrimExpr>> steps,            \
                                       tvm::tir::Stmt body) {                                \
      ICHECK_EQ(vars.size(), 1);                                                             \
      ICHECK_EQ(doms.size(), 1);                                                             \
      ICHECK_EQ(steps.size(), 1);                                                            \
      return tvm::tir::For(vars[0], doms[0]->min, doms[0]->extent, Kind, body, std::nullopt, \
                           annotations.value_or(ffi::Map<ffi::String, Any>()), steps[0]);    \
    };                                                                                       \
    return ForFrame(n);                                                                      \
  }

TVM_TIR_IR_BUILDER_FOR_FRAME(Serial, tvm::tir::ForKind::kSerial);
TVM_TIR_IR_BUILDER_FOR_FRAME(Parallel, tvm::tir::ForKind::kParallel);
TVM_TIR_IR_BUILDER_FOR_FRAME(Vectorized, tvm::tir::ForKind::kVectorized);
TVM_TIR_IR_BUILDER_FOR_FRAME(Unroll, tvm::tir::ForKind::kUnrolled);

#undef TVM_TIR_IR_BUILDER_FOR_FRAME

ForFrame ThreadBinding(PrimExpr start, PrimExpr stop, ffi::String thread,
                       ffi::Optional<ffi::Map<ffi::String, Any>> annotations) {
  using namespace tvm::tir;
  PrimExpr min = start;
  PrimExpr extent = arith::Analyzer().Simplify(stop - start);
  ObjectPtr<ForFrameNode> n = ffi::make_object<ForFrameNode>();
  int bits = std::max(min.dtype().bits(), extent.dtype().bits());
  DataType dtype = DataType(min.dtype().code(), bits, 1);
  n->vars = {Var("v", dtype)};
  n->doms = {Range::FromMinExtent(min, extent)};
  n->steps = {std::nullopt};
  n->f_make_for_loop = [annotations, thread, dtype](ffi::Array<Var> vars, ffi::Array<Range> doms,
                                                    ffi::Array<ffi::Optional<PrimExpr>> steps,
                                                    Stmt body) -> For {
    ICHECK_EQ(vars.size(), 1);
    ICHECK_EQ(doms.size(), 1);
    ICHECK(steps.size() == 1 && (!steps[0].has_value() || is_one(*steps[0])));
    IterVar iter_var(Range(nullptr), Var("iter", dtype), IterVarType::kThreadIndex, thread);
    return For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kThreadBinding, body, iter_var,
               annotations.value_or(ffi::Map<ffi::String, ffi::Any>()), std::nullopt);
  };
  return ForFrame(n);
}

ForFrame Grid(ffi::Array<PrimExpr> extents) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = ffi::make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  n->steps.resize(extents.size());
  for (const auto& extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop = [](ffi::Array<Var> vars, ffi::Array<Range> doms,
                          ffi::Array<ffi::Optional<PrimExpr>> steps, Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    ICHECK_EQ(vars.size(), steps.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      body = For(var, dom->min, dom->extent, ForKind::kSerial, std::move(body),
                 /*thread_binding=*/std::nullopt, /*annotations=*/{}, /*step=*/steps[i]);
    }
    return body;
  };
  return ForFrame(n);
}

AssertFrame Assert(PrimExpr condition, ffi::String message) {
  ObjectPtr<AssertFrameNode> n = ffi::make_object<AssertFrameNode>();
  n->condition = condition;
  n->message = tvm::tir::StringImm(message);
  return AssertFrame(n);
}

LetFrame LetStmt(PrimExpr value, ffi::Optional<Type> type_annotation, ffi::Optional<Var> var) {
  ObjectPtr<LetFrameNode> n = ffi::make_object<LetFrameNode>();
  if (var.defined()) {
    n->var = var.value();
  } else if (type_annotation.defined()) {
    n->var = Var("v", type_annotation.value());
  } else {
    n->var = Var("v", value.dtype());
  }
  n->value = value;
  return LetFrame(n);
}

LetFrame LegacyLetStmt(Var var, PrimExpr value) {
  ObjectPtr<LetFrameNode> n = ffi::make_object<LetFrameNode>();
  n->var = var;
  n->value = value;
  return LetFrame(n);
}

LaunchThreadFrame LaunchThread(Var var, PrimExpr extent) {
  IterVar iter_var{nullptr};

  if (ffi::Optional<PrimFuncFrame> opt_frame = IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    if (ffi::Optional<IterVar> opt_iter_var = opt_frame.value()->env_threads.Get(var)) {
      iter_var = opt_iter_var.value();
    } else {
      LOG(FATAL) << "ValueError: " << var->name_hint
                 << " is not an env_thread created using T.env_thread.";
    }
  } else {
    LOG(FATAL) << "LaunchThread can only be used inside a PrimFunc";
  }
  ObjectPtr<LaunchThreadFrameNode> n = ffi::make_object<LaunchThreadFrameNode>();
  if (!iter_var->dom.defined()) {
    const_cast<tvm::tir::IterVarNode*>(iter_var.get())->dom =
        Range(tvm::tir::make_zero(extent.dtype()), extent);
  } else if (!arith::Analyzer().CanProveEqual(iter_var->dom->extent, extent)) {
    LOG(FATAL) << "ValueError: Inconsistent extents of environment thread. "
               << iter_var->dom->extent << " vs " << extent;
  }
  n->iter_var = iter_var;
  n->extent = extent;
  n->attr_key = iter_var->thread_tag == "vthread" ? "virtual_thread" : "thread_extent";
  return LaunchThreadFrame(n);
}

LaunchThreadFrame LaunchThread(ffi::String thread_tag, PrimExpr extent) {
  return LaunchThread(EnvThread(thread_tag, extent.dtype()), extent);
}

RealizeFrame Realize(tvm::tir::BufferRegion buffer_slice, ffi::String storage_scope,
                     PrimExpr condition) {
  ObjectPtr<RealizeFrameNode> n = ffi::make_object<RealizeFrameNode>();
  n->buffer_slice = buffer_slice;
  n->storage_scope = storage_scope;
  n->condition = condition;
  return RealizeFrame(n);
}

AllocateFrame Allocate(ffi::Array<PrimExpr> extents, DataType dtype, ffi::String storage_scope,
                       ffi::Optional<PrimExpr> condition,
                       ffi::Optional<ffi::Map<ffi::String, Any>> annotations) {
  ObjectPtr<AllocateFrameNode> n = ffi::make_object<AllocateFrameNode>();
  n->extents = extents;
  n->dtype = dtype;
  n->storage_scope = storage_scope;
  n->condition = condition.value_or(tvm::Bool(true));
  n->annotations = annotations.value_or(ffi::Map<ffi::String, Any>());
  n->buffer_var = Var("", tvm::PointerType(tvm::PrimType(dtype), storage_scope));
  return AllocateFrame(n);
}

AllocateConstFrame AllocateConst(tvm::runtime::Tensor data, DataType dtype,
                                 ffi::Array<PrimExpr> extents,
                                 ffi::Optional<ffi::Map<ffi::String, Any>> annotations) {
  ObjectPtr<AllocateConstFrameNode> n = ffi::make_object<AllocateConstFrameNode>();
  n->dtype = dtype;
  n->extents = extents;
  n->data = data;
  n->annotations = annotations.value_or(ffi::Map<ffi::String, Any>());
  n->buffer_var = Var("", tvm::PointerType(tvm::PrimType(dtype)));
  return AllocateConstFrame(n);
}

AttrFrame Attr(ffi::Any node, ffi::String attr_key, PrimExpr value) {
  // convert POD value to PrimExpr
  if (node.type_index() < ffi::TypeIndex::kTVMFFISmallStr) {
    node = node.cast<PrimExpr>();
  }
  ObjectPtr<AttrFrameNode> n = ffi::make_object<AttrFrameNode>();
  n->node = std::move(node);
  n->attr_key = attr_key;
  n->value = value;
  return AttrFrame(n);
}

WhileFrame While(PrimExpr condition) {
  ObjectPtr<WhileFrameNode> n = ffi::make_object<WhileFrameNode>();
  n->condition = condition;
  return WhileFrame(n);
}

IfFrame If(PrimExpr condition) {
  ObjectPtr<IfFrameNode> n = ffi::make_object<IfFrameNode>();
  n->condition = condition;
  n->then_stmts = std::nullopt;
  n->else_stmts = std::nullopt;
  return IfFrame(n);
}

ThenFrame Then() {
  ObjectPtr<ThenFrameNode> n = ffi::make_object<ThenFrameNode>();
  return ThenFrame(n);
}

ElseFrame Else() {
  ObjectPtr<ElseFrameNode> n = ffi::make_object<ElseFrameNode>();
  return ElseFrame(n);
}

Var EnvThread(ffi::String thread_tag, DataType dtype) {
  IterVar iter_var(Range{nullptr}, Var("", dtype), tvm::tir::IterVarType::kThreadIndex, thread_tag);
  Var var = iter_var->var;
  if (ffi::Optional<PrimFuncFrame> opt_frame = IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    opt_frame.value()->env_threads.Set(var, iter_var);
  } else {
    LOG(FATAL) << "EnvThread can only be used inside a PrimFunc";
  }
  return var;
}

void BufferStore(Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                 ffi::Optional<PrimExpr> predicate = std::nullopt) {
  runtime::DataType buffer_dtype = buffer->dtype;
  bool is_index_scalable = indices.empty() ? false : indices.back().dtype().is_scalable_vector();
  bool is_buffer_dtype_scalable = buffer_dtype.is_scalable_vector();

  ICHECK(!(is_index_scalable && is_buffer_dtype_scalable))
      << "Index dtype and buffer dtype can't both be scalable.";

  int index_lanes;
  if (indices.empty()) {
    index_lanes = 1;
  } else if (is_index_scalable) {
    index_lanes = indices.back().dtype().vscale_factor();
  } else {
    index_lanes = indices.back().dtype().lanes();
  }

  int buffer_lanes = is_buffer_dtype_scalable ? buffer_dtype.vscale_factor() : buffer_dtype.lanes();

  runtime::DataType lhs_dtype;
  if (is_buffer_dtype_scalable || is_index_scalable) {
    lhs_dtype = buffer_dtype.with_scalable_vscale_factor(buffer_lanes * index_lanes);
  } else {
    lhs_dtype = buffer_dtype.with_lanes(buffer_dtype.lanes() * index_lanes);
  }

  runtime::DataType rhs_dtype = value->dtype;

  if (lhs_dtype != rhs_dtype) {
    ICHECK(lhs_dtype.is_scalable_vector() == rhs_dtype.is_scalable_vector())
        << "Can't mix scalable and fixed length vectors in a statement";

    bool lanes_match = false;
    if (lhs_dtype.is_scalable_vector()) {
      lanes_match = lhs_dtype.vscale_factor() == rhs_dtype.vscale_factor();
    } else {
      lanes_match = lhs_dtype.lanes() == rhs_dtype.lanes();
    }

    if (!lanes_match) {
      LOG(FATAL) << "TypeError: Incompatible types in BufferStore"
                 << ": LHS is `" << lhs_dtype << "`, RHS is `" << rhs_dtype
                 << "`, indexing lanes: " << index_lanes;
    }
    if (lhs_dtype.code() != rhs_dtype.code()) {
      if (
          // Case 1. lhs is handle, and rhs needs to be casted to handle.
          (lhs_dtype.code() == runtime::DataType::kHandle) ||
          // Case 2. rhs is handle, and it needs to be casted to non-handle.
          (rhs_dtype.code() == runtime::DataType::kHandle) ||
          // Case 3. rhs is float or bfloat, and casting to non-float can lose precision.
          ((lhs_dtype.code() == runtime::DataType::kInt ||
            lhs_dtype.code() == runtime::DataType::kUInt) &&
           (rhs_dtype.code() == runtime::DataType::kFloat ||
            rhs_dtype.code() == runtime::DataType::kBFloat))) {
        LOG(WARNING) << "Casting in BufferStore may lose precision"
                     << ": LHS is `" << lhs_dtype << "`, RHS is `" << rhs_dtype
                     << "`, indexing lanes: " << index_lanes;
      }
    }
    value = tvm::cast(lhs_dtype, value);
  }
  AddToParent(tvm::tir::BufferStore(buffer, value, indices, predicate));
}

DeclBufferFrame DeclBuffer(ffi::Array<PrimExpr> shape, DataType dtype, ffi::String buffer_name,
                           ffi::Optional<Var> data, ffi::Optional<ffi::Array<PrimExpr>> strides,
                           ffi::Optional<PrimExpr> elem_offset, ffi::String storage_scope,
                           int align, int offset_factor, ffi::String buffer_type,
                           ffi::Optional<ffi::Array<IntImm>> axis_separators) {
  ObjectPtr<DeclBufferFrameNode> n = ffi::make_object<DeclBufferFrameNode>();
  n->buffer = BufferDecl(shape, dtype, buffer_name, data, strides, elem_offset, storage_scope,
                         align, offset_factor, buffer_type, axis_separators);
  n->allocated = data.defined();
  return DeclBufferFrame(n);
}

void Evaluate(PrimExpr value) { AddToParent(tvm::tir::Evaluate(value)); }

PrimExpr Ptr(runtime::DataType dtype, ffi::String storage_scope = "global",
             bool is_size_var = false) {
  PointerType type_annotation(PrimType(dtype), storage_scope);
  return is_size_var ? tvm::tir::SizeVar("", type_annotation) : tvm::tir::Var("", type_annotation);
}

using tvm::script::ir_builder::details::Namer;

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::BufferNode>([](const ObjectRef& node, ffi::String name) -> void {
      tvm::tir::BufferNode* buffer =
          const_cast<tvm::tir::BufferNode*>(node.as<tvm::tir::BufferNode>());
      buffer->name = name;
      Namer::Name(buffer->data, name);
      int n = buffer->strides.size();
      for (int i = 0; i < n; ++i) {
        PrimExpr e = buffer->strides[i];
        if (const auto* v = e.as<tvm::tir::VarNode>()) {
          ffi::String new_name = !v->name_hint.empty() ? v->name_hint : (name + "_s" + std::to_string(i));
          Namer::Name(ffi::GetRef<tvm::tir::Var>(v), ffi::String(new_name));
        }
      }
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::SizeVarNode>([](const ObjectRef& node, ffi::String name) -> void {
      using namespace tvm::tir;
      SizeVarNode* var = const_cast<SizeVarNode*>(node.as<SizeVarNode>());
      var->name_hint = ffi::String(name);
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::VarNode>([](const ObjectRef& node, ffi::String name) -> void {
      using namespace tvm::tir;
      VarNode* var = const_cast<VarNode*>(node.as<VarNode>());
      var->name_hint = name;
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::IterVarNode>([](const ObjectRef& node, ffi::String name) -> void {
      using namespace tvm::tir;
      IterVarNode* var = const_cast<IterVarNode*>(node.as<IterVarNode>());
      Namer::Name(var->var, name);
    });

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Buffer", BufferDecl)
      .def("script.ir_builder.tir.PrimFunc", PrimFunc)
      .def("script.ir_builder.tir.Arg",
           [](ffi::String name, ObjectRef obj) -> ObjectRef {
             using namespace tvm::tir;
             if (auto var = obj.as<Var>()) {
               return Arg(name, var.value());
             }
             if (auto buffer = obj.as<Buffer>()) {
               return Arg(name, buffer.value());
             }
             LOG(FATAL) << "ValueError: Unexpected type for TIR Arg: " << obj->GetTypeKey();
             throw;
           })
      .def("script.ir_builder.tir.FuncName", FuncName)
      .def("script.ir_builder.tir.FuncAttrs", FuncAttrs)
      .def("script.ir_builder.tir.FuncRet", FuncRet)
      .def("script.ir_builder.tir.MatchBuffer", MatchBuffer)
      .def("script.ir_builder.tir.Block", Block)
      .def("script.ir_builder.tir.Init", Init)
      .def("script.ir_builder.tir.Where", Where)
      .def("script.ir_builder.tir.Reads", Reads)
      .def("script.ir_builder.tir.Writes", Writes)
      .def("script.ir_builder.tir.BlockAttrs", BlockAttrs)
      .def("script.ir_builder.tir.AllocBuffer", AllocBuffer)
      .def("script.ir_builder.tir.AxisSpatial", axis::Spatial)
      .def("script.ir_builder.tir.AxisReduce", axis::Reduce)
      .def("script.ir_builder.tir.AxisScan", axis::Scan)
      .def("script.ir_builder.tir.AxisOpaque", axis::Opaque)
      .def("script.ir_builder.tir.AxisRemap", axis::Remap)
      .def("script.ir_builder.tir.Serial", Serial)
      .def("script.ir_builder.tir.Parallel", Parallel)
      .def("script.ir_builder.tir.Vectorized", Vectorized)
      .def("script.ir_builder.tir.Unroll", Unroll)
      .def("script.ir_builder.tir.ThreadBinding", ThreadBinding)
      .def("script.ir_builder.tir.Grid", Grid)
      .def("script.ir_builder.tir.Assert", Assert)
      .def("script.ir_builder.tir.LetStmt", LetStmt)
      .def("script.ir_builder.tir.LegacyLetStmt", LegacyLetStmt)
      .def("script.ir_builder.tir.Allocate", Allocate)
      .def("script.ir_builder.tir.AllocateConst", AllocateConst)
      .def("script.ir_builder.tir.Realize", Realize)
      .def("script.ir_builder.tir.Attr", Attr)
      .def("script.ir_builder.tir.While", While)
      .def("script.ir_builder.tir.If", If)
      .def("script.ir_builder.tir.Then", Then)
      .def("script.ir_builder.tir.Else", Else)
      .def("script.ir_builder.tir.DeclBuffer", DeclBuffer)
      .def("script.ir_builder.tir.LaunchThread",
           [](ffi::Variant<tvm::tir::Var, ffi::String> thread_tag_or_var, PrimExpr extent) {
             if (auto var = thread_tag_or_var.as<tvm::tir::Var>()) {
               return LaunchThread(var.value(), extent);
             } else if (auto str = thread_tag_or_var.as<ffi::String>()) {
               return LaunchThread(str.value(), extent);
             } else {
               LOG(FATAL) << "ValueError: Unexpected type for TIR LaunchThread: "
                          << thread_tag_or_var.GetTypeKey();
               throw;
             }
           })
      .def("script.ir_builder.tir.EnvThread", EnvThread)
      .def("script.ir_builder.tir.BufferStore", BufferStore)
      .def("script.ir_builder.tir.Evaluate", Evaluate)
      .def("script.ir_builder.tir.Ptr", Ptr);
}

#define TVM_TMP_STR(x) #x

#define TVM_FFI_REFL_DEF_GLOBAL_SIZE(Prefix, DType) \
  def(Prefix TVM_TMP_STR(8), DType##8)              \
      .def(Prefix TVM_TMP_STR(16), DType##16)       \
      .def(Prefix TVM_TMP_STR(32), DType##32)       \
      .def(Prefix TVM_TMP_STR(64), DType##64)

#define TVM_FFI_REFL_DEF_GLOBAL_LANES(Prefix, Func) \
  def(Prefix TVM_TMP_STR(x2), Func##x2)             \
      .def(Prefix TVM_TMP_STR(x4), Func##x4)        \
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
      .def("script.ir_builder.tir.BFloat16", BFloat16)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZE("script.ir_builder.tir.Float", Float)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZE("script.ir_builder.tir.UInt", UInt)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZE("script.ir_builder.tir.Int", Int)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZES_LANES("script.ir_builder.tir.Float", Float)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZES_LANES("script.ir_builder.tir.UInt", UInt)
      .TVM_FFI_REFL_DEF_GLOBAL_SIZES_LANES("script.ir_builder.tir.Int", Int)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.BFloat16", BFloat16);
}

// Float8 variants
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float8E3M4", Float8E3M4)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float8E3M4", Float8E3M4);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float8E4M3", Float8E4M3)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float8E4M3", Float8E4M3);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float8E4M3B11FNUZ", Float8E4M3B11FNUZ)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float8E4M3B11FNUZ", Float8E4M3B11FNUZ);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float8E4M3FN", Float8E4M3FN)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float8E4M3FN", Float8E4M3FN);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float8E4M3FNUZ", Float8E4M3FNUZ)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float8E4M3FNUZ", Float8E4M3FNUZ);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float8E5M2", Float8E5M2)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float8E5M2", Float8E5M2);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float8E5M2FNUZ", Float8E5M2FNUZ)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float8E5M2FNUZ", Float8E5M2FNUZ);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float8E8M0FNU", Float8E8M0FNU)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float8E8M0FNU", Float8E8M0FNU);
}

// Float6 variants
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float6E2M3FN", Float6E2M3FN)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float6E2M3FN", Float6E2M3FN);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float6E3M2FN", Float6E3M2FN)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float6E3M2FN", Float6E3M2FN);
}

// Float4 variant
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Float4E2M1FN", Float4E2M1FN)
      .TVM_FFI_REFL_DEF_GLOBAL_LANES("script.ir_builder.tir.Float4E2M1FN", Float4E2M1FN);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.tir.Boolean", Boolean)
      .def("script.ir_builder.tir.Handle", Handle)
      .def("script.ir_builder.tir.TensormapHandle", TensormapHandle)
      .def("script.ir_builder.tir.Void", Void)
      .def("script.ir_builder.tir.min",
           [](PrimExpr a, PrimExpr b) -> PrimExpr { return tvm::min(a, b); })
      .def("script.ir_builder.tir.max",
           [](PrimExpr a, PrimExpr b) -> PrimExpr { return tvm::max(a, b); });
}
}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
