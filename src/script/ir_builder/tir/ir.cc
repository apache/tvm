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
#include <tvm/script/ir_builder/tir/ir.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

using tvm::tir::IterVar;

Buffer BufferDecl(Array<PrimExpr> shape, DataType dtype, String buffer_name, Optional<Var> data,
                  Optional<Array<PrimExpr>> strides, Optional<PrimExpr> elem_offset,
                  String storage_scope, int align, int offset_factor, String buffer_type,
                  Optional<Array<IntImm>> axis_separators) {
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
  return Buffer(buffer_data, dtype, shape, strides.value_or(Array<PrimExpr>()),
                elem_offset.value_or(PrimExpr()), buffer_name, align, offset_factor,
                (buffer_type == "auto" ? tvm::tir::kAutoBroadcast : tvm::tir::kDefault),
                axis_separators.value_or(Array<IntImm>()));
}

PrimFuncFrame PrimFunc(bool is_private) {
  ObjectPtr<PrimFuncFrameNode> n = make_object<PrimFuncFrameNode>();
  n->name = NullOpt;
  n->is_private = is_private;
  n->args.clear();
  n->ret_type = NullOpt;
  n->buffer_map.clear();
  n->attrs = {};
  n->env_threads.clear();
  n->root_alloc_buffers.clear();
  return PrimFuncFrame(n);
}

Var Arg(String name, Var var) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.Arg");
  details::Namer::Name(var, name);
  frame->args.push_back(var);
  return var;
}

Buffer Arg(String name, Buffer buffer) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.Arg");
  details::Namer::Name(buffer, name);
  Var handle(buffer->name + "_handle", DataType::Handle());
  frame->args.push_back(handle);
  frame->buffer_map.Set(handle, buffer);
  return buffer;
}

void FuncName(String name) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_name");
  if (frame->name.defined()) {
    LOG(FATAL) << "ValueError: Duplicate prim func name, previous one is " << frame->name.value();
  }
  frame->name = name;
}

void FuncAttrs(Map<String, ObjectRef> new_attrs) {
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

Buffer MatchBuffer(ObjectRef param, Array<PrimExpr> shape, DataType dtype, Optional<Var> data,
                   Array<PrimExpr> strides, PrimExpr elem_offset, String storage_scope, int align,
                   int offset_factor, String buffer_type_str, Array<IntImm> axis_separators) {
  Buffer buffer = BufferDecl(shape, dtype, "", data, strides, elem_offset, storage_scope, align,
                             offset_factor, buffer_type_str, axis_separators);
  if (const auto* var = param.as<tvm::tir::VarNode>()) {
    PrimFuncFrame frame = FindPrimFuncFrame("T.match_buffer");
    Var v = GetRef<Var>(var);
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
        buffer, BufferRegionFromLoad(GetRef<tvm::tir::BufferLoad>(buffer_load))));
  } else if (const auto* buffer_region = param.as<tvm::tir::BufferRegionNode>()) {
    BlockFrame frame = FindBlockFrame("T.match_buffer");
    frame->match_buffers.push_back(
        tvm::tir::MatchBufferRegion(buffer, GetRef<tvm::tir::BufferRegion>(buffer_region)));
  } else {
    LOG(FATAL) << "ValueError: Unexpected type for TIR MatchBuffer.";
  }
  return buffer;
}

BlockFrame Block(String name, bool no_realize) {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->name = name;
  n->iter_vars.clear();
  n->reads = NullOpt;
  n->writes = NullOpt;
  n->init = NullOpt;
  n->alloc_buffers.clear();
  n->match_buffers.clear();
  n->annotations = NullOpt;
  n->iter_values.clear();
  n->predicate = NullOpt;
  n->no_realize = no_realize;
  return BlockFrame(n);
}

BlockInitFrame Init() { return BlockInitFrame(make_object<BlockInitFrameNode>()); }

void Where(PrimExpr predicate) {
  BlockFrame frame = FindBlockFrame("T.where");
  if (frame->predicate.defined()) {
    LOG(FATAL) << "ValueError: Duplicate block predicate declaration, previous one is "
               << frame->predicate;
  }
  frame->predicate = predicate;
}

void Reads(Array<ObjectRef> buffer_slices) {
  using namespace tvm::tir;
  BlockFrame frame = FindBlockFrame("T.reads");
  if (frame->reads.defined()) {
    LOG(FATAL) << "ValueError: Duplicate read region declaration, previous one is " << frame->reads;
  }
  Array<BufferRegion> reads;
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

void Writes(Array<ObjectRef> buffer_slices) {
  using namespace tvm::tir;
  BlockFrame frame = FindBlockFrame("T.writes");
  if (frame->writes.defined()) {
    LOG(FATAL) << "ValueError: Duplicate write region declaration, previous one is "
               << frame->writes;
  }
  Array<BufferRegion> writes;
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

void BlockAttrs(Map<String, ObjectRef> attrs) {
  BlockFrame frame = FindBlockFrame("T.block_attr");
  if (frame->annotations.defined()) {
    LOG(FATAL) << "ValueError: Duplicate block annotations, previous one is " << frame->annotations;
  }
  frame->annotations = attrs;
}

Buffer AllocBuffer(Array<PrimExpr> shape, DataType dtype, Optional<Var> data,
                   Array<PrimExpr> strides, PrimExpr elem_offset, String storage_scope, int align,
                   int offset_factor, String buffer_type_str, Array<IntImm> axis_separators) {
  Buffer buffer = BufferDecl(shape, dtype, "", data, strides, elem_offset, storage_scope, align,
                             offset_factor, buffer_type_str, axis_separators);
  IRBuilder builder = IRBuilder::Current();
  if (Optional<BlockFrame> frame = builder->GetLastFrame<BlockFrame>()) {
    frame.value()->alloc_buffers.push_back(buffer);
  } else if (Optional<PrimFuncFrame> frame = builder->GetLastFrame<PrimFuncFrame>()) {
    frame.value()->root_alloc_buffers.push_back(buffer);
  } else {
    LOG(FATAL) << "ValueError: Block frame or PrimFunc frame not find. Please ensure "
                  "'T.alloc_buffer' is called under T.block() or T.prim_func()";
  }
  return buffer;
}
namespace axis {

IterVar PushBlockVar(IterVar iter_var, PrimExpr binding) {
  if (Optional<BlockFrame> opt_frame = IRBuilder::Current()->GetLastFrame<BlockFrame>()) {
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

Array<Var> Remap(String kinds, Array<PrimExpr> bindings, DataType dtype) {
  using namespace tvm::tir;
  Array<Var> results;
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
    ICHECK(dom.defined()) << "TypeError: Variable is not in the loop: " << GetRef<Var>(v);
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

#define TVM_TIR_IR_BUILDER_FOR_FRAME(Method, Kind)                                                \
  ForFrame Method(PrimExpr start, PrimExpr stop, Optional<Map<String, ObjectRef>> annotations) {  \
    PrimExpr min = start;                                                                         \
    PrimExpr extent = arith::Analyzer().Simplify(stop - start);                                   \
    ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();                                      \
    int bits = std::max(min.dtype().bits(), extent.dtype().bits());                               \
    n->vars = {Var("v", DataType(min.dtype().code(), bits, 1))};                                  \
    n->doms = {Range::FromMinExtent(min, extent)};                                                \
    n->f_make_for_loop = [annotations](Array<Var> vars, Array<Range> doms, tvm::tir::Stmt body) { \
      ICHECK_EQ(vars.size(), 1);                                                                  \
      ICHECK_EQ(doms.size(), 1);                                                                  \
      return tvm::tir::For(vars[0], doms[0]->min, doms[0]->extent, Kind, body, NullOpt,           \
                           annotations.value_or(Map<String, ObjectRef>()));                       \
    };                                                                                            \
    return ForFrame(n);                                                                           \
  }

TVM_TIR_IR_BUILDER_FOR_FRAME(Serial, tvm::tir::ForKind::kSerial);
TVM_TIR_IR_BUILDER_FOR_FRAME(Parallel, tvm::tir::ForKind::kParallel);
TVM_TIR_IR_BUILDER_FOR_FRAME(Vectorized, tvm::tir::ForKind::kVectorized);
TVM_TIR_IR_BUILDER_FOR_FRAME(Unroll, tvm::tir::ForKind::kUnrolled);

#undef TVM_TIR_IR_BUILDER_FOR_FRAME

ForFrame ThreadBinding(PrimExpr start, PrimExpr stop, String thread,
                       Optional<Map<String, ObjectRef>> annotations) {
  using namespace tvm::tir;
  PrimExpr min = start;
  PrimExpr extent = arith::Analyzer().Simplify(stop - start);
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  int bits = std::max(min.dtype().bits(), extent.dtype().bits());
  DataType dtype = DataType(min.dtype().code(), bits, 1);
  n->vars = {Var("v", dtype)};
  n->doms = {Range::FromMinExtent(min, extent)};
  n->f_make_for_loop = [annotations, thread, dtype](Array<Var> vars, Array<Range> doms,
                                                    Stmt body) -> For {
    ICHECK_EQ(vars.size(), 1);
    ICHECK_EQ(doms.size(), 1);
    IterVar iter_var(Range(nullptr), Var("iter", dtype), IterVarType::kThreadIndex, thread);
    return For(vars[0], doms[0]->min, doms[0]->extent, ForKind::kThreadBinding, body, iter_var,
               annotations.value_or(Map<String, ObjectRef>()));
  };
  return ForFrame(n);
}

ForFrame Grid(Array<PrimExpr> extents) {
  using namespace tvm::tir;
  ObjectPtr<ForFrameNode> n = make_object<ForFrameNode>();
  n->vars.reserve(extents.size());
  n->doms.reserve(extents.size());
  for (const auto& extent : extents) {
    DataType dtype = extent.dtype();
    n->vars.push_back(Var("v", extent.dtype()));
    n->doms.push_back(Range(make_const(dtype, 0), extent));
  }
  n->f_make_for_loop = [](Array<Var> vars, Array<Range> doms, Stmt body) -> Stmt {
    ICHECK_EQ(vars.size(), doms.size());
    int n = vars.size();
    for (int i = n - 1; i >= 0; --i) {
      Range dom = doms[i];
      Var var = vars[i];
      body = For(var, dom->min, dom->extent, ForKind::kSerial, std::move(body),
                 /*thread_binding=*/NullOpt, /*annotations=*/{});
    }
    return body;
  };
  return ForFrame(n);
}

AssertFrame Assert(PrimExpr condition, String message) {
  ObjectPtr<AssertFrameNode> n = make_object<AssertFrameNode>();
  n->condition = condition;
  n->message = tvm::tir::StringImm(message);
  return AssertFrame(n);
}

LetFrame LetStmt(PrimExpr value, Optional<Type> type_annotation, Optional<Var> var) {
  ObjectPtr<LetFrameNode> n = make_object<LetFrameNode>();
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
  ObjectPtr<LetFrameNode> n = make_object<LetFrameNode>();
  n->var = var;
  n->value = value;
  return LetFrame(n);
}

LaunchThreadFrame LaunchThread(Var var, PrimExpr extent) {
  IterVar iter_var{nullptr};

  if (Optional<PrimFuncFrame> opt_frame = IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    if (Optional<IterVar> opt_iter_var = opt_frame.value()->env_threads.Get(var)) {
      iter_var = opt_iter_var.value();
    } else {
      LOG(FATAL) << "ValueError: " << var->name_hint
                 << " is not an env_thread created using T.env_thread.";
    }
  } else {
    LOG(FATAL) << "LaunchThread can only be used inside a PrimFunc";
  }
  ObjectPtr<LaunchThreadFrameNode> n = make_object<LaunchThreadFrameNode>();
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

LaunchThreadFrame LaunchThread(String thread_tag, PrimExpr extent) {
  return LaunchThread(EnvThread(thread_tag, extent.dtype()), extent);
}

RealizeFrame Realize(tvm::tir::BufferRegion buffer_slice, String storage_scope,
                     PrimExpr condition) {
  ObjectPtr<RealizeFrameNode> n = make_object<RealizeFrameNode>();
  n->buffer_slice = buffer_slice;
  n->storage_scope = storage_scope;
  n->condition = condition;
  return RealizeFrame(n);
}

AllocateFrame Allocate(Array<PrimExpr> extents, DataType dtype, String storage_scope,
                       Optional<PrimExpr> condition, Optional<Map<String, ObjectRef>> annotations) {
  ObjectPtr<AllocateFrameNode> n = make_object<AllocateFrameNode>();
  n->extents = extents;
  n->dtype = dtype;
  n->storage_scope = storage_scope;
  n->condition = condition.value_or(tvm::Bool(true));
  n->annotations = annotations.value_or(Map<String, ObjectRef>());
  n->buffer_var = Var("", tvm::PointerType(tvm::PrimType(dtype), storage_scope));
  return AllocateFrame(n);
}

AllocateConstFrame AllocateConst(tvm::runtime::NDArray data, DataType dtype,
                                 Array<PrimExpr> extents,
                                 Optional<Map<String, ObjectRef>> annotations) {
  ObjectPtr<AllocateConstFrameNode> n = make_object<AllocateConstFrameNode>();
  n->dtype = dtype;
  n->extents = extents;
  n->data = data;
  n->annotations = annotations.value_or(Map<String, ObjectRef>());
  n->buffer_var = Var("", tvm::PointerType(tvm::PrimType(dtype)));
  return AllocateConstFrame(n);
}

AttrFrame Attr(ObjectRef node, String attr_key, PrimExpr value) {
  ObjectPtr<AttrFrameNode> n = make_object<AttrFrameNode>();
  n->node = node;
  n->attr_key = attr_key;
  n->value = value;
  return AttrFrame(n);
}

WhileFrame While(PrimExpr condition) {
  ObjectPtr<WhileFrameNode> n = make_object<WhileFrameNode>();
  n->condition = condition;
  return WhileFrame(n);
}

IfFrame If(PrimExpr condition) {
  ObjectPtr<IfFrameNode> n = make_object<IfFrameNode>();
  n->condition = condition;
  n->then_stmts = NullOpt;
  n->else_stmts = NullOpt;
  return IfFrame(n);
}

ThenFrame Then() {
  ObjectPtr<ThenFrameNode> n = make_object<ThenFrameNode>();
  return ThenFrame(n);
}

ElseFrame Else() {
  ObjectPtr<ElseFrameNode> n = make_object<ElseFrameNode>();
  return ElseFrame(n);
}

Var EnvThread(String thread_tag, DataType dtype) {
  IterVar iter_var(Range{nullptr}, Var("", dtype), tvm::tir::IterVarType::kThreadIndex, thread_tag);
  Var var = iter_var->var;
  if (Optional<PrimFuncFrame> opt_frame = IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    opt_frame.value()->env_threads.Set(var, iter_var);
  } else {
    LOG(FATAL) << "EnvThread can only be used inside a PrimFunc";
  }
  return var;
}

void BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices,
                 Optional<PrimExpr> predicate = NullOpt) {
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

void Prefetch(Buffer buffer, Array<Range> bounds) {
  AddToParent(tvm::tir::Prefetch(buffer, bounds));
}

DeclBufferFrame DeclBuffer(Array<PrimExpr> shape, DataType dtype, String buffer_name,
                           Optional<Var> data, Optional<Array<PrimExpr>> strides,
                           Optional<PrimExpr> elem_offset, String storage_scope, int align,
                           int offset_factor, String buffer_type,
                           Optional<Array<IntImm>> axis_separators) {
  ObjectPtr<DeclBufferFrameNode> n = make_object<DeclBufferFrameNode>();
  n->buffer = BufferDecl(shape, dtype, buffer_name, data, strides, elem_offset, storage_scope,
                         align, offset_factor, buffer_type, axis_separators);
  n->allocated = data.defined();
  return DeclBufferFrame(n);
}

void Evaluate(PrimExpr value) { AddToParent(tvm::tir::Evaluate(value)); }

PrimExpr Ptr(runtime::DataType dtype, String storage_scope = "global", bool is_size_var = false) {
  PointerType type_annotation(PrimType(dtype), storage_scope);
  return is_size_var ? tvm::tir::SizeVar("", type_annotation) : tvm::tir::Var("", type_annotation);
}

using tvm::script::ir_builder::details::Namer;

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::BufferNode>([](const ObjectRef& node, String name) -> void {
      tvm::tir::BufferNode* buffer =
          const_cast<tvm::tir::BufferNode*>(node.as<tvm::tir::BufferNode>());
      buffer->name = name;
      Namer::Name(buffer->data, name);
      int n = buffer->strides.size();
      for (int i = 0; i < n; ++i) {
        PrimExpr e = buffer->strides[i];
        if (auto v = e.as<tvm::tir::Var>()) {
          Namer::Name(v.value(), name + "_s" + std::to_string(i));
        }
      }
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::SizeVarNode>([](const ObjectRef& node, String name) -> void {
      using namespace tvm::tir;
      SizeVarNode* var = const_cast<SizeVarNode*>(node.as<SizeVarNode>());
      var->name_hint = name;
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::VarNode>([](const ObjectRef& node, String name) -> void {
      using namespace tvm::tir;
      VarNode* var = const_cast<VarNode*>(node.as<VarNode>());
      var->name_hint = name;
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::tir::IterVarNode>([](const ObjectRef& node, String name) -> void {
      using namespace tvm::tir;
      IterVarNode* var = const_cast<IterVarNode*>(node.as<IterVarNode>());
      Namer::Name(var->var, name);
    });

TVM_REGISTER_GLOBAL("script.ir_builder.tir.Buffer").set_body_typed(BufferDecl);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.PrimFunc").set_body_typed(PrimFunc);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Arg")
    .set_body_typed([](String name, ObjectRef obj) -> ObjectRef {
      using namespace tvm::tir;
      if (auto var = obj.as<Var>()) {
        return Arg(name, var.value());
      }
      if (auto buffer = obj.as<Buffer>()) {
        return Arg(name, buffer.value());
      }
      LOG(FATAL) << "ValueError: Unexpected type for TIR Arg: " << obj->GetTypeKey();
      throw;
    });
TVM_REGISTER_GLOBAL("script.ir_builder.tir.FuncName").set_body_typed(FuncName);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.FuncAttrs").set_body_typed(FuncAttrs);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.FuncRet").set_body_typed(FuncRet);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.MatchBuffer").set_body_typed(MatchBuffer);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.Block").set_body_typed(Block);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Init").set_body_typed(Init);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Where").set_body_typed(Where);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Reads").set_body_typed(Reads);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Writes").set_body_typed(Writes);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.BlockAttrs").set_body_typed(BlockAttrs);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.AllocBuffer").set_body_typed(AllocBuffer);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.AxisSpatial").set_body_typed(axis::Spatial);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.AxisReduce").set_body_typed(axis::Reduce);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.AxisScan").set_body_typed(axis::Scan);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.AxisOpaque").set_body_typed(axis::Opaque);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.AxisRemap").set_body_typed(axis::Remap);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.Serial").set_body_typed(Serial);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Parallel").set_body_typed(Parallel);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Vectorized").set_body_typed(Vectorized);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Unroll").set_body_typed(Unroll);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.ThreadBinding").set_body_typed(ThreadBinding);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Grid").set_body_typed(Grid);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.Assert").set_body_typed(Assert);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.LetStmt").set_body_typed(LetStmt);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.LegacyLetStmt").set_body_typed(LegacyLetStmt);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Allocate").set_body_typed(Allocate);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.AllocateConst").set_body_typed(AllocateConst);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Realize").set_body_typed(Realize);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Attr").set_body_typed(Attr);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.While").set_body_typed(While);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.If").set_body_typed(If);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Then").set_body_typed(Then);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Else").set_body_typed(Else);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.DeclBuffer").set_body_typed(DeclBuffer);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.LaunchThread")
    .set_body_typed([](ObjectRef thread_tag_or_var, PrimExpr extent) {
      if (auto var = thread_tag_or_var.as<tvm::tir::Var>()) {
        return LaunchThread(var.value(), extent);
      } else if (auto str = thread_tag_or_var.as<String>()) {
        return LaunchThread(str.value(), extent);
      } else {
        LOG(FATAL) << "ValueError: Unexpected type for TIR LaunchThread: "
                   << thread_tag_or_var->GetTypeKey();
        throw;
      }
    });
TVM_REGISTER_GLOBAL("script.ir_builder.tir.EnvThread").set_body_typed(EnvThread);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.BufferStore").set_body_typed(BufferStore);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Prefetch").set_body_typed(Prefetch);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Evaluate").set_body_typed(Evaluate);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.Ptr").set_body_typed(Ptr);

#define TVM_TMP_STR(x) #x

#define TVM_REGISTER_GLOBAL_SIZE(Prefix, DType)                          \
  TVM_REGISTER_GLOBAL(Prefix TVM_TMP_STR(8)).set_body_typed(DType##8);   \
  TVM_REGISTER_GLOBAL(Prefix TVM_TMP_STR(16)).set_body_typed(DType##16); \
  TVM_REGISTER_GLOBAL(Prefix TVM_TMP_STR(32)).set_body_typed(DType##32); \
  TVM_REGISTER_GLOBAL(Prefix TVM_TMP_STR(64)).set_body_typed(DType##64);

TVM_REGISTER_GLOBAL_SIZE("script.ir_builder.tir.Float", Float);
TVM_REGISTER_GLOBAL_SIZE("script.ir_builder.tir.UInt", UInt);
TVM_REGISTER_GLOBAL_SIZE("script.ir_builder.tir.Int", Int);

#define TVM_REGISTER_GLOBAL_LANES(Prefix, Func)                           \
  TVM_REGISTER_GLOBAL(Prefix TVM_TMP_STR(x4)).set_body_typed(Func##x4);   \
  TVM_REGISTER_GLOBAL(Prefix TVM_TMP_STR(x8)).set_body_typed(Func##x8);   \
  TVM_REGISTER_GLOBAL(Prefix TVM_TMP_STR(x16)).set_body_typed(Func##x16); \
  TVM_REGISTER_GLOBAL(Prefix TVM_TMP_STR(x32)).set_body_typed(Func##x32); \
  TVM_REGISTER_GLOBAL(Prefix TVM_TMP_STR(x64)).set_body_typed(Func##x64);

#define TVM_REGISTER_GLOBAL_SIZES_LANES(Prefix, DType)          \
  TVM_REGISTER_GLOBAL_LANES(Prefix TVM_TMP_STR(8), DType##8);   \
  TVM_REGISTER_GLOBAL_LANES(Prefix TVM_TMP_STR(16), DType##16); \
  TVM_REGISTER_GLOBAL_LANES(Prefix TVM_TMP_STR(32), DType##32); \
  TVM_REGISTER_GLOBAL_LANES(Prefix TVM_TMP_STR(64), DType##64);

TVM_REGISTER_GLOBAL_SIZES_LANES("script.ir_builder.tir.Float", Float);
TVM_REGISTER_GLOBAL_SIZES_LANES("script.ir_builder.tir.UInt", UInt);
TVM_REGISTER_GLOBAL_SIZES_LANES("script.ir_builder.tir.Int", Int);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.E4M3Float8").set_body_typed(E4M3Float8);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.E5M2Float8").set_body_typed(E5M2Float8);
TVM_REGISTER_GLOBAL_LANES("script.ir_builder.tir.E4M3Float8", E4M3Float8);
TVM_REGISTER_GLOBAL_LANES("script.ir_builder.tir.E5M2Float8", E5M2Float8);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.Boolean").set_body_typed(Boolean);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Handle").set_body_typed(Handle);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Void").set_body_typed(Void);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.min")
    .set_body_typed([](PrimExpr a, PrimExpr b) -> PrimExpr { return tvm::min(a, b); });
TVM_REGISTER_GLOBAL("script.ir_builder.tir.max")
    .set_body_typed([](PrimExpr a, PrimExpr b) -> PrimExpr { return tvm::max(a, b); });
}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
