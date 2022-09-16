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
                (buffer_type == "auto_broadcast") ? tvm::tir::kAutoBroadcast : tvm::tir::kDefault,
                axis_separators.value_or(Array<IntImm>()));
}

PrimFuncFrame PrimFunc() {
  ObjectPtr<PrimFuncFrameNode> n = make_object<PrimFuncFrameNode>();
  n->name = NullOpt;
  n->args.clear();
  n->ret_type = NullOpt;
  n->buffer_map.clear();
  n->preflattened_buffer_map.clear();
  n->attrs = NullOpt;
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

void FuncAttrs(Map<String, ObjectRef> attrs) {
  using namespace tvm::tir;
  PrimFuncFrame frame = FindPrimFuncFrame("T.func_attr");
  if (frame->attrs.defined()) {
    LOG(FATAL) << "ValueError: Duplicate prim func annotations, previous one is " << frame->attrs;
  }
  frame->attrs = attrs;
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

void PreflattenedBuffer(Buffer postflattened_buffer, Array<PrimExpr> shape, DataType dtype,
                        Optional<Var> data, Array<PrimExpr> strides, PrimExpr elem_offset,
                        String storage_scope, int align, int offset_factor, String buffer_type_str,
                        Array<IntImm> axis_separators) {
  PrimFuncFrame frame = FindPrimFuncFrame("T.preflattened_buffer");
  for (auto const& p : frame->buffer_map) {
    if (p.second.same_as(postflattened_buffer)) {
      String buffer_name(postflattened_buffer->name + "_preflatten");
      Buffer buffer =
          BufferDecl(shape, dtype, buffer_name, data.value_or(p.second->data), strides, elem_offset,
                     storage_scope, align, offset_factor, buffer_type_str, axis_separators);
      details::Namer::Name(buffer, buffer_name);
      frame->preflattened_buffer_map.Set(p.first, buffer);
      return;
    }
  }
  LOG(FATAL) << "ValueError: postflattened buffer " << postflattened_buffer->name
             << " does not exist.";
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
    n->vars = {Var("v", DataType::Int(bits))};                                                    \
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
  n->vars = {Var("v", DataType::Int(bits))};
  n->doms = {Range::FromMinExtent(min, extent)};
  n->f_make_for_loop = [annotations, thread](Array<Var> vars, Array<Range> doms, Stmt body) -> For {
    ICHECK_EQ(vars.size(), 1);
    ICHECK_EQ(doms.size(), 1);
    IterVar iter_var(Range(nullptr), Var("iter", DataType::Int(32)), IterVarType::kThreadIndex,
                     thread);
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

void Evaluate(PrimExpr value) { AddToParent(tvm::tir::Evaluate(value)); }

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
        if (const tvm::tir::VarNode* v = e.as<tvm::tir::VarNode>()) {
          Namer::Name(GetRef<tvm::tir::Var>(v), name + "_s" + std::to_string(i));
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

TVM_REGISTER_GLOBAL("script.ir_builder.tir.BufferDecl").set_body_typed(BufferDecl);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.PrimFunc").set_body_typed(PrimFunc);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Arg")
    .set_body_typed([](String name, ObjectRef obj) -> ObjectRef {
      using namespace tvm::tir;
      if (const auto* var = obj.as<VarNode>()) {
        return Arg(name, GetRef<tvm::tir::Var>(var));
      }
      if (const auto* buffer = obj.as<BufferNode>()) {
        return Arg(name, GetRef<Buffer>(buffer));
      }
      LOG(FATAL) << "ValueError: Unexpected type for TIR Arg: " << obj->GetTypeKey();
      throw;
    });
TVM_REGISTER_GLOBAL("script.ir_builder.tir.FuncName").set_body_typed(FuncName);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.FuncAttrs").set_body_typed(FuncAttrs);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.FuncRet").set_body_typed(FuncRet);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.MatchBuffer").set_body_typed(MatchBuffer);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.PreflattenedBuffer").set_body_typed(PreflattenedBuffer);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.Block").set_body_typed(Block);

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

TVM_REGISTER_GLOBAL("script.ir_builder.tir.Evaluate").set_body_typed(Evaluate);

TVM_REGISTER_GLOBAL("script.ir_builder.tir.Int8").set_body_typed(Int8);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Int16").set_body_typed(Int16);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Int32").set_body_typed(Int32);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Int64").set_body_typed(Int64);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.UInt8").set_body_typed(UInt8);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.UInt16").set_body_typed(UInt16);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.UInt32").set_body_typed(UInt32);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.UInt64").set_body_typed(UInt64);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Float8").set_body_typed(Float8);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Float16").set_body_typed(Float16);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Float32").set_body_typed(Float32);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Float64").set_body_typed(Float64);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Int32x4").set_body_typed(Int32x4);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Int32x8").set_body_typed(Int32x8);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Int32x16").set_body_typed(Int32x16);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Boolean").set_body_typed(Boolean);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Handle").set_body_typed(Handle);
TVM_REGISTER_GLOBAL("script.ir_builder.tir.Void").set_body_typed(Void);
}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
