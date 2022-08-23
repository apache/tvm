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

#include "tvm/ir/expr.h"

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/data_type.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include "../utils.h"
#include "./tir.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::StringImm>([](TracedObject<tir::StringImm> s, IRDocsifier p) {
      auto value = s.GetAttr(&tir::StringImmNode::value);
      return LiteralDoc::Str(value);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<IntImm>([](TracedObject<IntImm> i, IRDocsifier p) -> ExprDoc {
      const IntImm& node = i.Get();
      if (node->dtype == DataType::Int(32)) {
        return LiteralDoc::Int(i);
      } else if (node->dtype.is_bool()) {
        return LiteralDoc::Boolean(MakeTraced(i.Get()->value != 0, i.GetPath()));
      } else {
        String type_name = runtime::DLDataType2String(node->dtype);
        return TIR(p)
            ->Attr(MakeTraced(type_name, i.GetAttr(&PrimExprNode::dtype).GetPath()))
            ->Call({LiteralDoc::Int(i)});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<FloatImm>([](TracedObject<FloatImm> f, IRDocsifier p) {
      String type_name = runtime::DLDataType2String(f.Get()->dtype);
      return TIR(p)
          ->Attr(MakeTraced(type_name, f.GetAttr(&PrimExprNode::dtype).GetPath()))
          ->Call({LiteralDoc::Float(f)});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Cast>([](TracedObject<tir::Cast> cast, IRDocsifier p) {
      auto value = cast.GetAttr(&tir::CastNode::value);
      auto dtype = cast.GetAttr(&tir::CastNode::dtype);
      return TIR(p)->Attr("cast")->Call({p->AsExprDoc(value), DType2Literal(dtype)});
    });

using OpKind = OperationDocNode::Kind;

template <typename BinOpType, OpKind op_kind>
ExprDoc PrintBinOp(TracedObject<BinOpType> expr, IRDocsifier p) {
  using NodeType = typename BinOpType::ContainerType;
  auto a = expr.GetAttr(&NodeType::a);
  auto b = expr.GetAttr(&NodeType::b);
  return OperationDoc(op_kind, {p->AsExprDoc(a), p->AsExprDoc(b)});
}

#define TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(Op, DocOpKind) \
  TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<Op>(PrintBinOp<Op, DocOpKind>);

TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::Add, OpKind::kAdd);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::Sub, OpKind::kSub);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::Mul, OpKind::kMult);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::Div, OpKind::kDiv);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::FloorDiv, OpKind::kFloorDiv);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::FloorMod, OpKind::kMod);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::LT, OpKind::kLt);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::LE, OpKind::kLtE);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::GT, OpKind::kGt);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::GE, OpKind::kGtE);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::EQ, OpKind::kEq);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::NE, OpKind::kNotEq);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::And, OpKind::kAnd);
TVM_SCRIPT_PRINTER_SET_TIR_BINARY_OP(tir::Or, OpKind::kOr);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Not>([](TracedObject<tir::Not> e, IRDocsifier p) {
      return OperationDoc(OpKind::kNot, {p->AsExprDoc(e.GetAttr(&tir::NotNode::a))});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Select>([](TracedObject<tir::Select> expr, IRDocsifier p) {
      auto condition = expr.GetAttr(&tir::SelectNode::condition);
      auto true_value = expr.GetAttr(&tir::SelectNode::true_value);
      auto false_value = expr.GetAttr(&tir::SelectNode::false_value);
      return TIR(p)->Attr("Select")->Call(
          {p->AsExprDoc(condition), p->AsExprDoc(true_value), p->AsExprDoc(false_value)});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferLoad>([](TracedObject<tir::BufferLoad> expr, IRDocsifier p) {
      auto buffer = expr.GetAttr(&tir::BufferLoadNode::buffer);
      auto indices = expr.GetAttr(&tir::BufferLoadNode::indices);

      ExprDoc base = p->AsExprDoc(buffer);
      return base[AsDocArray<Doc>(indices, p)];
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerLoad>([](TracedObject<tir::ProducerLoad> e, IRDocsifier p) -> Doc {
      LOG(FATAL)
          << "Cannot print a tir.ProducerLoad as it is not valid in TIR Primfuncs. You need to "
             "lower this function first.";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Load>([](TracedObject<tir::Load> e, IRDocsifier p) -> Doc {
      LOG(FATAL) << "Cannot print a tir.Load";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Ramp>([](TracedObject<tir::Ramp> expr, IRDocsifier p) {
      auto base = expr.GetAttr(&tir::RampNode::base);
      auto stride = expr.GetAttr(&tir::RampNode::stride);
      auto lanes = expr.GetAttr(&tir::RampNode::lanes);
      return TIR(p)->Attr("ramp")->Call(
          {p->AsExprDoc(base), p->AsExprDoc(stride), LiteralDoc::Int(lanes)});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Broadcast>([](TracedObject<tir::Broadcast> expr, IRDocsifier p) {
      auto value = expr.GetAttr(&tir::BroadcastNode::value);
      auto lanes = expr.GetAttr(&tir::BroadcastNode::lanes);
      return TIR(p)->Attr("broadcast")->Call({p->AsExprDoc(value), LiteralDoc::Int(lanes)});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Let>([](TracedObject<tir::Let> expr, IRDocsifier p) {
      TIRGeneralFrame frame;
      WithCtx with_frame = p->WithFrame(frame);

      auto var = expr.GetAttr(&tir::LetNode::var);
      auto value = expr.GetAttr(&tir::LetNode::value);
      auto body = expr.GetAttr(&tir::LetNode::body);

      auto value_doc = p->AsExprDoc(value);
      IdDoc var_doc = DefineTIRVar(var, frame, p);
      return TIR(p)->Attr("let")->Call({var_doc, value_doc, p->AsExprDoc(body)});
    });

ExprDoc PrintCall(TracedObject<tir::Call> call, IRDocsifier p) {
  auto op_or_global_var = call.GetAttr(&tir::CallNode::op);

  if (op_or_global_var.IsInstance<Op>()) {
    // TODO(yelite): Call PrintOpCall once it's finished
    TracedObject<String> op_name = op_or_global_var.Downcast<Op>().GetAttr(&OpNode::name);
    Array<ExprDoc> arg_docs{LiteralDoc::Str(op_name)};
    TracedArray<PrimExpr> args = call.GetAttr(&tir::CallNode::args);
    arg_docs = Concat(arg_docs, AsExprDocArray(args, p));
    return TIR(p)->Attr("call")->Call(arg_docs);
  } else {
    auto op_gvar = op_or_global_var.Downcast<GlobalVar>();
    auto name_hint = op_gvar.GetAttr(&GlobalVarNode::name_hint);
    auto args = call.GetAttr(&tir::CallNode::args);

    IdDoc name_doc(name_hint.Get());
    name_doc->source_paths.push_back(name_hint.GetPath());

    return name_doc->Call(AsExprDocArray(args, p));
  }
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Call>(PrintCall);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Shuffle>([](TracedObject<tir::Shuffle> expr, IRDocsifier p) {
      auto vectors = expr.GetAttr(&tir::ShuffleNode::vectors);
      auto indices = expr.GetAttr(&tir::ShuffleNode::indices);
      return TIR(p)->Attr("shuffle")->Call({AsListDoc(vectors, p), AsListDoc(indices, p)});
    });

ExprDoc PrintCommReducer(TracedObject<tir::CommReducer> expr, IRDocsifier p) {
  TIRGeneralFrame frame;
  WithCtx with_frame = p->WithFrame(frame);

  auto lhs = expr.GetAttr(&tir::CommReducerNode::lhs);
  auto rhs = expr.GetAttr(&tir::CommReducerNode::rhs);

  Array<IdDoc> reducer_args;
  for (TracedObject<tir::Var> v_lhs : lhs) {
    IdDoc var_doc = DefineTIRVar(v_lhs, frame, p);
    reducer_args.push_back(var_doc);
  }
  for (TracedObject<tir::Var> v_rhs : rhs) {
    IdDoc var_doc = DefineTIRVar(v_rhs, frame, p);
    reducer_args.push_back(var_doc);
  }

  auto result = expr.GetAttr(&tir::CommReducerNode::result);

  ExprDoc reducer_body = rhs.size() == 1 ? p->AsExprDoc(result[0]) : AsTupleDoc(result, p);

  LambdaDoc reducer{reducer_args, reducer_body};

  auto identity_element = expr.GetAttr(&tir::CommReducerNode::identity_element);
  ListDoc identity_elements_doc = AsListDoc(identity_element, p);

  return TIR(p)->Attr("comm_reducer")->Call({reducer, identity_elements_doc});
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::CommReducer>(PrintCommReducer);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Reduce>([](TracedObject<tir::Reduce> expr, IRDocsifier p) {
      auto combiner = expr.GetAttr(&tir::ReduceNode::combiner);
      auto source = expr.GetAttr(&tir::ReduceNode::source);
      auto axis = expr.GetAttr(&tir::ReduceNode::axis);
      auto value_index = expr.GetAttr(&tir::ReduceNode::value_index);

      Array<ExprDoc> axis_docs;
      for (const auto& iter_var : axis) {
        axis_docs.push_back(IterVarStandaloneDef(iter_var, p));
      }
      ListDoc axis_list_doc = ListDoc(axis_docs);
      axis_list_doc->source_paths.push_back(axis.GetPath());

      return TIR(p)->Attr("reduce")->Call({p->AsExprDoc(combiner), AsListDoc(source, p),
                                           axis_list_doc, LiteralDoc::Int(value_index)});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Range>([](TracedObject<Range> expr, IRDocsifier p) {
      auto min = expr.GetAttr(&RangeNode::min);
      auto extent = expr.GetAttr(&RangeNode::extent);
      auto max = MakeTraced(min.Get() + extent.Get(), extent.GetPath());
      return SliceDoc(p->AsExprDoc(min), p->AsExprDoc(max), NullOpt);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Any>([](TracedObject<tir::Any> e, IRDocsifier p) -> Doc {
      LOG(FATAL) << "Cannot print tir::Any";
      throw;
    });

ExprDoc PrintBufferRegion(TracedObject<tir::BufferRegion> buffer_region, IRDocsifier p) {
  auto region = buffer_region.GetAttr(&tir::BufferRegionNode::region);

  Array<Doc> indices;

  for (TracedObject<Range> range : region) {
    auto extent = range.GetAttr(&RangeNode::extent);
    if (tir::is_one(extent.Get())) {
      auto index = p->AsExprDoc(range.GetAttr(&RangeNode::min));
      index->source_paths.push_back(extent.GetPath());
      indices.push_back(std::move(index));
    } else {
      indices.push_back(p->AsDoc<SliceDoc>(range));
    }
  }

  auto buffer = buffer_region.GetAttr(&tir::BufferRegionNode::buffer);
  return p->AsExprDoc(buffer)[indices];
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferRegion>(PrintBufferRegion);
}  // namespace printer
}  // namespace script
}  // namespace tvm
