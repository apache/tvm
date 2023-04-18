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
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>

#include "../../support/str_escape.h"

namespace tvm {

#define TVM_LEGACY_REPR_PRINTER_DEF_OP(Type)                        \
  ReprLegacyPrinter& operator<<(ReprLegacyPrinter& p, Type value) { \
    p.Stream() << value;                                            \
    return p;                                                       \
  }

TVM_LEGACY_REPR_PRINTER_DEF_OP(int);
TVM_LEGACY_REPR_PRINTER_DEF_OP(int64_t);
TVM_LEGACY_REPR_PRINTER_DEF_OP(float);
TVM_LEGACY_REPR_PRINTER_DEF_OP(double);
TVM_LEGACY_REPR_PRINTER_DEF_OP(char);
TVM_LEGACY_REPR_PRINTER_DEF_OP(const char*);
TVM_LEGACY_REPR_PRINTER_DEF_OP(const std::string&);
TVM_LEGACY_REPR_PRINTER_DEF_OP(runtime::DataType);
TVM_LEGACY_REPR_PRINTER_DEF_OP(const void*);
TVM_LEGACY_REPR_PRINTER_DEF_OP(const String&);

std::ostream& ReprLegacyPrinter::Stream() const { return stream; }

ReprLegacyPrinter& operator<<(ReprLegacyPrinter& p, const ObjectRef& value) {
  p.Stream() << AsLegacyRepr(value);
  return p;
}

ReprLegacyPrinter& operator<<(ReprLegacyPrinter& out, tir::ForKind type) {  // NOLINT(*)
  using tvm::tir::ForKind;
  switch (type) {
    case ForKind::kSerial:
      out << "for";
      break;
    case ForKind::kParallel:
      out << "parallel";
      break;
    case ForKind::kUnrolled:
      out << "unrolled";
      break;
    case ForKind::kVectorized:
      out << "vectorized";
      break;
    case ForKind::kThreadBinding:
      out << "launch_thread";
      break;
  }
  return out;
}

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<ArrayNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const ArrayNode*>(node.get());
      (*p) << '[';
      for (size_t i = 0; i < op->size(); ++i) {
        if (i != 0) {
          (*p) << ", ";
        }
        p->Print(op->at(i));
      }
      (*p) << ']';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<MapNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const MapNode*>(node.get());
      (*p) << '{';
      for (auto it = op->begin(); it != op->end(); ++it) {
        if (it != op->begin()) {
          (*p) << ", ";
        }
        if (it->first->IsInstance<StringObj>()) {
          (*p) << '\"' << Downcast<String>(it->first) << "\": ";
        } else {
          p->Print(it->first);
          (*p) << ": ";
        }
        p->Print(it->second);
      }
      (*p) << '}';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<ShapeTupleObj>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const ShapeTupleObj*>(node.get());
      (*p) << '[';
      for (size_t i = 0; i < op->size; ++i) {
        if (i != 0) {
          (*p) << ", ";
        }
        (*p) << op->data[i];
      }
      (*p) << ']';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<IntImmNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const IntImmNode*>(node.get());
      if (op->dtype == DataType::Int(32)) {
        (*p) << op->value;
      } else {
        (*p) << "(" << op->dtype << ")" << op->value;
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<FloatImmNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const FloatImmNode*>(node.get());
      switch (op->dtype.bits()) {
        case 64:
          (*p) << op->value;
          break;
        case 32:
          (*p) << op->value << 'f';
          break;
        case 16:
          (*p) << op->value << 'h';
          break;
        default:
          LOG(FATAL) << "Unknown float type bits=" << op->dtype.bits();
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<RangeNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const RangeNode*>(node.get());
      (*p) << "range(min=" << op->min << ", ext=" << op->extent << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<PrimTypeNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const PrimTypeNode*>(ref.get());
      (*p) << node->dtype;
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<PointerTypeNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const PointerTypeNode*>(ref.get());
      if (!node->storage_scope.empty()) {
        (*p) << node->storage_scope << " ";
      }
      p->Print(node->element_type);
      (*p) << '*';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<TupleTypeNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const TupleTypeNode*>(ref.get());
      (*p) << "TupleTypeNode(" << node->fields << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<IncompleteTypeNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const IncompleteTypeNode*>(ref.get());
      (*p) << "IncompleteTypeNode(" << node->kind << ", " << node << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<DictAttrsNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const DictAttrsNode*>(node.get());
      (*p) << op->dict;
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<GlobalVarNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const GlobalVarNode*>(ref.get());
      (*p) << "GlobalVar(" << node->name_hint << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<IRModuleNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const IRModuleNode*>(ref.get());
      (*p) << "IRModule(" << node->functions << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<TypeVarNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const TypeVarNode*>(ref.get());
      (*p) << "TypeVar(" << node->name_hint << ", " << node->kind << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<GlobalTypeVarNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const GlobalTypeVarNode*>(ref.get());
      (*p) << "GlobalTypeVar(" << node->name_hint << ", " << node->kind << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<FuncTypeNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const FuncTypeNode*>(ref.get());
      (*p) << "FuncType(" << node->type_params << ", " << node->arg_types << ", " << node->ret_type
           << ", " << node->type_constraints << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<RelayRefTypeNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const RelayRefTypeNode*>(ref.get());
      (*p) << "RelayRefTypeNode(" << node->value << ")";
    });

}  // namespace tvm

namespace tvm {
namespace tir {

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<BufferNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const BufferNode*>(node.get());
      (*p) << "buffer(" << op->name << ", " << op << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<VarNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const VarNode*>(node.get());
      // omit the type
      // stream << op->name << "." << op->type;
      (*p) << op->name_hint;
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<SizeVarNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const SizeVarNode*>(node.get());
      (*p) << "{" << op->name_hint << "|" << op->name_hint << ">=0}";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<IterVarNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const IterVarNode*>(node.get());
      (*p) << "iter_var(";
      if (op->var->name_hint.length() != 0) {
        (*p) << op->var->name_hint << ", ";
      }
      if (op->dom.defined()) {
        (*p) << op->dom;
      }
      if (op->thread_tag.length() != 0) {
        (*p) << ", " << op->thread_tag;
      }
      (*p) << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<StringImmNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const StringImmNode*>(node.get());
      (*p) << '\"' << support::StrEscape(op->value) << '\"';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<CastNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const CastNode*>(node.get());
      (*p) << op->dtype << '(';
      p->Print(op->value);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<AddNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const AddNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " + ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<SubNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const SubNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " - ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<MulNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const MulNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << "*";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<DivNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const DivNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << "/";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<ModNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const ModNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " % ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<FloorDivNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const FloorDivNode*>(node.get());
      (*p) << "floordiv(" << op->a << ", " << op->b << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<FloorModNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const FloorModNode*>(node.get());
      (*p) << "floormod(" << op->a << ", " << op->b << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<MinNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const MinNode*>(node.get());
      (*p) << "min(";
      p->Print(op->a);
      (*p) << ", ";
      p->Print(op->b);
      (*p) << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<MaxNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const MaxNode*>(node.get());
      (*p) << "max(";
      p->Print(op->a);
      (*p) << ", ";
      p->Print(op->b);
      (*p) << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<EQNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const EQNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " == ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<NENode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const NENode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " != ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<LTNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const LTNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " < ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<LENode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const LENode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " <= ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<GTNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const GTNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " > ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<GENode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const GENode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " >= ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<AndNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const AndNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " && ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<OrNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const OrNode*>(node.get());
      (*p) << '(';
      p->Print(op->a);
      (*p) << " || ";
      p->Print(op->b);
      (*p) << ')';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<NotNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const NotNode*>(node.get());
      (*p) << '!';
      p->Print(op->a);
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<SelectNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const SelectNode*>(node.get());
      (*p) << "select(";
      p->Print(op->condition);
      (*p) << ", ";
      p->Print(op->true_value);
      (*p) << ", ";
      p->Print(op->false_value);
      (*p) << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<RampNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const RampNode*>(node.get());
      (*p) << "ramp(";
      p->Print(op->base);
      (*p) << ", ";
      p->Print(op->stride);
      (*p) << ", " << op->lanes << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<BroadcastNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const BroadcastNode*>(node.get());
      (*p) << "x" << op->lanes << "(";
      p->Print(op->value);
      (*p) << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<LetNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const LetNode*>(node.get());
      (*p) << "(let " << op->var << " = ";
      p->Print(op->value);
      (*p) << " in ";
      p->Print(op->body);
      (*p) << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<CallNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const CallNode*>(node.get());
      if (auto* ptr_op = op->op.as<OpNode>()) {
        (*p) << ptr_op->name << "(";
      } else {
        auto* ptr_gvar = op->op.as<GlobalVarNode>();
        ICHECK(ptr_gvar != nullptr);
        (*p) << "@" << ptr_gvar->name_hint << "(";
      }
      for (size_t i = 0; i < op->args.size(); ++i) {
        p->Print(op->args[i]);
        if (i < op->args.size() - 1) {
          (*p) << ", ";
        }
      }
      (*p) << ")";
    });

template <typename T>
void PrintList(const Array<T>& exprs, ReprLegacyPrinter* p) {
  for (size_t i = 0; i < exprs.size(); ++i) {
    p->Print(exprs[i]);
    if (i < exprs.size() - 1) {
      (*p) << ", ";
    }
  }
}

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<ShuffleNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const ShuffleNode*>(node.get());
      (*p) << "shuffle(";
      PrintList(op->vectors, p);
      (*p) << ", ";
      PrintList(op->indices, p);
      (*p) << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<CommReducerNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const CommReducerNode*>(node.get());
      (*p) << "comm_reducer(result=" << op->result << ", lhs=" << op->lhs << ", rhs=" << op->rhs
           << ", identity_element=" << op->identity_element << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<ReduceNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const ReduceNode*>(node.get());
      (*p) << "reduce(combiner=" << op->combiner;
      (*p) << ", source=" << op->source;
      (*p) << ", init=" << op->init;
      (*p) << ", axis=" << op->axis;
      (*p) << ", where=" << op->condition;
      (*p) << ", value_index=" << op->value_index;
      (*p) << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<AnyNode>([](const ObjectRef& node, ReprLegacyPrinter* p) { (*p) << "?"; });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<BufferLoadNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const BufferLoadNode*>(node.get());
      (*p) << op->buffer->name << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) {
          (*p) << ", ";
        }
      }
      (*p) << "]";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<ProducerLoadNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const ProducerLoadNode*>(node.get());
      (*p) << op->producer->GetNameHint() << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) {
          (*p) << ", ";
        }
      }
      (*p) << "]";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<PrimFuncNode>([](const ObjectRef& ref, ReprLegacyPrinter* p) {
      auto* node = static_cast<const PrimFuncNode*>(ref.get());
      (*p) << "PrimFunc(" << node->params << ") ";
      if (node->attrs.defined()) {
        (*p) << "attrs=" << node->attrs;
      }
      (*p) << " {\n";
      p->indent += 2;
      p->Print(node->body);
      p->indent -= 2;
      (*p) << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<LetStmtNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const LetStmtNode*>(node.get());
      p->PrintIndent();
      (*p) << "let " << op->var << " = ";
      p->Print(op->value);
      (*p) << '\n';
      p->Print(op->body);
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<AttrStmtNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const AttrStmtNode*>(node.get());
      p->PrintIndent();
      (*p) << "// attr [";
      p->Print(op->node);
      (*p) << "] " << op->attr_key << " = ";
      p->Print(op->value);
      (*p) << '\n';
      p->Print(op->body);
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<AssertStmtNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const AssertStmtNode*>(node.get());
      p->PrintIndent();
      (*p) << "assert(";
      p->Print(op->condition);
      (*p) << ", ";
      p->Print(op->message);
      (*p) << ")\n";
      p->Print(op->body);
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<ForNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const ForNode*>(node.get());
      p->PrintIndent();
      (*p) << op->kind << " (" << op->loop_var << ", ";
      p->Print(op->min);
      (*p) << ", ";
      p->Print(op->extent);
      (*p) << ") {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      (*p) << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<WhileNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const WhileNode*>(node.get());
      p->PrintIndent();
      (*p) << "while(" << op->condition << ") {\n";
      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;
      p->PrintIndent();
      (*p) << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<ProducerStoreNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const ProducerStoreNode*>(node.get());
      p->PrintIndent();
      (*p) << op->producer->GetNameHint() << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) (*p) << ", ";
      }
      (*p) << "]";
      (*p) << " =";
      p->Print(op->value);
      (*p) << '\n';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<AllocateNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const AllocateNode*>(node.get());
      const auto* ptr_type = op->buffer_var->type_annotation.as<PointerTypeNode>();
      ICHECK(ptr_type) << "The provided variable is not of pointer type";
      p->PrintIndent();
      (*p) << "allocate " << op->buffer_var << "[" << op->dtype;
      for (size_t i = 0; i < op->extents.size(); ++i) {
        (*p) << " * ";
        p->Print(op->extents[i]);
      }
      (*p) << "], storage_scope = " << ptr_type->storage_scope;
      if (!is_one(op->condition)) {
        (*p) << " if ";
        p->Print(op->condition);
      }
      (*p) << "\n";
      p->Print(op->body);
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<AllocateConstNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const AllocateConstNode*>(node.get());
      p->PrintIndent();
      (*p) << "constant " << op->buffer_var << "[" << op->dtype;
      for (size_t i = 0; i < op->extents.size(); ++i) {
        (*p) << " * ";
        p->Print(op->extents[i]);
      }
      (*p) << "]";
      (*p) << "\n";
      p->Print(op->body);
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<DeclBufferNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const DeclBufferNode*>(node.get());
      p->PrintIndent();
      (*p) << "decl_buffer " << op->buffer << "\n";
      (*p) << op->body;
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<ProducerRealizeNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const ProducerRealizeNode*>(node.get());
      p->PrintIndent();
      (*p) << "producer_realize " << op->producer->GetNameHint() << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        (*p) << "[";
        p->Print(op->bounds[i]->min);
        (*p) << ", ";
        p->Print(op->bounds[i]->extent);
        (*p) << "]";
        if (i < op->bounds.size() - 1) (*p) << ", ";
      }
      (*p) << ")";
      if (!is_one(op->condition)) {
        (*p) << " if ";
        p->Print(op->condition);
      }
      (*p) << " {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      (*p) << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<PrefetchNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const PrefetchNode*>(node.get());
      p->PrintIndent();
      (*p) << "prefetch " << op->buffer << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        (*p) << "[";
        p->Print(op->bounds[i]->min);
        (*p) << ", ";
        p->Print(op->bounds[i]->extent);
        (*p) << "]";
        if (i < op->bounds.size() - 1) (*p) << ", ";
      }
      (*p) << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<SeqStmtNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const SeqStmtNode*>(node.get());
      for (Stmt stmt : op->seq) {
        p->Print(stmt);
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<IfThenElseNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const IfThenElseNode*>(node.get());
      p->PrintIndent();
      while (true) {
        (*p) << "if (" << op->condition << ") {\n";
        p->indent += 2;
        p->Print(op->then_case);
        p->indent -= 2;

        if (!op->else_case) {
          break;
        }

        if (const IfThenElseNode* nested_if = op->else_case.as<IfThenElseNode>()) {
          p->PrintIndent();
          (*p) << "} else ";
          op = nested_if;
        } else {
          p->PrintIndent();
          (*p) << "} else {\n";
          p->indent += 2;
          p->Print(op->else_case);
          p->indent -= 2;
          break;
        }
      }
      p->PrintIndent();
      (*p) << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<EvaluateNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const EvaluateNode*>(node.get());
      p->PrintIndent();
      p->Print(op->value);
      (*p) << "\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<BufferStoreNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const BufferStoreNode*>(node.get());
      p->PrintIndent();
      (*p) << op->buffer->name << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) (*p) << ", ";
      }
      (*p) << "]";
      (*p) << " = ";
      p->Print(op->value);
      (*p) << '\n';
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<BufferRealizeNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const BufferRealizeNode*>(node.get());
      p->PrintIndent();
      (*p) << "buffer_realize " << op->buffer->name << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        (*p) << "[";
        p->Print(op->bounds[i]->min);
        (*p) << ", ";
        p->Print(op->bounds[i]->extent);
        (*p) << "]";
        if (i < op->bounds.size() - 1) (*p) << ", ";
      }
      (*p) << ")";
      if (!is_one(op->condition)) {
        (*p) << " if ";
        p->Print(op->condition);
      }
      (*p) << " {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      (*p) << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<BufferRegionNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const BufferRegionNode*>(node.get());
      (*p) << op->buffer->name;
      (*p) << "[";
      for (size_t i = 0; i < op->region.size(); ++i) {
        const auto& range = op->region[i];
        p->Print(range->min);
        if (!is_one(range->extent)) {
          (*p) << ":";
          p->Print(range->min + range->extent);
        }
        if (i != op->region.size() - 1) (*p) << ", ";
      }
      (*p) << "]";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<MatchBufferRegionNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const MatchBufferRegionNode*>(node.get());
      p->PrintIndent();
      (*p) << op->buffer->name << " = match_buffer(";
      p->Print(op->source);
      (*p) << ")\n";
    });

void PrintBlockTitle(const BlockNode* op, ReprLegacyPrinter* p) {
  (*p) << "block " << op->name_hint << "(";
  for (size_t i = 0; i < op->iter_vars.size(); i++) {
    p->Print(op->iter_vars[i]);
    if (i < op->iter_vars.size() - 1) (*p) << ", ";
  }
  (*p) << ")";
}

void PrintBlockSignature(const BlockNode* op, ReprLegacyPrinter* p) {
  // print read/write regions
  p->PrintIndent();
  (*p) << "reads(";
  p->Print(op->reads);
  (*p) << ")\n";
  p->PrintIndent();
  (*p) << "writes(";
  p->Print(op->writes);
  (*p) << ")\n";
  // Print alloc_buffers
  for (const auto& alloc_buf : op->alloc_buffers) {
    p->PrintIndent();
    (*p) << alloc_buf->name << " = alloc_buffer(" << alloc_buf->dtype << "[";
    for (size_t i = 0; i < alloc_buf->shape.size(); ++i) {
      if (i > 0) (*p) << ", ";
      p->Print(alloc_buf->shape[i]);
    }
    (*p) << "])\n";
  }
  // Print match_buffer_regions
  for (const auto& match_buf : op->match_buffers) {
    p->Print(match_buf);
  }
  if (!op->annotations.empty()) {
    p->PrintIndent();
    (*p) << "annotations(" << op->annotations << ")\n";
  }
}

void PrintBlockBody(const BlockNode* op, ReprLegacyPrinter* p) {
  // Print init
  if (op->init.defined()) {
    p->PrintIndent();
    (*p) << "with init() {\n";
    p->indent += 2;
    p->Print(op->init.value());
    p->indent -= 2;
    p->PrintIndent();
    (*p) << "}\n";
  }
  // Print body
  p->Print(op->body);
}

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<BlockNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const BlockNode*>(node.get());
      p->PrintIndent();
      PrintBlockTitle(op, p);
      (*p) << " {\n";
      p->indent += 2;

      // Print block elements (e.g. reads/writes, etc)
      PrintBlockSignature(op, p);
      // Print block init and body
      PrintBlockBody(op, p);

      p->indent -= 2;
      p->PrintIndent();
      (*p) << "}\n";
    });

TVM_STATIC_IR_FUNCTOR(ReprLegacyPrinter, vtable)
    .set_dispatch<BlockRealizeNode>([](const ObjectRef& node, ReprLegacyPrinter* p) {
      auto* op = static_cast<const BlockRealizeNode*>(node.get());
      auto* block_op = op->block.get();
      p->PrintIndent();
      PrintBlockTitle(block_op, p);
      (*p) << " {\n";
      p->indent += 2;

      // Print binding iter_values
      for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
        p->PrintIndent();
        (*p) << "bind(";
        p->Print(block_op->iter_vars[i]->var);
        (*p) << ", ";
        p->Print(op->iter_values[i]);
        (*p) << ")\n";
      }
      // Print predicate
      if (!is_one(op->predicate)) {
        p->PrintIndent();
        (*p) << "where(";
        p->Print(op->predicate);
        (*p) << ")\n";
      }
      // Print block elements (e.g. reads/writes, etc)
      PrintBlockSignature(block_op, p);
      // Print block init and body
      PrintBlockBody(block_op, p);

      p->indent -= 2;
      p->PrintIndent();
      (*p) << "}\n";
    });

}  // namespace tir
}  // namespace tvm
