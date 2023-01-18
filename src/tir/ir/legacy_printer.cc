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
namespace tir {

std::string LegacyTIRPrint(const ObjectRef& obj) {
  using namespace tvm::tir;
  class LegacyTIRPrinter : private tir::ExprVisitor {
   public:
    explicit LegacyTIRPrinter(std::ostream& os) : stream(os) {}

    void Print(const ObjectRef& obj) {
      if (const auto* op = obj.as<CommReducerNode>()) {
        Print_(op);
      } else if (const auto* op = obj.as<IterVarNode>()) {
        Print_(op);
      } else if (const auto* op = obj.as<RangeNode>()) {
        Print_(op);
      } else if (const auto* op = obj.as<OpNode>()) {
        Print_(op);
      } else {
        VisitExpr(Downcast<PrimExpr>(obj));
      }
    }

   private:
    void VisitExpr_(const VarNode* op) final { stream << op->name_hint; }

    void VisitExpr_(const SizeVarNode* op) final {
      stream << "{" << op->name_hint << "|" << op->name_hint << ">=0}";
    }

    void VisitExpr_(const IntImmNode* op) final {
      if (op->dtype == DataType::Int(32)) {
        stream << op->value;
      } else {
        stream << "(" << op->dtype << ")" << op->value;
      }
    }

    void VisitExpr_(const FloatImmNode* op) final {
      switch (op->dtype.bits()) {
        case 64:
          stream << op->value;
          break;
        case 32:
          stream << op->value << 'f';
          break;
        case 16:
          stream << op->value << 'h';
          break;
        default:
          LOG(FATAL) << "Unknown float type bits=" << op->dtype.bits();
      }
    }
    void VisitExpr_(const StringImmNode* op) final {
      stream << '\"' << support::StrEscape(op->value) << '\"';
    }
    void VisitExpr_(const CastNode* op) final {
      stream << op->dtype << '(';
      VisitExpr(op->value);
      stream << ')';
    }
    void VisitExpr_(const AddNode* op) final { PrintBinary(op->a, op->b, " + "); }
    void VisitExpr_(const SubNode* op) final { PrintBinary(op->a, op->b, " - "); }
    void VisitExpr_(const MulNode* op) final { PrintBinary(op->a, op->b, "*"); }
    void VisitExpr_(const DivNode* op) final { PrintBinary(op->a, op->b, "/"); }
    void VisitExpr_(const ModNode* op) final { PrintBinary(op->a, op->b, " % "); }
    void VisitExpr_(const FloorDivNode* op) final { PrintCall("floordiv", op->a, op->b); }
    void VisitExpr_(const FloorModNode* op) final { PrintCall("floormod", op->a, op->b); }
    void VisitExpr_(const MinNode* op) final { PrintCall("min", op->a, op->b); }
    void VisitExpr_(const MaxNode* op) final { PrintCall("max", op->a, op->b); }
    void VisitExpr_(const EQNode* op) final { PrintBinary(op->a, op->b, " == "); }
    void VisitExpr_(const NENode* op) final { PrintBinary(op->a, op->b, " != "); }
    void VisitExpr_(const LTNode* op) final { PrintBinary(op->a, op->b, " < "); }
    void VisitExpr_(const LENode* op) final { PrintBinary(op->a, op->b, " <= "); }
    void VisitExpr_(const GTNode* op) final { PrintBinary(op->a, op->b, " > "); }
    void VisitExpr_(const GENode* op) final { PrintBinary(op->a, op->b, " >= "); }
    void VisitExpr_(const AndNode* op) final { PrintBinary(op->a, op->b, " && "); }
    void VisitExpr_(const OrNode* op) final { PrintBinary(op->a, op->b, " || "); }

    void VisitExpr_(const NotNode* op) final {
      stream << "!";
      VisitExpr(op->a);
    }

    void VisitExpr_(const SelectNode* op) final {
      stream << "select(";
      VisitExpr(op->condition);
      stream << ", ";
      VisitExpr(op->true_value);
      stream << ", ";
      VisitExpr(op->false_value);
      stream << ')';
    }

    void VisitExpr_(const RampNode* op) final {
      stream << "ramp(";
      VisitExpr(op->base);
      stream << ", ";
      VisitExpr(op->stride);
      stream << ", " << op->lanes << ')';
    }

    void VisitExpr_(const BroadcastNode* op) final {
      stream << "x" << op->lanes << "(";
      VisitExpr(op->value);
      stream << ")";
    }

    void VisitExpr_(const LetNode* op) final {
      stream << "(let " << op->var << " = ";
      VisitExpr(op->value);
      stream << " in ";
      VisitExpr(op->body);
      stream << ")";
    }

    void VisitExpr_(const CallNode* op) final {
      if (auto* ptr_op = op->op.as<OpNode>()) {
        stream << ptr_op->name << "(";
      } else {
        auto* p = op->op.as<GlobalVarNode>();
        ICHECK(p != nullptr);
        stream << "@" << p->name_hint << "(";
      }
      for (size_t i = 0; i < op->args.size(); ++i) {
        VisitExpr(op->args[i]);
        if (i < op->args.size() - 1) {
          stream << ", ";
        }
      }
      stream << ")";
    }

    void VisitExpr_(const ShuffleNode* op) final {
      stream << "shuffle(";
      PrintList(op->vectors.GetArrayNode());
      stream << ", ";
      PrintList(op->indices.GetArrayNode());
      stream << ")";
    }

    void VisitExpr_(const ReduceNode* op) final {
      stream << "reduce(combiner=";
      Print_(op->combiner.get());
      stream << ", source=";
      PrintList(op->source.GetArrayNode());
      stream << ", init=";
      PrintList(op->init.GetArrayNode());
      stream << ", axis=";
      PrintList(op->axis.GetArrayNode());
      stream << ", where=";
      VisitExpr(op->condition);
      stream << ", value_index=" << op->value_index;
      stream << ")";
    }

    void VisitExpr_(const AnyNode* op) final { stream << "?"; }

    void VisitExpr_(const BufferLoadNode* op) final {
      stream << op->buffer->name << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        VisitExpr(op->indices[i]);
        if (i < op->indices.size() - 1) {
          stream << ", ";
        }
      }
      stream << "]";
    }

    void VisitExpr_(const ProducerLoadNode* op) final {
      stream << op->producer->GetNameHint() << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        VisitExpr(op->indices[i]);
        if (i < op->indices.size() - 1) {
          stream << ", ";
        }
      }
      stream << "]";
    }

   private:
    void Print_(const CommReducerNode* op) {
      stream << "comm_reducer(result=";
      PrintList(op->result.GetArrayNode());
      stream << ", lhs=";
      PrintList(op->lhs.GetArrayNode());
      stream << ", rhs=";
      PrintList(op->rhs.GetArrayNode());
      stream << ", identity_element=";
      PrintList(op->identity_element.GetArrayNode());
      stream << ")";
    }

    void Print_(const IterVarNode* op) {
      stream << "{" << op->var->name_hint << "|" << op->var->name_hint << " in [";
      VisitExpr(op->dom->min);
      stream << ", ";
      VisitExpr(op->dom->extent);
      stream << ")}";
    }

    void Print_(const RangeNode* op) {
      stream << "range(min=" << op->min << ", ext=" << op->extent << ')';
    }

    void Print_(const OpNode* op) { stream << "Op(" << op->name << ")"; }

   private:
    void PrintBinary(const PrimExpr& a, const PrimExpr& b, const std::string& sign) {
      stream << '(';
      VisitExpr(a);
      stream << sign;
      VisitExpr(b);
      stream << ')';
    }

    void PrintCall(const std::string& call, const PrimExpr& a, const PrimExpr& b) {
      stream << call << '(';
      VisitExpr(a);
      stream << ", ";
      VisitExpr(b);
      stream << ')';
    }

    void PrintList(const ArrayNode* exprs) {
      int n = static_cast<int>(exprs->size());
      for (int i = 0; i < n; ++i) {
        VisitExpr(Downcast<PrimExpr>(exprs->at(i)));
        if (i < n - 1) {
          stream << ", ";
        }
      }
    }

    std::ostream& stream;
  };
  std::ostringstream os;
  LegacyTIRPrinter(os).Print(obj);
  return os.str();
}

}  // namespace tir
}  // namespace tvm
