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

/*!
 * \file src/tvm/relay/dataflow_matcher.cc
 * \brief The dataflow pattern matcher for Relay.
 */

#include <tvm/relay/dataflow_pattern_functor.h>

namespace tvm {
namespace relay {

// DFPatternVisitor

void DFPatternVisitor::VisitDFPattern(const DFPattern& pattern) {
  if (this->visited_.count(pattern.get()) == 0) {
    visited_.insert(pattern.get());
    DFPatternFunctor::VisitDFPattern(pattern);
  }
}

void DFPatternVisitor::VisitDFPattern_(const AltPatternNode* op) {
  VisitDFPattern(op->left);
  VisitDFPattern(op->right);
}

void DFPatternVisitor::VisitDFPattern_(const AttrPatternNode* op) { VisitDFPattern(op->pattern); }

void DFPatternVisitor::VisitDFPattern_(const CallPatternNode* op) {
  VisitDFPattern(op->op);
  if (op->args.defined()) {
    for (auto arg : op->args) {
      VisitDFPattern(arg);
    }
  }
}

void DFPatternVisitor::VisitDFPattern_(const DataTypePatternNode* op) {
  VisitDFPattern(op->pattern);
}

void DFPatternVisitor::VisitDFPattern_(const DominatorPatternNode* op) {
  VisitDFPattern(op->parent);
  VisitDFPattern(op->path);
  VisitDFPattern(op->child);
}

void DFPatternVisitor::VisitDFPattern_(const ExprPatternNode* op) {}

void DFPatternVisitor::VisitDFPattern_(const FunctionPatternNode* op) {
  if (op->params.defined()) {
    for (auto param : op->params) {
      VisitDFPattern(param);
    }
  }
  VisitDFPattern(op->body);
}

void DFPatternVisitor::VisitDFPattern_(const ShapePatternNode* op) { VisitDFPattern(op->pattern); }

void DFPatternVisitor::VisitDFPattern_(const TupleGetItemPatternNode* op) {
  VisitDFPattern(op->tuple);
}

void DFPatternVisitor::VisitDFPattern_(const TuplePatternNode* op) {
  if (op->fields.defined()) {
    for (auto field : op->fields) {
      VisitDFPattern(field);
    }
  }
}

void DFPatternVisitor::VisitDFPattern_(const IfPatternNode* op) {
  VisitDFPattern(op->cond);
  VisitDFPattern(op->true_branch);
  VisitDFPattern(op->false_branch);
}

void DFPatternVisitor::VisitDFPattern_(const LetPatternNode* op) {
  VisitDFPattern(op->var);
  VisitDFPattern(op->value);
  VisitDFPattern(op->body);
}

void DFPatternVisitor::VisitDFPattern_(const TypePatternNode* op) { VisitDFPattern(op->pattern); }

void DFPatternVisitor::VisitDFPattern_(const VarPatternNode* op) {}

void DFPatternVisitor::VisitDFPattern_(const ConstantPatternNode* op) {}

void DFPatternVisitor::VisitDFPattern_(const WildcardPatternNode* op) {
  if (op->pattern) {
    VisitDFPattern(op->pattern.value());
  }
}

}  // namespace relay
}  // namespace tvm
