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

/*! \file stmt_extension.h
 * \brief Narrow operation contexts for dialect-owned statement transform hooks.
 */
#ifndef TVM_TIRX_TRANSFORM_STMT_EXTENSION_H_
#define TVM_TIRX_TRANSFORM_STMT_EXTENSION_H_

#include <functional>
#include <vector>

#include "../ir_mutator_with_analyzer.h"
#include "ir_utils.h"

namespace tvm {
namespace tirx {

// Expose only the scope and remap operations needed at extension definition sites.
class SSAStmtMutator : public StmtExprMutator {
 public:
  using StmtExprMutator::VTable;
  static void RegisterExtension(void (*init)(VTable*)) { Extensions().push_back(init); }
  SSAStmtMutator() : StmtExprMutator(GlobalVTable()) {}
  virtual Stmt WithScope(const std::function<Stmt()>& body) = 0;
  virtual Var DefineVar(Var var) = 0;
  virtual BufferVar RemapBuffer(BufferVar buffer) = 0;

 protected:
  static void InitVTable(VTable* table) {
    StmtExprMutator::InitVTable(table);
    for (auto init : Extensions()) init(table);
  }

 private:
  static std::vector<void (*)(VTable*)>& Extensions() {
    static std::vector<void (*)(VTable*)> extensions;
    return extensions;
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
};

// Buffer geometry stays in the flattener; extensions identify definitions and regions.
class FlattenStmtMutator : public IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::VTable;
  static void RegisterExtension(void (*init)(VTable*)) { Extensions().push_back(init); }
  explicit FlattenStmtMutator(const arith::Analyzer& analyzer)
      : IRMutatorWithAnalyzer(analyzer.get(), GlobalVTable()) {}
  virtual BufferVar DefineBuffer(BufferVar buffer) = 0;
  virtual BufferRegion RewriteRegion(BufferRegion region) = 0;

 protected:
  static void InitVTable(VTable* table) {
    IRMutatorWithAnalyzer::InitVTable(table);
    for (auto init : Extensions()) init(table);
  }

 private:
  static std::vector<void (*)(VTable*)>& Extensions() {
    static std::vector<void (*)(VTable*)> extensions;
    return extensions;
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
};

class IndexDomainVisitor : public StmtExprVisitor {
 public:
  using StmtExprVisitor::VTable;
  static void RegisterExtension(void (*init)(VTable*)) { Extensions().push_back(init); }
  IndexDomainVisitor() : StmtExprVisitor(GlobalVTable()) {}
  virtual void BindDomain(const Var& var, const Range& domain) = 0;

 protected:
  static void InitVTable(VTable* table) {
    StmtExprVisitor::InitVTable(table);
    for (auto init : Extensions()) init(table);
  }

 private:
  static std::vector<void (*)(VTable*)>& Extensions() {
    static std::vector<void (*)(VTable*)> extensions;
    return extensions;
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
};

class StorageAlignVisitor : public StmtExprVisitor {
 public:
  using StmtExprVisitor::VTable;
  static void RegisterExtension(void (*init)(VTable*)) { Extensions().push_back(init); }
  StorageAlignVisitor() : StmtExprVisitor(GlobalVTable()) {}
  virtual void RecordAlignment(const Var& buffer, const StorageAlignTuple& annotation) = 0;

 protected:
  static void InitVTable(VTable* table) {
    StmtExprVisitor::InitVTable(table);
    for (auto init : Extensions()) init(table);
  }

 private:
  static std::vector<void (*)(VTable*)>& Extensions() {
    static std::vector<void (*)(VTable*)> extensions;
    return extensions;
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
};

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_TRANSFORM_STMT_EXTENSION_H_
