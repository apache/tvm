/*!
 * Copyright (c) 2017 by Contributors
 * \file storage_access.h
 * \brief Common data structure for storage access analysis.
 */
#ifndef TVM_PASS_STORAGE_ACCESS_H_
#define TVM_PASS_STORAGE_ACCESS_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <vector>
#include <unordered_map>
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {
namespace storage {
// The storage scope.
using runtime::StorageScope;
/*! \brief Storage access type */
enum AccessType {
  kRead,
  kWrite,
  kOpaque,
  kSync,
  kAlloc
};
/*! \brief The access entry */
struct AccessEntry {
  /*! \brief The buffer variable, if any */
  const Variable* buffer{nullptr};
  /*! \brief The access index */
  Expr index;
  /*! \brief The type of access */
  AccessType type;
  /*! \brief The storage scope */
  StorageScope scope;
  // constructor
  AccessEntry() {}
  AccessEntry(const Variable* buffer,
              Expr index,
              AccessType type,
              StorageScope scope)
      : buffer(buffer), index(index), type(type), scope(scope) {}
};
/*! \brief The access info about a statment */
struct StmtEntry {
  /*! \brief The statement */
  const Node* stmt;
  /*! \brief access patterns in the statement */
  std::vector<AccessEntry> access;
};
}  // namespace storage
}  // namespace ir
}  // namespace tvm
#endif  // TVM_PASS_STORAGE_ACCESS_H_
