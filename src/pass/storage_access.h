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

using runtime::StorageScope;
using runtime::StorageRank;
/*!
 * \brief Base class of storage access analysis
 */
class StorageAccessVisitor : public IRVisitor {
 public:
  /*! \brief Storage access type */
  enum AccessType {
    kRead,
    kWrite,
    kSync,
    kAlloc,
    // acquired version of read, only need to handle WAR dep.
    kReadAcquire
  };
  /*! \brief An access entry */
  struct AccessEntry {
    /*! \brief The thread index that access this entry */
    Array<IterVar> threads;
    /*! \brief The buffer variable, if any */
    VarExpr buffer;
    /*! \brief The access data type */
    Type dtype;
    /*! \brief The touched access range */
    arith::IntSet touched;
    /*! \brief The type of access */
    AccessType type;
    /*! \brief The storage scope */
    StorageScope scope;
    /*! \brief Whether the access is double buffer write */
    bool double_buffer_write{false};
  };
  /*! \brief Access pattern about a single statement */
  struct StmtEntry {
    /*! \brief The statement */
    const Node* stmt;
    /*! \brief access patterns in the statement */
    std::vector<AccessEntry> access;
  };
  // override visitor pattern
  void Visit_(const Load* op) final;
  void Visit_(const Store* op) final;
  void Visit_(const Evaluate* op) final;
  void Visit_(const AttrStmt* op) final;
  void Visit_(const For* op) final;
  void Visit_(const IfThenElse* op) final;
  void Visit_(const Call* op) final;

 protected:
  StorageAccessVisitor() {
    scope_.push_back(std::vector<StmtEntry>());
  }
  /*! \return number of conditions in the current scope. */
  int condition_counter() const {
    return condition_counter_;
  }
  /*! \return whether we are in device environment. */
  bool in_device_env() const {
    return in_device_env_;
  }
  /*! \return environment threads */
  const Array<IterVar>& env_threads() const {
    return env_threads_;
  }
  /*!
   * \brief Whether we need analyze the buffer in current scope.
   * \param buffer The buffer to be checked
   * \param scope The scope of the buffer.
   * \return Whether the analysis of buffer is enabled.
   */
  virtual bool Enabled(const Variable* buffer,
                       const StorageScope& scope) const {
    return true;
  }
  /*!
   * \brief Summarize the sequence of operations into parent.
   *
   *  Insert synchronization if necessary and remove un-necessary
   *  memory access which are already synced.
   *
   * \param seq The sequence of the access operations.
   * \param loop Pass loop node if it is a loop, otherwise nullptr.
   * \return The summarized sequence that represent access that
   *  the parent should taken care of to synchronize.
   */
  virtual std::vector<AccessEntry> Summarize(
      std::vector<StmtEntry> seq, const For* loop) = 0;
  /*!
   * \brief Get the scope of the buffer array.
   * \return The scope of the final buffer array.
   */
  StorageScope GetScope(const Variable* buf) const;
  // access scope
  std::vector<std::vector<StmtEntry> > scope_;

 private:
  // whether access appending is enabled.
  bool allow_append_{false};
  // Whether we are in device environment
  bool in_device_env_{false};
  // Whether we are inside condition.
  int condition_counter_{0};
  // The current double buffer write scope.
  const Variable* double_buffer_write_{nullptr};
  // the current free stmt entry.
  StmtEntry curr_stmt_;
  // The involving threads
  Array<IterVar> env_threads_;
  // The storage scope of each buffer
  std::unordered_map<const Variable*, StorageScope> storage_scope_;
};

}  // namespace ir
}  // namespace tvm
#endif  // TVM_PASS_STORAGE_ACCESS_H_
