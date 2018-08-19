/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/compiler/intern_table.h
 * \brief A table which maps string keys to data.
 *
 * These are useful for mapping user-readable names
 * to globally unique allocations which use pointer
 * equality for comparsion.
 */
#ifndef TVM_RELAY_COMPILER_INTERN_TABLE_H_
#define TVM_RELAY_COMPILER_INTERN_TABLE_H_

#include <string>
#include <unordered_map>
#include "dmlc/logging.h"

namespace tvm {
namespace relay {

struct KeyNotFound : dmlc::Error {
  explicit KeyNotFound(std::string msg) : dmlc::Error(msg) {}
};

template <typename T>
class InternTable {
private:
  /*! \brief The internal table mapping from strings to T. */
  std::unordered_map<std::string, T> table_;

 public:
  /*! \brief Insert a new key into the table.
   * \note Attempting to reinsert a key triggers an error.
   */
  void Insert(const std::string& key, const T& value) {
    if (table_.find(key) == table_.end()) {
      table_.insert({key, value});
    } else {
      throw dmlc::Error(
          std::string("you have previously interred a value for: ") + key);
    }
  }

  /*! \brief Lookup the data in the table. */
  const T& Lookup(std::string key) {
    if (table_.find(key) != table_.end()) {
      return table_.at(key);
    } else {
      throw KeyNotFound(std::string("could not find match") + key);
    }
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_COMPILER_INTERN_TABLE_H_
