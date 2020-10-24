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
 * \file src/relay/analysis/call_graph.h
 * \brief Define data structures for the call graph of a IRModule. It borrows
 * the idea how LLVM constructs CallGraph.
 *
 * https://llvm.org/doxygen/CallGraph_8h_source.html
 */

#ifndef TVM_RELAY_ANALYSIS_CALL_GRAPH_H_
#define TVM_RELAY_ANALYSIS_CALL_GRAPH_H_

#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/runtime/object.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {

class CallGraphEntry;
class CallGraph;

class CallGraphNode : public Object {
  using CallGraphMap =
      std::unordered_map<GlobalVar, std::unique_ptr<CallGraphEntry>, ObjectPtrHash, ObjectPtrEqual>;
  // Create iterator alias for a CallGraphNode object.
  using iterator = CallGraphMap::iterator;
  using const_iterator = CallGraphMap::const_iterator;

 public:
  /*! \brief The IR module for creating a CallGraphNode. */
  IRModule module;

  /*! \brief Default constructor. */
  CallGraphNode() {}

  void VisitAttrs(AttrVisitor* v) { v->Visit("module", &module); }

  /*!
   * \brief Print the call graph.
   *
   * \param os The stream for printing.
   */
  void Print(std::ostream& os) const;

  /*! \return The begin iterator. */
  iterator begin() { return call_graph_.begin(); }
  /*! \return The end iterator. */
  iterator end() { return call_graph_.end(); }
  /*! \return The begin iterator. */
  const_iterator begin() const { return call_graph_.begin(); }
  /*! \return The end iterator. */
  const_iterator end() const { return call_graph_.end(); }

  /*!
   * \brief Get an element from the CallGraphNode using a GlobalVar.
   *
   * \param gv The GlobalVar used for indexing.
   *
   * \return The fetched element.
   */
  const CallGraphEntry* operator[](const GlobalVar& gv) const;
  /*!
   * \brief Get an element from the CallGraphNode using a GlobalVar.
   *
   * \param gv The GlobalVar used for indexing.
   *
   * \return The fetched element.
   */
  CallGraphEntry* operator[](const GlobalVar& gv);
  /*!
   * \brief Get an element from the CallGraphNode using the global function name.
   *
   * \param gvar_name The global function name used for indexing.
   *
   * \return The fetched element.
   */
  const CallGraphEntry* operator[](const std::string& gvar_name) const {
    return (*this)[module->GetGlobalVar(gvar_name)];
  }
  /*!
   * \brief Get an element from the CallGraphNode using the global function name.
   *
   * \param gvar_name The global function name used for indexing.
   *
   * \return The fetched element.
   */
  CallGraphEntry* operator[](const std::string& gvar_name) {
    return (*this)[module->GetGlobalVar(gvar_name)];
  }

  /*!
   * \brief Get the global function corresponding to the variable.
   *
   * \param var The global variable.
   *
   * \return The found global function.
   */
  BaseFunc GetGlobalFunction(const GlobalVar& var) const;

  /*!
   * \brief Get the entries/root nodes of CallGraphNode.
   *
   *  Entry functions are never referenced by other functions.
   *  Note these functions can be recursive as well.
   *
   * \return The list of CallGraphEntry that represent entry nodes.
   */
  std::vector<CallGraphEntry*> GetEntryGlobals() const;

  /*!
   * \brief Remove a GlobalVar in a given CallGraphEntry from the current
   *        IR module.
   *
   * \param cg_node The CallGraphEntry that contains a global function to be
   *        removed.
   * \param update_call_graph Indicate if we will update the CallGraph as well
   *        since updating is costly. We are only able to remove a leaf function
   *        when update_call_graph is disabled because the edges pointing to
   *        functions being removed are not updated.
   *
   * \return The GlobalVar removed from the current module.
   */
  GlobalVar RemoveGlobalVarFromModule(CallGraphEntry* cg_node, bool update_call_graph = false);

  /*!
   * \brief Lookup a GlobalVar for the CallGraphNode. It creates an entry for
   *        the GlobalVar if it doesn't exist.
   *
   * \param gv The GlobalVar for query.
   *
   * \return The queried entry.
   */
  CallGraphEntry* LookupGlobalVar(const GlobalVar& gv);

  /*!
   * \brief Get the entries from the CallGraphNode in the topological order.
   *
   *  This is useful for various module-level optimizations/analysis. For example,
   *  inlining requires the correct order of the functions being processed, i.e.
   *  callee should be always handled before callers.
   *
   * \return The list of collected entries that are sorted in the topological order.
   */
  std::vector<CallGraphEntry*> TopologicalOrder() const;

  static constexpr const char* _type_key = "relay.CallGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallGraphNode, Object);

 private:
  /*!
   * \brief Create a CallGraphEntry for a global function and add it to the
   *        CallGraphNode.
   *
   * \param gv The global var.
   * \param func The global function corresponding to `gv`.
   */
  void AddToCallGraph(const GlobalVar& gv, const Function& func);

  /*! \brief A record contains GlobalVar to CallGraphEntry mapping. */
  CallGraphMap call_graph_;

  friend CallGraph;
};

/*!
 * \brief The class that represents the call graph of a Relay IR module. It also
 * provides a variety of utility functions for users to query, view, and update
 * a call graph.
 */
class CallGraph : public ObjectRef {
  using CallGraphMap =
      std::unordered_map<GlobalVar, std::unique_ptr<CallGraphEntry>, ObjectPtrHash, ObjectPtrEqual>;
  // Create iterator alias for a CallGraph object.
  using iterator = CallGraphMap::iterator;
  using const_iterator = CallGraphMap::const_iterator;

 public:
  /*!
   * \brief Construct a CallGraph from a IR module.
   *
   * \param module The IR module
   */
  explicit CallGraph(IRModule module);

  /*!
   * \brief Construct from an object pointer.
   * \param n The object pointer.
   */
  explicit CallGraph(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*! \return The begin iterator. */
  iterator begin() {
    auto* n = operator->();
    ICHECK(n);
    return n->begin();
  }
  /*! \return The end iterator. */
  iterator end() {
    auto* n = operator->();
    ICHECK(n);
    return n->end();
  }
  /*! \return The begin iterator. */
  const_iterator begin() const {
    const auto* n = operator->();
    ICHECK(n);
    return n->begin();
  }
  /*! \return The end iterator. */
  const_iterator end() const {
    const auto* n = operator->();
    ICHECK(n);
    return n->end();
  }

  /*!
   * \brief Get an element from the CallGraph using a GlobalVar.
   *
   * \param gv The GlobalVar used for indexing.
   *
   * \return The fetched element.
   */
  const CallGraphEntry* operator[](const GlobalVar& gv) const {
    const auto* n = operator->();
    ICHECK(n);
    return (*n)[gv];
  }
  /*!
   * \brief Get an element from the CallGraph using a GlobalVar.
   *
   * \param gv The GlobalVar used for indexing.
   *
   * \return The fetched element.
   */
  CallGraphEntry* operator[](const GlobalVar& gv) {
    auto* n = operator->();
    ICHECK(n);
    return (*n)[gv];
  }
  /*!
   * \brief Get an element from the CallGraph using the global function name.
   *
   * \param gvar_name The global function name used for indexing.
   *
   * \return The fetched element.
   */
  const CallGraphEntry* operator[](const std::string& gvar_name) const {
    const auto* n = operator->();
    ICHECK(n);
    return (*n)[gvar_name];
  }
  /*!
   * \brief Get an element from the CallGraph using the global function name.
   *
   * \param gvar_name The global function name used for indexing.
   *
   * \return The fetched element.
   */
  CallGraphEntry* operator[](const std::string& gvar_name) {
    auto* n = operator->();
    ICHECK(n);
    return (*n)[gvar_name];
  }

  /*! \return mutable pointers to the node. */
  CallGraphNode* operator->() const {
    auto* ptr = get_mutable();
    ICHECK(ptr != nullptr);
    return static_cast<CallGraphNode*>(ptr);
  }

 private:
  /*! \brief Overload the << operator to print a call graph. */
  friend std::ostream& operator<<(std::ostream& os, const CallGraph&);
};

/*!
 * \brief A node in the call graph. It maintains the edges from a caller to
 * all callees.
 */
class CallGraphEntry {
 public:
  using CallGraphEntryPair = std::pair<GlobalVar, CallGraphEntry*>;
  using CallGraphEntryVector = std::vector<CallGraphEntryPair>;
  using CallGraphEntrySet = std::unordered_set<const CallGraphEntry*>;
  // Create iterator alias for a CallGraphEntry object.
  using iterator = std::vector<CallGraphEntryPair>::iterator;
  using const_iterator = std::vector<CallGraphEntryPair>::const_iterator;

  /*!
   * \brief Construct from a GlobalVar.
   *
   * \param gv The GlobalVar to create a CallGraphEntry.
   */
  explicit CallGraphEntry(const GlobalVar& gv) : global_(gv) {}
  /*!
   * \brief Delete copy constructor.
   */
  CallGraphEntry(const CallGraphEntry&) = delete;
  /*! \brief Delete assignment. */
  CallGraphEntry& operator=(const CallGraphEntry&) = delete;

  /*! \return The begin iterator */
  iterator begin() { return called_globals_.begin(); }
  /*! \return The end iterator */
  iterator end() { return called_globals_.end(); }
  /*! \return The const begin iterator */
  const_iterator begin() const { return called_globals_.begin(); }
  /*! \return The const end iterator */
  const_iterator end() const { return called_globals_.end(); }

  /*!
   * \brief Return if the list of called nodes is empty.
   *
   * \return true if the list is empty. Otherwise, false.
   */
  bool empty() const { return called_globals_.empty(); }

  /*!
   * \brief Return the size of the list that represents the nodes are called by
   * the current node.
   *
   * \return The number of called nodes.
   */
  uint32_t size() const { return static_cast<uint32_t>(called_globals_.size()); }

  /*!
   * \brief Fetch the i-th CallGraphEntry from the list of nodes that are called
   * by the current function.
   *
   * \param i The index.
   *
   * \return The fetched CallGraphEntry.
   */
  CallGraphEntry* operator[](size_t i) const {
    ICHECK_LT(i, called_globals_.size()) << "Invalid Index";
    return called_globals_[i].second;
  }

  /*!
   * \brief Print the call graph that is stemmed from the current CallGraphEntry.
   *
   * \param os The stream for printing.
   */
  void Print(std::ostream& os) const;

  /*!
   * \brief Return the number of times the global function is referenced.
   *
   * \return The count.
   */
  uint32_t GetRefCount() const { return ref_cnt_; }

  /*!
   * \brief Return the GlobalVar stored in the current CallGraphEntry.
   *
   * \return The GlobalVar.
   */
  GlobalVar GetGlobalVar() const { return global_; }

  /*!
   * \brief Return the name hint of the GlobalVar stored in the CallGraphEntry.
   *
   * \return The name hint of the global function.
   */
  std::string GetNameHint() const { return global_->name_hint; }

  /*!
   * \brief Return if the global function corresponding to the current
   * CallGraphEntry is a recursive function.
   *
   * \return true if it is recursive. Otherwise, false.
   */
  bool IsRecursive() const { return is_recursive_; }

  /*!
   * \brief Return if the global function corresponding to the current
   * CallGraphEntry is both a recursive function and an entry function. This type
   * of function only has one reference which is called by itself.
   *
   * \return true if it is both a recursive function and an entry. Otherwise, false.
   */
  bool IsRecursiveEntry() const { return GetRefCount() == 1 && IsRecursive(); }

  /*!
   * \brief Return the topological order of the CallGraphEntry.
   *
   * \param visited A set of CallGraphEntry objects that have been visited.
   *
   * \return The list of CallGraphEntry that is represented in topological order.
   */
  std::vector<CallGraphEntry*> TopologicalOrder(
      CallGraphEntrySet* visited = new CallGraphEntrySet()) const;

  /*!
   * \brief Remove all edges from the current CallGraphEntry to any global
   * function it calls.
   */
  void CleanCallGraphEntries();

  /*!
   * \brief Add a node to the list of nodes that are being called by the current
   * global function.
   *
   * \param cg_node The CallGraphEntry that will be added to the call list.
   */
  void AddCalledGlobal(CallGraphEntry* cg_node);

  /*!
   * \brief Remove a call edge to the global function from the current
   * function.
   *
   * \param callee The function that is being called.
   */
  void RemoveCallTo(const GlobalVar& callee);

  /*!
   * \brief Remove all the edges that represent that calls to the global function
   * stored in a given CallGraphEntry.
   *
   * \param callee The function that is being called.
   */
  void RemoveAllCallTo(CallGraphEntry* callee);

 private:
  /*! \brief Decrement the reference counter by 1. */
  void DecRef() {
    ICHECK_GT(ref_cnt_, 0);
    --ref_cnt_;
  }
  /*! \brief Increment the reference counter by 1. */
  void IncRef() { ++ref_cnt_; }

  /*!
   * \brief Mark if the global function stored in the CallGraphEntry is
   * recursive function.
   */
  bool is_recursive_{false};
  /*! \brief Count the number of times the global function is referenced. */
  uint32_t ref_cnt_{0};
  /*! \brief The GlobalVar stored in the current CallGraphEntry. */
  GlobalVar global_;
  /*! \brief The list of entries called by the current CallGraphEntry. */
  CallGraphEntryVector called_globals_;

  friend class CallGraph;
  /*! \brief Overload the << operator to print a call graph node. */
  friend std::ostream& operator<<(std::ostream& os, const CallGraphEntry&);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ANALYSIS_CALL_GRAPH_H_
