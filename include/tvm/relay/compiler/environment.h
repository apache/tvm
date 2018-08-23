/*!
 *  Copyright (c) 2018 by Contributors
 * \file environment.h
 * \brief The global environment containing
 */
#ifndef TVM_RELAY_ENVIRONMENT_H_
#define TVM_RELAY_ENVIRONMENT_H_

#include <string>
#include <vector>
#include "tvm/relay/compiler/intern_table.h"
#include "../expr.h"
#include "../type.h"
#include "../op.h"
#include "../error.h"
// #include "tvm/relay/options.h"
// #include "tvm/relay/source_map.h"

namespace tvm {
namespace relay {

struct Environment;

/*! \brief The global environment of Relay programs.
 *
 *  The global environment contains all the global
 *  information needed to compile a Relay program,
 *  including the set of operators, the set of
 *  global functions, and configuration options.
 *
 *  Many operations require acess to the global
 *  Environment. We mostly pass the argument by value
 *  in a functional style as an explicit argument.
 *
 *  This means users can construct custom environments
 *  easily, for example a fresh environment for each
 *  thread while auto-tuning.
 * */

class EnvironmentNode : public RelayNode {
 private:
  /*! A map from string names to GlobalIds, ensures global uniqueness. */
  InternTable<GlobalVar> global_map_;

  // /*! \brief A map from file names to source fragments. */
  // SourceMap source_map_
  // /*! \brief A list of the errors reported during the current run. */
  // std::vector<Error> errors_;

 public:
  /*! \brief A map from ids to all global functions. */
  tvm::Map<GlobalVar, Function> items;

  // Options options;

  EnvironmentNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final {}

  TVM_DLL static Environment make(
      std::unordered_map<GlobalVar, Function, NodeHash> global_funcs);

  /*! Add an operator to the Enviroment. */
  void register_op(const Operator& op);
  // void add(const Operator& op, bool update = false);

  // void try_add(const Item& item, bool update=false);
  // void update(const Item& item);
  // void remove(const GlobalId& id);

  // GlobalId global_id(const std::string& str);
  Operator op(const std::string& str);

  /*! \brief Lookup a global function by its name. */
  Function lookup(const GlobalVar& id);

  /*! \brief Add a source fragment to the environment. */
  // FileId add_source(std::string file_name, std::string source);

  void report_error(std::string msg, Span sp);
  void display_errors();

  static constexpr const char* _type_key = "relay.Environment";
  TVM_DECLARE_NODE_TYPE_INFO(EnvironmentNode, Node);
};

struct Environment : public NodeRef {
  Environment() {}
  explicit Environment(std::shared_ptr<tvm::Node> p) : NodeRef(p) {}

  inline EnvironmentNode* operator->() const {
    return static_cast<EnvironmentNode*>(node_.get());
  }

  using ContainerType = EnvironmentNode;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ENVIRONMENT_H_
