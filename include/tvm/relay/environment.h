/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/environment.h
 * \brief The global environment: contains information needed to
 * compile & optimize Relay programs.
 */
#ifndef TVM_RELAY_ENVIRONMENT_H_
#define TVM_RELAY_ENVIRONMENT_H_

#include <string>
#include <vector>
#include "./expr.h"
#include "./type.h"
#include "./op.h"
#include "./error.h"
#include "tvm/relay/source_map.h"

namespace tvm {
namespace relay {

struct Environment;

/*! \brief The global environment of Relay programs.
 *
 *  The global environment contains the global
 *  information needed to compile a Relay program.
 * 
 *  It contains all global functions, and configuration 
 *  options.
 *
 *  Many operations require acess to the global
 *  Environment. We pass the Environment by value
 *  in a functional style as an explicit argument,
 *  but we will mutate the Environment while optimizing
 *  Relay programs.
 *
 *  The functional style allows users to construct custom 
 *  environments easily, for example each thread can store
 *  an Environment while auto-tuning.
 * */

class EnvironmentNode : public RelayNode {
 private:
  /*! \brief A map from string names to global variables ensures global uniqueness. */
  tvm::Map<std::string, GlobalVar> global_map_;
  /*! \brief A map from file names to source fragments. */
  SourceMap source_map_;
  /*! \brief A list of the errors reported during the current run. */
  std::vector<SpannedError> errors_;

 public:
  /*! \brief A map from ids to all global functions. */
  tvm::Map<GlobalVar, Function> functions;

  EnvironmentNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final {}

  TVM_DLL static Environment make(
      tvm::Map<GlobalVar, Function> global_funcs);

  void Add(const GlobalVar& var, const Function & func, bool update = false);
  void Update(const GlobalVar& var, const Function & func);
  void Remove(const GlobalVar& var);

  /*! \brief Lookup a global function by its variable. */
  GlobalVar GetGlobalVar(const std::string& str);

  /*! \brief Lookup a global function by its variable. */
  Function Lookup(const GlobalVar& id);

  /*! \brief Lookup a global function by its string name */
  Function Lookup(const std::string & s);
  
  // TODO(@jroesch, @tqchen): what are the semantics here
  void Merge(const Environment & env);

  /*! \brief Add a source fragment to the environment. */
  SourceName AddSource(std::string file_name, std::string source);

  void ReportError(std::string msg, Span sp);
  void DisplayErrors();

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
