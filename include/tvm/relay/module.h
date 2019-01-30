/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/module.h
 * \brief The global environment: contains information needed to
 * compile & optimize Relay programs.
 */
#ifndef TVM_RELAY_MODULE_H_
#define TVM_RELAY_MODULE_H_

#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/type.h>
#include <string>
#include <vector>

namespace tvm {
namespace relay {

struct Module;

/*! \brief The global environment of Relay programs.
 *
 *  The global environment contains the global
 *  information needed to compile a Relay program.
 *
 *  It contains all global functions, and configuration
 *  options.
 *
 *  Many operations require access to the global
 *  Module. We pass the Module by value
 *  in a functional style as an explicit argument,
 *  but we mutate the Module while optimizing
 *  Relay programs.
 *
 *  The functional style allows users to construct custom
 *  environments easily, for example each thread can store
 *  an Module while auto-tuning.
 * */

class ModuleNode : public RelayNode {
 public:
  /*! \brief A map from ids to all global functions. */
  tvm::Map<GlobalVar, Function> functions;

  /*! \brief The entry function (i.e. "main"). */
  GlobalVar entry_func;

  ModuleNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("functions", &functions);
    v->Visit("global_var_map_", &global_var_map_);
    v->Visit("entry_func", &entry_func);
  }

  TVM_DLL static Module make(tvm::Map<GlobalVar, Function> global_funcs);

  /*!
   * \brief Add a function to the global environment.
   * \param var The name of the global function.
   * \param func The function.
   * \param update Controls whether you can replace a definition in the
   * environment.
   */
  void Add(const GlobalVar& var, const Function& func, bool update = false);

  /*!
   * \brief Add a function to the global environment.
   * \param var The name of the global function.
   * \param func The function.
   *
   * It does not do type inference as Add does.
   */
  void AddUnchecked(const GlobalVar& var, const Function& func);

  /*!
   * \brief Update a function in the global environment.
   * \param var The name of the global function to update.
   * \param func The new function.
   */
  void Update(const GlobalVar& var, const Function& func);

  /*!
   * \brief Remove a function from the global environment.
   * \param var The name of the global function to update.
   */
  void Remove(const GlobalVar& var);

  /*!
   * \brief Lookup a global function by its variable.
   * \param str The unique string specifying the global variable.
   * \returns The global variable.
   */
  GlobalVar GetGlobalVar(const std::string& str);

  /*!
   * \brief Lookup a global function by its variable.
   * \param var The global var to lookup.
   * \returns The function named by the variable argument.
   */
  Function Lookup(const GlobalVar& var);

  /*!
   * \brief Lookup a global function by its string name
   * \param name The name of the function.
   * \returns The function named by the argument.
   */
  Function Lookup(const std::string& name);

  /*!
   * \brief Update the functions inside this environment by
   *        functions in another environment.
   * \param other The other environment.
   */
  void Update(const Module& other);

  /*! \brief Construct a module from a standalone expression.
   *
   * Allows one to optionally pass a global function map as
   * well.
   *
   * \param expr The expression to set as the entry point to the module.
   * \param global_funcs The global function map.
   *
   * \returns A module with expr set as the entry point.
   */
  static Module FromExpr(
    const Expr& expr,
    const tvm::Map<GlobalVar, Function>& global_funcs = {});

  static constexpr const char* _type_key = "relay.Module";
  TVM_DECLARE_NODE_TYPE_INFO(ModuleNode, Node);

 private:
  /*! \brief A map from string names to global variables that
   * ensures global uniqueness.
   */
  tvm::Map<std::string, GlobalVar> global_var_map_;
};

struct Module : public NodeRef {
  Module() {}
  explicit Module(NodePtr<tvm::Node> p) : NodeRef(p) {}

  inline ModuleNode* operator->() const {
    return static_cast<ModuleNode*>(node_.get());
  }

  using ContainerType = ModuleNode;
};


}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_MODULE_H_
