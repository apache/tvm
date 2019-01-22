/*!  * Copyright (c) 2019 by Contributors
 * \file tvm/relay/pass/optimizer.h
 * \brief The optimizer manages a sequence of Relay-to-Relay transformation
 * passes over a particlar unit of AST. The design is largely inspired from
 * LLVM's pass manager.
 */
#ifndef TVM_RELAY_PASS_OPTIMIZER_H_
#define TVM_RELAY_PASS_OPTIMIZER_H_

#include <functional>
#include <string>

#include <tvm/attrs.h>
#include <tvm/ir.h>
#include <tvm/node/node.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/module.h>
#include <tvm/relay/type.h>

namespace tvm {
namespace relay {

enum PassKind : int {
  kModuleKind,
  kFunctionKind,
  kExprKind
};

class PassState;

class PassStateNode : public RelayNode {
 public:
  Module mod;
  GlobalVar current_func;
  // TODO(zhiics) error reporter could be added when it is ready.
  // ErrorReporter err_reporter;

  PassStateNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("mod", &mod);
    v->Visit("current_func", &current_func);
  }

  TVM_DLL static PassState make(Module mod, GlobalVar current_func);

  static constexpr const char* _type_key = "relay.PassState";
  TVM_DECLARE_NODE_TYPE_INFO(PassStateNode, RelayNode);
};

class PassState : public NodeRef{
 public:
  PassState() = default;
  explicit PassState(NodePtr<tvm::Node> p) : NodeRef(p) {}

  const PassStateNode* operator->() const {
    return static_cast<PassStateNode*>(this->node_.get());
  }

  using ContainerType = PassStateNode;
};

class Pass;

/*!
 * \brief PassNode is the base type of the Relay type hierarchy. It is
 * implemented by different pass subclasses.
 */
class PassNode : public RelayNode {
 public:
  std::string name;
  PassKind pass_kind;
  // TODO(zhiics) error reporter could be added here or in the pass state when it is ready.
  // ErrorReporter err_reporter;

  virtual void run(const Module& mod) const = 0;
  virtual void run(const PassState& mod) const = 0;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("passkind", &pass_kind);
  }

  static constexpr const char* _type_key = "relay.Pass";
  TVM_DECLARE_BASE_NODE_INFO(PassNode, RelayNode);
};

class Pass : public NodeRef {
 public:
  Pass() = default;
  explicit Pass(NodePtr<tvm::Node> p) : NodeRef(p) {}

  const PassNode* operator->() const {
    return static_cast<PassNode*>(this->node_.get());
  }

  using ContainerType = PassNode;
};

/*!
 * \brief A helper macro to define sub-classes of Pass.
 */
#define RELAY_DEFINE_PASS(TypeName, NodeName)                     \
  template <typename T>                                           \
  class TypeName : public Pass {                                  \
   public:                                                        \
    TypeName() {}                                                 \
    explicit TypeName(::tvm::NodePtr<::tvm::Node> n) : Pass(n) {} \
    const NodeName* operator->() const {                          \
      return static_cast<const NodeName*>(node_.get());           \
    }                                                             \
    operator bool() { return this->defined(); }                   \
    using ContainerType = NodeName;                               \
  };

// We are going to use currying here. The module or pass state is captured as
// the source for optimizations. It produces a function from FromT to ToT. For
// example, PassFunc<Module, Function, Function> indicates we perform Function
// to Function transformation on the captured Module.
template <typename SrcT, typename FromT = Function, typename ToT = Function,
          typename = typename std::enable_if<
              std::is_same<SrcT, Module>::value ||
              std::is_same<SrcT, PassState>::value>::type>
using PassFunc = std::function<std::function<ToT(FromT)>(const SrcT&)>;

template<typename T = Module>
class ModulePass;

template <typename T = Module>
class ModulePassNode : public PassNode {
  static_assert(std::is_same<T, Module>::value ||
                    std::is_same<T, PassState>::value,
                "class template can only be the type of Module or PassState");

 public:
  ModulePassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("pass_kind", &pass_kind);
  }

  // TODO(zhiics) what should ModulePass do? Module->(Module->Module)?
  // void run(const Module& mod) const override;

  TVM_DLL static ModulePass<T> make(std::string name);

  static constexpr const char* _type_key = "relay.ModulePass";
  TVM_DECLARE_NODE_TYPE_INFO(ModulePassNode, PassNode);
};

RELAY_DEFINE_PASS(ModulePass, ModulePassNode<T>);

template<typename T = Module>
class FunctionPass;

/*!
 * \brief FunctionPass is used to implement various global optimizations for a
 * given Relay module. It fetechs function one at a time from the function
 * list in the module for optimization.
 */
template <typename T = Module>
class FunctionPassNode : public PassNode {
  static_assert(std::is_same<T, Module>::value ||
                    std::is_same<T, PassState>::value,
                "class template can only be the type of Module or PassState");

 public:
  PassFunc<T> pass_func;

  FunctionPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("pass_kind", &pass_kind);
  }

  /*
   * !\brief Runt a function pass on a certain unit that can be either a module
   * or a pass state.
   */
  void run(const Module& unit) const override;
  void run(const PassState& unit) const override;
  TVM_DLL static FunctionPass<T> make(std::string name,
                                   PassFunc<T> pass_func);

  static constexpr const char* _type_key = "relay.FunctionPass";
  TVM_DECLARE_NODE_TYPE_INFO(FunctionPassNode, PassNode);

 protected:
  bool SkipFunction(const Function& func) const;
};

RELAY_DEFINE_PASS(FunctionPass, FunctionPassNode<T>);

template<typename T = Module>
class ExprPass;

template <typename T = Module>
class ExprPassNode : public PassNode {
  static_assert(std::is_same<T, Module>::value ||
                    std::is_same<T, PassState>::value,
                "class template can only be the type of Module or PassState");

 public:
  PassFunc<T, Expr, Expr> pass_func;

  ExprPassNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("pass_kind", &pass_kind);
  }

  void run(const Module& mod) const override;
  void run(const PassState& mod) const override;

  TVM_DLL static ExprPass<T> make(std::string name,
                               PassFunc<T, Expr, Expr> pass_func);

  static constexpr const char* _type_key = "relay.ExprPass";
  TVM_DECLARE_NODE_TYPE_INFO(ExprPassNode, PassNode);
};

RELAY_DEFINE_PASS(ExprPass, ExprPassNode<T>);

class PassManager;

/*!
 * \brief The Relay optimizer contains a set of passes which transform Relay
 * programs.
 */
class PassManagerNode : public RelayNode {
 public:
  // TODO(@jroesch): For the time being we are using a very simple design for
  // the optimizer it will just store a list of passes run in order.
  //
  // As we move forward we need to generalize the ability to have constraints
  // between them.
  //
  // We also need to store side information.
  tvm::Array<Pass> passes;

  PassManagerNode() = default;

  void AddPass(const Pass& pass) {
    passes.push_back(pass);
  }

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("passes", &passes);
  }

  TVM_DLL static PassManager make(tvm::Array<Pass> passes);

  static constexpr const char* _type_key = "relay.PassManager";
  TVM_DECLARE_NODE_TYPE_INFO(PassManagerNode, RelayNode);
};

class PassManager : public NodeRef {
 public:
  PassManager() = default;
  explicit PassManager(NodePtr<tvm::Node> p) : NodeRef(p) {}

  const PassManagerNode* operator->() const {
    return static_cast<PassManagerNode*>(this->node_.get());
  }

  using ContainerType = PassManagerNode;
};

/*!
 * \brief Optimizes the functions in the module.
 * \param mod The module for optimization.
 * \param passes The optimization passes.
 */
void Optimize(const Module& mod, tvm::Array<Pass> passes);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_PASS_OPTIMIZER_H_
