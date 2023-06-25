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
 * \file src/relay/transforms/device_planner.cc
 * \brief Determines a unique \p VirtualDevice to hold the result of every Relay sub-expression.
 * This pass can be run multiple times, and can be run both before and after lowering.
 *
 * We say a Relay expression E is 'on device D' if the result of executing E is stored on D.
 * We represent D by an \p VirtualDevice, which means we can track anywhere from an arbitrary device
 * of some \p DLDeviceType to a specific memory scope on a specific (virtual) \p Device who's
 * code is compiled with a specific \p Target.
 *
 * Note that 'stored on device D' is almost but not quite the same as 'executes on device D',
 * see below.
 *
 * This pass works by collecting and solving device constraints, using defaulting heuristics to
 * resolve any remaining undetermined devices, and encoding the results on the output in a form
 * that's reasonably friendly to downstream passes.
 *
 * Specific \p VirtualDevices flow into the constraints from five places:
 *  - Existing "device_copy" CallNodes (with a \p DeviceCopyAttrs attribute) specify a
 *    'src_virtual_device' and 'dst_virtual_device' \p VirtualDevice. Those constrain the argument
 * and context of the call respectively. It is ok if source and destination devices are the same,
 * such no-op copies will be removed after accounting for the device preference.
 *  - Existing "on_device" CallNodes (with a \p OnDeviceAttrs attribute) specify an
 * 'virtual_device', which constrains the argument of the call, but (usually, see below) leaves the
 * context unconstrained. These are called 'annotations' in the rest of the code, have no
 * operational significance by themselves, but may trigger the insertion of a new "device_copy" call
 * by this pass. In two situations the result of an "on_device" CallNode may also be constrained to
 * the given 'virtual_device':
 *     - The "on_device" call occurs at the top-level of a function body, or occurs as an
 *       immediately let-bound expression. In this situation the extra degree of freedom in
 *       the function result and let-binding leads to surprising device copies, so we simply
 *       force the function result or let-bound variable to the given device.
 *     - The \p OnDeviceAttrs has an \p is_fixed field of \p true, which indicates we inserted
 *       it ourselves during an earlier invocation of this pass. This helps make this pass
 *       idempotent.
 *  - Some special operators require their arguments or results to be on the 'host' (typcially
 *    a CPU) \p VirtualDevice, see below.
 *  - Any \p PrimFuncs in the \p IRModule (if \p LowerTEPass has already run) may constrain their
 *    argument buffers to have a specific memory scope, which is part of \p VirtualDevice.
 *  - Annotations left over from a previous run of this pass, such as 'param_virtual_devices' and
 *    'result_virtual_device' function attributes we introduce below. This is so the pass is
 * idempotent and can be re-run to flow additional memory scope constraints.
 *
 * We proceed in five phases:
 *
 * Phase 0
 * -------
 * We rewrite the programs to handle some special cases:
 *  - "on_device" calls at the top-level of function or immediately let-bound are rewritten
 *    to have \code is_fixed=true \endcode.
 *  - We wish to treat \code on_device(expr, device_type=d).0 \endcode as if it were written
 *    \code on_device(expr.0, device_type_d) \endcode. I.e. we prefer to copy the projection from
 *    the tuple rather than project from a copy of the tuple. We'll do this by rewriting.
 *  - We are prepared to insert device_copies on the arguments and result of calls to PrimFuncs,
 *    on the assumption a) we already ran PlanDevices before lowering so we are not allowing
 *    any new cross-device copies, but b) after lowering we may have new memory scope constraits
 *    to deal with.
 *
 * Phase 1
 * -------
 * We iteratively process the programs and find nodes with conflicting virtual devices. If the
 * virtual devices ( \p d1 and \p d2 ) are joinable, they are replaced with a joined device \p d. If
 * they are unjoinable, a "device_copy" CallNode is inserted to copy the node output to the second
 * device.
 *
 * Phase 2
 * -------
 * We flow constraints from the "on_device" and "device_copy" calls, PrimFunc buffer memory scopes,
 * and some special ops, to all other Relay sub-expressions.
 *
 * For a primitive such as \code add(e1, e2) \endcode all arguments and results must be on the
 * same device. However each call site can use a different device. In other words primitives are
 * 'device polymorphic' since we compile and execute them for each required device. ADT constructors
 * are similarly polymorphic, but require all constructor args to be on the same device.
 *
 * For most Relay expressions the device for the overall expression is the same as the device
 * for its sub-expressions. E.g. each field of a tuple must be on the same device as the tuple
 * itself, the condition and arms of an \p if must all be on the same device as the overall \p if,
 * and so on.
 *
 * Some special ops (or 'dialects') are handled:
 *  - Relay supports computing the shape of tensors and operators at runtime using "shape_of"
 *    and "reshape_tensor". Shapes must only be held on the CPU, but the tensors they describe
 *    may reside on any device.
 *  - Explicit memory allocation is done using the "alloc_storage" and "alloc_tensor". Again
 *    shapes reside on the CPU, but the allocated tensors may reside on any device.
 *
 * Two Relay expression have special handling:
 *  - For \code let x = e1; e2 \endcode the result of \p e2 must be on the same device as the
 *    overall let. However the result of \p e1 may be on a different device.
 *  - For a function \code fn(x, y) { body } \endcode the result of the function must be on the
 *    same device as \p body. However parameters \p x and \p may be on different devices, even
 *    different from each other. Every call to the function must use the same choice of parameter
 *    and result devices -- there is no 'device polymorphism' for Relay functions.
 *
 * Currently \p PrimFuncs and external functions do not carry over their parameter and result
 * devices from their original Relay Function representations. However we know all calls to those
 * functions are device-consistent, thus no information is lost.
 *
 * Phase 3
 * -------
 * After flowing constraints we apply some defaulting heuristics (using a global default \p
 * VirtualDevice) to fix the device for any as-yet unconstrained sub-expressions.
 *  - Unconstrained function result devices default to the global default device.
 *  - Unconstrained function parameters devices default to the device for the function result.
 *  - Unconstrained let-bound expression devices default to the device for the overall let.
 * TODO(mbs): These are very simple minded heuristics, and ultimately we'd like to treat the
 * assignment of the remaining unconstrained sub-expressions as an optimiziation problem in itself.
 * This requires a formal notion of 'choicepoint' inside the compiler which can integrate with
 * automation.
 *
 * Phase 4
 * -------
 * Finally, the result of this analysis is reified into the result as:
 *  - Additional "param_virtual_devices" (an \p Array<VirtualDevice>) and "result_virtual_device"
 * (an \p VirtualDevice) attributes for every function (both top-level and local). These describe
 * the devices for the function's parameters and the result.
 *  - Additional "device_copy" CallNodes where a copy is required in order to respect the
 *    intent of the original "on_device" CallNodes.
 *  - Additional "on_device" CallNodes where the device type of an expression is not trivially
 *    implied by the lexically enclosing "on_device" CallNode or function attribute. In practice
 *    this means "on_device" CallNodes may appear in two places:
 *     - On let-bound expressions. It is tempting to elide the "on_device" if the let-bound value
 *       has the same device as the overall let expression. However this would mean passes which
 *       inline let-bound values, such as FoldConstant and DeadCodeElimination, would need to us
 *       a DeviceAware visitor which in turn requires the expression to be in ANF to avoid
 *       deep recursion. To minimize disruption we always include the "on_device" so that it
 *       can follow the inline.
 *     - On a call argument if its device differs from the call result. In particular, the
 *       argument to a "device_copy" call will always be wrapped in an "on_device". (That may
 *       seem pedantic but simplifies downstream handling.)
 *    However since we make it easy to track devices for variables we never wrap an "on_device"
 *    around a var or global var. These uses of "on_device" imply both the argument and result are
 *    on the same device. We signal this by setting the 'is_fixed' OnDeviceAttrs field to true,
 *    which helps make this pass idempotent.
 *  - The buffer maps for called PrimFuncs are updated to capture memory scopes.
 *
 * Helper visitors (in device_aware_visitors.h) can be used by downstream transforms to recover
 * the device for any expression for their own use, e.g. during memory planning. All downstream
 * passes must preserve the lexical scoping of the "on_device" CallNodes. E.g. conversion
 * to ANF must respect the lexical scoping convention:
 * \code
 *   f(on_device(g(h(a, b), c), virtual_device=CPU))
 *   ==>
 *   let %x0 = on_device(h(a, b), virtual_device=CPU)
 *   let %x1 = on_device(g(%x0), virtual_device=CPU)
 *   f(on_device(%x1, virtual_device=CPU))
 * \endcode
 *
 * This pass can be run before FuseOps so that it can use device-specific fusion rules.
 *
 * 'Stored on' vs 'Executes on'
 * ----------------------------
 * Obviously for a primitive call \code add(x, y) \endcode we can execute the primitive on the
 * same device as will hold its result. Thus 'executes on' is the same as 'stored on' for
 * primitives.
 *
 * But what about for arbitrary Relay expressions? Most backends (interpreter, graph, VM) are
 * implicitly executed on the 'host' CPU, with only primitive evaluation handed off to specific
 * devices, thus the notion of 'executes on' is mute. AOT backends on the other hand need to
 * know exactly which device (possibly one of a number of available 'CPU'-like devices) is
 * responsible for execution. Currently that's handled independently by the \p AnnotateTargets
 * pass, but we'd like to fold that into device planning here to ensure everything is consistent.
 *
 * Obviously since tensors are passed-by-pointer it's quite possible to execute a Relay
 * expression (eg an \p if expression) on one device even though the tensor data resides on
 * another. But for AOT that flexibility seems excessive. So we'd like to just take 'executes on'
 * to be 'stored on' exactly. In particular, for a Relay function, we'd like to be able to just
 * compile the function body for the function's result device.
 *
 * This works after conversion to ANF provided the compilation for a let expression is prepared
 * to make a cross-device call. However we leave it to a downstream transformation to heuristically
 * minimize cross-device calls by moving device copies out of functions. E.g.:
 * \code
 *   def @f() {  // execute on CPU
 *     let x = on_device(...GPU computation..., virtual_device=GPU);
 *     device_copy(...GPU computation..., src_dev_type=GPU, dst_dev_type=CPU)
 *   }
 *   def @main() {
 *     ... call @f() on CPU ...
 *   }
 * \endcode
 * could be rewritten to:
 * \code
 *   def @f() {  // execute on GPU
 *     let x = ...GPU computation...;
 *     ...GPU computation...
 *   }
 *   def @main() {
 *     let x = device_copy(@f(), src_dev_type=GPU, dst_dev_type=CPU)
 *     ... use x on CPU ...
 *   }
 * \endcode
 *
 * Higher-order shenanigans
 * ------------------------
 * Relay is a 'mostly' higher-order language -- we can let-bind functions, pass functions
 * as arguments (even anonymous functions), return functions, evaluate conditional expressions
 * over functions, and so on. We handle this during constraint solving using the domain:
 * \code
 *   D  ::= <specific device type>   -- first-order
 *        | fn(D,...,D):D            -- higher-order
 * \endcode
 * In this way we can determine the device for all function parameters and results. E.g. for
 * \code
 *   let f = fn(x, y) { ... }
 *   let g = fn(f, z) { f(z, z) }
 *   g(f, on_device(..., virtual_device=CPU))
 * \endcode
 * the parameters \p x and \p y will be on the CPU.
 *
 * But now look closely at the call \code e1(e2, e3) \endcode. We know \p e1 must evaluate to a
 * function. Our analysis must guarantee that the function's parameters and result devices are
 * consistent for \p e2, \p e3, and the context of the call. But:
 *  - Which device holds the closure result of evaluating \p e1 ?
 *  - If \p e2 is of function type, what does that mean when we say every function parameter
 *    is on a device?
 *  - If \p e1 returns a function, what does that mean when we say every function result is
 *    on a device?
 *
 * Since higher-order aspects are later compiled away (by 'defunctionalization'
 * aka 'firstification') we'd prefer not to have to answer any of those questions. In particular,
 * we really don't want our domain \p D to allow for yet another device for the function closure.
 * So we'll just force the 'device for a function' to be the same as the device for the function's
 * result using the notion of the 'result domain' for a domain:
 * \code
 *   result_domain(<specific device type>) = <specific device type>
 *   result_domain(fn(D1,...,Dn):Dr)       = result_domain(Dr)
 * \endcode
 *
 * Similarly the domain does not have entries for tuples, references, or ADTs. Whenever the
 * analysis encounters a function inside one of those it simply forces all argument and result
 * devices for the function to match the device for the first-order expression. For example,
 * if the tuple \code (fn(x, y) { ... }, 3) \endcode is on the GPU then the inner function
 * parameters and result must similarly be on the GPU.
 *
 * -------
 * | AOR |  This pass supports all of Relay.
 * -------
 *    ^
 *    |
 *    `-- Mark's stamp of completeness :-)
 *
 * TODO(mbs): Proper diagnostics for unification failure using spans.
 * TODO(mbs): We may want some 'device polymorphism' for Relay functions. Eg it's ok for the
 * function to be called with params/result on different (virtual) device ids provided the target
 * and memory scopes are consistent.
 */

#include <tvm/ir/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>

#include "../../tir/analysis/device_constraint_utils.h"
#include "../op/annotation/annotation.h"
#include "../op/memory/device_copy.h"
#include "../op/memory/on_device.h"
#include "./device_domains.h"

namespace tvm {
namespace relay {
namespace transform {

namespace {

/* =============== Phase 0 =============== */

/*!
 * \brief Rewrites "on_device" calls to handle some special cases.
 *
 *  - Don't let the device for %x remain unconstrained:
 *    \code
 *      let %x = on_device(e, virtual_device=d)
 *      ==> let %x = on_device(e, virtual_device=d, constraint=kBoth)
 *    \endcode
 *
 *  - Don't let the function result remain unconstrained:
 *    \code
 *      fn(%x) { on_device(e, virtual_device=d) }
 *      ==> fn(%x) { on_device(e, virtual_device=d, constraint=kBoth)
 *    \endcode
 *
 *  - Project-then-copy rather than copy-then-project:
 *    \code
 *      on_device(e).0
 *      ==> on_device(e.0)
 *    \endcode
 *
 *  - Be prepared to copy arguments and results on primitive call boundaries in case memory
 *    scopes don't line up. We'll use the 'fully unconstrained' version of on_device so that
 *    we can allow for a device_copy without knowing the specific device for the arguments.
 *    \code
 *      call_lowered(@prim, (a, b))
 *      ==> copy_ok(call_lowered(@prim, (copy_ok(a), copy_ok(b))))
 *      where
 *        copy_ok(x) = on_device(x, virtual_device=VirtualDevice::FullyUnconstrained,
 *                               constrain_body=False, constrain_result=False)
 *    \endcode
 */
class RewriteOnDevices : public ExprMutator {
 public:
  explicit RewriteOnDevices(IRModule mod) : mod_(std::move(mod)) {}

 private:
  Expr VisitExpr_(const TupleGetItemNode* tuple_get_item_node) final {
    Expr tuple = VisitExpr(tuple_get_item_node->tuple);
    OnDeviceProps props = GetOnDeviceProps(tuple);

    Expr tuple_get_item = WithFields(GetRef<TupleGetItem>(tuple_get_item_node), tuple);
    if (props.body.defined() && props.is_normal()) {
      VLOG(2) << "wrapping tuple get item:" << std::endl
              << PrettyPrint(GetRef<TupleGetItem>(tuple_get_item_node)) << std::endl
              << "with \"on_device\" for VirtualDevice " << props.virtual_device;
      return OnDeviceWithProps(tuple_get_item, props);
    } else {
      return tuple_get_item;
    }
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    auto expr = GetRef<Expr>(let_node);
    std::vector<std::tuple<Let, Expr>> bindings;
    while (auto opt = expr.as<Let>()) {
      auto inner_let = opt.value();
      Expr value = VisitExpr(inner_let->value);
      OnDeviceProps props = GetOnDeviceProps(value);
      if (props.body.defined() && props.is_normal()) {
        VLOG(2) << "revising let-bound expression of let:" << std::endl
                << PrettyPrint(expr) << std::endl
                << "to be fixed to VirtualDevice " << props.virtual_device;
        value = MaybeOnDeviceFixed(props.body, props.virtual_device);
      }
      bindings.emplace_back(inner_let, value);
      expr = inner_let->body;
    }
    expr = VisitExpr(expr);
    for (auto itr = bindings.rbegin(); itr != bindings.rend(); ++itr) {
      expr = WithFields(/*let=*/std::get<0>(*itr), /*opt_var=*/{},
                        /*opt_value=*/std::get<1>(*itr), /*opt_body=*/expr);
    }
    return expr;
  }

  Expr VisitExpr_(const FunctionNode* function_node) final {
    Expr body = VisitExpr(function_node->body);
    OnDeviceProps props = GetOnDeviceProps(body);
    if (props.body.defined() && props.is_normal()) {
      VLOG(2) << "revising body of function:" << std::endl
              << PrettyPrint(GetRef<Function>(function_node)) << std::endl
              << "to be fixed to VirtualDevice " << props.virtual_device;
      body = MaybeOnDeviceFixed(props.body, props.virtual_device);
    }
    return WithFields(GetRef<Function>(function_node), function_node->params, body);
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    CallLoweredProps props = GetCallLoweredProps(call_node);
    if (props.lowered_func.defined()) {
      BaseFunc base_func = mod_->Lookup(props.lowered_func);
      if (base_func.as<tir::PrimFuncNode>()) {
        VLOG(2) << "allowing device_copy on PrimFunc arguments and result";
        Array<Expr> new_args;
        new_args.reserve(props.arguments.size());
        for (const auto& arg : props.arguments) {
          Expr new_arg = VisitExpr(arg);
          new_args.push_back(OnDeviceCopyOk(std::move(new_arg)));
        }
        Call new_call = CallLowered(std::move(props.lowered_func), std::move(new_args), props.attrs,
                                    call_node->span);
        return OnDeviceCopyOk(std::move(new_call));
      }
    }
    return ExprMutator::VisitExpr_(call_node);
  }

  /*! \brief Module we are rewriting, so we can lookup global definitions. */
  IRModule mod_;
};

/* =============== Phase 1 =============== */

/*!
 * \brief Add "device_copy" calls for nodes that have conflicting virtual devices.
 *
 * Eg Suppose an IRModule contains the following expr:
 * \code
 *   %0 = add(%a, %b);
 *   %1 = on_device(%0, virtual_device=d1);
 *   %2 = add(%b, %c);
 *   %3 = on_device(%2, virtual_device=d2);
 * \endcode
 * In the above example, node %b has two possible virtual devices: \p d1 and \p d2.
 *
 * - If \p d1 and \p d2 are joinable, replace \p d1 and \p d2 with the joined device \p d:
 * \code
 *   %0 = add(%a, %b);
 *   %1 = on_device(%0, virtual_device=d);
 *   %2 = add(%b, %c);
 *   %3 = on_device(%2, virtual_device=d);
 * \endcode
 *
 * - If \p d1 and \p d2 are unjoinable, insert a "device_copy" CallNode to copy \p %b to \p d2:
 * \code
 *   %0 = add(%a, %b);
 *   %1 = on_device(%0, virtual_device=d);
 *   %2 = device_copy(%b, src_dev_type=d1, dst_dev_type=d2);
 *   %3 = add(%2, %c);
 *   %4 = on_device(%3, virtual_device=d);
 * \endcode
 */
struct DeviceContext {
  VirtualDevice VirtualDeviceFor(const ExprNode* expr) {
    auto itr = expr_to_device.find(expr);
    if (itr != expr_to_device.end()) {
      return itr->second;
    }
    auto default_dev = VirtualDevice::FullyUnconstrained();
    expr_to_device.emplace(expr, default_dev);
    return default_dev;
  }

  bool Update(const ExprNode* expr, VirtualDevice dev) {
    bool success = true;
    auto pair = expr_to_device.emplace(expr, dev);
    if (!pair.second) {
      auto replaced_item = pair.first;
      auto joined_dev = VirtualDevice::Join(replaced_item->second, dev);
      if (joined_dev == nullptr) {
        success = false;
      } else {
        replaced_item->second = joined_dev.value();
      }
    }
    return success;
  }

  bool IsConflicted(const ExprNode* expr) {
    auto itr = conflicted_nodes.find(expr);
    return itr != conflicted_nodes.end();
  }

  std::unordered_set<const ExprNode*> conflicted_nodes;
  std::unordered_map<const ExprNode*, VirtualDevice> expr_to_device;
};

/*!
 * \brief Flow the device constraints over the module and find all the conflicted nodes. The
 * conflicted nodes only contain nodes that have no explicit constraints. For example, "on_device"
 * nodes are not considered as conflicted.
 */
class ConflictedNodeFinder : ExprVisitor {
 public:
  explicit ConflictedNodeFinder(IRModule mod)
      : mod_(std::move(mod)), dev_ctx_(std::make_unique<DeviceContext>()) {}

  std::unique_ptr<DeviceContext> Finder() {
    VLOG_CONTEXT << "ConflictedNodeFinder";
    for (const auto& kv : mod_->functions) {
      if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
        VisitExpr(GetRef<Function>(function_node));
      }
    }
    for (auto const node : dev_ctx_->conflicted_nodes) {
      if (node->IsInstance<CallNode>()) {
        auto call = Downcast<Call>(GetRef<Expr>(node));
        // "DeviceCapturer" will insert "device_copy" for "on_device" calls.
        // Therefore, "on_device" should not be considered as conflicted.
        if (call->op == OnDeviceOp()) {
          dev_ctx_->conflicted_nodes.erase(node);
        }
      }
    }
    return std::move(dev_ctx_);
  }

 private:
  void VisitExpr_(const CallNode* call_node) final {
    VLOG(2) << "Initial call node: " << std::endl << PrettyPrint(GetRef<Call>(call_node));
    auto call_dev = dev_ctx_->VirtualDeviceFor(call_node);
    auto body_dev = call_dev;

    auto on_dev_props = GetOnDeviceProps(call_node);
    auto dev_cp_props = GetDeviceCopyProps(call_node);
    if (call_node->op == OnDeviceOp()) {
      if (on_dev_props.constrain_body) {
        body_dev = on_dev_props.virtual_device;
      }
      if (on_dev_props.constrain_result) {
        call_dev = on_dev_props.virtual_device;
      }
    } else if (call_node->op == DeviceCopyOp()) {
      body_dev = dev_cp_props.src_virtual_device;
      call_dev = dev_cp_props.dst_virtual_device;
    }

    if (!dev_ctx_->Update(call_node, call_dev) && call_node->op != OnDeviceOp()) {
      LOG(FATAL) << "Mismatched device type after iterating args. Implied device: " << std::endl
                 << PrettyPrint(call_dev) << "and practial device:" << std::endl
                 << PrettyPrint(dev_ctx_->VirtualDeviceFor(call_node)) << std::endl
                 << "With CallNode: " << std::endl
                 << PrettyPrint(GetRef<Call>(call_node));
    }

    for (auto& arg : call_node->args) {
      VLOG(3) << "Handle call node arg: " << std::endl << PrettyPrint(arg);
      if (!dev_ctx_->Update(arg.get(), body_dev)) {
        VLOG(2) << "Conflicted node found:" << std::endl
                << PrettyPrint(GetRef<Expr>(arg.get())) << std::endl
                << "With corresponding Callee:" << std::endl
                << PrettyPrint(GetRef<Call>(call_node));
        dev_ctx_->conflicted_nodes.emplace(arg.get());
      }
    }
    for (auto& expr : call_node->args) {
      VisitExpr(expr);
    }
  }

  IRModule mod_;
  std::unique_ptr<DeviceContext> dev_ctx_;
};

/*!
 * \brief Insert "device_copy" CallNode for all the conflicted nodes found by \p
 * ConflictedNodeFinder.
 */
class ConflictedNodeRewriter : ExprMutator {
 public:
  ConflictedNodeRewriter(IRModule mod, CompilationConfig config,
                         std::unique_ptr<DeviceContext> dev_ctx)
      : mod_(mod), config_(config), dev_ctx_(std::move(dev_ctx)) {}

  IRModule Rewrite() {
    VLOG_CONTEXT << "ConflictedNodeRewriter";
    IRModule result(/*functions=*/{}, mod_->type_definitions, mod_->Imports(), mod_->source_map,
                    mod_->attrs);
    for (const auto& kv : mod_->functions) {
      if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
        auto func = Mutate(GetRef<Function>(function_node));
        result->Add(kv.first, Downcast<Function>(func));
      } else {
        result->Add(kv.first, kv.second);
      }
    }

    return result;
  }

 private:
  Expr VisitExpr_(const CallNode* call_node) final {
    VLOG(3) << "Initial call node:" << std::endl << PrettyPrint(GetRef<Call>(call_node));
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    tvm::Array<Expr> call_args;
    call_args.reserve(call_node->args.size());
    for (auto arg : call->args) {
      if (dev_ctx_->IsConflicted(arg.get())) {
        auto src_dev = config_->CanonicalVirtualDevice(dev_ctx_->VirtualDeviceFor(arg.get()));
        auto dst_dev = config_->CanonicalVirtualDevice(dev_ctx_->VirtualDeviceFor(call_node));
        call_args.push_back(MaybeDeviceCopy(arg, src_dev, dst_dev));
        VLOG(2) << "Adding DeviceCopy Op: " << std::endl << PrettyPrint(call_args.back());
      } else {
        call_args.push_back(arg);
      }
    }
    auto new_call = WithFields(GetRef<Call>(call_node), call_node->op, call_args);
    VLOG(3) << "Final call node:" << std::endl << PrettyPrint(GetRef<Call>(call_node));
    return new_call;
  }

  IRModule mod_;
  CompilationConfig config_;
  std::unique_ptr<DeviceContext> dev_ctx_;
};

/* =============== Phase 2 =============== */

/*
 * \brief Collects the system of device constraints for all sub-expressions in a module.
 * It is possible some devices remain free and will need to be defaulted by \p DeviceDefaulter.
 *
 * Eg from \code add(%x, %y) \endcode we know \p %x and \p %y must be on the same device. Later,
 * from \code on_device(%x, virtual_device=d) \endcode we know \p %x must be on device \p d, and
 * thus so must \p %y.
 *
 * Constraints can flow in interesting ways. E.g. in:
 * \code
 *   let %f = fn(%x, %y) { add(%x, on_device(%y, virtual_device=d)) }
 *   let %g = fn(%f, %x, %y) { %f(%x, %y) }
 *   %g(%f, %a, %b)
 * \endcode
 * we discover \p %b must be on device \p d.
 */
class DeviceAnalyzer : public MixedModeVisitor {
 public:
  DeviceAnalyzer(IRModule mod, CompilationConfig config)
      : mod_(std::move(mod)), domains_(std::make_unique<DeviceDomains>(std::move(config))) {}

  /*!
   * \brief Returns the expression-to-device-domain map for all expressions in all the global
   * function definitions in the module. Expressions may have free domains, these will be resolved
   * by \p DeviceDefaulter below.
   */
  std::unique_ptr<DeviceDomains> Analyze() {
    VLOG_CONTEXT << "DeviceAnalyzer";
    for (const auto& kv : mod_->functions) {
      // The global variable and what it is bound to must obviously agree on domain.
      if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
        VLOG(2) << "collecting constraints from Relay Function '" << kv.first->name_hint << "'";
        domains_->UnifyExprExact(kv.first, kv.second);
        VisitExpr(GetRef<Function>(function_node));
      } else if (auto prim_func = kv.second.as<tir::PrimFunc>()) {
        VLOG(2) << "collecting constraints from TIR PrimFunc '" << kv.first->name_hint << "'";
        domains_->UnifyExprExact(kv.first, DomainForPrimFunc(kv.first, prim_func.value()));
      } else {
        VLOG(2) << "skipping '" << kv.first->name_hint << "'";
      }
    }
    return std::move(domains_);
  }

 private:
  /*!
   * \brief Return the domain representing \p prim_func which, before lowering, had
   * the Relay \p type.
   */
  DeviceDomainPtr DomainForPrimFunc(const GlobalVar& global_var, const tir::PrimFunc& prim_func) {
    // CAUTION: The prim_func->checked_type() is currently w.r.t. the flattened and DPS form
    // of the prim func, however here we wish to remain within the Relay view of all functions.
    // Thus we'll use the global var who's checked_type is in Relay form.
    auto func_domain = domains_->DomainFor(global_var);  // higher-order

    // TODO(mbs): We don't visit the body of the function -- there's currently nothing to be done.
    const auto* func_type_node = global_var->checked_type().as<FuncTypeNode>();
    ICHECK(func_type_node);
    ICHECK_EQ(func_domain->function_arity(), func_type_node->arg_types.size());

    Array<VirtualDevice> virtual_devices =
        tir::GetPrimFuncArgAndResultConstraints(prim_func, GetRef<FuncType>(func_type_node));

    // Build the implied domain (in terms of the function's Relay type) implied by any memory scope
    // constrains in the function's buffers, for both arguments and results.
    std::vector<DeviceDomainPtr> args_and_result_domains;
    args_and_result_domains.reserve(virtual_devices.size());
    for (size_t i = 0; i < func_type_node->arg_types.size(); ++i) {
      const VirtualDevice& param_virtual_device = virtual_devices[i];
      VLOG(2) << "param_virtual_device[" << i << "] = " << param_virtual_device;
      args_and_result_domains.push_back(domains_->MakeFirstOrderDomain(param_virtual_device));
    }
    const VirtualDevice& ret_virtual_device = virtual_devices.back();
    VLOG(2) << "ret_virtual_device = " << ret_virtual_device;
    args_and_result_domains.push_back(domains_->MakeFirstOrderDomain(ret_virtual_device));

    return domains_->MakeHigherOrderDomain(std::move(args_and_result_domains));
  }

  void VisitExpr_(const CallNode* call_node) final {
    auto call = GetRef<Call>(call_node);

    // We don't care if the call is in pre- or post-lowered form.
    auto vanilla_call = GetAnyCall(call_node);

    // Find the higher-order domain for the callee. See DomainForCallee for the special rules
    // for primitives.
    VisitExpr(vanilla_call->op);
    auto func_domain = domains_->DomainForCallee(call);  // higher-order

    // Build the domain for the function implied by its arguments and call context.
    ICHECK_EQ(func_domain->function_arity(), vanilla_call->args.size()) << PrettyPrint(call);
    std::vector<DeviceDomainPtr> args_and_result_domains;
    args_and_result_domains.reserve(vanilla_call->args.size() + 1);
    for (const auto& arg : vanilla_call->args) {
      args_and_result_domains.emplace_back(domains_->DomainFor(arg));
    }
    args_and_result_domains.emplace_back(domains_->DomainFor(call));
    auto implied_domain =
        domains_->MakeHigherOrderDomain(std::move(args_and_result_domains));  // higher-order

    VLOG(2) << "initial call function domain:" << std::endl
            << domains_->ToString(func_domain) << std::endl
            << "and implied domain:" << std::endl
            << domains_->ToString(implied_domain) << std::endl
            << "for call:" << std::endl
            << PrettyPrint(call);

    // The above must match.
    if (domains_->UnifyOrNull(func_domain, implied_domain) == nullptr) {  // higher-order
      // TODO(mbs): Proper diagnostics.
      LOG(FATAL)
          << "Function parameters and result VirtualDevices do not match those of call. Call:"
          << std::endl
          << PrettyPrint(call) << std::endl
          << "with function virtual devices:" << std::endl
          << domains_->ToString(func_domain) << std::endl
          << "and implied call virtual devices:" << std::endl
          << domains_->ToString(implied_domain);
    }

    VLOG(2) << "final call function domain:" << std::endl
            << domains_->ToString(func_domain) << std::endl
            << "for call:" << std::endl
            << PrettyPrint(call);
  }

  void VisitExpr_(const LetNode* let_node) final {
    Expr expr = GetRef<Let>(let_node);
    // Iteratively visit let nodes to avoid stack overflow.
    while (expr->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(expr);
      // Let var must be same device as value it is bound to.
      domains_->UnifyExprExact(let->var, let->value);  // may be higher-order
      // Let body must be same device as overall let.
      domains_->UnifyExprExact(let, let->body);  // may be higher-order

      VisitExpr(let->var);
      VisitExpr(let->value);

      expr = let->body;
    }

    // Visit the last body
    VisitExpr(expr);
  }

  void VisitExpr_(const FunctionNode* function_node) final {
    auto function = GetRef<Function>(function_node);
    auto func_domain = domains_->DomainFor(function);  // higher-order
    ICHECK_EQ(func_domain->function_arity(), function_node->params.size());

    VLOG(2) << "initial function domain:" << std::endl
            << domains_->ToString(func_domain) << std::endl
            << "and function body domain:" << std::endl
            << domains_->ToString(domains_->DomainFor(function_node->body)) << std::endl
            << "for function:" << std::endl
            << PrettyPrint(function);

    // The function body domain must match the function result domain.
    domains_->UnifyExprExact(function_node->body,
                             func_domain->function_result());  // may be higher-order
    if (!function_node->virtual_device()->IsFullyUnconstrained()) {
      // The function body domain must match any existing virtual device annotation.
      domains_->UnifyExprExact(function_node->body,
                               domains_->ForVirtualDevice(function_node->body->checked_type(),
                                                          function_node->virtual_device()));
    }

    for (size_t i = 0; i < function_node->params.size(); ++i) {
      const auto& param = function_node->params[i];
      // The parameter domain must match the function argument domain.
      domains_->UnifyExprExact(param,
                               func_domain->function_param(i));  // may be higher-order
      if (!param->virtual_device()->IsFullyUnconstrained()) {
        // The parameter domain must match any existing virtual device annotation.
        domains_->UnifyExprExact(
            param, domains_->ForVirtualDevice(param->checked_type(), param->virtual_device()));
      }
      VisitExpr(param);
    }

    // No need to step into the body of Primitive functions.
    if (!function_node->HasNonzeroAttr(attr::kPrimitive)) {
      VisitExpr(function_node->body);
    }

    VLOG(2) << "final function domain:" << std::endl
            << domains_->ToString(func_domain) << std::endl
            << "and function body domain:" << std::endl
            << domains_->ToString(domains_->DomainFor(function_node->body)) << std::endl
            << "for function:" << std::endl
            << PrettyPrint(function);
  }

  void VisitExpr_(const TupleNode* tuple_node) final {
    Tuple tuple = GetRef<Tuple>(tuple_node);
    for (size_t i = 0; i < tuple->fields.size(); i++) {
      auto domain = domains_->DomainFor(tuple->fields[i]);  // may be higher-order
      domains_->UnifyExprCollapsed(tuple, domain);          // collapse to first-order if needed
    }
  }

  void VisitExpr_(const TupleGetItemNode* tuple_get_item_node) final {
    TupleGetItem tuple_get_item = GetRef<TupleGetItem>(tuple_get_item_node);
    auto domain = domains_->DomainFor(tuple_get_item);  // may be higher-order
    domains_->UnifyExprCollapsed(tuple_get_item_node->tuple,
                                 domain);  // collapse to first-order if needed
  }

  class DevicePatternAnalyzer : public PatternVisitor {
   public:
    DevicePatternAnalyzer(DeviceDomains* domains, const ExprNode* adt_node)
        : domains_(domains), adt_node_(adt_node) {}

   private:
    void VisitPattern_(const PatternVarNode* pattern_var_node) final {
      auto var_domain = domains_->DomainFor(pattern_var_node->var);  // may be higher order
      domains_->UnifyExprCollapsed(GetRef<Expr>(adt_node_),
                                   var_domain);  // collapse to first-order if needed
    }

    /*! \brief (Mutable borrow of) the domains for all expressions processed so far. */
    DeviceDomains* domains_;
    /*! \brief The expression for the ADT we are matching over. */
    const ExprNode* adt_node_;
  };

  void VisitPattern(const Pattern& pattern) final {}

  void VisitExpr_(const MatchNode* match_node) final {
    // For match node, we unify the value and the rhs of each clause
    Match match = GetRef<Match>(match_node);
    auto match_domain = domains_->DomainFor(match);  // may be higher-order
    DevicePatternAnalyzer pattern_analyzer(domains_.get(), match->data.get());
    domains_->UnifyExprCollapsed(match->data, match_domain);  // collapse to first-order if needed
    for (const auto& clause : match->clauses) {
      pattern_analyzer.VisitPattern(clause->lhs);
      domains_->UnifyExprExact(clause->rhs, match_domain);
      VisitExpr(clause->rhs);
    }
    VisitExpr(match_node->data);
  }

  void VisitExpr_(const GlobalVarNode* global_var_node) final {
    domains_->DomainFor(GetRef<GlobalVar>(global_var_node));
  }

  void VisitExpr_(const VarNode* var_node) final { domains_->DomainFor(GetRef<Var>(var_node)); }

  void VisitExpr_(const ConstantNode* constant_node) final {
    domains_->DomainFor(GetRef<Constant>(constant_node));
  }

  void VisitExpr_(const ConstructorNode* constructor_node) final {
    // no-op, constructors are handled at their call-sites.
    // TODO(mbs): Assumes eta-expansion
  }

  void VisitExpr_(const IfNode* if_node) final {
    auto ife = GetRef<If>(if_node);
    auto domain = domains_->DomainFor(ife);               // may be higher-order
    domains_->UnifyExprCollapsed(if_node->cond, domain);  // collapse to first-order if needed
    domains_->UnifyExprExact(if_node->true_branch, domain);
    domains_->UnifyExprExact(if_node->false_branch, domain);
    VisitExpr(if_node->cond);
    VisitExpr(if_node->true_branch);
    VisitExpr(if_node->false_branch);
  }

  void VisitExpr_(const OpNode* op) final {
    // no-op, primitive operators are handled at their call-sites.
  }

  void VisitExpr_(const RefCreateNode* ref_create_node) final {
    auto ref_create = GetRef<RefCreate>(ref_create_node);
    auto domain = domains_->DomainFor(ref_create_node->value);  // may be higher-order
    domains_->UnifyExprCollapsed(ref_create, domain);           // collapse to first-order if needed
    VisitExpr(ref_create_node->value);
  }

  void VisitExpr_(const RefReadNode* ref_read_node) final {
    auto ref_read = GetRef<RefRead>(ref_read_node);
    auto domain = domains_->DomainFor(ref_read);               // may be higher-order
    domains_->UnifyExprCollapsed(ref_read_node->ref, domain);  // collapse to first-order if needed
    VisitExpr(ref_read_node->ref);
  }

  void VisitExpr_(const RefWriteNode* ref_write_node) final {
    auto ref_write = GetRef<RefWrite>(ref_write_node);
    auto domain = domains_->DomainFor(ref_write->value);   // may be higher-order
    domains_->UnifyExprCollapsed(ref_write->ref, domain);  // collapse to first-order if needed
    domains_->UnifyExprCollapsed(ref_write, domain);       // collapse to first-order if needed
    VisitExpr(ref_write_node->ref);
    VisitExpr(ref_write_node->value);
  }

  /*! \brief The module we are analyzing. */
  IRModule mod_;
  /*! \brief The domains for all expressions processed so far. */
  std::unique_ptr<DeviceDomains> domains_;
};

/* =============== Phase 3 =============== */

/*!
 * \brief Calls to 'free' "on_device" annotations (ie where both constrain_body=false and
 * constrain_result=false) indicate a device_copy is allowed if required, but no particular
 * device is imposed on the body or the context. At this stage we can attempt to unify the
 * body and device contexts. In this way we can avoid the defaulting rules in \p DeviceDefaulter
 * from choosing default devices which are only going to induce a device copy.
 *
 * TODO(mbs): The order in which we encounter the "on_device" calls can influence the final global
 * device assignment. However we visit global functions in hash map order.
 */
class FreeOnDeviceDefaulter : public ExprVisitor {
 public:
  FreeOnDeviceDefaulter(IRModule mod, std::unique_ptr<DeviceDomains> domains)
      : mod_(std::move(mod)), domains_(std::move(domains)) {}

  std::unique_ptr<DeviceDomains> Default() {
    VLOG_CONTEXT << "FreeOnDeviceDefaulter";
    VLOG(0) << "unifying free on_device annotations";
    for (const auto& kv : mod_->functions) {
      if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
        VLOG(2) << "unifying for '" << kv.first->name_hint << "'";
        VisitExpr(GetRef<Function>(function_node));
      } else {
        VLOG(2) << "skipping '" << kv.first->name_hint << "'";
      }
    }
    return std::move(domains_);
  }

 private:
  void VisitExpr_(const CallNode* call_node) final {
    auto call = GetRef<Call>(call_node);
    OnDeviceProps props = GetOnDeviceProps(call_node);
    ExprVisitor::VisitExpr_(call_node);
    if (props.body.defined() && !props.constrain_body && !props.constrain_result) {
      domains_->OptionalUnifyExprExact(call, props.body);
    }
  }

  /*! \brief The module we are processing. */
  IRModule mod_;
  /*! \brief The domains for all expressions.  */
  std::unique_ptr<DeviceDomains> domains_;
};

/*!
 * \brief Ensures every sub-expression in a module has a device type, using both the global
 * default and some local heuristics to avoid unnecessary additional "device_copy" CallNodes.
 *
 * E.g. in:
 * \code
 *   def @main(%x, %y, %z) {
 *     let %a = add(%x, %y);
 *     multiply(%a, on_device(%z, virtual_device=d))
 *   }
 * \endcode
 * we know the parameter \p %z must be on device \p d, but the devices for \p %x and \p %y,
 * and the device for the function result, are still 'free'. The global 'default' device type
 * is first used to 'fix' \p main's result type, which in turn 'fixes' \p %x and \p %y, which
 * in turn 'fixes' the device on which the \p add and \p multiply are executed.
 *
 * TODO(mbs): I think this is deterministic? We do however visit the top-level defs in hashmap
 * order.
 */
class DeviceDefaulter : public ExprVisitor {
 public:
  DeviceDefaulter(IRModule mod, std::unique_ptr<DeviceDomains> domains)
      : mod_(std::move(mod)), domains_(std::move(domains)) {}

  std::unique_ptr<DeviceDomains> Default() {
    VLOG_CONTEXT << "DeviceDefaulter";
    VLOG(0) << "defaulting to VirtualDevice "
            << domains_->config()->default_primitive_virtual_device;
    for (const auto& kv : mod_->functions) {
      if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
        VLOG(2) << "defaulting devices for '" << kv.first->name_hint << "'";
        VisitExpr(GetRef<Function>(function_node));
      } else {
        VLOG(2) << "skipping '" << kv.first->name_hint << "'";
      }
    }
    return std::move(domains_);
  }

 private:
  void VisitExpr_(const FunctionNode* function_node) final {
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      return;
    }

    auto function = GetRef<Function>(function_node);
    auto func_domain = domains_->DomainFor(function);  // higher-order
    ICHECK_EQ(func_domain->function_arity(), function_node->params.size());
    if (!domains_->IsFullyConstrained(func_domain)) {
      VLOG(2) << "before defaulting function:" << std::endl << domains_->ToString(func_domain);
      domains_->SetResultDefaultThenParams(func_domain,
                                           domains_->config()->default_primitive_virtual_device);
      VLOG(2) << "after defaulting function:" << std::endl << domains_->ToString(func_domain);
    }
    VisitExpr(function_node->body);
  }

  void VisitExpr_(const CallNode* call_node) final {
    auto call = GetRef<Call>(call_node);

    // We don't care if the call is pre- or post-lowered.
    auto vanilla_call = GetAnyCall(call_node);

    auto func_domain = domains_->DomainForCallee(call);  // higher-order
    ICHECK_EQ(func_domain->function_arity(), vanilla_call->args.size());
    if (!domains_->IsFullyConstrained(func_domain)) {
      // For calls to Relay functions this step is identical to that for VisitExpr_(FunctionNode*)
      // above. But for calls to primitives we may still need to force free domains to be
      // defaulted.
      VLOG(2) << "before defaulting callee:" << std::endl
              << PrettyPrint(call_node->op) << std::endl
              << "of domain:" << std::endl
              << domains_->ToString(func_domain);
      domains_->SetResultDefaultThenParams(func_domain,
                                           domains_->config()->default_primitive_virtual_device);
      VLOG(2) << "after defaulting callee:" << std::endl
              << PrettyPrint(call_node->op) << std::endl
              << "of domain:" << std::endl
              << domains_->ToString(func_domain);
    }
    return ExprVisitor::VisitExpr_(call_node);
  }

  void VisitExpr_(const LetNode* let_node) final {
    Expr expr = GetRef<Let>(let_node);
    // Iteratively visit let nodes to avoid stack overflow.
    while (expr->IsInstance<LetNode>()) {
      Let let = Downcast<Let>(expr);
      // If the let-var device is still free force it to match the overall let.
      auto let_domain = domains_->DomainFor(let);  // may be higher-order
      VirtualDevice let_virtual_device = domains_->ResultVirtualDevice(let_domain);
      ICHECK(!let_virtual_device->IsFullyUnconstrained());
      auto let_var_domain = domains_->DomainFor(let->var);  // may be higher-order
      if (!domains_->IsFullyConstrained(let_var_domain)) {
        VLOG(2) << "before defaulting let-var:" << std::endl << domains_->ToString(let_var_domain);
        domains_->SetDefault(let_var_domain, let_virtual_device);
        VLOG(2) << "after defaulting let-var:" << std::endl << domains_->ToString(let_var_domain);
      }
      VisitExpr(let->var);
      VisitExpr(let->value);
      expr = let->body;
    }
    VisitExpr(expr);
  }

  /*! \brief The module we are processing. */
  IRModule mod_;
  /*! \brief The domains for all expressions.  */
  std::unique_ptr<DeviceDomains> domains_;
};

/* =============== Phase 4 =============== */
/*!
 * \brief Inserts missing "device_copy" CallNodes, and ensures the device type of every
 * sub-expression in a module can be easily recovered by a later transformation using simple
 * lexical scoping rules (e.g. for memory planning).
 *
 * - Discard any existing "on_device" CallNodes since their job is done. Similarly, discard
 *   any existing "device_copy" CallNodes which are no-ops.
 *
 * - The result virtual device for a function is stored in the function's virtual_device_ field
 *   and the virtual devices of the function's parameters are stored in the parameter's
 *   virtual_device_ field.
 *
 * - Additional "device_copy" CallNodes are inserted wherever there's a transition between
 *   storage device types. Since the DeviceAnalyzer phase succeeded this can only happen
 *   where the original program explicitly allowed a transition using an "on_device" CallNode.
 *   That is, we do not not try to 'fix' a program with inconsistent devices.
 *
 * - Additional "on_device" CallNodes are inserted so that a later transform can discover
 *   the device for an arbitrary sub-expression by looking only for the lexically enclosing
 *   "on_device" CallNode or "on_device" function attribute. In particular, since function
 *   arguments and let-bound expressions can be on a device different from the function
 *   or let body itself we will insert "on_device" CallNodes to spell out any differences. This
 *   applies even to the argument to a "device_copy" CallNode, which may look pedantic but
 *   keeps downstream processing simple. The "on_device" calls should be removed before code gen,
 *   which is easily done on-the-fly.
 *
 * - Update memory scopes in PrimFunc buffer maps.
 *
 * For example, we'll end up with programs that look like:
 * \code
 *   def @main(%x, %y, param_virtual_devices=[...], result_virtual_device=...) {
 *     let %a = on_device(..., virtual_device=..., is_fixed=True)
 *     @f(%a, device_copy(on_device(..., virtual_device=..., is_fixed=True),
 *                        src_virtual_device=..., dst_virtual_device=...))
 *   }
 * \endcode
 */
class DeviceCapturer : public ExprMutator {
 public:
  DeviceCapturer(IRModule mod, std::unique_ptr<DeviceDomains> domains)
      : mod_(std::move(mod)), domains_(std::move(domains)) {}

  IRModule Capture() {
    VLOG_CONTEXT << "CaptureDevices";
    IRModule result(/*functions=*/{}, mod_->type_definitions, mod_->Imports(), mod_->source_map,
                    mod_->attrs);
    for (const auto& kv : mod_->functions) {
      if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
        VLOG(2) << "capturing devices for Relay Function '" << kv.first->name_hint << "'";
        result->Add(kv.first, Downcast<Function>(Mutate(GetRef<Function>(function_node))));
      } else if (auto prim_func = kv.second.as<tir::PrimFunc>()) {
        VLOG(2) << "capturing devices for TIR PrimFunc '" << kv.first->name_hint << "'";
        tir::PrimFunc new_prim_func = UpdatePrimFunc(kv.first, prim_func.value());
        VLOG(2) << "Rewritten prim func:" << std::endl
                << PrettyPrint(prim_func) << std::endl
                << "to:" << std::endl
                << PrettyPrint(new_prim_func);
        result->Add(kv.first, std::move(new_prim_func));
      } else {
        VLOG(2) << "skipping '" << kv.first->name_hint << "'";
        result->Add(kv.first, kv.second);
      }
    }
    return result;
  }

 private:
  /*!
   * \brief Returns \p prim_func updated to capture any memory scope's implied by its device
   * domain.
   */
  tir::PrimFunc UpdatePrimFunc(const GlobalVar& global_var, const tir::PrimFunc& prim_func) {
    // CAUTION: Same caution as for DeviceAnalyzer::DomainForPrimFunc.
    auto func_domain = domains_->DomainFor(global_var);
    ICHECK(func_domain->is_higher_order());

    const auto* func_type_node = global_var->checked_type().as<FuncTypeNode>();
    ICHECK(func_type_node);
    ICHECK_EQ(func_domain->function_arity(), func_type_node->arg_types.size());

    std::vector<VirtualDevice> arg_and_result_virtual_devices;
    arg_and_result_virtual_devices.reserve(func_type_node->arg_types.size() + 1);
    for (size_t i = 0; i < func_type_node->arg_types.size(); ++i) {
      VirtualDevice param_virtual_device =
          domains_->ResultVirtualDevice(func_domain->function_param(i));
      VLOG(2) << "param_virtual_device[" << i << "] = " << param_virtual_device;
      arg_and_result_virtual_devices.push_back(param_virtual_device);
    }
    VirtualDevice ret_virtual_device =
        domains_->ResultVirtualDevice(func_domain->function_result());
    VLOG(2) << "ret_virtual_device = " << ret_virtual_device;
    arg_and_result_virtual_devices.push_back(ret_virtual_device);

    return tir::ApplyPrimFuncArgAndResultConstraints(prim_func, GetRef<FuncType>(func_type_node),
                                                     arg_and_result_virtual_devices);
  }

  // Nothing interesting for VarNode, ConstantNode, GlobalVarNode, OpNode and ConstructorNode

  Expr VisitExpr_(const TupleNode* tuple_node) final {
    auto tuple = GetRef<Tuple>(tuple_node);
    Array<Expr> fields;
    fields.reserve(tuple_node->fields.size());
    for (const auto& field : tuple_node->fields) {
      fields.push_back(VisitChild(tuple, field));
    }
    return WithFields(tuple, fields);
  }

  Expr VisitExpr_(const FunctionNode* function_node) final {
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Function>(function_node);
    }

    auto function = GetRef<Function>(function_node);
    auto func_domain = domains_->DomainFor(function);  // higher-order
    VLOG(2) << "capturing function:" << std::endl
            << PrettyPrint(function) << std::endl
            << "with domain:" << std::endl
            << domains_->ToString(func_domain);

    // Gather the parameter and result device types for the function attributes.
    ICHECK_EQ(func_domain->function_arity(), function_node->params.size());
    VirtualDevice result_virtual_device = domains_->ResultVirtualDevice(func_domain);
    ICHECK(!result_virtual_device->IsFullyUnconstrained());

    // Map the function parameters to a new variable annotated with a virtual device so
    // we can substitute them later.
    Map<Var, Expr> annotated_bind_map;
    Array<Var> annotated_params;
    annotated_params.reserve(function_node->params.size());
    for (size_t i = 0; i < function_node->params.size(); ++i) {
      VirtualDevice param_virtual_device =
          domains_->ResultVirtualDevice(func_domain->function_param(i));
      VLOG(4) << "Param: " << function_node->params[i];
      Var annotated_var = WithFields(function_node->params[i], {}, {}, param_virtual_device);
      VLOG(4) << "Annotated param: " << annotated_var;
      VLOG(4) << "VirtualDevice: " << annotated_var->virtual_device();
      ICHECK(!param_virtual_device->IsFullyUnconstrained());
      annotated_bind_map.Set(function_node->params[i], annotated_var);
      annotated_params.push_back(annotated_var);
    }
    // Eventually we probably want to bind before visiting, but for now this is causing an issue
    // with the GetVirtualDevice utility, so leaving as is for now.

    // Rewrite the body. Note that the body may have begun with an "on_device" so
    // be prepared to insert a "device_copy".
    Expr body = VisitChild(
        /*lexical_virtual_device=*/result_virtual_device,
        /*expected_virtual_device=*/result_virtual_device,
        /*child_virtual_device=*/GetVirtualDevice(function_node->body), function_node->body);
    VLOG(4) << "Visited body: " << body;
    Function func = WithFields(GetRef<Function>(function_node), function_node->params, body);
    VLOG(4) << "New function: " << func;
    func = SubstituteBoundVars(func, annotated_bind_map);
    VLOG(4) << "Func with bound params: " << func;
    func->virtual_device_ = result_virtual_device;
    VLOG(4) << "Func with bound params & result vid set: " << func;
    return std::move(func);
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = GetRef<Call>(call_node);

    // We don't care if the call is pre- or post-lowered
    // (However we'll preserve the form in the result below.)
    auto vanilla_call = GetAnyCall(call_node);

    VirtualDevice call_virtual_device = GetVirtualDevice(call);

    auto on_device_props = GetOnDeviceProps(call_node);
    if (on_device_props.body.defined()) {
      // We're done with the original "on_device" calls and can pinch them out.
      // Note that this step has already been simulated by GetDeviceType.
      return VisitExpr(on_device_props.body);
    }

    DeviceCopyProps device_copy_props = GetDeviceCopyProps(call_node);
    if (device_copy_props.body.defined()) {
      VirtualDevice src_virtual_device =
          domains_->config()->CanonicalVirtualDevice(device_copy_props.src_virtual_device);
      VirtualDevice dst_virtual_device =
          domains_->config()->CanonicalVirtualDevice(device_copy_props.dst_virtual_device);
      ICHECK_EQ(call_virtual_device, dst_virtual_device);
      if (src_virtual_device == dst_virtual_device) {
        // We can pinch out existing "device_copy" CallNodes if their source and destinations
        // match.
        return VisitExpr(device_copy_props.body);
      } else {
        return VisitChild(/*lexical_virtual_device=*/dst_virtual_device,
                          /*expected_virtual_device=*/dst_virtual_device,
                          /*child_virtual_device=*/src_virtual_device, device_copy_props.body);
      }
    }

    // Generic call.
    auto func_domain = domains_->DomainForCallee(call);  // higher-order
    VLOG(2) << "considering call:" << std::endl
            << PrettyPrint(call) << std::endl
            << "in virtual device " << call_virtual_device
            << " with function virtual devices:" << std::endl
            << domains_->ToString(func_domain);
    VirtualDevice result_virtual_device = domains_->ResultVirtualDevice(func_domain);
    ICHECK(!result_virtual_device->IsFullyUnconstrained());

    // The callee is on the current device.
    Expr op = VisitChild(
        /*lexical_virtual_device=*/call_virtual_device,
        /*expected_virtual_device=*/call_virtual_device,
        /*child_virtual_device=*/result_virtual_device, vanilla_call->op);

    // Each argument can be on the device for the corresponding function parameter. However if
    // any of those differ from the overall call device then wrap them in an "on_device" to
    // help downstream transforms track devices lexically.
    Array<Expr> args;
    args.reserve(vanilla_call->args.size());
    ICHECK_EQ(func_domain->function_arity(), vanilla_call->args.size());
    for (size_t i = 0; i < vanilla_call->args.size(); ++i) {
      VirtualDevice param_virtual_device =
          domains_->ResultVirtualDevice(func_domain->function_param(i));
      ICHECK(!param_virtual_device->IsFullyUnconstrained())
          << "for parameter " << i << " for call:" << std::endl
          << PrettyPrint(call);
      args.push_back(VisitChild(/*lexical_virtual_device=*/call_virtual_device,
                                /*expected_virtual_device=*/param_virtual_device,
                                /*child_virtual_device=*/GetVirtualDevice(vanilla_call->args[i]),
                                vanilla_call->args[i]));
    }

    if (call_node->op == CallLoweredOp()) {
      Call new_call =
          CallLowered(Downcast<GlobalVar>(op), args, /*call_lowered_attrs=*/{}, /*span=*/{});
      return WithFields(call, new_call->op, new_call->args);
    } else {
      return WithFields(call, op, args);
    }
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    Expr expr = GetRef<Expr>(let_node);
    // Iterate through chained lets, provided they all agree on their device type.
    VirtualDevice let_virtual_device = GetVirtualDevice(expr);
    std::vector<std::tuple<Var, Expr, Span>> bindings;
    while (const auto* inner_let = expr.as<LetNode>()) {
      if (GetVirtualDevice(GetRef<Let>(inner_let)) != let_virtual_device) {
        // We have a device transition which needs to be handled.
        break;
      }
      // The let-bound value can be on a different device than the overall let.
      // By using the fully-unconstrained virtual device for the 'lexical' scope we'll force the
      // let-bound value to *always* be wrapped by an "on_device" (see introductory comment for
      // motivation.)
      Expr value =
          VisitChild(/*lexical_virtual_device=*/VirtualDevice::FullyUnconstrained(),
                     /*expected_virtual_device=*/GetVirtualDevice(inner_let->var),
                     /*child_virtual_device=*/GetVirtualDevice(inner_let->value), inner_let->value);
      bindings.emplace_back(inner_let->var, value, inner_let->span);
      expr = inner_let->body;
    }
    Expr body = VisitChild(/*lexical_virtual_device=*/let_virtual_device,
                           /*expected_virtual_device=*/let_virtual_device,
                           /*child_virtual_device=*/GetVirtualDevice(expr), expr);
    for (auto itr = bindings.rbegin(); itr != bindings.rend(); ++itr) {
      body = Let(/*var=*/std::get<0>(*itr), /*value=*/std::get<1>(*itr), body,
                 /*span=*/std::get<2>(*itr));
    }
    return body;
  }

  Expr VisitExpr_(const IfNode* if_node) final {
    auto ife = GetRef<If>(if_node);
    Expr cond = VisitChild(ife, if_node->cond);
    Expr true_branch = VisitChild(ife, if_node->true_branch);
    Expr false_branch = VisitChild(ife, if_node->false_branch);
    return WithFields(ife, cond, true_branch, false_branch);
  }

  Expr VisitExpr_(const TupleGetItemNode* tuple_get_item_node) final {
    auto tuple_get_item = GetRef<TupleGetItem>(tuple_get_item_node);
    Expr tuple = VisitChild(tuple_get_item, tuple_get_item_node->tuple);
    return WithFields(tuple_get_item, tuple);
  }

  Expr VisitExpr_(const RefCreateNode* ref_create_node) final {
    auto ref_create = GetRef<RefCreate>(ref_create_node);
    Expr value = VisitChild(ref_create, ref_create_node->value);
    return WithFields(ref_create, value);
  }

  Expr VisitExpr_(const RefReadNode* ref_read_node) final {
    auto ref_read = GetRef<RefRead>(ref_read_node);
    Expr ref = VisitChild(ref_read, ref_read_node->ref);
    return WithFields(ref_read, ref);
  }

  Expr VisitExpr_(const RefWriteNode* ref_write_node) final {
    auto ref_write = GetRef<RefWrite>(ref_write_node);
    Expr ref = VisitChild(ref_write, ref_write_node->ref);
    Expr value = VisitChild(ref_write, ref_write_node->value);
    return WithFields(ref_write, ref, value);
  }

  Expr VisitExpr_(const MatchNode* match_node) final {
    auto match = GetRef<Match>(match_node);
    Expr data = VisitChild(match, match_node->data);
    Array<Clause> clauses;
    clauses.reserve(match_node->clauses.size());
    for (const auto& clause : match_node->clauses) {
      Pattern lhs = VisitPattern(clause->lhs);  // actually a no-op, so we're not checking vars
      Expr rhs = VisitChild(match, clause->rhs);
      clauses.push_back(Clause(lhs, rhs));
    }
    return WithFields(match, data, clauses);
  }

  VirtualDevice GetVirtualDevice(const Expr& expr) {
    // Look through any "on_device" CallNodes, to mimic how we will be pinching them out.
    OnDeviceProps props = GetOnDeviceProps(expr);
    Expr true_expr = props.body.defined() ? props.body : expr;
    ICHECK(domains_->contains(true_expr));
    // If expr is higher order we'll return only the result domain's device.
    VirtualDevice virtual_device = domains_->ResultVirtualDevice(domains_->DomainFor(true_expr));
    ICHECK(!virtual_device->IsFullyUnconstrained())
        << "no VirtualDevice was determined for expression:" << std::endl
        << PrettyPrint(true_expr);
    return std::move(virtual_device);
  }

  /*!
   * \brief Reconcile the \p child_virtual_device for \p child with both the \p
   * expected_virtual_device (as required by the expression context the \p child is in) and the \p
   * lexical_virtual_device (as a downstream transform would infer based only on lexically enclosing
   * "on_device" CallNodes and function attributes.) Generally \p lexical_virtual_device and \p
   * expected_virtual_device are the same by definition, but may differ in arguments to  functions
   * and let-bound expressions.
   *
   * If \p child_virtual_device differs from \p expected_virtual_device, wrap it as:
   * \code
   *   device_copy(on_device(child', virtual_device=child_virtual_device),
   *               src_dev_type=child_virtual_device, dst_dev_type=expected_virtual_device)
   * \endcode
   * (where child is rewritten to child'). Note the pedantic spelling out of "on_device" on the
   * child.
   *
   * If \p expected_virtual_device differs from \p lexical_virtual_device, then (also) wrap
   * the expression as:
   * \code
   *   on_device(..., virtual_device=expected_virtual_device)
   * \endcode
   *
   * TODO(mbs): There's no attempt at sharing here. If usage of child's node could be wrapped
   * by a "device_copy", even though those copies will generally all be to the same destination
   * device.
   */
  Expr VisitChild(const VirtualDevice& lexical_virtual_device,
                  const VirtualDevice& expected_virtual_device,
                  const VirtualDevice& child_virtual_device, const Expr& child) {
    ICHECK(!expected_virtual_device->IsFullyUnconstrained());
    if (child->IsInstance<OpNode>() || child->IsInstance<ConstructorNode>()) {
      // Primitive operators and contructors don't need to be rewritten and can have a
      // different domain at each call site.
      return child;
    }
    Expr result = VisitExpr(child);
    if (child_virtual_device != expected_virtual_device) {
      VLOG(2) << "creating " << DeviceCopyOp()->name << " from virtual device "
              << child_virtual_device << " to virtual device " << expected_virtual_device
              << " for:" << std::endl
              << PrettyPrint(result);
      // Also wrap the child in an "on_device" so downstream transforms can track devices
      // lexically.
      result = MaybeOnDeviceFixed(result, child_virtual_device);
      result = DeviceCopy(result, child_virtual_device, expected_virtual_device);
    }
    if (expected_virtual_device != lexical_virtual_device) {
      VLOG(2) << "creating " << OnDeviceOp()->name << " for virtual device "
              << expected_virtual_device << " for:" << std::endl
              << PrettyPrint(result);
      result = MaybeOnDeviceFixed(result, expected_virtual_device);
    }
    return result;
  }

  /*!
   * Common case of visiting a direct \p child of \p parent where by default the \p child
   * is expected to be on the same device as the \p parent.
   */
  Expr VisitChild(const Expr& parent, const Expr& child) {
    VirtualDevice expected_virtual_device = GetVirtualDevice(parent);
    VirtualDevice child_virtual_device = GetVirtualDevice(child);
    return VisitChild(expected_virtual_device, expected_virtual_device, child_virtual_device,
                      child);
  }

  /*! \brief Module we are rewriting, so we can lookup global variables. */
  IRModule mod_;
  /*! \brief Device domain for every expression from DeviceAnalyzer. */
  std::unique_ptr<DeviceDomains> domains_;
};

/*! \brief Rewrite the "on_device" calls (and implicitly re-type-check). */
tvm::transform::Pass Rewrite() {
  auto pass_func = [](Function f, IRModule m, transform::PassContext ctxt) {
    auto attrs = m->attrs;
    auto r = Downcast<Function>(RewriteOnDevices(std::move(m)).Mutate(f));
    return attrs.defined() ? WithAttrs(r, {attrs->dict}) : r;
  };
  return tvm::relay::transform::CreateFunctionPass(pass_func, 0, "PlanDevicesRewrite", {});
}

/*! \brief Check the conflicted nodes and add "device_copy" calls. */
tvm::transform::Pass Check(CompilationConfig config) {
  return tvm::transform::CreateModulePass(
      [config = std::move(config)](IRModule mod,
                                   tvm::transform::PassContext pass_cnxt) -> IRModule {
        auto dev_ctx = ConflictedNodeFinder(mod).Finder();
        return ConflictedNodeRewriter(mod, config, std::move(dev_ctx)).Rewrite();
      },
      /*opt_level=*/0, "PlanDevicesCheckConflicts", {});
}

/*! \brief Run the remaining phases. */
tvm::transform::Pass PlanDevicesCore(CompilationConfig config) {
  return tvm::transform::CreateModulePass(
      [config = std::move(config)](IRModule mod,
                                   tvm::transform::PassContext pass_cnxt) -> IRModule {
        // Collect the system of constraints for every sub-expression using existing "on_device"
        // and "device_copy" calls.
        std::unique_ptr<DeviceDomains> domains = DeviceAnalyzer(mod, config).Analyze();
        VLOG(3) << "Domains after analysis:" << std::endl << domains->ToString();

        // Choose sensible default devices for every sub-expression if otherwise unconstrained
        // by existing "on_device" or "device_copy" calls.
        domains = FreeOnDeviceDefaulter(mod, std::move(domains)).Default();
        domains = DeviceDefaulter(mod, std::move(domains)).Default();
        VLOG(3) << "Domains after defaulting: " << std::endl << domains->ToString();

        // Insert "device_copy" and "on_device" CallNodes where needed to unambiguously capture
        // the above map, and attach additional "param_virtual_devices" and "result_virtual_device"
        // attributes to all function definitions.
        return DeviceCapturer(mod, std::move(domains)).Capture();
      },
      /*opt_level=*/0, "PlanDevicesCore", {});
}

}  // namespace

/* =============== Driver =============== */

// This function is declared in the public <tvm/relay/transform.h>.
tvm::transform::Pass PlanDevices(CompilationConfig config) {
  std::vector<Pass> passes;
  passes.emplace_back(Rewrite());
  passes.emplace_back(Check(config));
  passes.emplace_back(InferType());
  passes.emplace_back(PlanDevicesCore(config));
  return tvm::transform::Sequential(passes, "PlanDevices");
}

TVM_REGISTER_GLOBAL("relay._transform.PlanDevices").set_body_typed(PlanDevices);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
