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
 * \file src/relay/analysis/device_planner.cc
 * \brief Determines a unique device to hold the result of every Relay sub-expression.
 *
 * We say a Relay expression E is 'on device D' if the result of executing E is stored on D.
 * Currently we only track the 'device_type' of D and not its 'device id'. We do not track the
 * specific target associated with D (this is recovered independently via a TargetMap), and we
 * do not track the storage scope within D (this is yet to be implemented).
 *
 * Note that 'stored on device D' is almost but not quite the same as 'executes on device D',
 * see below.
 *
 * This pass assumes the module already contains some "on_device" and/or "device_copy" CallNodes:
 *  - "device_copy" CallNodes (with a \p DeviceCopyAttrs attribute) specify a 'src_dev_type' and
 *    'dst_dev_type' device type, which constrain the argument and context of the call
 *     respectively. It is ok if source and destination devices are the same, such no-op copies
 *     will be removed after accounting for the device preference.
 *  - "on_device" CallNodes (with a \p OnDeviceAttrs attribute) specify a 'device_type', which
 *    constrains the argument of the call, but (usually, see below) leaves the context
 *    unconstrained. These are called 'annotations' in the rest of the code, have no operational
 *    significance by themselves, but may trigger the insertion of a new "device_copy".
 *  - In two situations the result of an "on_device" CallNode may also be constrained to the
 *    given device:
 *     - The "on_device" call occurs at the top-level of a function body, or occurs as an
 *       immediately let-bound expression. In this situation the extra degree of freedom in
 *       the function result and let-binding leads to surprising device copies, so we simply
 *       force the function result or let-bound variable to the given device.
 *     - The \p OnDeviceAttrs has an \p is_fixed field of \p true, which indicates we inserted
 *       it ourselves during an earlier invocation of this pass. This helps make this pass
 *       idempotent.
 *
 * We proceed in four phases:
 *
 * Phase 0
 * -------
 * We rewrite the programs to handle some special cases:
 *  - "on_device" calls at the top-level of function or immediately let-bound are rewritten
 *    to have \code is_fixed=true \endcode.
 *  - We wish to treat \code on_device(expr, device_type=d).0 \endcode as if it were written
 *    \code on_device(expr.0, device_type_d) \endcode. I.e. we prefer to copy the projection from
 *    the tuple rather than project from a copy of the tuple. We'll do this by rewriting.
 *
 * Phase 1
 * -------
 * We flow constraints from the "on_device" and "device_copy" calls (and some special ops, see
 * below) to all other Relay sub-expressions. (For idempotence we also respect any existing
 * "param_device_types" and "result_device_type" function attributes we introduce below.)
 *
 * For a primitive such as \code add(e1, e2) \endcode all arguments and results must be on the
 * same device. However each call site can use a different device. In other words primitives are
 * 'device polymorphic' since we compile and execute them for each required device.
 *
 * For most Relay expressions the device for the overall expression is the same as the device
 * for it's sub-expressions. E.g. each field of a tuple must be on the same device as the tuple
 * itself, the condition and arms of an \p if must all be on the same device as the overall if,
 * and so on.
 *
 * Some special ops (or 'dialects') are handled:
 *  - Relay supports computing the shape of tensors and operators at runtime using "shape_of",
 *    "shape_func", and "reshape_tensor". Shapes must only be held on the CPU, but the tensors
 *    they describe may reside on any device.
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
 * Phase 2
 * -------
 * After flowing constraints we apply some defaulting heuristics (using a global default device)
 * to fix the device for any as-yet unconstrained sub-expressions.
 *  - Unconstrained function result devices default to the global default device.
 *  - Unconstrained function parameters devices default to the device for the function result.
 *  - Unconstrained let-bound expression devices default to the device for the overall let.
 * TODO(mbs): I may have over-innovated here and we simply want to bind all free domaints to
 * the global default device. Worth a design doc with motivating examples I think.
 *
 * Phase 3
 * -------
 * Finally, the result of this analysis is reified into the result as:
 *  - Additional "param_device_types" (an Array<Integer>) and "result_device_type" (Integer)
 *    attributes for every function (both top-level and local). These describe the devices for
 *    the function's parameters and the result.
 *  - Additional "device_copy" CallNodes where a copy is required in order to respect the
 *    intent of the original "on_device" CallNodes.
 *  - Additional "on_device" CallNodes where the device type of an expression does not match
 *    that of the lexically enclosing "on_device" CallNode or function attribute. In practice
 *    this means "on_device" CallNodes may appear in two places:
 *     - On a let-bound expression if its device differs from the overall let expression.
 *     - On a call argument if its device differs from the call result. In particular, the
 *       argument to a "device_copy" call will always be wrapped in an "on_device". (That may
 *       seem pedantic but simplifies downstream handling.)
 *    However since we make it easy to track devices for variables we never wrap an "on_device"
 *    around a var or global var. These uses of "on_device" imply both the argument and result are
 *    on the same device. We signal this by setting the 'is_fixed' OnDeviceAttrs field to true,
 *    which helps make this pass idempotent.
 *
 * Helper visitors (in device_aware_visitors.h) can be used by downstream transforms to recover
 * the device for any expression for their own use, e.g. during memory planning. All downstream
 * passes must preserve the lexical scoping of the "on_device" CallNodes. E.g. conversion
 * to ANF must respect the lexical scoping convention:
 * \code
 * f(on_device(g(h(a, b), c), device_type=CPU))
 * ==>
 * let %x0 = on_device(h(a, b), device_type=CPU)
 * let %x1 = on_device(g(%x0), device-type=CPU)
 * f(on_device(%x1, device_type=CPU))
 * \endcode
 *
 * This pass can be run before FuseOps it can use device-specific fusion rules.
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
 * expression (eg an if expression) on one device even though the tensor data resides on
 * another. But for AOT that flexibility seems excessive. So we'd like to just take 'executes on'
 * to be 'stored on' exactly. In particular, for a Relay function, we'd like to be able to just
 * compile the function body for the function's result device.
 *
 * This works after conversion to ANF provided the compilation for a let expression is prepared
 * to make a cross-device call. However we leave it to a downstream transformation to heuristically
 * minimize cross-device calls by moving device copies out of functions. E.g.:
 * \code
 *   def @f() {  // execute on CPU
 *     let x = on_device(...GPU computation..., device_type=GPU);
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
 *   g(f, on_device(..., device_type=CPU))
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
 * TODO(mbs):
 *  * Though on_device is the identity for all types we can't wrap it around functions/constructors
 *    taking type args (or at least not without changing type_infer.cc to see through them).
 *    This is not currently handled generally.
 *  * Proper diagnostics for unification failure using spans.
 *  * Make sure the pass is idempotent even after FuseOps etc.
 *  * Support application of constructors properly. Are they device polymorphic?
 *  * Replace DLDeviceType with TargetDevice, and unify 'target annotation' with 'device planning'.
 *  * Support running the pass post FuseOps (so need to understand primitive functions, both
 *    outlines and lined) and post the VM transforms (probably need to support more intrinsic
 *    forms?).
 *  * Don't hardcode the 'CPU' device for shape funcs etc, and distinguish between the default
 *    device for primitives vs the default device for the rest of Relay.
 *  * We'll probably need some support for partial 'device polymorphism' for functions once we
 *    incorporate targets and memory scopes into the domain. For example it's ok for the function
 *    body to be executed on different device ids provided they have the same target and memory
 *    scope.
 *  * Might be simpler to just let every type have a device annotation rather than work in
 *    a separate domain?
 *  * Switch to expr.CopyWith(...) form once implemented to avoid unnecessary copies.
 *  * The original device_annotation.cc RewriteAnnotatedOps removed all "on_device" calls
 *    in tuples at the top level of function bodies or main expression, irrespective of the
 *    "on_device" body. What's up with that?
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

#include <unordered_map>

#include "../op/annotation/annotation.h"
#include "../op/memory/device_copy.h"
#include "./device_domains.h"

namespace tvm {
namespace relay {
namespace transform {

namespace {

/******
******* Phase 0
*******/

/*!
 * \brief Rewrites "on_device" calls to handle some special cases.
 *
 * \code
 * let %x = on_device(e, device_type=d)
 * ==> let %x = on_device(e, device_type=d, is_fixed=True)
 *
 * fn(%x) { on_device(e, device_type=d) }
 * ==> fn(%x) { on_device(e, device_type=d, is_fixed=True)
 *
 * on_device(e).0
 * ==> on_device(e.0)
 * \endcode
 */
class RewriteOnDevices : public ExprMutator {
 public:
  RewriteOnDevices() = default;

 private:
  Expr VisitExpr_(const TupleGetItemNode* tuple_get_item_node) final {
    Expr tuple = VisitExpr(tuple_get_item_node->tuple);
    // TODO(mbs): Avoid copy.
    Expr tuple_get_item =
        TupleGetItem(tuple, tuple_get_item_node->index, tuple_get_item_node->span);
    auto props = GetOnDeviceProps(tuple);
    if (props.body.defined() && !props.is_fixed) {
      VLOG(1) << "wrapping tuple get item:" << std::endl
              << PrettyPrint(GetRef<TupleGetItem>(tuple_get_item_node)) << std::endl
              << "with \"on_device\" for device " << props.device_type;
      return OnDevice(tuple_get_item, props.device_type, /*is_fixed=*/false);
    } else {
      return tuple_get_item;
    }
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    auto expr = GetRef<Expr>(let_node);
    std::vector<std::tuple<Var, Expr, Span>> bindings;
    while (const auto* inner_let_node = expr.as<LetNode>()) {
      Expr inner_let = GetRef<Let>(inner_let_node);
      Expr value = VisitExpr(inner_let_node->value);
      auto props = GetOnDeviceProps(value);
      if (props.body.defined() && !props.is_fixed) {
        VLOG(1) << "revising let-bound expression of let:" << std::endl
                << PrettyPrint(expr) << std::endl
                << "to be fixed to device " << props.device_type;
        value = OnDevice(props.body, props.device_type, /*is_fixed=*/true);
      }
      bindings.emplace_back(inner_let_node->var, value, inner_let_node->span);
      expr = inner_let_node->body;
    }
    expr = VisitExpr(expr);
    // TODO(mbs): Avoid copy.
    for (auto itr = bindings.rbegin(); itr != bindings.rend(); ++itr) {
      expr = Let(/*var=*/std::get<0>(*itr), /*value=*/std::get<1>(*itr), expr,
                 /*span=*/std::get<2>(*itr));
    }
    return expr;
  }

  Expr VisitExpr_(const FunctionNode* function_node) final {
    Expr body = VisitExpr(function_node->body);
    auto props = GetOnDeviceProps(body);
    if (props.body.defined() && !props.is_fixed) {
      VLOG(1) << "revising body of function:" << std::endl
              << PrettyPrint(GetRef<Function>(function_node)) << std::endl
              << "to be fixed to device " << props.device_type;
      body = OnDevice(props.body, props.device_type, /*is_fixed=*/true);
    }
    // TODO(mbs): Avoid copy
    return Function(function_node->params, body, function_node->ret_type,
                    function_node->type_params, function_node->attrs, function_node->span);
  }
};

/******
******* Phase 1
*******/

/*
 * \brief Collects the system of device constraints for all sub-expressions in a module.
 * It is possible some devices remain free and will need to be defaulted by \p DeviceDefaulter.
 *
 * Eg from \code add(%x, %y) \endcode we know \p %x and \p %y must be on the same device. Later,
 * from \code on_device(%x, device_type=d) \endcode we know \p %x must be on device \p d, and thus
 * so must \p %y.
 *
 * Constraints can flow in interesting ways. E.g. in:
 * \code
 *   let %f = fn(%x, %y) { add(%x, on_device(%y, device_type=d)) }
 *   let %g = fn(%f, %x, %y) { %f(%x, %y) }
 *   %g(%f, %a, %b)
 * \endcode
 * we discover \p %b must be on device \p d.
 */
class DeviceAnalyzer : public ExprVisitor {
 public:
  explicit DeviceAnalyzer(IRModule mod)
      : mod_(std::move(mod)), domains_(std::make_unique<DeviceDomains>()) {}

  /*!
   * \brief Returns the expression-to-device-domain map for all expressions in all the global
   * function definitions in the module. Expressions may have free domains, these will be resolved
   * by \p DeviceDefaulter below.
   */
  std::unique_ptr<DeviceDomains> Analyze() {
    VLOG_CONTEXT << "DeviceAnalyzer";
    for (const auto& pair : mod_->functions) {
      VLOG(1) << "collecting constraints for '" << PrettyPrint(pair.first) << "'";
      domains_->UnifyExprExact(pair.first, pair.second);
      VisitExpr(pair.second);
    }
    return std::move(domains_);
  }

 private:
  void VisitExpr_(const CallNode* call_node) final {
    auto call = GetRef<Call>(call_node);

    // Find the higher-order domain for the callee. See DomainForCallee for the special rules
    // for primitives.
    VisitExpr(call_node->op);
    auto func_domain = domains_->DomainForCallee(call);  // higher-order

    // Build the domain for the function implied by its arguments and call context.
    ICHECK_EQ(func_domain->function_arity(), call_node->args.size());
    std::vector<DeviceDomainPtr> args_and_result_domains;
    args_and_result_domains.reserve(call_node->args.size() + 1);
    for (const auto& arg : call_node->args) {
      args_and_result_domains.emplace_back(domains_->DomainFor(arg));
      VisitExpr(arg);
    }
    args_and_result_domains.emplace_back(domains_->DomainFor(call));
    auto implied_domain =
        DeviceDomains::MakeDomain(std::move(args_and_result_domains));  // higher-order

    VLOG(1) << "initial call function domain:" << std::endl
            << domains_->ToString(func_domain) << std::endl
            << "and implied domain:" << std::endl
            << domains_->ToString(implied_domain) << std::endl
            << "for call:" << std::endl
            << PrettyPrint(call);

    // The above must match.
    try {
      domains_->Unify(func_domain, implied_domain);  // higher-order
    } catch (const Error& e) {
      // TODO(mbs): Proper diagnostics.
      LOG(FATAL) << "Function parameters and result devices do not match those of call. Call:"
                 << std::endl
                 << PrettyPrint(call) << std::endl
                 << "with function devices:" << std::endl
                 << domains_->ToString(func_domain) << std::endl
                 << "and implied call devices:" << std::endl
                 << domains_->ToString(implied_domain) << std::endl
                 << e.what();
    }

    VLOG(1) << "final call function domain:" << std::endl
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
    // No need to step into fused primitive functions as they are lowered individually according
    // to the devices of all their call sites.
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      return;
    }

    auto function = GetRef<Function>(function_node);
    auto func_domain = domains_->DomainFor(function);  // higher-order

    // The function body domain must match the function result domain.
    domains_->UnifyExprExact(function_node->body,
                             func_domain->function_result());  // may be higher-order

    VLOG(1) << "initial function domain:" << std::endl
            << domains_->ToString(func_domain) << std::endl
            << "and function body domain:" << std::endl
            << domains_->ToString(domains_->DomainFor(function_node->body)) << std::endl
            << "for function:" << std::endl
            << PrettyPrint(function);

    ICHECK_EQ(func_domain->function_arity(), function_node->params.size());
    for (size_t i = 0; i < function_node->params.size(); ++i) {
      // The parameter domains must match the function argument domains.
      domains_->UnifyExprExact(function_node->params[i],
                               func_domain->function_param(i));  // may be higher-order
      VisitExpr(function_node->params[i]);
    }

    // If the function already has device attributes then we can further constrain the
    // function's domain to match them.
    if (GetFunctionResultDeviceType(function_node) != kInvalidDeviceType) {
      std::vector<DeviceDomainPtr> args_and_result;
      for (size_t i = 0; i < function_node->params.size(); ++i) {
        args_and_result.emplace_back(
            domains_->ForDeviceType(function_node->params[i]->checked_type(),
                                    GetFunctionParamDeviceType(function_node, i)));
      }
      args_and_result.emplace_back(domains_->ForDeviceType(
          function_node->body->checked_type(), GetFunctionResultDeviceType(function_node)));
      auto annotation_domain = domains_->MakeDomain(std::move(args_and_result));
      try {
        domains_->Unify(func_domain, annotation_domain);  // higher-order
      } catch (const Error& e) {
        // TODO(mbs): Proper diagnostics.
        LOG(FATAL)
            << "Function devices are incompatible with its \"on_device\" annotation. Function:"
            << std::endl
            << PrettyPrint(function) << std::endl
            << "with function devices:" << std::endl
            << domains_->ToString(func_domain) << std::endl
            << "and annotation devices:" << std::endl
            << domains_->ToString(annotation_domain) << std::endl
            << e.what();
      }
    }

    VisitExpr(function_node->body);

    VLOG(1) << "final function domain:" << std::endl
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
      VisitExpr(tuple->fields[i]);
    }
  }

  void VisitExpr_(const TupleGetItemNode* tuple_get_item_node) final {
    TupleGetItem tuple_get_item = GetRef<TupleGetItem>(tuple_get_item_node);
    auto domain = domains_->DomainFor(tuple_get_item);  // may be higher-order
    domains_->UnifyExprCollapsed(tuple_get_item_node->tuple,
                                 domain);  // collapse to first-order if needed
    VisitExpr(tuple_get_item_node->tuple);
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

/******
******* Phase 2
*******/

/*!
 * \brief Ensures every sub-expression in a module has a device type, using both the global
 * default and some local heuristics to avoid unnecessary additional "device_copy" CallNodes.
 *
 * E.g. in:
 * \code
 *   def @main(%x, %y, %z) {
 *     let %a = add(%x, %y);
 *     multiply(%a, on_device(%z, device_type=d))
 * \endcode
 * we know the parameter \p %z must be on device \p d, but the devices for \p %x and \p %y,
 * and the device for the function result, are still 'free'. The global 'default' device type
 * is first used to 'fix' \p @main's result type, which in turn 'fixes' \p %x and \p %y, which
 * in turn 'fixes' the device on which the \p add and \p multiply are executed.
 *
 * TODO(mbs): I think this is deterministic? We do however visit the top-level defs in hashmap
 * order.
 */
class DeviceDefaulter : public ExprVisitor {
 public:
  DeviceDefaulter(IRModule mod, std::unique_ptr<DeviceDomains> domains,
                  DLDeviceType default_device_type)
      : mod_(std::move(mod)),
        domains_(std::move(domains)),
        default_device_type_(default_device_type) {}

  std::unique_ptr<DeviceDomains> Default() {
    VLOG_CONTEXT << "DeviceDefaulter";
    VLOG(0) << "using default device type " << default_device_type_;
    for (const auto& pair : mod_->functions) {
      VLOG(1) << "defaulting devices for '" << PrettyPrint(pair.first) << "'";
      VisitExpr(pair.second);
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
    if (domains_->AnyFree(func_domain)) {
      VLOG(1) << "before defaulting function:" << std::endl << domains_->ToString(func_domain);
      domains_->SetResultDefaultThenParams(func_domain, default_device_type_);
      VLOG(1) << "after defaulting function:" << std::endl << domains_->ToString(func_domain);
    }
    VisitExpr(function_node->body);
  }

  void VisitExpr_(const CallNode* call_node) final {
    auto call = GetRef<Call>(call_node);
    auto func_domain = domains_->DomainForCallee(call);  // higher-order
    ICHECK_EQ(func_domain->function_arity(), call_node->args.size());
    if (domains_->AnyFree(func_domain)) {
      // For calls to Relay functions this step is identical to that for VisitExpr_(FunctionNode*)
      // above. But for calls to primitives we may still need to force free domains to be
      // defaulted.
      VLOG(1) << "before defaulting callee:" << std::endl << domains_->ToString(func_domain);
      domains_->SetResultDefaultThenParams(func_domain, default_device_type_);
      VLOG(1) << "after defaulting callee:" << std::endl << domains_->ToString(func_domain);
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
      DLDeviceType let_device_type = domains_->ResultDeviceType(let_domain);
      ICHECK_NE(let_device_type, kInvalidDeviceType);
      auto let_var_domain = domains_->DomainFor(let->var);  // may be higher-order
      if (domains_->AnyFree(let_var_domain)) {
        VLOG(1) << "before defaulting let-var:" << std::endl << domains_->ToString(let_var_domain);
        domains_->SetDefault(let_var_domain, let_device_type);
        VLOG(1) << "after defaulting let-var:" << std::endl << domains_->ToString(let_var_domain);
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
  /*! \brief The default device type. */
  DLDeviceType default_device_type_;
};

/******
******* Phase 3
*******/

/*!
 * \brief Inserts missing "device_copy" CallNodes, and ensures the device type of every
 * sub-expression in a module can be easily recovered by a later transformation using simple
 * lexical scoping rules (e.g. for memory planning).
 *
 * - Discard any existing "on_device" CallNodes since their job is done. Similarly, discard
 *   any existing "device_copy" CallNodes which are no-ops.
 *
 * - Functions are given "param_device_types" and "result_device_type" attributes to capture
 *   the device type for its parameters and result.
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
 * For example, we'll end up with programs that look like:
 * \code
 *   def @main(%x, %y, param_device_types=[...], result_device_type=...) {
 *     let %a = on_device(..., device_type=..., is_fixed=True)
 *     @f(%a, device_copy(on_device(..., device_type=..., is_fixed=True),
 *                        src_device_type=..., dst_device_type=...))
 *   }
 * \endcode
 */
class DeviceCapturer : public ExprMutator {
 public:
  DeviceCapturer(IRModule mod, std::unique_ptr<DeviceDomains> domains)
      : mod_(std::move(mod)), domains_(std::move(domains)) {}

  IRModule Capture() {
    VLOG_CONTEXT << "CaptureDevices";
    IRModule result(/*functions=*/{}, mod_->type_definitions, mod_->Imports(), mod_->source_map);
    for (const auto& pair : mod_->functions) {
      VLOG(1) << "capturing devices for '" << PrettyPrint(pair.first) << "'";
      result->Add(pair.first, Downcast<BaseFunc>(Mutate(pair.second)));
    }
    return result;
  }

 private:
  // Nothing interesting for VarNode, ConstantNode, GlobalVarNode, OpNode and ConstructorNode

  Expr VisitExpr_(const TupleNode* tuple_node) final {
    auto tuple = GetRef<Tuple>(tuple_node);
    Array<Expr> fields;
    fields.reserve(tuple_node->fields.size());
    for (const auto& field : tuple_node->fields) {
      fields.push_back(VisitChild(tuple, field));
    }
    // TODO(mbs): Avoid copy
    return Tuple(std::move(fields), tuple_node->span);
  }

  Expr VisitExpr_(const FunctionNode* function_node) final {
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Function>(function_node);
    }

    auto function = GetRef<Function>(function_node);
    auto func_domain = domains_->DomainFor(function);  // higher-order
    VLOG(1) << "capturing function:" << std::endl
            << PrettyPrint(function) << std::endl
            << "with domain:" << std::endl
            << domains_->ToString(func_domain);

    // Gather the parameter and result device types for the function attributes.
    ICHECK_EQ(func_domain->function_arity(), function_node->params.size());
    DLDeviceType result_device_type = domains_->ResultDeviceType(func_domain);
    ICHECK_NE(result_device_type, kInvalidDeviceType);
    Array<Integer> param_device_types;
    param_device_types.reserve(function_node->params.size());
    for (size_t i = 0; i < function_node->params.size(); ++i) {
      DLDeviceType param_device_type = domains_->ResultDeviceType(func_domain->function_param(i));
      ICHECK_NE(param_device_type, kInvalidDeviceType);
      param_device_types.push_back(param_device_type);
    }

    // Rewrite the body. Note that the body may have begun with an "on_device" so
    // be prepared to insert a "device_copy".
    Expr body = VisitChild(
        /*lexical_device_type=*/result_device_type,
        /*expected_device_type=*/result_device_type,
        /*child_device_type=*/GetDeviceType(function_node->body), function_node->body);

    // TODO(mbs): Avoid copy
    Function func = Function(function_node->params, body, function_node->ret_type,
                             function_node->type_params, function_node->attrs, function_node->span);
    return FunctionOnDevice(func, param_device_types, result_device_type);
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = GetRef<Call>(call_node);
    DLDeviceType call_device_type = GetDeviceType(call);

    auto on_device_props = GetOnDeviceProps(call_node);
    if (on_device_props.body.defined()) {
      // We're done with the original "on_device" calls and can pinch them out.
      // Note that this step has already been simulated by GetDeviceType.
      return VisitExpr(on_device_props.body);
    }

    auto device_copy_props = GetDeviceCopyProps(call_node);
    if (device_copy_props.body.defined()) {
      DLDeviceType src_device_type = device_copy_props.src_dev_type;
      ICHECK_EQ(call_device_type, device_copy_props.dst_dev_type);
      if (call_device_type == src_device_type) {
        // We can pinch out existing "device_copy" CallNodes if their source and destinations
        // match.
        return VisitExpr(device_copy_props.body);
      }
      // else: handle as for any other call.
    }

    auto func_domain = domains_->DomainForCallee(call);  // higher-order
    VLOG(1) << "considering call:" << std::endl
            << PrettyPrint(call) << std::endl
            << "on device " << call_device_type << " with function domain:" << std::endl
            << domains_->ToString(func_domain);
    DLDeviceType result_device_type = domains_->ResultDeviceType(func_domain);
    ICHECK_NE(result_device_type, kInvalidDeviceType);

    // The callee is on the current device.
    Expr op = VisitChild(
        /*lexical_device_type=*/call_device_type,
        /*expected_device_type=*/call_device_type,
        /*child_device_type=*/result_device_type, call_node->op);

    // Each argument can be on the device for the corresponding function parameter. However if
    // any of those differ from the overall call device then wrap them in an "on_device" to
    // help downstream transforms track devices lexically.
    Array<Expr> args;
    args.reserve(call_node->args.size());
    ICHECK_EQ(func_domain->function_arity(), call->args.size());
    for (size_t i = 0; i < call_node->args.size(); ++i) {
      DLDeviceType param_device_type = domains_->ResultDeviceType(func_domain->function_param(i));
      ICHECK_NE(param_device_type, kInvalidDeviceType)
          << "for parameter " << i << " for call:" << std::endl
          << PrettyPrint(call);
      args.push_back(VisitChild(/*lexical_device_type=*/call_device_type,
                                /*expected_device_type=*/param_device_type,
                                /*child_device_type=*/GetDeviceType(call_node->args[i]),
                                call_node->args[i]));
    }
    // TODO(mbs): Avoid copy
    return Call(std::move(op), std::move(args), call_node->attrs, call_node->type_args,
                call_node->span);
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    Expr expr = GetRef<Expr>(let_node);
    // Iterate through chained lets, provided they all agree on their device type.
    DLDeviceType let_device_type = GetDeviceType(expr);
    std::vector<std::tuple<Var, Expr, Span>> bindings;
    while (const auto* inner_let_node = expr.as<LetNode>()) {
      Expr inner_let = GetRef<Let>(inner_let_node);
      if (GetDeviceType(inner_let) != let_device_type) {
        // We have a device transition which needs to be handled.
        break;
      }
      // The let-bound value can be on a different device than the overall let. However if those
      // devices don't agree wrap the let-bound value in an "on_device" to help downstream
      // transforms track devices lexically.
      Expr value = VisitChild(/*lexical_device_type=*/let_device_type,
                              /*expected_device_type=*/GetDeviceType(inner_let_node->var),
                              /*child_device_type=*/GetDeviceType(inner_let_node->value),
                              inner_let_node->value);
      bindings.emplace_back(inner_let_node->var, value, inner_let_node->span);
      expr = inner_let_node->body;
    }
    Expr body = VisitChild(/*lexical_device_type=*/let_device_type,
                           /*expected_device_type=*/let_device_type,
                           /*child_device_type=*/GetDeviceType(expr), expr);
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
    // TODO(mbs): Avoid copy
    return If(cond, true_branch, false_branch, if_node->span);
  }

  Expr VisitExpr_(const TupleGetItemNode* tuple_get_item_node) final {
    auto tuple_get_item = GetRef<TupleGetItem>(tuple_get_item_node);
    Expr tuple = VisitChild(tuple_get_item, tuple_get_item_node->tuple);
    // TODO(mbs): Avoid copy
    return TupleGetItem(tuple, tuple_get_item_node->index, tuple_get_item_node->span);
  }

  Expr VisitExpr_(const RefCreateNode* ref_create_node) final {
    auto ref_create = GetRef<RefCreate>(ref_create_node);
    Expr value = VisitChild(ref_create, ref_create_node->value);
    // TODO(mbs): Avoid copy
    return RefCreate(value, ref_create_node->span);
  }

  Expr VisitExpr_(const RefReadNode* ref_read_node) final {
    auto ref_read = GetRef<RefRead>(ref_read_node);
    Expr ref = VisitChild(ref_read, ref_read_node->ref);
    // TODO(mbs): Avoid copy
    return RefRead(ref, ref_read_node->span);
  }

  Expr VisitExpr_(const RefWriteNode* ref_write_node) final {
    auto ref_write = GetRef<RefWrite>(ref_write_node);
    Expr ref = VisitChild(ref_write, ref_write_node->ref);
    Expr value = VisitChild(ref_write, ref_write_node->value);
    // TODO(mbs): Avoid copy
    return RefWrite(ref, value, ref_write_node->span);
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
    // TODO(mbs): Avoid copy
    return Match(data, std::move(clauses), match_node->complete, match_node->span);
  }

  DLDeviceType GetDeviceType(const Expr& expr) {
    // Look through any "on_device" CallNodes, to mimic how we will be pinching them out.
    auto props = GetOnDeviceProps(expr);
    Expr true_expr = props.body.defined() ? props.body : expr;
    ICHECK(domains_->contains(true_expr));
    // If expr is higher order we'll return only the result domain's device type.
    DLDeviceType device_type = domains_->ResultDeviceType(domains_->DomainFor(true_expr));
    ICHECK_NE(device_type, kInvalidDeviceType)
        << "no device type was determined for expression:" << std::endl
        << PrettyPrint(true_expr);
    return device_type;
  }

  /*!
   * \brief Reconcile the \p child_device_type for \p child with both the \p expected_device_type
   * (as required by the expression context the \p child is in) and the \p lexical_device_type
   * (as a downstream transform would infer based only on lexically enclosing "on_device"
   * CallNodes and function attributes.) Generally \p lexical_device_type and \p
   * expected_device_type are the same by definition, but may differ in arguments to  functions
   * and let-bound expressions.
   *
   * If \p child_device_type differs from \p expected_device_type, wrap it as:
   * \code
   *   device_copy(on_device(child', device_type=child_device_type),
   *               src_dev_type=child_device_type, dst_dev_type=expected_device_type)
   * \endcode
   * (where child is rewritten to child'). Note the pedantic spelling out of "on_device" on the
   * child.
   *
   * If \p expected_device_type differs from \p lexical_device_type, then (also) wrap
   * the expression as:
   * \code
   *   on_device(..., device_type=expected_device_type)
   * \endcode
   *
   * TODO(mbs): There's no attempt at sharing here. If usage of child's node could be wrapped
   * by a "device_copy", even though those copies will generally all be to the same destination
   * device.
   */
  Expr VisitChild(DLDeviceType lexical_device_type, DLDeviceType expected_device_type,
                  DLDeviceType child_device_type, const Expr& child) {
    ICHECK_NE(lexical_device_type, kInvalidDeviceType);
    ICHECK_NE(expected_device_type, kInvalidDeviceType);
    if (child->IsInstance<OpNode>()) {
      // Primitive operators don't need to be rewritten and can have a different domain for
      // each call site.
      return child;
    }
    Expr result = VisitExpr(child);
    if (child_device_type != expected_device_type) {
      VLOG(1) << "creating " << DeviceCopyOp()->name << " from device type " << child_device_type
              << " to device type " << expected_device_type << " for:" << std::endl
              << PrettyPrint(result);
      // Also wrap the child in an "on_device" so downstream transforms can track devices
      // lexically.
      result = MaybeOnDevice(result, child_device_type, /*is_fixed=*/true);
      result = DeviceCopy(result, child_device_type, expected_device_type);
    }
    if (expected_device_type != lexical_device_type) {
      VLOG(1) << "creating " << OnDeviceOp()->name << " for device type " << expected_device_type
              << " for:" << std::endl
              << PrettyPrint(result);
      result = MaybeOnDevice(result, expected_device_type, /*is_fixed=*/true);
    }
    return result;
  }

  /*!
   * Common case of visiting a direct \p child of \p parent where by default the \p child
   * is expected to be on the same device as the \p parent.
   */
  Expr VisitChild(const Expr& parent, const Expr& child) {
    DLDeviceType expected_device_type = GetDeviceType(parent);
    DLDeviceType child_device_type = GetDeviceType(child);
    return VisitChild(expected_device_type, expected_device_type, child_device_type, child);
  }

  /*! \brief Module we are rewriting, so we can lookup global variables. */
  IRModule mod_;
  /*! \brief Device domain for every expression from DeviceAnalyzer. */
  std::unique_ptr<DeviceDomains> domains_;
};

/*! \brief Rewrite the "on_device" calls (and implicitly re-type-check). */
tvm::transform::Pass Rewrite() {
  auto pass_func = [](Function f, IRModule m, transform::PassContext ctxt) {
    return Downcast<Function>(RewriteOnDevices().Mutate(f));
  };
  return tvm::relay::transform::CreateFunctionPass(pass_func, 0, "PlanDevicesRewrite", {});
}

/*! \brief Run the remaining phases. */
tvm::transform::Pass PlanDevicesCore(DLDeviceType default_device_type) {
  return tvm::transform::CreateModulePass(
      [=](IRModule mod, tvm::transform::PassContext pass_cnxt) -> IRModule {
        // Collect the system of constraints for every sub-expression using existing "on_device"
        // and "device_copy" calls.
        std::unique_ptr<DeviceDomains> domains = DeviceAnalyzer(mod).Analyze();
        VLOG(1) << "Domains after analysis:" << std::endl << domains->ToString();

        // Choose sensible default devices for every sub-expression if otherwise unconstrained
        // by existing "on_device" or "device_copy" calls.
        domains = DeviceDefaulter(mod, std::move(domains), default_device_type).Default();
        VLOG(1) << "Domains after defaulting: " << std::endl << domains->ToString();

        // Insert "device_copy" and "on_device" CallNodes where needed to unambiguously capture
        // the above map, and attach additional "param_device_types" and "result_device_type"
        // attributes to all function definitions.
        return DeviceCapturer(mod, std::move(domains)).Capture();
      },
      /*opt_level=*/0, "PlanDevicesCore", {});
}

}  // namespace

/******
******* Overall composite Pass
*******/

// This function is declared in the public <tvm/relay/transform.h>.
TVM_DLL tvm::transform::Pass PlanDevices(DLDeviceType default_device_type) {
  std::vector<Pass> passes;
  passes.emplace_back(Rewrite());
  passes.emplace_back(PlanDevicesCore(default_device_type));
  return tvm::transform::Sequential(std::move(passes), "PlanDevices");
}

TVM_REGISTER_GLOBAL("relay._transform.PlanDevices")
    .set_body_typed([](const Device& default_device) {
      return PlanDevices(default_device.device_type);
    });

}  // namespace transform
}  // namespace relay
}  // namespace tvm
