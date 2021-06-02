..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _relay-add-pass:

Adding a Compiler Pass to Relay
===============================

Compiler passes are the primary interface for both extending Relay's feature
set and for performing optimizations on Relay programs. By writing a compiler
pass, you can modify the AST or collect information about the AST,
depending on your goal. Indeed, some of Relay's most important built-in
features (e.g., autodiff and type inference) are nothing more than "standard"
compiler passes.

At a high level, there are two key components to writing a pass:

- Creating one or more C++ classes that traverse the program
- Wrapping the traversal implementation and its metadata in the pass manager API so it can neatly interface with the :ref:`pass-infra`

To begin, we'll give an overview of the key mechanisms for writing a compiler
pass. Then, we'll walk through a concrete example of the constant-folding
pass in Relay.

AST Traversers
--------------

The base class used to traverse Relay programs is ``ExprFunctor``. The public
interface it provides is a ``VisitExpr`` method that takes an expression and
zero or more arguments and returns an instance of some type. When you extend
this class, you define the AST traversal pattern by overriding
implementations of ``VisitExpr_`` for each type of expression.

The relation between ``VisitExpr`` and ``VisitExpr_`` has to do with
dispatch. Each ``VisitExpr_`` definition targets a specific type of
expression, but you don't always know which node type you'll be visiting.
To remedy this, ``ExprFunctor`` provides a ``VisitExpr`` function which
routes from the given expression to the ``VisitExpr_`` case that handles it.
Although C++ already provides dynamic dispatch, ``ExprFunctor`` defines its
own vtable, which ``VisitExpr`` uses. By defining our own vtable, we have
more control over dispatch. For example, if we wanted to define a
``PrintVisitor`` traverser that printed "Here" before every visit, we
could override ``VisitExpr``:

.. code:: c

    void PrintVisitor::VisitExpr(const Expr& expr) {
      std::cout << "Here" << std::endl;
      ExprFunctor::VisitExpr(expr);
    }

``ExprFunctor`` itself is a very general class, which is why more often than
not, you will be extending ``ExprVisitor`` or ``ExprMutator``. These classes
extend ``ExprFunctor`` and provide default implementations of ``VisitExpr_``
that capture common traversal patterns for each expression type. Having these
default implementations means we only need to provide overriding
implementations for the expression types where we want different behavior. We
describe each subclass on its own in the following sections.

Expression Visitors
~~~~~~~~~~~~~~~~~~~

``ExprVisitor`` is for passes that don't modify the program and instead
perform program analyses and collect information. With this class,
``VisitExpr`` and the private counterparts return nothing. The ``VisitExpr_``
implementations provided by this class simply visit all of the expression's
fields that are expressions. The default implementation for ``IfNode`` is
shown below.

.. code:: c

    void ExprVisitor::VisitExpr_(const IfNode* op) {
      this->VisitExpr(op->cond);
      this->VisitExpr(op->true_branch);
      this->VisitExpr(op->false_branch);
    }

Note that we're calling ``VisitExpr`` and not ``VisitExpr_`` here, so we can
use the vtable in ``ExprFunctor`` for routing.

Now, if we wanted to write a class ``CallChecker`` that checks if any
function calls appear in the program, we would only need to extend
``ExprVisitor`` and define the following ``VisitExpr_`` method:

.. code:: c

    void VisitExpr_(const CallNode* n) final {
      result_ = true;
    }

where ``result_`` is a field. In this case, we don't need to further recurse
on the fields of the ``CallNode``, because ``result_`` is already true and we
now know the original expression contains a call. To make this visitor
usable, we would provide the following public method:

.. code:: c

    bool Check(const Expr& expr) final {
      result_ = false;
      VisitExpr(expr);
      return result_;
    }

And that's all we need. It is very common to define a public interface that
performs some bookkeeping before invoking the top-level recursion. We could
of course further wrap the API by making a standalone procedure that creates
a ``CallChecker`` instance and calls ``Check`` on it, but the takeaway is
that we've achieved our goal with very little effort.

Expression Mutators
~~~~~~~~~~~~~~~~~~~

``ExprMutator`` is for passes that transform the program in some way. With
this class, ``VisitExpr`` and its private counterparts return ``Expr``. The
default ``VisitExpr_`` implementations provided by this class visit all of
the expression's fields that are expressions and set the fields to be the
result of visiting them. The default implementation for ``TupleGetItemNode``
is shown below.

.. code:: c

    Expr ExprMutator::VisitExpr_(const TupleGetItemNode* g) {
      auto t = this->Mutate(g->tuple);
      if (g->tuple == t) {
        return GetRef<Expr>(g);
      } else {
        return TupleGetItem(t, g->index);
      }
    }

There are a few things to notice here. First, ``Mutate`` is an alias for
``VisitExpr`` in ``ExprMutator``. Second, we only return a new node if the
call to ``Mutate`` modified the ``tuple`` field. This method of update is
called a functional update and doing so avoids unnecessary allocations.

One feature ``ExprMutator`` has that ``ExprVisitor`` doesn't is a built-in
``memo_`` field for caching results. It makes sense that ``ExprMutator`` has
a memoizer, because we know which types of results we're caching (i.e.,
``Expr``), whereas the visit methods of ``ExprVisitor`` don't return
anything. Usually, when we want to cache results in a subclass of
``ExprVisitor``, we need to define the cache ourselves.

Now, if we wanted to write a class ``IfCollapser`` that replaces every if
statement with its true branch, we would override ``VisitExpr_`` for
``IfNode``:

.. code:: c

    Expr ExprMutator::VisitExpr_(const IfNode* op) {
      return this->Mutate(op->true_branch);
    }

Note that the returned expression will not necessarily be an ``IfNode``, and
this is fine, because the return type is ``Expr``. Now, we create the public
interface:

.. code:: c

    Expr CollapseIfs(const Expr& expr) final {
      return this->Mutate(expr);
    }

With this mutator, we didn't need to do any bookkeeping, but we still want to
follow the convention of having a descriptive method as the interface.

Example: Constant Folding
-------------------------

In order to better understand the process of writing a pass, we will look at
the constant folding pass (found in `src/relay/transforms/fold_constant.cc`_)
as a guide, because it is a relatively simple pass that incorporates
both types of traversals.

Constant folding involves evaluating expressions in the program that only
involve constant values, then replacing those expressions with the result
of evaluating them. The goal of this pass is to frontload all of the
computations that we can. To achieve this, the constant folding pass makes
use of a visitor (``ConstantChecker``) and a mutator (``ConstantFolder``).

The ``ConstantChecker`` Visitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This visitor is used to check if an expression is constant. In Relay, we
define an expression to be constant if it is a ``ConstantNode`` or it is a
``TupleNode`` with only constant fields.

We use a ``memo_`` field to map from nodes to whether they are constant and
to cache these results. Below are the ``VisitExpr_`` definitions in the
``ConstantChecker``.

.. code:: c

    void VisitExpr_(const ConstantNode* n) final {
      memo_[GetRef<Constant>(n)] = true;
    }

    void VisitExpr_(const TupleNode* n) final {
      bool result = true;
      for (const auto& field : n->fields) {
        if (!Check(field)) {
          result = false;
          break;
        }
      }
      memo_[GetRef<Tuple>(n)] = result;
    }

The bookkeeping used to coordinate these definitions is a ``Check`` method
that returns whether the given expression is considered constant.

.. code:: c

    bool Check(const Expr& expr) {
      const auto it = memo_.find(expr);
      if (it != memo_.end())
        return it->second;
      VisitExpr(expr);
      return memo_[expr];
    }

We don't modify ``memo_`` for every node we encounter; instead we only modify
``memo_`` when the encountered node could potentially be constant. Then we
rely on the default value being false when ``memo_`` doesn't contain
``expr``.

The ``ConstantFolder`` Mutator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This mutator performs the bulk of the constant folding pass and internally
uses ``ConstantChecker``. In Relay, there are three node types that are
involved in constant folding: ``LetNode``, ``TupleItemGetNode``, and
``CallNode``. In the following paragraphs, we explain the roles of each in
the pass.

.. code:: c

    Expr VisitExpr_(const LetNode* op) final {
      Expr value = this->Mutate(op->value);
      if (value.as<ConstantNode>()) {
        memo_[op->var] = value;
        return this->Mutate(op->body);
      } else {
        Var var = Downcast<Var>(this->Mutate(op->var));
        Expr body = this->Mutate(op->body);
        if (var.same_as(op->var) &&
            value.same_as(op->value) &&
            body.same_as(op->body)) {
          return GetRef<Expr>(op);
        } else {
          return Let(var, value, body);
        }
      }
    }

In the ``LetNode`` case, we first attempt to const-fold the value being bound
in the expression. If we can, then we populate ``memo_`` and return the
result of visiting the body---essentially, propagating the bound value to its
use sites in the body. If we can't const-fold the bound value, we mimic the
default implementation.

.. code:: c

    Expr VisitExpr_(const TupleGetItemNode* op) final {
      Expr res = ExprMutator::VisitExpr_(op);
      op = res.as<TupleGetItemNode>();
      if (const auto* tuple = op->tuple.as<TupleNode>()) {
        return tuple->fields[op->index];
      } else {
        return res;
      }
    }

In the ``TupleItemGetNode`` case, we check if ``op->tuple`` field is a
``TupleNode``. If so, we replace the tuple get with the field of the tuple
pointed to by ``op->index``. The reason we need to check is because
``op->tuple`` might evaluate to a tuple, without itself being a tuple.

.. code:: c

    Expr VisitExpr_(const CallNode* call) final {
      static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");
      Expr res = ExprMutator::VisitExpr_(call);
      call = res.as<CallNode>();
      // We don't constant fold function with zero arguments.
      // This is a heuristic that is useful.
      // For example it is harmful to fold ones(shape=(4, 5)).
      if (call->args.size() == 0) return res;
      const OpNode* op = call->op.as<OpNode>();
      if (op == nullptr) return res;
      // skip stateful ops.
      if (op_stateful.get(GetRef<Op>(op), false)) return res;
      bool all_const_args = true;
      for (Expr arg : call->args) {
        if (!checker_.Check(arg)) {
          all_const_args = false;
        }
      }
      if (all_const_args) {
        return ConstEvaluate(res);
      } else {
        return res;
      }
    }

In the ``CallNode`` case, we first use the ``VisitExpr_`` of ``ExprMutator``
to visit the call, which const-folds all of the fields of the call. We use
``ExprMutator::VisitExpr_`` instead of ``VisitExpr``, because we want to
bypass the vtable (to avoid an infinite loop) and use the default
implementation provided by ``ExprMutator``. Then we evaluate the call only if
all of the arguments are constant (using ``ConstantChecker``). Evaluating the
call produces a **value**, so we use a helper method ``ValueToExpr`` to allow
us to place the evaluated expression back into the AST.

Now, we construct a more convenient interface ``FoldConstant`` for our constant
folder. ``FoldConstant`` is a standalone function outside of the ``ConstantFolder``
class that takes an expression and internally creates and uses a
``ConstantFolder`` instance (the full definition can be found in
`src/relay/transforms/fold_constant.cc`_).


Registering a Pass with the Pass Manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Note: please see the documentation on the :ref:`pass-infra` for more specific detail on this subject.*

With the AST traversers written, the pass can be registered to become a TVM
API endpoint with the following code:

.. code:: c

    namespace transform {

    Pass FoldConstant() {
      runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
        [=](Function f, Module m, PassContext pc) {
          return Downcast<Function>(FoldConstant(f));
      };
      return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
    }

    }  // namespace transform

If the ``Pass`` object produced by the above code is given to the pass infrastructure,
it will ensure that the AST traversal is applied to every function in the
given Relay module, which is the behavior one would expect for a constant folding
pass (it should fold all constants where possible).

The function ``CreateFunctionPass``
allows for registering the optimization level of the pass (in this case, 2), which can
be used to group together passes based on their general utility, a name for the pass,
and any dependencies for the pass. A pass's dependencies are given as a list of any passes
whose results are necessary to be able to run the current pass. ``FoldConstant`` does not
have any dependencies, but many Relay passes do depend on having type information,
so ``InferType`` is a common dependency; others may depend on the program's being in
A-normal form, via the ``ToANormalForm`` pass.

Note that the ``PassContext`` object contains information a pass uses for
error reporting and configuration options; ``FoldConstant`` does not need
this information but other passes may reference their ``PassContext`` objects.

The pass can now be invoked via the pass infrastructure, though it's a good idea to
also add a Python binding for the pass, as in this code snippet:

.. code:: c

   TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
   .set_body_typed(FoldConstant);

Once ``Pass`` objects are defined in the above fashion, they can be invoked using the
pass infrastructure's ``Sequential`` construct, which takes a list of passes and applies
them in sequence to a Relay module, obtaining a transformed module as a result. For example,
the below code applies both the ``FoldConstant`` and ``ToANormalForm`` passes
(one after the other) to each function in ``mod`` and obtains a new module.

.. code:: python

    seq = transform.Sequential([
        relay.transform.FoldConstant(),
        relay.transform.ToANormalForm()
    ])
    new_mod = seq(mod)

More detail about registration can be found in :ref:`tvm-runtime-system` and more
information about the pass manager interface can be found in :ref:`pass-infra`.
Relay's standard passes are listed in `include/tvm/relay/transform.h`_ and implemented
in `src/relay/transforms/`_.

.. _include/tvm/relay/transform.h: https://github.com/apache/tvm/blob/main/include/tvm/relay/transform.h

.. _src/relay/transforms/: https://github.com/apache/tvm/tree/main/src/relay/transforms

.. _src/relay/transforms/fold_constant.cc: https://github.com/apache/tvm/blob/main/src/relay/transforms/fold_constant.cc
