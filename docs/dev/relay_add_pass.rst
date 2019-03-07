.. _relay-add-pass:

Adding a Compiler Pass to Relay
===============================

Compiler passes can be used to collect data about Relay programs or transform
them in various ways. The base class used to traverse programs is
``ExprFunctor``. Depending on what you want to do, there are subclasses
``ExprVisitor`` and ``ExprMutator`` that collect information or mutate the
AST, respectively.

At a high level, there are three key components to writing a pass:

- Creating one or more C++ classes that traverse the program
- Registering an API endpoint with the ``TVM_REGISTER_API`` macro that performs the pass
- Wrapping the Python API hook in a neater interface

In order to better understand the process of writing a pass, we will look at
the constant folding pass (found in ``src/relay/pass/fold_constant.cc`` and
in ``python/tvm/relay/ir_pass.py``) as a guide, because it is a relatively
simple pass that incorporates both types of traversals.

Background
----------

Constant folding involves evaluating expressions in the program that only
involve constant values, then replacing those expressions with the result
of evaluating them. The goal of this pass is to frontload all of the
computations that we can. With that background, we now present an example of
an ``ExprVisitor``, then an example of an ``ExprMutator``.

Writing an Expression Visitor
-----------------------------

The constant folding pass makes use of a ``ConstantChecker``, which extends
``ExprVisitor``. The public interface to this class is a ``Check`` method
that returns whether the given expression is considered constant.
In Relay, we define an expression to be constant if it is a
``ConstantNode`` or it is a ``TupleNode`` with only constant fields.

Internally, the ``ConstantChecker`` uses the ``VisitExpr_`` functions to
check for constness. Because ``VisitExpr_`` returns nothing, we need to keep
track of our results in our own data structure. So we use a ``memo_`` field
to map from nodes to whether they are constant. Additionally, ``memo_``
serves as a cache for later ``Check`` calls on the same expression. Below are
the ``VisitExpr_`` definitions in the ``ConstantChecker``.

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

Note that we don't need to define a ``VisitExpr_`` function for every type of
node, because ``ExprVisitor`` defines a default implementation for every
node. This default implementation simply visits all of its fields that are
nodes. So we can default to returning false and only define ``VisitExpr_``
methods for the node types that could possibly change the result to true
(i.e., the ones listed in our constness condition earlier).

Writing an Expression Mutator
-----------------------------

The second component of the constant folding pass is the ``ConstantFolder``,
which extends ``ExprMutator`` and internally uses ``ConstantChecker``. The
``ConstantFolder`` class is used to implement the ``FoldConstant`` function,
so it can be registered to become an API endpoint.

In the ``ConstantChecker``, we defined our own ``memo_`` field, but we don't
need to do the same in ``ConstantFolder``. ``ExprMutator`` already provides a
memoizer, because we know its ``VisitExpr_`` functions will always take
expressions and return expressions.

Similarly to ``ExprVisitor``, ``ExprMutator`` also defines default
implementations for all node types. Each ``VisitExpr_`` method visits all of
its node fields and creates a new node if any of the resulting fields have
changed. Otherwise, the original node is returned.

In Relay, there are three types of nodes that we want to perform constant
folding on:

- ``LetNode``s
- ``TupleItemGetNode``s
- ``CallNode``s

For ``LetNode``s, we first attempt to const-fold the value being bound in the
expression. If we can, then we populate ``memo_`` and return the result of
visiting the body---essentially, propagating the bound value to its use sites
in the body. If we can't const-fold the bound value, we mimic the default
implementation.

For ``TupleItemGetNode``s, we check if the index expression can be
const-folded. If so, we replace the tuple get with the field the evaluated
constant points to.

For ``CallNode``s, we first use the ``VisitExpr_`` of ``ExprMutator`` to
visit the call, which const-folds all of the call parameters. Then we
evaluate the call only if all of the arguments are constant (using the
``ConstantChecker``). Evaluating the call produces a value, so we use a
helper method ``ValueToExpr`` to allow us to place the evaluated expression
back into the AST.

Wrapping the API in Python
-----------------------------

TODO: The content of this section should be almost identical to the
corresponding section in ``docs/dev/relay_add_op.rst``. Can we just link to
there?

Summary
-------
This was a walkthrough of a single compiler pass in Relay, and hopefully it
gave a sufficient background to both understand the structure of other passes
and to write your own.
