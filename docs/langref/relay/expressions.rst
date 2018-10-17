===========
Expressions
===========

The Relay IR is a pure, expression-oriented language with distinct dataflow
and control flow language fragments. Although Relay's representation is an abstract syntax
tree, it is possible to view the dataflow fragment as graph for purposes of writing and
expressing transformations.

The below sections make an attempt to clearly split the expressions which
are pure dataflow (equivalent to traditional computation graphs) from
the extended expressions which contain control flow.

====================
Dataflow Expressions
====================

First we will cover the set of expressions which do not involve control flow;
this fragment of the language is semantically equivalent to pure computation graphs
without control flow.

Constants
~~~~~~~~~

Relay programs can contain constant Tensor values. This node represents
a constant tensor value (see :py:mod:~tvm.relay.Value for more details).
The constants are represented as :py:class:`~tvm.NDArray`, allowing us to utilize
TVM operators for constant evaluation.

See :py:class:`~tvm.relay.expr.Constant` for its definition and documentation.

Tuple
~~~~~

We support tuple constructors; the tuple node builds a finite (i.e statically known size) sequence of
heterogeneous data. These tuples match closely to Python's and enable efficient projection of their
members due to their fixed length.

.. code-block:: python

    (a, b, c) : Tuple<A, B, C>

    (a + b + c, d) : Tuple<Tensor<f32, (10, 10)>, Tensor<f32, 100, 100>>

See :py:class:`~tvm.relay.expr.Tuple` for its definition and documentation.

Function
~~~~~~~~

A function node represents a function; it contains a sequence of
parameters, a return type, and a body.

.. code-block:: python

    fun (x : Float, y: Float) -> Float { x + y }

Functions are first class in Relay, and can be used in any expression
position. Functions expressions are the same as global functions, but do not
have a globally unique name. You can use a function in conjunction with a let
binding to define locally recursive functions.

.. code-block:: python

    let fact = fun (x : Float) -> Float {
        if (x == 0) {
            0
        } else {
            x * fact(x - 1)
        }
    };
    fact(10)

See :py:class:`~tvm.relay.expr.Function` for its definition and documentation.

Variables
~~~~~~~~~

Both global variables, and local variables are valid expressions, one may use them
anywhere an expression may appear.

For example the below fragment of code is a valid expression.

.. code-block:: python
    %ret = @global(op_name, %local)

See :py:class:`~tvm.relay.expr.LocalVar` and :py:class:`~tvm.expr.GlobalVar` for its definition
and documentation.

Let Binding
~~~~~~~~~~~

An immutable variable binding, allows the user to bind an
expression to a name. A let binding contains a local variable,
an optional type annotation, a value, and body expression
which may reference the bound identifier.

We will first introduce a single binding without
type annotations:

.. code-block:: python
    let %x = %a + %b;
    x

The value of a let binding is the value of the final expression
after evaluating the bindings it depends on.

A sequence of let bindings can be viewed as a dataflow graph,
where the bindings are a series of sub-graphs connected
by bound variables. Since these binding sequences are
pure, we can evaluate them in any order up to the program
dataflow.

For example the below Relay program is equivalent to the
below NNVM program.

.. code-block:: python
    let %y_pred = %x * %w + %b;
    let %loss = pow(%y - %y_pred, 2);
    ret %loss

.. code-block:: python
    TODO

See :py:class:`~tvm.relay.expr.Let` for its definition and documentation.

=======================
Control Flow Expression
=======================

Control flow expressions enable network topology to change based
based on the value of previously executed expressions.

Call
~~~~

Expressions with function types in Relay are "callable", i.e they can be invoked using
a function call.

All Relay functions are typed with function types, as well as all Relay operators.

For example we can call the previously defined `fact` because it has a function
type:

.. code-block:: python
    fact(10)

See :py:class:`~tvm.relay.expr.Call` for its definition and documentation.

If-Then-Else
~~~~~~~~~~~~

Relay has a simple if-then-else expression which allows programs to branch
on a single value of type :code:`bool`, i.e a zero-rank
tensor of booleans (:code:`Tensor[(), bool]`).

.. code-block:: python
    if (sum(equal(t, u))) {
        return x:
    } else {
        return y;
    }

See :py:class:`~tvm.relay.expr.If` for its definition and documentation.
