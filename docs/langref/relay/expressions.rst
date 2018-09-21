==================
Expressions
==================

The Relay IR is a pure, expression oriented language, with a dataflow fragment 
and structured control flow. Although Relay's representation is a tree, it is 
possible to view the dataflow fragments as graph for purposes of writing and
expressing transformations.

The below sections make an attempt to clearly split the dataflow
fragment from the control fragment.

==================
Dataflow Expressions
==================

First we will cover the set of nodes which do not involve control flow,
this fragment of the language is semantically equivalent to pure
computation graphs without control flow.

Constants
~~~~~~~~~

Relay programs can contain constant Tensor values. This node represents
a constant tensor value (values are either Tensors, Products, or Closures in Relay).
The constants are represented as :py:class:`~tvm.NDArray`, allowing us to utilize 
TVM operators for constant evaluation.

See :py:class:`~tvm.relay.expr.Constant` for its definition and documentation.

Tuple
~~~~~~~~~~~~~~~

We support tuple constructors; the tuple node builds a finite (i.e statically known size) sequence of 
heterogenous data.  These tuples match closely to Python's and enable efficient projection of their m
embers due to their fixed length.

.. code-block:: python

    (a, b, c) : Tuple<A, B, C>

    (a + b + c, d) : Tuple<Tensor<f32, (10, 10)>, Tensor<f32, 100, 100>>

See :py:class:`~tvm.relay.expr.Tuple` for its definition and documentation.

Function
~~~~~~~~

A function node represents a function, it contains a seqeuence of
parameters, a return type, and a body.

.. code-block:: python

    fun (x : Float, y: Float) -> Float { x + y }

Functions are first class in Relay, and can be used in any expression
position. Functions are the same as global functions, but do not have
an explicit name. You can use a function in conjunction with a let
binding to define locally recursive functions.

.. code-block:: python

    let fact = fun (x : Float) -> Float {
        if (x == 0) {
            0
        } else {
            x * fact(x - 1)
    };
    fact(10)

See :py:class:`~tvm.relay.expr.Function` for its definition and documentation.

Variables
~~~~~~~~~~~

Both global variables, and local variables, are valid expressions, one may use them
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

We will first introduce a single binding with no type
anntoations:

.. code-block:: python
    let %x = %a + %b;
    x

The value of a let binding is the value of the final expression
after evaluating the bindings it depends on.

A user can write a sequence of let bindings, we can view
these blocks and pure dataflow
single binding. These blocks are pure dataflow, and can
be evaluated in any order, reordered up to dataflow.

We support a sequence of bindings followed by a body which
is the continutation after executing the sequence of bindings.

I believe this representation will be easier to manipulate then
the mixed dataflow/control flow comptuation graphs.
Data flow and control flow are strictly seperated in this representation
and we can easily syntactically discriminate. When in ANF there should only be
general control flow between `Assignment` nodes and not within the values bound
in bindings.

This representation also makes it easy to apply reverse more since
sequences of assignments where the only control flow is call instructions
are treated by the algorithm uniformly, and each control flow construct
must be handled individualy.

See :py:class:`~tvm.relay.expr.Let` for its definition and documentation.

=======================
Control Flow Expression
=======================

Control flow expressions change network topology based on values
computed by previous expressions.

Call
~~~~

Terms with function types in Relay are "callable", i.e they can be invoked like
a function in a typical programming language by supplying a set of arguments.

All Relay functions are typed with function types, as well as all Relay operators.

.. code-block:: python
    fact(10)

See :py:class:`~tvm.relay.expr.Call` for its definition and documentation.

If-Then-Else
~~~~~~~~~~~~

Relay has a simple if/then/else expression which allows programs to branch
on a single control value which must be of type :code:`bool`, i.e a zero-rank
tensor of booleans (:code:`Tensor[(), bool]`).

.. code-block:: python
    if (sum(equal(t, u))) {
        jreturn x:
    } else { 
        return y;
    }

See :py:class:`~tvm.relay.expr.If` for its definition and documentation.
