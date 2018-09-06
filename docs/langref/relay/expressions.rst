==================
Expressions
==================

Relay's IR is a pure expression oriented language, that has a
dataflow fragment and structured control flow. Although Relay's
representation is a tree, it is possible to view the dataflow
fragments as graph for purposes of writing and expressing
transformations.

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
Relay programs can contain constant Tensor values, since in Relay all
values are either Tensors, Products, or Closures. We will discuss the
later two later, but we represent Tensor constants as `tvm.NDArray`,
allowing us to utilize normal operators for constant evaluation.


Constructors
~~~~~~~~

Relay supports a handful of constructors which we will cover below. A
constructor enables programs to build new values from arbitrary Relay
expressions.


We support four types of literals, literals are type polymorphic and can
assigned any base type. If we can not solve for a concrete type we apply
a defaulting rule.

We support signed and unsigned integers, floating point numbers, booleans,
and tensor literals.

The base type literals are designed to closely model literals in TVM's
expressions langauge.

### Boolean Literals
TODO: don't have these in any form right now

### Integer Literals
TODO: don't have these in any form right now

Tensor Constructor
~~~~~~~~~~~~~~~

A tensor literal allows us to build a Tensor from other expressions.

TODO: Example here


Tuple Constructor
~~~~~~~~~~~~~~~

We support tuple constructors which allows us to build a fixed-k sized
sequence of heterogenous data. These tuples match closely to Python's
and enable efficient projection of their members due to their fixed length.

    (a, b, c) : Tuple<A, B, C>

    (a + b + c, d) : Tuple<Tensor<f32, (10, 10)>, Tensor<f32, 100, 100>>

Function
~~~~~~~~

A function node represents a function, it contains a seqeuence of
parameters, a return type, and a body.

    fun (x : Float, y: Float) -> Float { x + y }

Functions are first class in Relay, and can be used in any expression
position. Functions are the same as global functions, but do not have
an explicit name. You can use a function in conjunction with a let
binding to define locally recursive functions.

    let fact = fun (x : Float) -> Float {
        if (x == 0) {
            0
        } else {
            x * fact(x - 1)
    };
    fact(10)

Identifiers
~~~~~~~~~~~

All of the identifiers are valid expressions, you can use a local identifier,
global identifier, or intrinsic identifier anywhere an expression may appear.

For example the below fragment of code is a valid expression.

    %ret = @global(intrinsic, %local)

Let Binding
~~~~~~~~~~~

An immutable variable binding, allows the user to bind an
expression to a name. A let binding contains a local identifier,
an optional type, a value, and body expression which may
reference the bound identifier.

We will first introduce a single binding with no type
anntoations::
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

TODO Add Ref, ReadRef, WriteRef, Projection,

Gradient
~~~~~~~~

The `Reverse` acts as a marker node, when the compiler encounters it
we will apply the reverse mode transformation to the enclosed function.

We will employ static analysis and constant evaluation in order to
simplify the node's argument to a known function call target.


You can compute the reverse node of a function node like so:

Cast
~~~~~

Cast the type of the `node` to `ty`.

=======================
Control Flow Expression
=======================
Control flow expressions change network topology based on values
computed by previous expressions.

Call
~~~~

Terms with function types in Relay are "callable", that can be invoked like
a function in a typical programming language by supplying a set of arguments.

Instrinsics with functions types, definitions, and functions are all callable.

If-Then-Else
~~~~~~~~~~~~

Relay has a simple if/then/else expression which allows programs to branch
on a single control value which must be of type `Bool`, i.e a zero-rank
tensor of booleans.
