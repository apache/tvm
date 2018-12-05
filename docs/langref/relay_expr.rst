====================
Expressions in Relay
====================

The Relay IR is a pure, expression-oriented language with distinct
dataflow and control flow language fragments.
The dataflow fragments of a program (i.e., those without
calls to recursive functions or branching) can be considered
as a more traditional computation graph when writing and
expressing transformations.

The below sections describe the different expressions in Relay
as well as give details of their semantics.

Variables
=========

Relay allows for local and global variables. Our design is based on
that of LLVM, which differentiates between identifier types; a writer of
optimizations can thus determine a lot of information about what a
variable references simply by knowing the kind of identifier.

Global variables are written with `@`, local variables are written
with `%`, and variables written without a sigil correspond to operator
names.

The distinction between global and local identifiers
makes certain kinds of transformation easier. For example,
inlining a global definition requires no analysis: simply inlining
the definitions suffices.

Global Variable
~~~~~~~~~~~~~~~~~~

Global identifiers are prefixed by the `@` sigil, such as "`@global`".
A global identifier always references a globally visibly definition contained in the environment. The names of global identifiers must be unique.

See :py:class:`~tvm.relay.expr.GlobalVar` for its implementation
and documentation.

Local Variable
~~~~~~~~~~~~~~~~~

Local identifiers are prefixed by the :code:`%` sigil,
such as ":code:`%local`". A local identifier always references
a function argument or a variable bound in a `let` expression.
A local variable will be scoped to the function where it
appears or the `let` expression where it is bound, respectively.

Suppose the local variable :code:`%a` has been defined in a scope
and within that scope, a `let` expression binding to a variable
:code:`%a` appears. This is permitted, as in most functional languages.
In the scope of the `let` expression (the inner scope),
the name :code:`%a` is "shadowed," meaning all references to
:code:`%a` in the inner scope refer to the later defintion, while
references to :code:`%a` in the outer scope continue to refer to
the previous one.

(Note that in Relay's implemetnation, each definition of a local variable
creates a new :py:class:`~tvm.relay.expr.Var`, so a shadowed local variable,
despite having the same name as one in an outer scope, will be a different
object. This allows for comparing local variables by pointer identity with the
knowledge that the same local variable object corresponds to a different binding site.)

See :py:class:`~tvm.relay.expr.Var` for its implementation
and documentation.

Global Functions
================

A function is no different from a procedure or function in a
typical programming language and generalizes the concept of a
named subgraph.
A function definition consists of a name, arguments, return type,
type parameters, and any applicable type relations.
A function's return type and the types of parameters may be omitted;
Relay will attempt to infer the most general types where types
are omitted.

Functions defined in the manner described in this subsection are
of global scope; anonymous functions will be discussed later, though
their mechanics are nearly identical. Note that global functions may be
recursive; that is, within the function's body, the function's
identifier refers back to the function unless it is shadowed in a `let`
expression.

A definition minimally consists of an identifier :code:`@id`, an empty set of
parameters, and a body expression (:py:class:`~tvm.relay.expr.Expr`)
contained by curly braces.

.. code-block:: python

    def @id() { body }

A definition may contain any number of parameters. For example, a
simple function that invokes the `add` operator:

.. code-block:: python

    def @plus(%x, %y) { add(%x, %y) }

Notice that within the function's body, the parameters are local
variables, just like those bound in a `let` expression.

It is also possible for us to annotate explicit types on definitions.
For example, we can restrict the above definition to only work
on certain types:

.. code-block:: python

    def @add(%x: Tensor<Float, (10, 10)>, %y: Tensor<Float, (10, 10)>) -> Tensor<Float, (10, 10)> {
        %x + %y
    }

A parameter is just a pairing of a :py:class:`~tvm.relay.expr.LocalVar` and optional :py:class:`~tvm.relay.ty.Type`. They represent
the formal parameters of functions and definitions and are written as :code:`%x : T`.

Parameters may only appear in function literals and definitions
and have no relation to parameters in the machine learning.

When the type information is omitted, we will attempt to infer the most general type
for the users. This property is known as generalization: for a definition without
explicit annotations, we will attempt to assign the most general type. When the
return type is omitted, we will infer the return type based on the text of the
program.

We can directly construct type-polymorphic definitions by writing down
a set of type parameters for a definition. For example, one can define a
polymorphic identity function for tensors as follows:

.. code-block:: python
    def @id<s: Shape, bt: BaseType>(%x: Tensor<bt, s>) {
        %x
    }

Notice that the return type is omitted and will be inferred.

.. *Note: polymorphism is not yet implemented.*

A function may also be subject to one or more type relations, such as in
the following:

.. code-block:: python

    def @plus(%x, %y) where Broadcast { add(%x, %y) }

In the above definition, the types of `%x` and `%y` and the return type
are subject to the `Broadcast` relation, meaning all three must be tensors
and their shapes follow the elementwise broadcast relation. As with
operators, the definitions of relations are not transparent to Relay
and they are instead implemented externally.

As in the case of `Broadcast`, relations are used to express complicated
constraints on types (especially tensor shapes).
All function relations must hold at all call sites;
type checking is thus treated as a constraint-solving problem.
For more detail on type relations and their implementations,
please see the documentation on typing in Relay.

See :py:class:`~tvm.relay.expr.Function` for the definition and documentation of function nodes.

Operators
=========

An operator is a primitive operation that is not defined in the Relay
language but is registered in the global environment in either Python
or C++. Implementations of operators are typically backed by TVM's TOPI.

An operator requires a user to provide an implementation
of the operator, its type, and any other desired metadata.
The operator registry is simply a column-based store where
operators are keys, so any metadata (which might be referenced
by optimization passes) may be registered as a new column.

From the perspective of Relay's type system, an operator is a function,
so operators may be called like any other function and have function
types. In particular, operator types are registered using a single
type relation (see the above subsection), typically a relation
specialized to that operator. For example, the :code:`add` operator
is registered with the `Broadcast` relation, indicating that the
arguments of :code:`add` must be tensors and that the return type
is a tensor whose shape depends on those of its arguments.

Operators are rendered without a sigil (e.g :code:`add`, :code:`subtract`)
when pretty-printing Relay programs.
Operators are explicitly contained in the program and are uniquely
identifiable by pointer during a run of the Relay compiler.

See :py:class:`~tvm.relay.op.Op` for the definition and documentation
of operator nodes, demonstrating the infrastructure for registering
operator metadata. The other files in :py:class:`~tvm.relay.op` give
handles for generating a call to various pre-registered operators.

Dataflow Fragment
=================

This subsection covers the set of Relay expressions that do not involve
control flow. That is, any portion of a program comprised only of these
expressions corresponds to a pure computation graph without control flow.
Note that global and local variables are also part of the dataflow fragment.

Constants
~~~~~~~~~

This node represents a constant tensor value
(see :py:mod:~tvm.relay.Value for more details).
The constants are represented as :py:class:`~tvm.NDArray`,
allowing Relay to utilize TVM operators for constant evaluation.

See :py:class:`~tvm.relay.expr.Constant` for its definition and documentation.

Tuple
~~~~~

The tuple node builds a finite (that is, of statically known size) sequence of heterogeneous data. 
These tuples match Python's closely. Their fixed length allows for efficient projection of their
members.

.. code-block:: python

    (a, b, c) : Tuple<A, B, C>

    (add(add(a, b), c), d) : Tuple<Tensor<float32, (10, 10)>, Tensor<float32, (100, 100)>>

See :py:class:`~tvm.relay.expr.Tuple` for its definition and documentation.

Tuple Projection
~~~~~~~~~~~~~~~~

A tuple must be indexed by an integer constant in order to extract a
particular member of the tuple. Projections are 0-indexed.

For example, the below projection evaluates to :code:`b`:

.. code-block:: python

   (a, b, c).1

See :py:class:`~tvm.relay.expr.TupleGetItem` for its definition and documentation.

Function Expressions
~~~~~~~~~~~~~~~~~~~~

Functions are first class in Relay and can be used in any expression
position. Function expressions are the same as global functions and
use nearly the same syntax (besides not including a name), but do
not have a globally unique name.

.. code-block:: python

    fun (%x : Tensor<float32, (10, 10)>, y: Tensor<float32, (10, 10)>
                -> Tensor<float32, (10, 10)> { add(%x, %y) }

Note that function expressions evaluate to closure values. Closures
store values for all free variables in their body. A free variable
is a variable defined outside the scope of the function's body,
which means that if a function expression references a local variable
in the outer scope that is not shadowed by the parameters,
the closure will store the value for that local variable and
use that value when the function is called, even if original
local variable has gone out of scope.

For example, in the below example, :code:`%z` will evaluate to a tensor
of zero values because the closure for :code:`%f` stores the value of
:code:`%x` at the pointer where :code:`%f` was defined.

.. code-block::

    let %g = fun () {
      let %x = Constant(0, float32, (10, 10));
      # x is a free variable in the below function
      fun (%y) { multiply(%y, %x) }
    };
    # the %x in %g's body is not in scope anymore
    # %f is a closure where %x maps to Constant(0, float32, (10, 10))
    let %f = %g();
    let %x = Constant(1, float32, (10, 10));
    let %z = %f(%x)

A recursive function expression can be defined using a `let` binding,
as here:

.. code-block:: python

    let %fact = fun (%x : Tensor<float32, (10, 10)>) -> Tensor<float32, (10, 10)> {
        if (equal(%x, Constant(1, float32, (10, 10))) {
            Constant(0, float32, (10, 10))
        } else {
            multiply(%x,  %fact(subtract(%x, Constant(1, float32, (10, 10))))
        }
    };
    %fact(10)

See :py:class:`~tvm.relay.expr.Function` for its definition and documentation.

Let Binding
~~~~~~~~~~~

A `let` binding is an immutable local variable binding,
allowing the user to bind an expression to a name.

A `let` binding contains a local variable,
an optional type annotation, a value, and a body expression
that may reference the bound identifier. If a type annotation
on the bound variable is omitted, Relay attempts to infer the
most general type permitted for the variable.

The bound variable in a `let` expression is scoped to the let
expression's body. Note that the bound variable can only be
recursively referenced in the value in the case of a function
expression, as in the above subsection.

The value of a `let` binding is the value of the final expression
after evaluating the bindings it depends on. For example, in the
following example the entire expression evaluates to a tensor
of shape (10, 10) where all elements are 2:

.. code-block:: python

   let %x : Tensor<float32, (10, 10)> = Consantt(1, float32, (10, 10));
   add(%x, %x)

A sequence of `let` bindings can be considered as a dataflow graph,
where the bindings are a series of sub-graphs connected
by bound variables. Since these binding sequences are
pure, they can be evaluated in any order according to the program
dataflow. For example, the first and second `let` bindings below
may be evaluated in either order because neither has a dataflow
dependency on the other:

.. code-block:: python

   let %x = add(%a, %b);
   let %y = add(%c, %d);
   multiply(%x, %y)

See :py:class:`~tvm.relay.expr.Let` for its definition and documentation.

=======================
Control Flow Expression
=======================

Control flow expressions allow the network topology to change
based on the value of previously executed expressions.

Call
~~~~

Expressions with function types in Relay are "callable,"
meaning that they can be invoked via a function call. These consist of
any expression that evaluates to a closure (i.e., function expressions
or global functions) and Relay operators.

When a closure is called, the body is evaluated in the closure's stored
environment (i.e., using the stored values for free variables) with
local variable bindings added for each argument; the final value
obtained by evaluating the body is the call's return value.
In the case of operators, the implementation is opaque to Relay,
so the result is left up to the registered TVM implementation.

For example, we can call the previously defined `%fact` because it
has a function type:

.. code-block:: python

   %fact(10)

A type-polymorphic function can also include type arguments at a call
site. The type arguments are substituted for type parameters when
type checking.

.. *Note: polymorphism is not yet implemented.*

Note that all type relations in the function type must hold at each
call site. Because relations are checked at call sites, this means that
the relations are checked against the types of the particular arguments
at that call site, so this is also allows for a form of polymorphism.

See :py:class:`~tvm.relay.expr.Call` for its definition and documentation.

If-Then-Else
~~~~~~~~~~~~

Relay has a simple if-then-else expression that allows programs to branch
on a single value of type :code:`bool`, i.e., a zero-rank
tensor of booleans (:code:`Tensor<bool, ()>`).

.. code-block:: python

    if (sum(equal(t, u))) {
        return x:
    } else {
        return y;
    }

Since if-then-else branches are expressions, they may appear inline
wherever any other expression may be expected, like invocations of
the ternary operator in C-like languages. The if-then-else expression
evaluates to the value of the "then" branch if the condition value is
`True` and evaluates to the value of the "else" branch otherwise.

See :py:class:`~tvm.relay.expr.If` for its definition and documentation.
