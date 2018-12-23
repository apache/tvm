====================
Expressions in Relay
====================

The Relay IR is a pure, expression-oriented language with distinct
dataflow and control flow language fragments.
Each dataflow fragment of a program (i.e., the portions of the
program without recursive calls or branching) can be viewed as a
traditional computation graph when writing and expressing transformations.

The below sections describe the different expressions in Relay
as well as give details of their semantics.

Variables
=========

Inspired by LLVM, Relay explicitly distinguishes between local and
global variables both in the AST and in the text format. In the text format,
global and local variables are distinguished by prefixes, or sigils.
Global variables are prefixed with :code:`@` and local variables with :code:`%`.

This explicit distinction makes certain optimizations easier to implement.
For example, inlining a global definition requires no analysis: simply
substituting the definition suffices.

Global Variable
~~~~~~~~~~~~~~~~~~

Global identifiers are prefixed by the :code:`@` sigil, such as ":code:`@global`".
A global identifier always references a globally visible definition contained in the environment.
Global identifiers must be unique.

See :py:class:`~tvm.relay.expr.GlobalVar` for its implementation
and documentation.

Local Variable
~~~~~~~~~~~~~~

Local identifiers are prefixed by the :code:`%` sigil,
such as ":code:`%local`". A local identifier always references
a function argument or a variable bound in a :code:`let` expression,
and will be scoped to the function where it appears or the :code:`let`
expression where it is bound, respectively.

In the below code segment, notice that :code:`%a` is defined twice. This is
permitted, as in most functional languages; in the scope of the second
:code:`let` expression, the name :code:`%a` is "shadowed," meaning all
references to :code:`%a` in the inner scope refer to the later defintion, while
references to :code:`%a` in the outer scope continue to refer to
the first one.

.. code-block:: python
  def @f(%a) {
    let %b = %a;
    let %a = add(%a, %a);
    multiply(%a, %b)
  }

(Note that in Relay's implementation, each definition of a local variable
creates a new :py:class:`~tvm.relay.expr.Var`, so a shadowed local variable,
despite having the same name as one in an outer scope, will be a different
object. This allows for comparing local variables by pointer identity with the
knowledge that the same local variable object corresponds to a different binding site.)

See :py:class:`~tvm.relay.expr.Var` for its implementation
and documentation.

Global Functions
================

A function definition consists of a name, arguments, return type,
type parameters, and any applicable type relations.
A function's return type and parameter types may be omitted;
Relay will attempt to infer the most general types where types
are omitted.

Functions in Relay act similarly to procedures or functions in
other programming languages and serve to generalize the concept
of a named subgraph.

Functions defined in the manner described in this subsection have
global scope; anonymous functions will be discussed later, though
their mechanics are nearly identical. Note that global functions may be
recursive; that is, within the function's body, the function's
identifier refers back to the function unless it is shadowed in a :code:`let`
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
variables, just like those bound in a :code:`let` expression.

One may also annotate explicit types on definitions.
For example, we can restrict the above definition to only work
on certain types:

.. code-block:: python

    def @plus(%x: Tensor[(10, 10), float32], %y: Tensor[(10, 10), float32])
               -> Tensor[(10, 10), float32] {
        add(%x, %y)
    }

A function parameter is just a local variable (:py:class:`~tvm.relay.expr.LocalVar`) optionally
annotated with a type. Parameters are written as :code:`%x : T`.

When the type information is omitted, we will attempt to infer the most general type
for the users. This property is known as generalization: for a definition without
explicit annotations, we will attempt to assign the most general type to the
parameters and return type based on the function body and call sites.

.. *Note: type parameter syntax is not yet supported in the text format.*

A function may also be given a set of type parameters, which can be
substituted for specific types at call sites. Functions with
type parameters are *type polymorphic*; their return type or the types
of arguments they will accept can vary based on the type arguments
given at call sites.

For example, one can define a polymorphic identity function for
any Relay type as follows:

.. code-block:: python
    def @id<t: Type>(%x : t) -> t {
        %x
    }

The below definition is also polymorphic, but restricts its
arguments to tensor types:

.. code-block:: python
    def @id<s: Shape, bt: BaseType>(%x: Tensor[s, bt]) {
        %x
    }

Notice that the return type is omitted and will be inferred.

.. *Note: :code:`where` syntax is not yet supported in the text format.*

A function may also be subject to one or more type relations, such as in
the following:

.. code-block:: python

    def @plus(%x, %y) where Broadcast { add(%x, %y) }

In the above definition, the types of `%x` and `%y` and the return type
are subject to the `Broadcast` relation, meaning all three must be tensors
and their shapes follow the elementwise broadcast relation. As with
operators, the definitions of relations are not transparent to Relay
and they are instead implemented externally in either C++ or Python.

As in the case of `Broadcast`, relations are used to express complicated
constraints on types (especially tensor shapes).
All function relations must hold at all call sites;
type checking is thus treated as a constraint-solving problem.
For more detail on type relations and their implementations,
please see the documentation on typing in Relay.

See :py:class:`~tvm.relay.expr.Function` for the definition and documentation of function nodes.

Operators
=========

An operator is a primitive operation not defined in the Relay
language. Operators are declared in the global operator
registry in C++. Many common operators are backed by TVM's
Tensor Operator Inventory (`TOPI <https://github.com/dmlc/tvm/tree/master/topi>`__).

An operator requires a user to provide an implementation
of the operator, its type, and any other desired metadata.
The operator registry is a column-based store where
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
identifiable by pointer.

See :py:class:`~tvm.relay.op.Op` for the definition and documentation
of operator nodes, demonstrating the infrastructure for registering
operator metadata. The other files in :py:class:`~tvm.relay.op` give
handles for generating a call to various pre-registered operators.
The `tutorial on adding operators to Relay`__ shows how to add further
operators into the language.

__ `Adding an Operator to Relay`_

Dataflow Fragment
=================

This subsection covers the set of Relay expressions that do not involve
control flow. That is, any portion of a program comprised only of these
expressions corresponds to a pure computation graph without control flow.
Note that global and local variables are also part of the dataflow fragment.

Constants
~~~~~~~~~

This node represents a constant tensor value
(see :py:mod:`~tvm.relay.Value` for more details).
The constants are represented as :py:class:`~tvm.NDArray`,
allowing Relay to utilize TVM operators for constant evaluation.

See :py:class:`~tvm.relay.expr.Constant` for its definition and documentation.

Tuple
~~~~~

The tuple node builds a finite (that is, of statically known size) sequence of heterogeneous data. 
These tuples match Python's closely. Their fixed length allows for efficient projection of their
members.

.. code-block:: python

    (a, b, c) : (Tensor[(10, 10), float32], Tensor[(10, 10), float32], Tensor[(10, 10), float32])
    d : Tensor[(100, 100), float32]

    (add(add(a, b), c), d) : (Tensor[(10, 10), float32], Tensor[(100, 100), float32])

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
use nearly the same syntax, but do not have a globally unique name.

.. code-block:: python

    fn (%x : Tensor[(10, 10), float32], y: Tensor[(10, 10), float32]
                -> Tensor[(10, 10), float32] { add(%x, %y) }

Note that function expressions evaluate to a closure. Closures
are values that are represented as a pair of a local environment
(storing the values for all variables defined outside the scope
of the function's body) and the function itself.

For example, in the below example, :code:`%z` will evaluate to a tensor
of zero values because the closure for :code:`%f` stores the value of
:code:`%x` at the pointer where :code:`%f` was defined.

.. code-block::

    let %g = fn () {
      let %x = Constant(0, (10, 10), float32);
      # x is a free variable in the below function
      fn (%y) { multiply(%y, %x) }
    };
    # the %x in %g's body is not in scope anymore
    # %f is a closure where %x maps to Constant(0, (10, 10), float32)
    let %f = %g();
    let %x = Constant(1, (10, 10), float32);
    let %z = %f(%x)

A recursive function expression can be defined using a :code:`let` binding,
as here:

.. code-block:: python

    let %fact = fun (%x : Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
        if (equal(%x, Constant(1, (10, 10), float32)) {
            Constant(0, (10, 10), float32)
        } else {
            multiply(%x, %fact(subtract(%x, Constant(1, (10, 10), float32))))
        }
    };
    %fact(10)

See :py:class:`~tvm.relay.expr.Function` for its definition and documentation.

Let Binding
~~~~~~~~~~~

A :code:`let` binding is an immutable local variable binding,
allowing the user to bind an expression to a name.

A :code:`let` binding contains a local variable,
an optional type annotation, a value, and a body expression
that may reference the bound identifier. If a type annotation
on the bound variable is omitted, Relay attempts to infer the
most general type permitted for the variable.

The bound variable in a :code:`let` expression is scoped to the let
expression's body. Note that the bound variable can only be
recursively referenced in the value in the case of a function
expression, as in the above subsection.

The value of a :code:`let` binding is the value of the final expression
after evaluating the bindings it depends on. For example, in the
following example the entire expression evaluates to a tensor
of shape (10, 10) where all elements are 2:

.. code-block:: python

   let %x : Tensor[(10, 10), float32] = Constant(1, (10, 10), float32);
   add(%x, %x)

A sequence of :code:`let` bindings can be considered as a dataflow graph,
where the bindings are a series of sub-graphs connected
by bound variables. Since these binding sequences are
pure, they can be evaluated in any order according to the program
dataflow. For example, the first and second :code:`let` bindings below
may be evaluated in either order because neither has a dataflow
dependency on the other:

.. code-block:: python

   let %x = add(%a, %b);
   let %y = add(%c, %d);
   multiply(%x, %y)

See :py:class:`~tvm.relay.expr.Let` for its definition and documentation.

Graph Bindings
~~~~~~~~~~~~~~

A :code:`let` binding creates a named variable that is bound to the given value
and scoped to the subsequent expression. By contrast, a graph binding allows for
explicitly constructing dataflow graphs in a Relay program by binding an expression
(graph node) directly to a temporary variable, which is not scoped. Each reference
to the variable corresponds to an edge in the dataflow graph. This has the
semantics of substituting the expression wherever the variable appears, even though
the graph node will only be evaluated once by the compiled program.

These bindings allow for a style of programming that corresponds to that already
employed by NNVM and other dataflow graph-based input formats. The fact that the variables
are not scoped offers some flexibility in evaluation order compared to :code:`let`
bindings, though this can also introduce some ambiguity in programs (the
`developer introduction to the Relay IR`__ includes more detailed discussion
of this nuance).

__ `Introduction to Relay IR`_

In Relay's text format, a graph binding can be written as below (note the lack of a
:code:`let` keyword and a semicolon):

.. code-block:: python

   %1 = add(%a, %b)
   %2 = add(%1, %1)
   multiply(%2, %2)

Graph bindings are not represented as an AST node in Relay, but rather as meta-variables set
to reference AST nodes. For example, a program like the above could be constructed in Relay's
Python front-end by setting *Python variables* equal to the corresponding Relay AST node and
using the variables repeatedly, as below (a C++ program using the corresponding API bindings
could accomplish the same thing):

.. code-block:: python

   sum1 = relay.add(a, b)
   sum2 = relay.add(sum1, sum1)
   relay.multiply(sum2, sum2)

For development purposes and to enable certain optimizations, Relay includes passes to
convert between dataflow graphs defined using graph bindings and programs with :code:`let`
bindings in A-normal form, employed by many compiler optimizations from the functional
programming community (see `"The Essence of Compiling with Continuations" by
Flanagan *et al*<https://slang.soe.ucsc.edu/cormac/papers/pldi93.pdf>`__ for a discussion
of the A-normal form).

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

Note that all type relations in the function type must hold at each
call site. Because relations are checked at call sites, this means that
the relations are checked against the types of the particular arguments
at that call site, so this also allows for a form of polymorphism.

See :py:class:`~tvm.relay.expr.Call` for its definition and documentation.

If-Then-Else
~~~~~~~~~~~~

Relay has a simple if-then-else expression that allows programs to branch
on a single value of type :code:`bool`, i.e., a zero-rank
tensor of booleans (:code:`Tensor[(), bool]`).

.. code-block:: python

    if (equal(%t, %u)) {
        %t
    } else {
        %u
    }

Since if-then-else branches are expressions, they may appear inline
wherever any other expression may be expected, like invocations of
the ternary operator in C-like languages. The if-then-else expression
evaluates to the value of the "then" branch if the condition value is
`True` and evaluates to the value of the "else" branch otherwise.

See :py:class:`~tvm.relay.expr.If` for its definition and documentation.
