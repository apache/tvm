==================
Introduction
==================

Relay is differentiable programming language with support for
closures, control-flow, and recursion. It has an advanced
static type system specifically designed for programs written by machine 
learning practioniers and researchers.

Relay is intended to replace computation graph based
intermediate representations currently employed by deep
learning frameworks and compilers. The deep learning community
has organically evolved from a representation that was useful
for the limited form of computation originally desired, i.e
a DAG of primitive functions.

This may be a useful way to describe computation, but is limited once 
you want to generalize to dynamic models.

The conflation of the description of a computation, the
representation used for optimizing it, and the data structure
used to execute it uncessarily hampers many of the goals
of a framework.

We believe having a high level, an expressive language designed
for compiler optimizations is essential to the future of an
end-to-end deep learning compiler stack.

Relay's design was influenced by the authors previous experince
building advanced optimizing compilers for high level languages,
as well as experinces with the current TVM stack, and NNVM.

Concretely NNVM has played the role of this high level IR.

for NNVM which address design flaws of the computation graph IRs
employed by common deep learning frameworks.
We address a few important challenges. First we present a new IR with first
class functions and closures, enabling the design of differentiable
arbitrary control flow constructs. Second we contribute a type system for
Relay which makes the important properties of computations involving tensors
explicit.

The result is a high-level IR for differentiable comptuation that can
be used to abstract over the details of execution including devices,
parallelism, and distribution.

We believe Relay will enable researchers and industry to design a
new class of DL framework frontends, differentiable programming languages,
deep probablistic programming languages, thus accelerating the pace of
research in applications.

==================
Language
==================

Relay is a purely functional, differentiable intermediate representation.

==================
IR Reference
==================

The IR has a global environment which stores the set of definitions,
constants, options, attributes, and provides access to features like
the type inferencer, constant evaluator, and more.

## Node

The fundmental unit of the IR is the node, all nodes must have
a type field.

.. code-block:: python
    class Node:
        def checked_type() -> Type:
            ...

==================
Variables
==================

Relay has three variable references local, global, and operators. Our design draws inspiration
from LLVM which differentiates between identifier types. This enables writers of 
optimizations to know precisely what an identifier references without needing information 
beyond the type of identifier.

Globals are written with `@`, locals are written with `%`, and operators are written with
no sigil like LLVM's intrinsics. The distinction between global and local identifiers
makes certain kinds of transformation easier. For example inlining a global definition
requires no analysis, you can write a pass that just directly inlines the defintions.
It also ensures there are no spooky action at a distance, introducing a new identifier
of any type will never introduce an ambiguity to the program.


Global Variable
~~~~~~~~~~~~~~~~~~

Global identifiers are prefixed by the ::@ sigil. A global identifier always
references a globally visibly definition contained in the environment. You
can write a global identifier as ::@global.

Local Variable
~~~~~~~~~~~~~~~~~

Local identifiers are prefixed by the :code:`%` sigil. A local identifier always
references a parameter, or let bound expression. You can write a local
identifier as :code:`%local`.


==================
Global Functions
==================

A definition consists of a name, type parameter, parameters, and an optional return
type. A Relay definition is similar to a procedure or function in a typical programming
language, but can also be viewed as a named subgraph.

A definition minimally consists of an identifier :code:`@id`, an empty set of
parameters, and a body expression contained by curly braces

.. code-block:: python

    def @id() { body }

A definiton may also contain any number of parameters, for example a
simple function which just adds two tensors

.. code-block:: python

    def @add(%x, %y) { %x + %y }

It is also possible for us to annotate explicit types on definitions, for example
we can restrict the above definition to only work on certain types

.. code-block:: python

    def @add(%x: Tensor<Float,10, 10>, %y: Tensor<Float, 10, 10>) -> Tensor<Float, 10, 10> {
        %x + %y
    } 

A parameter is just a pairing of a :py:class:`~tvm.relay.expr.LocalVar` and optional :py:class:`~tvm.relay.type.Type`. They represent
the formal parameters of functions and definitions, and are written as :code:`%x : T`.

They may only appear in function literals, and definitions, and have no relation
to parameters in the machine learning.

When the type information is omitted we will attempt to infer a most general type
for the users. This property is known as generalization, for a definition without
explicit annotations, we will attempt to assign the most general type. When the
return type is omitted we will infer the return type based on the text of the
program.

Finally we can directly construct type polymorphic definitions by writing down
a set of type parameters for a definition. To define a polymoprhic identity
function, the function which just returns its argument as so.
::
    def @id<s: Shape, bt: BaseType>(%x: Tensor<bt, s>) {
        %x
    }

Notice we can omit the return type, and it will still be inferred.

*Note: this is not yet implemented.*

Finally we allo a definition be prefixed by metadata, which adds
extra properties to the definition.

It is important to be able to annotate metadata that is external to
the computational behavior of a definition. For example we can use
this to add an `inline` or `noinline` attribute which the compiler
can consider when performing inlining.

For example we can set the attributes for :code:`@id_real`.::


    attributes id_real {
        inline: true
    }

    def id_real(%x:Real) { ret %x }


==================
Operators
==================

A primitive operation that is not defined in the Relay language but provided
externally. Currently we back these operators registrations with the operators
exposed by TVM's TOPI. An operator requires a user to provide an implementation 
of the operator, its type and various attributes required by Relay subsystems.

The input methods for Relay programs do not provide a way to describe operators in
Relay, they must be explicitly registered in the global environment via Python or C++.

Operators are rendered without a sigil (e.g :code:`add`, :code:`subtract`) when pretty 
printing Relay programs.  Operators are explicitly contained in the program and are uniquely 
identifiable by pointer during a run of the Relay compiler.

Programs
~~~~~~~~

Now that we have presented both global functions, and operators we have
everthing in hand to describe a complete Relay program. A Relay program consists of a 
registry of operators, one or more functions, as well as the global configuration 
stored in the environment.