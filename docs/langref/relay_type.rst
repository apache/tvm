===================
Relay's Type System
===================

We briefly introduced types while detailing Relay's expression language
, but have not yet described its type system. Relay is
a statically typed and type-inferred language, allowing programs to
be fully typed while requiring just a few explicit type annotations.

Static types are useful when performing compiler optimizations because they
communicate properties about the data a program manipulates, such as runtime
shape, data layout, and storage, without needing to run the program.

Relay's type system features a form of *dependent typing* for shapes. That is, its type system keeps track of the shapes of tensors in a Relay program. Treating tensor
shapes as types allows Relay to perform more powerful reasoning at compile time;
in particular, Relay can statically reason about operations whose output shapes
vary based on the input shapes in complex ways. Casting shape inference as a type
inference problem allows Relay to infer the shapes of all tensors at compile time,
including in programs that use branching and function calls.

Statically reasoning about shapes in this manner allows
Relay to be ahead-of-time compiled and provides much more information about
tensors for optimizations further in the compilation pipeline. Such optimizations
can be implemented as passes, which are Relay-to-Relay AST transformations, and
may use the inferred types (e.g., shape information) for making decisions about
program transformations. For instance, :code:`src/relay/pass/fuse_ops.cc` gives
an implementation of a pass that uses inferred tensor shapes to replace invocations
of operators in a Relay program with fused operator implementations.

Reasoning about tensor types in Relay is encoded using *type relations*, which means
that the bulk of type checking in Relay is constraint solving (ensuring that all
type relations are satisfied at call sites). Type relations offer a flexible and
relatively simple way of making the power of dependent typing available in Relay
without greatly increasing the complexity of its type system.

Types
=====

Below we detail the language of types in Relay and how they are assigned to Relay expressions.

Type
~~~~

The base type for all Relay types. All Relay types are sub-classes of this base type.

See :py:class:`~tvm.relay.ty.Type` for its definition and documentation.

Tensor Type
~~~~~~~~~~~

A concrete tensor type in Relay.

Tensors are typed according to data type and shape. At present, these use TVM's
data types and shapes, but in the future, Relay may include a separate AST for
shapes. In particular, data types include :code:`bool`, :code:`float32`, :code:`int8` and various
other bit widths and numbers of lanes. Shapes are given as tuples of dimensions (TVM :code:`IndexExpr`),
such as :code:`(5, 5)`; scalars are also given tuple types and have a shape of :code:`()`.

Note, though, that TVM shapes can also include variables and arithmetic expressions
including variables, so Relay's constraint solving phase will attempt to find
assignments to all shape variables to ensure all shapes will be concrete before
running a program.

For example, here is a simple concrete tensor type corresponding to a 10-by-10 tensor of 32-bit floats:

.. code-block:: python

   Tensor[(10, 10), float32]

See :py:class:`~tvm.relay.ty.TensorType` for its definition and documentation.

Tuple Type
~~~~~~~~~~

A type of a tuple in Relay.

Just as a tuple is simply a sequence of values of statically known length, the type
of a tuple consists of a sequence of the types corresponding to each member of the tuple.

Because a tuple type is of statically known size, the type of a tuple projection
is simply the corresponding index into the tuple type.

For example, in the below code, :code:`%t` is of type
:code:`(Tensor[(), bool], Tensor[(10, 10), float32])`
and :code:`%c` is of type :code:`Tensor[(10, 10), float32]`.

.. code-block:: python
   let %t = (False, Constant(1, (10, 10), float32));
   let %c = %t.1;
   %c

See :py:class:`~tvm.relay.ty.TupleType` for its definition and documentation.

.. _type-parameter:

Type Parameter
~~~~~~~~~~~~~~

Type parameters represent placeholder types used for polymorphism in functions.
Type parameters are specified according to *kind*, corresponding to the types
those parameters are allowed to replace: 

- :code:`Type`, corresponding to top-level Relay types like tensor types, tuple types, and function types
- :code:`BaseType`, corresponding to the base type of a tensor (e.g., :code:`float32`, :code:`bool`)
- :code:`Shape`, corresponding to a tensor shape
- :code:`ShapeVar`, corresponding to variables within a tensor shape

Relay's type system enforces that type parameters are only allowed to appear where their kind permits them,
so if type variable :code:`t` is of kind :code:`Type`, :code:`Tensor[t, float32]` is not a valid type.

.. *Note: At present, only type parameters of kind :code:`Type` are supported.*

Like normal parameters, concrete arguments must be given for type parameters at call sites.

.. *Note: type parameter syntax is not yet supported in the text format.*

For example, :code:`s` below is a type parameter of kind :code:`Shape` and it will
be substituted with :code:`(10, 10)` at the call site below:

.. code-block:: python

   def @plus<s : Shape>(%t1 : Tensor[s, float32], %t2 : Tensor[s, float32]) {
        add(%t1, %t2)
   }
   plus<(10, 10)>(%a, %b)

See :py:class:`~tvm.relay.ty.TypeVar` for its definition and documentation.

Type Constraint
~~~~~~~~~~~~~~~

This is an abstract class representing a type constraint, to be elaborated
upon in further releases. Currently, type relations are the only
type constraints provided; they are discussed below.

See :py:class:`~tvm.relay.ty.TypeConstraint` for its definition and documentation.

Function Type
~~~~~~~~~~~~~

A function type in Relay, see `tvm/relay/type.h` for more details.

This is the type assigned to functions in Relay. A function type
consists of a list of type parameters, a set of type constraints,
a sequence of argument types, and a return type.

We informally write function types as:
:code:`fn<type_params>(arg_types) -> ret_type where type_constraints`

A type parameter in the function type may appear in the argument
types or the return types. Additionally, each of the type constraints
must hold at every call site of the function. The type constraints
typically take the function's argument types and the function's return
type as arguments, but may take a subset instead.

See :py:class:`~tvm.relay.ty.FuncType` for its definition and documentation.

.. _type-relation:

Type Relation
~~~~~~~~~~~~~

A type relation is the most complex type system feature in Relay.
It allows users to extend type inference with new rules.
We use type relations to define types for operators that work with
tensor shapes in complex ways, such as broadcasting operators or
:code:`flatten`, allowing Relay to statically reason about the shapes
in these cases.

A type relation :code:`R` describes a relationship between the input and output types of a Relay function.
Namely, :code:`R` is a function on types that
outputs `true` if the relationship holds and `false`
if it fails to hold. Types given to a relation may be incomplete or
include shape variables, so type inference must assign appropriate
values to incomplete types and shape variables for necessary relations
to hold, if such values exist.

For example we can define an identity relation to be:

.. code-block:: prolog
    Identity(I, I) :- true

It is usually convenient to type operators
in Relay by defining a relation specific to that operator that
encodes all the necessary constraints on the argument types
and the return type. For example, we can define the relation for :code:`flatten`:

.. code-block:: prolog
    Flatten(Tensor(sh, bt), O) :-
      O = Tensor(sh[0], prod(sh[1:]))

If we have a relation like :code:`Broadcast` it becomes possible
to type operators like :code:`add`:

.. code-block:: python
    add : fn<t1 : Type, t2 : Type, t3 : Type>(t1, t2) -> t3
                where Broadcast

The inclusion of :code:`Broadcast` above indicates that the argument
types and the return type must be tensors where the shape of :code:`t3` is
the broadcast of the shapes of :code:`t1` and :code:`t2`. The type system will
accept any argument types and return type so long as they fulfill
:code:`Broadcast`.

Note that the above example relations are written in Prolog-like syntax,
but currently the relations must be implemented by users in C++
or Python. More specifically, Relay's type system uses an *ad hoc* solver
for type relations in which type relations are actually implemented as
C++ or Python functions that check whether the relation holds and
imperatively update any shape variables or incomplete types. In the current
implementation, the functions implementing relations should return :code:`False`
if the relation fails to hold and :code:`True` if the relation holds or if
there is not enough information to determine whether it holds or not.

The functions for all the relations are run as needed (if an input is updated)
until one of the following conditions holds:

1. All relations hold and no incomplete types remain (typechecking succeeds).
2. A relation fails to hold (a type error).
3. A fixpoint is reached where shape variables or incomplete types remain (either a type error or more type annotations may be needed).

Presently all of the relations used in Relay are implemented in C++.
See the files in :code:`src/relay/op` for examples of relations implemented
in C++.

See :py:class:`~tvm.relay.ty.TypeRelation` for its definition and documentation.

Incomplete Type
~~~~~~~~~~~~~~~

An incomplete type is a type or portion of a type that is not yet known.
This is only used during type inference. Any omitted type annotation is
replaced by an incomplete type, which will be replaced by another
type at a later point.

Incomplete types are known as "type variables" or "type holes" in the programming languages
literature. We use the name "incomplete type" in order to more clearly distinguish them from type
parameters: Type parameters must be bound to a function and are replaced with concrete type arguments (instantiated)
at call sites, whereas incomplete types may appear anywhere in the program and are filled in during type inference.

See :py:class:`~tvm.relay.ty.IncompleteType` for its definition and documentation.
