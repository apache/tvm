===================
Relay's Type System
===================

We have briefly introduced types while detailing the the expression language
of Relay, but have not yet described the type system. Relay is
a statically typed and type-inferred language, allowing programs to
be typed with a minimal requirement of explicit type information.

Static types are useful when performing compiler optimizations because they
communicate properties about the data we manipulate, such as runtime shape,
data layout, and storage without needing to run the program. Additionally,
static typing is useful for determining the 

Relay's type system features a form of depending typing for shapes to
replace the dynamic shape inference used in NNVM and most other machine
learning IRs. Treating tensor shapes as types allows Relay to perform
more powerful reasoning at compile time; in particular, Relay can statically
reason about operations whose output shapes vary based on the input shapes
in complex ways. Statically reasoning about shapes in this manner allows
Relay to be ahead-of-time compiled and provides much more information about
tensors for optimizations furhter in the compilation pipeline.

Reasoning about tensor types in Relay is encoded using type relations, which means
that the bulk of type checking in Relay is constraint solving (ensuring that all
type relations are satisfied at call sites). Type relations offer a flexible and
relatively simple way of making the power of dependent typing available in Relay
without greatly increasing the complexity of Relay's type AST.

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
shapes. In particular, data types include `bool`, `float32`, `int8` and various
other bit widths. Shapes are given as tuples of dimensions (TVM `IndexExpr`),
such as `(5, 5)`; note, though, that TVM shapes can also include variables and
arithmetic expressions including variables, so Relay's constraint solving phase
will attempt to find assignments to all shape variables to ensure all shapes will
be concrete before running a program.

For example, here is a simple concrete tensor type corresponding to a 10-by-10 tensor of 32-bit floats:

.. code-block:: python

   Tensor<float32, (10, 10)>

See :py:class:`~tvm.relay.ty.TensorType` for its definition and documentation.

Tuple Type
~~~~~~~~~~

A type of a tuple in Relay.

Just as a tuple is simply a sequence of values of statically known length, the type
of the tuple consists of a sequence of types corresponding to the type of each member
of the tuple.

Because a tuple type is of statically known size, the type of a tuple projection
is simply the corresponding index into the tuple type.

For example, in the below code, :code:`%t` is of type
`Tuple<Tensor<float32, (10, 10)>, Tensor<float32, (10, 10)>>`
and :code:`%c` is of type `Tensor<float32, (10, 10)>`.

.. code-block:: python
   let %t = (Constant(0, bool, ()), Constant(1, float32, (10, 10)));
   let %c = %t.1

See :py:class:`~tvm.relay.ty.TupleType` for its definition and documentation.

Type Parameter
~~~~~~~~~~~~~~

.. *note*: Type parameters are not yet implemented

Type parameters represent placeholder types used for polymorphism in functions.
Type parameters are specified according to *kind*: Kinds include general types
(such as tensor types and tuple types), shapes, and data types. Relay enforces
that types can only appear where their kind permits them; for instance, a
shape or data type can only appear inside a tensor type, while only general types
may appear inside a tensor or for a function's return and argument types.

Type parameters will be substituted with a concrete type at call sites.

For example, `s` below is a type paramter of kind `Shape` and it will
be substituted for `(10, 10)` at the call site below:

.. code-block:: python

   def @plus<s : Shape>(%t1 : Tensor<float32, s>, %t2 : Tensor<float32, s>) {
        add(%t1, %t2)
   }
   plus<(10, 10)>(%a, %b)
    

See :py:class:`~tvm.relay.ty.TypeParam` for its definition and documentation.

Type Constraint
~~~~~~~~~~~~~~~

Abstract class representing a type constraint, to be elaborated
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
`fun<type_params>(arg_types) -> ret_type where type_constraints`

A type parameter in the function type may appear in the argument
types or the return types. Additionally, each of the type constraints
must hold at every call site of the function. The type constraints
typically take the function's argument types and the function's return
type as arguments, but may take a subset instead.

See :py:class:`~tvm.relay.ty.FuncType` for its definition and documentation.

Type Relation
~~~~~~~~~~~~~

A type relation is the most complex type system feature in Relay.
It allows users to extend type inference with new rules.
We use type relations to define types for operators that work with
tensor shapes in complex ways, such as broadcasting operators or
:code:`flatten`, allowing Relay to statically reason about the shapes
in these cases.

A type relation :code:`R` is an n-ary-input, single-output relation over
types. Namely, :code:`R` specifies a relationship between its argument
types and outputs either `true` if the relationship holds and `false`
if it fails to hold. Types given to a relation may be incomplete or
include shape variables, so it may be possible to assign values to
incomplete types and shape variables such that a relation can hold.

For example we can define an identity relation to be:

.. code-block:: prolog
    Identity(I, I) :- true

Or we can define the relation for :code:`flatten`:

.. code-block:: prolog
    Flatten(Tensor(sh, bt), O) :-
      O = Tensor(sh[0], prod(sh[1:]))

If we have a relation like :code:`Broadcast` it becomes possible
to type operators like :code:`add`:

.. code-block:: python
    add : fun<t1 : Type, t2 : Type, t3 : Type>(t1, t2) -> t3
                where Broadcast

The inclusion of :code:`Broadcast` above indicates that the argument
types and the return type must be tensors where the shape of `t3` is
the broadcast of the shapes of `t1` and `t2`. The type system will
accept any argument types and return type so long as they fulfill
:code:`Broadcast`. Hence, it is usually convenient to type operators
in Relay by defining a relation specific to that operator that
encodes all the necessary constraints on the argument types
and the return type.

Note that the above example relations are written in Prolog-like syntax,
but currently the relations must be implemented by users in C++
or Python. More specifically, Relay's type system uses an *ad hoc* solver
for type relations in which type relations are actually implemented as
C++ or Python functions that check whether the relation holds and
imperatively update any shape variables or incomplete types. The
functions are run as needed (if an input is updated) until one of the following holds:

1. All relations hold (typechecking succeeds).
2. A relation fails to hold (a type error).
3. A fixpoint is reached where shape variables or incomplete types
remain (either a type error or more type annotations may be needed).

Presently all of the relations used in Relay are implemented in C++.
See the files in `src/relay/op` for examples of relations implemented
in C++.

See :py:class:`~tvm.relay.ty.TypeRelation` for its definition and documentation.

Incomplete Type
~~~~~~~~~~~~~~~

A type or portion of a type that is not yet known.
Only used during type inference: any omitted type annotation is
replaced by an incomplete type, which will be replaced by another
type at a later point.

.. note:: Known as a "type variable" in the type checking literature.

See :py:class:`~tvm.relay.ty.IncompleteType` for its definition and documentation.
