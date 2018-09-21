==================
Type System
==================

We have briefly introduced types while detailing the the expression language
of Relay, but have fully laid out the type system.

Although the majority of Relay programs are written without type annotations, 
Relay is statically typed.

Static types are useful because they enable efficient layout, memory reuse, and 
code generation. They aid in debugging program transformations, but can also
give us the expressivity afforded by more dynamic langauges. 

We are able to omit these type annotations by a process known as type inference.
Type inference is a technique that has its roots in the programming language
community, and can be viewed as a method for generalizing shape inference to
run over arbitrary user programs containing control flow and recursion.

Static types are useful when performing compiler optimization because they 
communicate properties about the data we manipulate, such as runtime shape, 
data layout, storage without needing to run the program.

Most current IRs use "shape inference" to recover Tensor dimensions from the user
provided program. Machine learning users have enjoyed shape inference for
tensors because it allows them to generate performant code without giving up
on the expressivity of the input language.

Because Relay is intended as an IR we require *some* type information to provide
full inference. We don't believe this to be an issue as many of the IR builder
inferfaces require some type information, or can generate IR based on their own
higher level inferences.

We view this limited shape inference as a simpler form of type
inference. Instead of relying on an ad-hoc procedure for recovering type
information from a potentially dynamic program, we apply ideas from compiler and IR design.

Below we briefly dicsuss the different kinds of types in Relay.

=====
Types
=====

Relay's type system has a "language of types" which allow us to write down the type of
a Relay program. Below we detail the langauge of types and how we assign them to Relay
programs.

Type
~~~~
The base type for all Relay types. All Relay types are sub-classes of this base type.

See :py:class:`~tvm.relay.type.Type` for its definition and documentation.

Tensor Type
~~~~~~~~~~

A concrete TensorType in Relay, see tvm/relay/type.h for more details.

This is the type assigned to tensor's with a known dype and shape. For
example a tensor of `float32` and `(5, 5)`.



Tensor values in Relay are typed with tensor types. A tensor type is
parametrized by a data type, and shape. The data type must be a base
type as enforced by the kind checking rules described in TODO.

This restriction importantly means

The shape may be any valid Relay shape as described in the below
section on shapes.


See :py:class:`~tvm.relay.type.TensorType` for its definition and documentation.

Kind
~~~~
The kind of a type parameter, represents a variable shape,
base type, type, or dimension.

This controls what a type parameter is allowed to be instantiated
with. For example one's of kind BaseType can only be `float32`, 
`int32`, and so on.
       
See :py:class:`~tvm.relay.type.Kind` for its definition and documentation.

Type Parameter
~~~~~~~~~~~~~~

A type parameter used for generic types in Relay,
see tvm/relay/type.h for more details.

A type parameter represents a type placeholder which will
be filled in later on. This allows the user to write
functions which are generic over types.

See :py:class:`~tvm.relay.type.TypeParam` for its definition and documentation.

Type Constriant
~~~~~~~~~~~~~~~

Abstract class representing a type constraint, to be elaborated
on in further releases.

See :py:class:`~tvm.relay.type.TypeConstraint` for its definition and documentation.

Function Type
~~~~~~~~~~~~~
A function type in Relay, see tvm/relay/type.h for more details.

This is the type assigned to functions in Relay. They consist of
a list of type parameters which enable the definition of generic
functions, a set of type constraints which we omit for the time
being, a sequence of argument types, and a return type.

We informally write them as:
`forall (type_params), (arg_types) -> ret_type where type_constraints`

See :py:class:`~tvm.relay.type.FuncType` for its definition and documentation.

Type Relation
~~~~~~~~~~~~~

A type relation is the most exotic type system feature in Relay. It allows 
users to extend type and shape checking/inference with new rules. We use
type relations to type operators with "hard" types such as broadcasting
operators, or special ones like :code:`flatten`.

A type relation :code:`R` is a n-ary input, single output relation over
types. To unpack that, it allows us to specify a relationship between
a set of input and output types.

For example we can define the identity relation to be:

.. code-block:: prolog
    Identity(I, I) :- true

Or we can define the relation for flatten:

.. code-block:: prolog
    Flatten(Tensor(sh, bt), O) :-
      O = Tensor(sh[0], prod(sh[1:]))

The above examples are written in Prolog-like syntax but currently the relations
must be implemented by users in C++ or Python.

If we have a relation such as :code:`Broadcast` it becomes possible to type things
such as :code:`elemwise_add`:

.. code-block:: python
    elemwise_add : forall (Lhs : Type) (Rhs : Type), (Lhs, Rhs) -> Broadcast(Lhs, Rhs)

You might ask why we write the relation in the return type but we use it as a 
notational convenience for:

.. code-block:: python
    elemwise_add : forall (Lhs : Type) (Rhs : Type) (Out : Type), Broadcast(Lhs, Rhs, Out) => (Lhs, Rhs) -> Out

That is the user may pick the type of the :code:`Lhs`, :code:`Rhs`, and :code:`Out` as long as we can
show :code:`Broadcast(Lhs, Rhs, Out)` holds.

See :py:class:`~tvm.relay.type.TypeRelation` for its definition and documentation.

Type Call
~~~~~~~~~

Apply a type relation to a set of input arguments, at the present momen the type 
call node represents the application of a :py:class:`~tvm.relay.type.TypeRelation`
to a set of input arguments. The result of type application is the output variable
of the type relation.

See :py:class:`~tvm.relay.type.TypeCall` for its definition and documentation.

Incomplete Type
~~~~~~~~~~~~~~~

A type, or portion of a type which is not known yet. Only used during type inference.

.. note:: Known as a "type variable" in the type checking literature.

See :py:class:`~tvm.relay.type.IncompleteType` for its definition and documentation.






