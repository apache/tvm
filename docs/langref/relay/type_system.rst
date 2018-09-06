==================
Type System
==================

We have briefly introduced types while detailing the the expression language
of Relay, but have fully laid out the type system.

Although the majority of Relay programs require no type annotations, Relay
is statically typed. Each expression in Relay has a precisely known type.

You might ask why we want a statically typed IR, there are multiple advantages.
- efficient layout and code generation for tensors
- TODO
- debugging transformations (most program transformations should be type perserving)

We are able to omit these type annotations by a process known as type inference.
Type inference is a technique that has its roots in the programming language
community, and can be viewed as a method for generalizing shape inference to
run over arbitrary user programs.

Static typing means we know before executing the program properties about
the values it manipulates. Static types are useful for compiler optimization
because they communicate properties about the data we manipulate, such as
runtime shape, data layout, storage.

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

BaseType
~~~~~~~~~~
Relay has a notion of a BaseType, which captures the set of types
that can be stored in a Tensor. Relay's base types map to the set
of types supported by TVM.

Each of the base types can be parametrized by number of bits, and
lanes for vectorization purposes. We support four base types any:`Bool`,
any:`Int`

Type Variables
~~~~~~~~~~~~~~

Type Parameters
~~~~~~
TODO: type parameter

Kind
~~~~

Function Types
~~~~~~~~~~
TODO: rename function type?

TypeQuantifier
~~~~~~~~~~~~~~
TODO

Placeholders
~~~~~~~~~~~~

TODO

Tuple Types
~~~~~~~~~~~~~

Reference Types
~~~~~~~~~~~~~~~

A reference type is simply a mutable memory location, since Relay is a pure
language by default we need a way to introduce limited mutability. In this
case mutable data is clearly marked in the type system as a reference type.

    Ref<T>

Tensor Type
~~~~~~~~~~~

Tensor values in Relay are typed with tensor types. A tensor type is
parametrized by a data type, and shape. The data type must be a base
type as enforced by the kind checking rules described in TODO.

This restriction importantly means

The shape may be any valid Relay shape as described in the below
section on shapes.


======
Shapes
======

Shape Singleton
~~~~~~~~~~~~~~~
I don't like this name

ShapeAttr
~~~~~~~~~
TODO

ShapeProjection
~~~~~~~~~~~~~~~
TODO

ShapeBinaryOp
~~~~~~~~~~~~~

enum ShapeOp : int {
  SHPLUS = 0,
  SHSUB = 1,
  SHMUL = 2,
  SHDIV = 3
};


Shape Sequence
~~~~~~~~
A sequence of shapes ...


ShapeBroadcast
~~~~~~~~~~~~~~
