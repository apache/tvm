=============================
Algebraic Data Types in Relay
=============================

Algebraic data types (ADTs, also known as sum types) are
a staple feature of functional programming languages, particularly
those derived from ML, because they express data structures in a
manner that is easy to reason about when writing recursive computations.
Because recursion is intended to be one of the primary mechanisms of control
flow in Relay, it is important that Relay include ADTs in order to handle
loops and other control flow structures.

Defining and Matching on an ADT
===============================

*Note: ADTs are not presently supported in the text format.
 The syntax here is speculative, based on ADTs in other languages.*

An ADT is defined with a name and a list of constructors, each of
which takes certain types as arguments and returns an instance of the
named ADT. A call to one of the constructors in the program where the
ADT is defined produces an instance of the ADT.

An ADT "value" simply contains the values of the arguments passed
to the constructor used to produce it. An ADT value is opaque until
it is *deconstructed*, allowing the arguments to the
constructor to be accessed again and used to compute new values. Because
a particular ADT can have multiple constructors with different signatures,
it is usually necessary to branch on the different possible constructors,
resulting in the *match* syntax for ADTs.

Note that ADTs are identified by name,
meaning that two ADTs with structurally identical constructors
will nevertheless be distinct data types from the point of view of
the typechecker.

Below is a simple example of defining an ADT and using it in a function
via a match construction.

.. code-block:: python

   # Defines an ADT named "Numbers"
   data Numbers {
     Empty : () -> Numbers
     Single : (Tensor[(), int32]) -> Numbers
     Pair : (Tensor[(), int32], Tensor[(), int32]) -> Numbers
   }
   # A Numbers value can be produced using an Empty, Single, or Pair
   # constructor, each with a signature given above

   def @sum(%n : Numbers[]) -> Tensor[(), int32] {
      # The match expression branches on the constructor that was
      # used to produce %n. The variables in each case are bound
      # if the constructor matches that used for %n
      match(%n) {
        case Empty() { 0 }
        case Single(x) { x }
        case Pair(x, y) { x + y }
      }
   }

   def @main() {
     let %s1 = @sum(Empty());    # evaluates to 0
     let %s2 = @sum(Single(3));  # evaluates to 3
     let %s3 = @sum(Pair(5, 6)); # evaluates to 11
     ()
   }

Type-Checking ADTs and Polymorphism
===================================

This section will go into more specific detail about the typing of ADTs.
Most of the complexity involved here results from the fact that ADTs
can be polymorphic and take type parameters, like functions.

For example, one of the standard ADTs commonly used in functional
programming languages is the option type, defined here:

.. code-block:: python

   data Option<a> {
     None : () -> Option
     Some : (a) -> Option
   }

Option types are commonly used as the return type for any operation
involving querying into a data structure (returning :code:`Some(v)`
if a value is found and :code:`None` if it isn't).
Taking a type parameter in the definition allows the same option type
to be used in a wide variety of situations, rather than having to
define a unique ADT for each different type that could be contained in
the option.

However, it is important to ensure that option types whose contents
are of different types can still be distinguished by the type system,
since it would violate type safety if a function expecting an option
containing a :code:`Tensor[(), int32]` instead receives an option
containing a :code:`Tensor[(3, 4), float32]`. As this example may
imply, an ADT instance is thus given a type that contains the
concrete type arguments for that instance, ensuring the information is
kept around. Let the below example illustrate:

.. code-block:: python

   # the signature for option indicates the type argument
   def @inc_scalar(%opt : Option[Tensor[(), int32]]) -> Tensor[(), int32] {
     match(%opt) {
       case None() { 1 }
       case Some(%s) { %s + 1 }
     }
   }

   def @main() {
     let %one : Option[Tensor[(), int32]] = Some(1);
     let %big : Option[Tensor[(10, 10), float32]]
       = Some(Constant(1, (10, 10), float32));
     let %two = inc_scalar(%one);
     # let %bigger = inc_scalar(%big); # type system rejects
     # None does not take an argument so it can always implicitly
     # be given the correct type arguments
     let %z = inc_scalar(None());
     ()
   }

The syntax for the annotated type arguments in the above examples is
called a "type call," treating the polymorphic ADT definition as a
type-level function (taking type params and returning a type, namely
the ADT). Any ADT appearing in a type annotation or function signature
must be annotated with type arguments (a non-polymorphic ADT must be
in a type call with no arguments).

Thus, we can say in general that if constructor :code:`C` that
takes arguments of types :code:`T1, ..., Tn` is a constructor
for an ADT :code:`D` that takes type arguments :code:`v1, ..., vn`,
then :code:`C` has
the type :code:`fun<v1, ..., vn>(T1, ..., Tn) -> D[v1, ..., vn]`.
This means that constructors behave like ordinary functions and
thus appear inside call nodes and can be passed to or returned by
other functions. In particular, the :code:`Some` example above has
the signature :code:`fun<a>(a) -> Option[a]`, while :code:`None`
has the signature :code:`fun<a>() -> Option[a]`.

Recursion with ADTs
===================

ADT definitions are allowed to be recursive, that is, a definition for
an ADT named :code:`D` can assume the existence of type :code:`D` and
use it as an argument to constructors. Recursion allows ADTs to
represent complex structures such as lists or trees; it is the source
of much of ADTs' power in functional programming, since an appropriately
designed data structure could make it easy to concisely express a
computation with a recursive function.

Many commonly used ADTs involve recursion; some of these are given
in `Stanard ADTs and Their Uses`_. As an example here, we will
examine the list ADT, ubiquitous in functional languages:

.. code-block:: python

   data List<a> {
      Nil : () -> List
      Cons : (a, List[a]) -> List
   }

(Notice that the recursive reference to :code:`List` is wrapped
in a type call even in the constructor.)

The above definition means that a list of values of a particular type
can be represented by nesting :code:`Cons` constructors until the
end of the list is reached, which can be indicated with a :code:`Nil`
(representing an empty list).

Lists represented in this manner can easily be recursively processed.
For example, the following function sums a list of integers:

.. code_block:: python

   def @list_sum(%l : List[Tensor[(), int32]]) -> Tensor[(), int32] {
     match(%l) {
       case Nil() { 0 } # base case
       # induction: add the head of the list to the sum of the tail
       case Cons(%h, %t) { %h + @list_sum(%t) }
     }
   }

As it happens, many recursive functions on lists like the one just given
share structures that can be factored out into generic, easily
usable functions that will be discussed under `Standard ADTs and Their Uses`_.

The above example refers to the :code:`Nil` match case as a "base case";
this is because the :code:`Nil` constructor does not require an instance
of a list to be called. It is possible to give an ADT definition where
every constructor case has an argument that is a recursive reference
to the ADT, but it will never be possible to create an instance of
such an ADT, since there is no way to start building up the nesting.

Pattern Matching in Match Expressions
=====================================

Standard ADTs and Their Uses
============================

Implementation Details: Module Type Data
========================================

Optimizations for ADTs
======================
