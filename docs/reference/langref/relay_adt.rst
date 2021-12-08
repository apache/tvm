
..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _adt-overview:

=============================
Algebraic Data Types in Relay
=============================

Algebraic data types (ADTs) are a staple feature of functional programming languages,
particularly those derived from ML, because they express data structures in a
manner that is easy to reason about when writing recursive computations.
Because recursion is intended to be one of the primary mechanisms of control
flow in Relay, it is important that Relay include ADTs in order to best express
loops and other control flow structures that must be implemented using recursion.

Defining and Matching on an ADT
===============================

*Note: ADTs are not presently supported in the text format. The syntax here is speculative, based on ADTs in other languages.*

ADTs can be understood as a generalized version of :code:`enum` and :code:`struct` types
from C-like languages. Like a C :code:`struct:`, an ADT instance is a container for fields
of specified types, but the type system allows for the same type to encode different possible
groupings of fields in a systematic manner, similar to C :code:`enum` types, which are
defined using a finite set of possible values named by the user.

Specifically, an ADT is defined as a named group of constructors, each of which is
a function that takes values of specified types as arguments and returns an instance
of the named ADT. An ADT instance simply contains the values of the arguments
passed to the constructor call used to produce it.

An ADT value is opaque until it is *deconstructed*, allowing the arguments to the
constructor to be accessed again and used to compute new values. Because
a particular ADT can have multiple constructors with different signatures,
it is usually necessary to branch on the different possible constructors,
resulting in the *match* syntax for ADTs. Hence, ADTs are sometimes called
"tagged unions" because an ADT instance is tagged by the name of the constructor
used to produce it and can later be inspected based on the tag.

Because each ADT has a finite set of constructors, it is straightforward to determine
whether a function processing an ADT instance is handling all possible cases.
In particular, the type system can ensure that types are properly assigned in all cases when
deconstructing an ADT instance, in contrast to :code:`union` types in C.
Hence, it is often easy to reason about ADTs.

*Implementation detail: Relay ADT definitions are global and are stored in the module, similarly to global function definitions. An ADT name is, in fact, a global type variable (just as a global function name is a global variable). The module keeps a mapping of ADT names (global type variables) to the list of constructors for that ADT.*

Below is a simple example of defining an ADT and using it in a function
via a match expression:

.. code-block::

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

   @sum(Empty())    # evaluates to 0
   @sum(Single(3))  # evaluates to 3
   @sum(Pair(5, 6)) # evaluates to 11

Note that ADTs are identified by name,
meaning that two ADTs with structurally identical constructors
will nevertheless be distinct data types from the point of view of
the typechecker.

.. code-block::

   # structurally identical constructors to Numbers
   data Numbers2 {
     Empty2 : () -> Numbers2
     Single2 : (Tensor[(), int32]) -> Numbers2
     Pair2 : (Tensor[(), int32], Tensor[(), int32]) -> Numbers2
   }

   # the below results in a type error because Numbers2
   # is a distinct type from Numbers
   # fn() { @sum(Empty2()) }

Type-Checking ADTs and Polymorphism
===================================

This section will go into more specific detail about the typing of ADTs.
Most of the complexity involved results from the fact that, as with functions, ADTs
can be polymorphic and take type parameters.

For example, one of the standard ADTs commonly used in functional
programming languages is the optional type, defined here:

.. code-block::

   # a is a type parameter
   data Optional<a> {
     None : () -> Optional
     Some : (a) -> Optional
   }

Optional types are commonly used as the return type for any operation
involving querying into a data structure (returning :code:`Some(v)`
if a value is found and :code:`None` if it isn't).
Taking a type parameter in the definition allows the same optional type
to be used in a wide variety of situations, rather than having to
define a unique ADT for each different type that could be contained in it.

However, it is important to ensure that option types whose contents
are of different types can still be distinguished by the type system,
since it would violate type safety if a function expecting an option
containing a :code:`Tensor[(), int32]` instead receives an option
containing a :code:`Tensor[(3, 4), float32]`. As this example may
imply, an ADT instance is thus given a type that contains the
concrete type arguments for that instance, ensuring the information is
kept around. Let the below example illustrate:

.. code-block::

   # the signature for option indicates the type argument
   def @inc_scalar(%opt : Optional[Tensor[(), int32]]) -> Tensor[(), int32] {
     match(%opt) {
       case None() { 1 }
       case Some(%s) { %s + 1 }
     }
   }

   def @main() {
     let %one : Optional[Tensor[(), int32]] = Some(1);
     let %big : Optional[Tensor[(10, 10), float32]]
       = Some(Constant(1, (10, 10), float32));
     let %two = inc_scalar(%one);
     # let %bigger = inc_scalar(%big); # type system rejects
     # None does not take an argument so it can always implicitly
     # be given the correct type arguments
     let %z = inc_scalar(None());
     ()
   }

The syntax for the annotated type arguments
(e.g., :code:`Optional[Tensor[(), int32]]`) in the above examples is
called a "type call," treating the polymorphic ADT definition as a
type-level function (taking type params and returning a type, namely
the ADT). Any ADT appearing in a type annotation or function signature
must be annotated with type arguments (a non-polymorphic ADT must be
in a type call with no arguments).

Thus, we can say in general that if constructor :code:`C` that
takes arguments of types :code:`T1, ..., Tn` is a constructor
for an ADT :code:`D` that takes type parameters :code:`v1, ..., vn`
(where :code:`T1, ..., Tn` may contain any of the :code:`v1, ..., vn`),
then :code:`C` has
the type :code:`fun<v1, ..., vn>(T1, ..., Tn) -> D[v1, ..., vn]`.
This means that constructors are typed like ordinary functions and
thus appear inside call nodes and can be passed to or returned by
other functions. In particular, the :code:`Some` example above has
the signature :code:`fun<a>(a) -> Optional[a]`, while :code:`None`
has the signature :code:`fun<a>() -> Optional[a]`.

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
in `Common ADT Uses`_. As an example here, we will
examine the list ADT, ubiquitous in functional languages:

.. code-block::

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

.. code-block::

   def @list_sum(%l : List[Tensor[(), int32]]) -> Tensor[(), int32] {
     match(%l) {
       case Nil() { 0 }
       # add the head of the list to the sum of the tail
       case Cons(%h, %t) { %h + @list_sum(%t) }
     }
   }

As it happens, many recursive functions on lists like the one just given
share structures that can be factored out into generic, easily
usable functions that will be discussed under `Common ADT Uses`_.

.. _adt-pattern:

Pattern Matching in Match Expressions
=====================================

Match expressions in Relay, as in other functional languages, are capable of
more versatile pattern matching than simply having one case for each constructor
for the datatype of the value being deconstructed.

In particular, the patterns in match cases can be built up recursively:

- Constructor patterns match for a particular ADT constructor. If a value matches the constructor, each argument to the constructor will be matched against a nested pattern.
- Wildcard patterns will match any value and will not bind to a variable.
- Variable patterns will match any value and bind it to a local variable, scoped to the match clause.

In the simple case of :code:`@list_sum` above, the first match case has a :code:`Nil` constructor pattern (with no nested arguments)
and the second has a :code:`Cons` constructor pattern that uses variable patterns for each of the arguments to :code:`Cons`.

The below example uses a wildcard pattern to ignore one of the arguments to :code:`Cons`:

.. code-block::

   def @first<a>(%l : List[a]) -> Optional[a] {
     match(%l) {
       case Nil() { None() }
       case Cons(%h, _) { Some(%h) } # list tail is unused and ignored
     }
   }

Here, a constructor pattern is nested inside another constructor pattern to avoid nested match expressions for a list option.
A top-level wildcard pattern is also used to handle all cases that do not match the first clause:

.. code-block::

   def @second_opt<a>(%ll : Optional[List[a]]) -> Optional[a] {
     match(%ll) {
       # we only need the second member of the list if there is one
       case Some(Cons(_, Cons(%s, _))) { Some(%s) }
       case _ { None() }
     }
   }

   # @second_opt(Some(Cons(1, Nil()))) evaluates to None()
   # @second_opt(Some(Cons(1, Cons(2, Nil())))) evaluates to Some(2)
   # @second_opt(Some(Nil())) evaluates to None()
   # @second_opt(None()) evaluates to None()

Note that a match expression checks its patterns in the order the cases are listed: the first clause whose pattern
that matches the input value is the one that is evaluated. Here, a top-level variable pattern binds the whole
input value:

.. code-block::

   def @match_order_beware<a>(%l : List[a]) -> List[a] {
     match(%l) {
       case %v { %v }
       # the above matches everything so neither of these runs
       case Cons(%h, %t) { Cons(%h, @match_order_beware(%t)) }
       case Nil() { Nil() }
     }
   }

Common ADT Uses
===============

In functional programming languages, certain ADTs provide useful facilities for writing common programs.
Parametric polymorphism and higher-order functions allow these ADTs to be easily reuseable and for generic
functions to manipulate them in common situations. Relay includes a "Prelude" of certain pre-defined ADTs
and functions for them that correspond to the indispensable ADTs of other languages.

The option type defined under `Type-Checking ADTs and Polymorphism`_ is one such ADT, used
whenever it can make sense for a function to only return a value under certain circumstances. Having
the option type allows for the type system to keep track of which functions always return a value
of a certain type versus returning an option of that type, ensuring that any options are always
explicitly checked (contrast with returning null pointers or throwing
exceptions as other ways to addressing that problem).

Lists (defined in `Recursion with ADTs`_) can be manipulated by generic functions in a manner similar to
list comprehensions and certain library functions in Python. Below are very common functions for iterating
through lists, which are included in Relay's Prelude. (These have all been extensively characterized
in the functional programming literature, and we do not attempt to reproduce that work in this document.)

.. code-block::

   # Map: for [h1, h2, ..., hn] returns [f(h1), f(h2), ..., f(hn)]
   def @map<a, b>(%f : fn(a) -> b, %l : List[a]) -> List[b] {
     match(%l) {
       case Nil() { Nil() }
       case Cons(%h, %t) { Cons(%f(%h), @map(%f, %t)) }
     }
   }

   # Left fold: for [h1, h2, ..., hn] returns f(...(f(f(z, h1), h2)...), hn)
   def @foldl<a, b>(%f : fn(b, a) -> b, %z : b, %l : List[a]) -> b {
     match(%l) {
       case Nil() { %z }
       case Cons(%h, %t) { @foldl(%f, %f(%z, %h), %t) }
     }
   }

   # Right fold: for [h1, h2, ..., hn] returns f(h1, f(h2, f(..., (f(hn, z)...)
   def @foldr<a, b>(%f : fn(a, b) -> b, %z : b, %l : List[a] -> b {
     match(%l) {
       case Nil() { %z }
       case Cons(%h, %t) { %f(%h, @foldr(%f, %z, %t)) }
     }
   }

Using these iteration constructs, many common operations over lists can be expressed compactly.
For example, the following map doubles all members of a list:

.. code-block::

   # directly written
   def @double(%l : List[Tensor[(), int32]]) -> List[Tensor[(), int32]] {
     match(%l) {
       case Nil() { Nil() }
       case Cons(%h, %t) { Cons(%h * 2, @double(%t)) }
     }
   }

   # map takes care of the recursion
   @map(fn(%i) { %i * 2 }, %l)

The following right fold concatenates two lists:

.. code-block::

   # directly written
   def @concat<a>(%l1 : List[a], %l2 : List[a]) -> List[a] {
     match(%l1) {
       case Nil() { %l2 }
       case Cons(%h, %t) { Cons(%h, @concat(%t, %l2) }
     }
   }

   # foldr takes care of the recursion
   @foldr(fn(%h, %z) { Cons(%h, %z) }, %l2, %l1)

The following left fold flattens a list of lists (using concatenation):

.. code-block::

  # directly written
  def @flatten<a>(%ll : List[List[a]]) -> List[a] {
    match(%ll) {
      case Cons(%h, %t) { @concat(%h, @flatten(%t)) }
      case Nil() { Nil() }
    }

  # foldl takes care of the recursion
  @foldl(@concat, Nil(), %ll)

Note that these iteration constructs can be implemented directly in Relay's
source language and more can easily be defined (and for more data types, like trees),
rather than being constructs built into the language (e.g.,
`"foreach" in MXNet <https://mxnet.apache.org/versions/master/tutorials/control_flow/ControlFlowTutorial.html>`__).
ADTs and their extensibility allow for a broad range of iterations and data structures to be expressed
in Relay and supported by the type system without having to modify the language implementation.

Implementing Neural Nets Using ADTs
===================================

In `this 2015 blog post <http://colah.github.io/posts/2015-09-NN-Types-FP/>`__, Christopher Olah notes that
many neural networks can be easily expressed using common functional programming constructs. Relay's ADTs
allow those examples to be implemented directly in TVM.

First let us suppose that we have a function corresponding to a trained recurrent neural net (RNN)
cell, which takes in a past state and an input value and returns a new state and output value. In
Relay, this would have the following signature:

.. code-block::

   @cell : fn<state_type, in_type, out_type>(state_type, in_type) -> (state_type, out_type)

We might consider a ReLU cell as a simple concrete example, with a trained version below:

.. code-block::

  def @linear(%x, %w, %b) { %w*%x + %b }

  def @relu_cell(%w, # weights
                 %b, # offsets
                 %s, # state
                 %x  # input
  ) {
    let %x2 = @linear(%x, %w.0, %b.0);
    let %s2 = @linear(%s, %w.1, %b.1);
    # doesn't change the state
    (%s, nn.relu(%x2 + %s2))
  }

  # this is a higher-order function because it returns a closure
  def @trained_cell(%w, %b) {
    fn(%x, %h) { @relu_cell(%w, %b, %x, %h) }
  }

Following Olah's example, we can encode a sequence (list) of inputs with the following left fold:

.. code-block::

   def @encode<state_type, in_type, out_type>(%cell, %input : List[in_type], %init : state_type) -> state_type {
     # not using the output
     @foldl(fn(%state, %in) { %cell(%state, %in).0 }, %init, %input)
   }

Using an *unfold* iterator (from Haskell's standard library), the same cell could be used to make
a generator network (which takes a single input and produces a sequence of outputs):

.. code-block::

   # included in Relay's Prelude
   def @unfoldr<a, b>(%f : fn(b) -> Optional[(a, b)], %z : b) -> List[a] {
     match(%f(%z)) {
       case Some(%pair) { Cons(%pair.0, @unfoldr(%f, %pair.1)) }
       case None() { Nil() }
     }
   }

   # we need some way of generating an input to the cell function given only a state
   def @gen_func<state_type, in_type, out_type>(%state : state_type) : Optional[(out_type, state_type)] {
     let %in : Optional[in_type] = @generate_input(%state);
     match(%in) {
       case Some(%n) {
         let %cell_out = @cell(%n, %state);
         Some((%cell_out.1, %cell_out.0)) # pair of output and state
       }
       case None() { None() }
     }
   }

   def @generator<state_type, in_type, out_type>(%cell, %init : state_type) -> List[out_type] {
     @unfoldr(fn(%state) { @gen_func(%cell, %state) }, %init)
   }

An accumulating map (a fold that simultaneously updates an accumulator value and a list
of outputs) can be used to write a general RNN (with an output for every input):

.. code-block::

   def @map_accumr<a, b, c>(%f : fn(a, b) -> (a, c), %acc : a, %l : List[b]) -> (a, List[c]) {
     match(%l) {
       case Nil() { (%acc, Nil()) }
       case Cons(%b, %t) {
         let %update = %f(%acc, %b);
         let %rest = @map_accumr(%f, %update.0, %t));
         (%rest.0, Cons(%update.1, %rest.1))
       }
     }
   }

   # can also be implemented as a right fold
   # (this version is included in Relay's Prelude)
   def @map_accumr_fold(%f, %acc, %l) {
     @foldr(fn(%b, %p) {
       let %f_out = %f(%p.0, %b);
       (%f_out.0, Cons(%f_out.1, %p.1))
     },
     (%acc, Nil()), %l)
   }

   def @general_rnn<state_type, in_type, out_type>(%cell, %init : state_type, %input : List[in_type])
     -> (state_type, List[out_type]) {
     @map_accumr(%cell, %init, %input)
   }

Olah also gives an example of a bidirectional neural network, in which two sets of
cells (which may have different weights) process the input in both directions and produce a
single set of outputs. The following is a Relay implementation of that example:

.. code-block::

   # creates a list of tuples from two lists
   # included in Relay's Prelude
   def @zip<a, b>(%l : List[a], %m : List[b]) -> List[(a, b)] {
     match(%l) {
       case Nil() { Nil() }
       case Cons(%a, %t1) {
         match(%m) {
           case Nil() { Nil() }
           case Cons(%b, %t2) { Cons((%a, %b), @zip(%t1, %t2)) }
         }
       }
     }
   }

   # analogous to map_accumr
   # included in Relay's Prelude
   def @map_accmul(%f, %acc, %l) {
     @foldl(fn(%p, %b){
       let %f_out = %f(%p.0, %b);
       (%f_out.0, Cons(%f_out.1, %p.1))
     }, (%acc, Nil()), %l)
   }

   def @bidirectional_rnn<state1_type, state2_type, in_type, out1_type, out2_type>
     (%cell1, %cell2, %state1 : state1_type, %state2 : state2_type, %input : List[in_type])
     -> List[(out1_type, out2_type)] {
     @zip(@map_accumr(%cell1, %state1, %input).1, @map_accuml(%cell2, %state2, %input).1)
   }
