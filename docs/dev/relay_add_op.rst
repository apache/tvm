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

.. _relay-add-op:

Adding an Operator to Relay
===========================

In order to use TVM operators from within the Relay IR, the
operators need to be registered in Relay in order to ensure
that they will be integrated into Relay's type system.

Registering an operator requires three steps:

- Using the ``RELAY_REGISTER_OP`` macro in C++ to register the operator's arity and type information
- Defining a C++ function to produce a call node for the operator and registering a Python API hook for the function
- Wrapping the above Python API hook in a neater interface

The file ``src/relay/op/tensor/binary.cc`` provides
examples of the first two steps, while
``python/tvm/relay/op/tensor.py`` gives examples of the
last.

Registering an Operator
-----------------------

TVM already has an operator registry, but Relay cannot properly
incorporate TVM operators without additional type information.

To allow for flexibility in registering operators and greater
expressivity and granularity in expressing types in Relay, operators
are typed using relations between input and output types. These relations
are represented as functions that take in a list of input types and
output types (any of these types may be incomplete) and return a list
of input and output types that satisfies the relation. Essentially, a
relation for an operator can enforce all the necessary typing rules
(namely by inspecting the input types) in addition to computing the
output type.

For example, see ``src/relay/op/type_relations.h`` and their
implementations. E.g., ``BroadcastRel`` takes two input types and an
output type, checks that they are all tensor types with the same underlying
data type, and finally ensures that the shape of the output type is the
broadcast of the input types' shapes.

It may be necessary to add another type relation to ``type_relations.h``
if the existing ones do not capture the behavior of the desired operator.

The ``RELAY_REGISTER_OP`` macro in C++ allows a developer
to specify the following information about an operator in Relay:

- Arity (number of arguments)
- Names and descriptions for positional arguments
- Support level (1 indicates an internal intrinsic; higher numbers indicate less integral or externally supported operators)
- A type relation for the operator

The below example is from ``binary.cc`` and uses a broadcasting
add for tensors:

.. code:: c

    RELAY_REGISTER_OP("add")
        .set_num_inputs(2)
        .add_argument("lhs", "Tensor", "The left hand side tensor.")
        .add_argument("rhs", "Tensor", "The right hand side tensor.")
        .set_support_level(1)
        .add_type_rel("Broadcast", BroadcastRel);

Creating a Call Node
--------------------

This step requires simply writing a function that takes
the arguments to the operator (as Relay expressions) and
returning a call node to the operator (i.e., the node that
should be placed into the Relay AST where the call to the
operator is intended).

At present call attributes and type arguments (the last two fields)
are not supported, so it suffices to use ``Op::Get`` to fetch
the operator's information from the operator registry and pass in
the arguments to the call node, as below.

.. code:: c

    TVM_REGISTER_API("relay.op._make.add")
        .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {
            static const Op& op = Op::Get("add");
          return CallNode::make(op, {lhs, rhs}, Attrs(), {});
        });

Including a Python API Hook
---------------------------

It is generally the convention in Relay, that functions exported
through ``TVM_REGISTER_API`` should be wrapped in a separate
Python function rather than called directly in Python. In the case
of the functions that produce calls to operators, it may be convenient
to bundle them, as in ``python/tvm/relay/op/tensor.py``, where
elementwise operators on tensors are all provided. For example,
the following is how the add function from the previous section is
exposed in Python:

.. code:: python

    def add(lhs, rhs):
        """Elementwise addition.

        Parameters
        ----------
        lhs : relay.Expr
            The left hand side input data
        rhs : relay.Expr
            The right hand side input data

        Returns
        -------
        result : relay.Expr
            The computed result.
        """
        return _make.add(lhs, rhs)

Note that these Python wrappers might also be good opportunities to
provide an easier interface to the operator. For example, the
``concat`` operator is registered as taking only one operator,
namely a tuple with the tensors to be concatenated, but the Python
wrapper takes the tensors as arguments and combines them into a tuple
before producing the call node:

.. code:: python

    def concat(*args):
        """Concatenate the input tensors along the zero axis.

        Parameters
        ----------
        args: list of Tensor

        Returns
        -------
        tensor: The concatenated tensor.
        """
        tup = Tuple(list(args))
        return _make.concat(tup)

Gradient Operators
------------------

Gradient operators are important for writing differentiable programs in
Relay. While it is the case that Relay's autodiff algorithm can differentiate
first-class language constructs, operators are opaque. Because Relay can't
look into the implementation, an explicit differentiation rule must be
provided.

Both Python and C++ can be used to write gradient operators, but we focus our
examples on Python, as it is more commonly used.

Adding a Gradient in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A collection of Python gradient operators can be found in
``python/tvm/relay/op/_tensor_grad.py``. We will walk through two
representative examples: ``sigmoid`` and ``multiply``.

.. code:: python

    @register_gradient("sigmoid")
    def sigmoid_grad(orig, grad):
        """Returns [grad * sigmoid(x) * (1 - sigmoid(x))]."""
        return [grad * orig * (ones_like(orig) - orig)]

The inputs here are the original operator ``orig`` and a gradient ``grad`` to
accumulate into. What we return is a list, where the element at the i'th
index is the derivative of the operator with respect to the operator's i'th
input. In general, the gradient will return a list with as many elements as
there are inputs to the base operator.

Before we further analyze this definition, first we should recall the
derivative of the sigmoid function: :math:`\frac{\partial \sigma}{\partial x}
= \sigma(x)(1 - \sigma(x))`. The definition above looks similar to the
mathematical definition, but there is one important addition, which we
describe below.

The term ``orig * (ones_like(orig) - orig)`` directly matches the derivative,
because ``orig`` here is the sigmoid function, but we're not just interested
in how to compute the gradient of this function. We're interested in
composing this gradient with other gradients, so we can accumulate the
gradient across an entire program. This is where the ``grad`` term comes in.
In the expression ``grad * orig * (ones_like(orig) - orig)``, multiplying by
``grad`` specifies how to compose the derivative with the gradient thus far.

Now, we consider ``multiply``, a slightly more interesting example:

.. code:: python

    @register_gradient("multiply")
    def multiply_grad(orig, grad):
        """Returns [grad * y, grad * x]"""
        x, y = orig.args
        return [collapse_sum_like(grad * y, x),
                collapse_sum_like(grad * x, y)]

In this example, there are two elements in the returned list, because
``multiply`` is a binary operator. And to recall, if :math:`f(x, y) = xy`, the
partial derivatives are :math:`\frac{\partial f}{\partial x} = y` and
:math:`\frac{\partial f}{\partial y} = x`.

There is one required step for ``multiply`` that is not required for
``sigmoid``, because ``multiply`` has broadcasting semantics. Since the shape
of ``grad`` might not match the shape of the inputs, we use
``collapse_sum_like`` to take the contents of the ``grad * <var>`` terms and
make the shape match the shape of the input we're differentiating with
respect to.

Adding a Gradient in C++
~~~~~~~~~~~~~~~~~~~~~~~~

Adding a gradient in C++ is similar to adding one in Python, but the
interface for registering is slightly different.

First, make sure ``src/relay/pass/pattern_util.h`` is included. It provides
helper functions for creating nodes in the Relay AST. Then, define the
gradient in a similar fashion as in the Python example:

.. code:: c

    tvm::Array<Expr> MultiplyGrad(const Expr& orig_call, const Expr& output_grad) {
        const Call& call = orig_call.Downcast<Call>();
        return { CollapseSumLike(Multiply(output_grad, call.args[1]), call.args[0]),
                 CollapseSumLike(Multiply(output_grad, call.args[0]), call.args[1]) };
    }

Notice that in C++ we can't use the same operator overloading that we have in
Python, and we need to downcast, so the implementation is more verbose. Even
so, we can easily verify that this definition mirrors the earlier example in
Python.

Now, instead of using a Python decorator, we need to tack a ``set_attr`` call
for "FPrimalGradient" onto the end of the base operator's registration, in
order to register the gradient.

.. code:: c

    RELAY_REGISTER_OP("multiply")
        // ...
        // Set other attributes
        // ...
        .set_attr<FPrimalGradient>("FPrimalGradient", MultiplyGrad);

Summary
-------

- A TVM operator can be registered in Relay using a relation to express the appropriate type information.
- Using an operator in Relay requires a function to produce a call node for the operator.
- It is best to have a simple Python wrapper for producing the call node.
