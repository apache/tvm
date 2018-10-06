Adding an Operator to Relay
===========================

In order to use TVM operators from within the Relay IR, the
operators need to be registered in Relay in order to ensure
that they will be integrated into Relay's type system.

Registering an operator requires three steps:

- Using the ``RELAY_REGISTER_OP`` macro in C++ to
register the operator's arity and type information
- Defining a C++ function to produce a call node for the
operator and registering a Python API hook for the function
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
- Support level (1 indicating an internal intrinsic, higher numbers
indicating operators that are not as integral to the framework or are
supported externally)
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

Summary
-------

- A TVM operator can be registered in Relay using a relation to express
the appropriate type information.
- Using an operator in Relay requires a function to produce a
call node for the operator.
- It is best to have a simple Python wrapper for producing the call node.
