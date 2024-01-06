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

In this document we will go over the steps needed to register a new TVM operator
in Relay. We will be following this PR which adds a `cumulative product`_ operation as an example.
The PR itself builds upon another PR which adds a `cumulative sum`_ operation.

.. _cumulative product: https://github.com/apache/tvm/pull/7722
.. _cumulative sum: https://github.com/apache/tvm/pull/7334

Registering a new operator requires a few steps:

1. Add an attribute node declaring fixed arguments which are known at compile time
2. Write a type relation for your operation to integrate into Relay's type system.
3. Use the ``RELAY_REGISTER_OP`` macro in C++ to register the operator's arity, type, and other hints for the compiler
4. Write how the operator is computed
5. Register the compute, schedule with the relay operator
6. Define a C++ function to produce a call node for the operator and registering a Python API hook for the function
7. Wrapping the above Python API hook in a neater interface
8. Writing tests for the new relay operator

1. Defining an Attribute Node
-----------------------------
Attributes are fixed arguments which are supposed to be known at compile time. The stride and dilation of a convolution
operator would be an appropriate example of fields which might belong in an attribute node for a convolution operator.

Attributes should be defined in a file within the folder `include/tvm/relay/attrs/`_.

.. _include/tvm/relay/attrs/: https://github.com/apache/tvm/tree/main/include/tvm/relay/attrs

Ultimately we want to create an operator whose interface can be seen clearly in the final python interface:

.. code:: python

    def cumprod(data, axis=None, dtype=None, exclusive=None):
        """Numpy style cumprod op. Return the cumulative inclusive product of the elements along
        a given axis.
        Parameters
        ----------
        data : relay.Expr
            The input data to the operator.
        axis : int, optional
            Axis along which the cumulative product is computed. The default (None) is to compute
            the cumprod over the flattened array.
        dtype : string, optional
            Type of the returned array and of the accumulator in which the elements are multiplied.
            If dtype is not specified, it defaults to the dtype of data.
        exclusive : bool, optional
            If true will return exclusive product in which the first element is not
            included. In other terms, if true, the j-th output element would be
            the product of the first (j-1) elements. Otherwise, it would be the product of
            the first j elements. The product of zero elements will be 1.
        Returns
        -------
        result : relay.Expr
            The result has the same size as data, and the same shape as data if axis is not None.
            If axis is None, the result is a 1-d array.
        """

A similiar interface exists for ``cumsum()``.

Therefore, when defining our attributes in ``include/tvm/relay/attrs/transform.h`` we choose the axis,
accumulation dtype, and exclusivity of the operation as appropriate fields for the struct.

.. code:: c++

  /*! \brief Attributes used in cumsum and cumprod operator */
  struct ScanopAttrs : public tvm::AttrsNode<ScanopAttrs> {
    Integer axis;
    DataType dtype;
    Bool exclusive = Bool(false);
    TVM_DECLARE_ATTRS(ScanopAttrs, "relay.attrs.ScanopAttrs") {
      TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
      TVM_ATTR_FIELD(dtype).describe("Output data type").set_default(NullValue<DataType>());
      TVM_ATTR_FIELD(exclusive)
          .describe("The first element is not included")
          .set_default(Bool(false));
    }
  };

2. Writing a Type Relation
--------------------------
To allow for flexibility in registering operators and greater
expressivity and granularity in expressing types in Relay, operators
are typed using relations between input and output types. These relations
are represented as functions that take in a list of input types and
output types (any of these types may be incomplete) and return a list
of input and output types that satisfies the relation. This includes shape
information which can be determined statically at compile time. Essentially, a
relation for an operator can enforce all the necessary typing rules
(namely by inspecting the input types) in addition to computing the
output type.

Type relation for the cumulative product and sum operators can be found in
``src/relay/op/tensor/transform.cc``:

.. code:: c++

    TVM_REGISTER_NODE_TYPE(ScanopAttrs);
    bool ScanopRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
        // types: [data, output]
        ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
        const auto* data = types[0].as<TensorTypeNode>();
        if (data == nullptr) {
            ICHECK(types[0].as<IncompleteTypeNode>())
            << "Scanop: expect input type to be TensorType but get " << types[0];
            return false;
        }

        const auto* param = attrs.as<ScanopAttrs>();

        auto dtype = param->dtype;
        if (dtype.is_void()) {
            dtype = data->dtype;
        }

        if (param->axis.defined()) {
            reporter->Assign(types[1], TensorType(data->shape, dtype));
        } else {
            auto prod = data->shape[0];
            for (size_t i = 1; i < data->shape.size(); ++i) {
                prod = prod * data->shape[i];
            }
            reporter->Assign(types[1], TensorType({prod}, dtype));
        }

        return true;
    }

3. Relating the Arity and Attributes to an Operation
----------------------------------------------------

We then register the name of our new ops and annotate them with the calling interface.
The ``RELAY_REGISTER_OP`` macro in C++ allows a developer
to specify the following information about an operator in Relay:

- Arity (number of arguments)
- Names and descriptions for positional arguments
- Support level (1 indicates an internal intrinsic; higher numbers indicate less integral or externally supported operators)
- A type relation for the operator
- Other annotations useful when optimizing the operation.

Once again we add this to ``src/relay/op/tensor/transform.cc``:

.. code:: c++

    RELAY_REGISTER_OP("cumsum")
        .describe(
            R"doc(Return the cumulative sum of the elements along a given axis.)doc" TVM_ADD_FILELINE)
        .set_num_inputs(1)
        .add_argument("data", "Tensor", "The input tensor.")
        .set_support_level(3)
        .add_type_rel("Cumsum", ScanopRel)
        .set_attr<TOpPattern>("TOpPattern", kOpaque);

    RELAY_REGISTER_OP("cumprod")
        .describe(
            R"doc(Return the cumulative product of the elements along a given axis.)doc" TVM_ADD_FILELINE)
        .set_num_inputs(1)
        .add_argument("data", "Tensor", "The input tensor.")
        .set_support_level(3)
        .add_type_rel("Cumprod", ScanopRel)
        .set_attr<TOpPattern>("TOpPattern", kOpaque);

In this case the ``TOpPattern`` is a hint to the compiler on the pattern of computation the operator does, which might be
useful for fusing operators. ``kOpaque`` tells TVM to not bother trying to fuse this operator.

4. Defining the Compute of the Operation
----------------------------------------

While we've now defined the interface for our operations we still need to define
how to perform the actual calculations for cumulative sum and product.

Writing this code is outside the scope of the tutorial. For now, we assume we
have a well tested implementation for the operation's compute. For more details
on how to do this, we recommend looking up the tutorials on :ref:`tensor
expressions <tutorial-tensor-expr-get-started>`, :ref:`TVM's operator inventory
(topi) <tutorial-topi>` and looking at the example cumulative sum and product
implementations found in `python/tvm/topi/scan.py`_ and the gpu versions in
`python/tvm/topi/cuda/scan.py`_. In the case of our cumulative sum and product
operations we write things directly in :ref:`TIR <api-python-tir>` which is the
representation where tensor expressions and topi will lower into.

.. _python/tvm/topi/scan.py: https://github.com/apache/tvm/blob/main/python/tvm/topi/scan.py
.. _python/tvm/topi/cuda/scan.py: https://github.com/apache/tvm/blob/main/python/tvm/topi/cuda/scan.py

5. Hooking up Compute and Strategy with Relay
---------------------------------------------

After you have implemented your compute function we now need to glue it to our
relay operation. Within TVM this means not only defining the computation, but also the schedule
for an operation. A strategy is a method which picks which computation and which schedule
to use. For example, for 2D convolutions we might recognize we are doing a depthwise convolution
and dispatch to a more efficient computation and schedule as a result. In our case however we have
no such need except for dispatching between our CPU and GPU implementations. In
``python/tvm/relay/op/strategy/generic.py`` and ``python/tvm/relay/op/strategy/cuda.py`` we
add the following strategies:

.. code:: python

    def wrap_compute_scanop(topi_compute):
        """Wrap scanop style topi compute"""

        def _compute_scanop(attrs, inputs, _):
            return [topi_compute(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]

        return _compute_scanop


    @override_native_generic_func("cumsum_strategy")
    def cumsum_strategy(attrs, inputs, out_type, target):
        """cumsum generic strategy"""
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_scanop(topi.cumsum),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="cumsum.generic",
        )
        return strategy


    @override_native_generic_func("cumprod_strategy")
    def cumprod_strategy(attrs, inputs, out_type, target):
        """cumprod generic strategy"""
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_scanop(topi.cumprod),
            wrap_topi_schedule(topi.generic.schedule_extern),
            name="cumprod.generic",
        )
        return strategy

    @cumsum_strategy.register(["cuda", "gpu"])
    def cumsum_strategy_cuda(attrs, inputs, out_type, target):
        """cumsum cuda strategy"""
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_scanop(topi.cuda.cumsum),
            wrap_topi_schedule(topi.cuda.schedule_scan),
            name="cumsum.cuda",
        )
        return strategy


    @cumprod_strategy.register(["cuda", "gpu"])
    def cumprod_strategy_cuda(attrs, inputs, out_type, target):
        """cumprod cuda strategy"""
        strategy = _op.OpStrategy()
        strategy.add_implementation(
            wrap_compute_scanop(topi.cuda.cumprod),
            wrap_topi_schedule(topi.cuda.schedule_scan),
            name="cumprod.cuda",
        )
        return strategy

Where in each strategy we define the compute we wrote and the schedule to use within ``add_implementation()``.
We finally link the strategy and compute with the defined relay operator in ``python/tvm/relay/op/_transform.py``:

.. code:: python

    # cumsum
    @_reg.register_compute("cumsum")
    def compute_cumsum(attrs, inputs, output_type):
        """Compute definition of cumsum"""
        return [topi.cumsum(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]


    _reg.register_strategy("cumsum", strategy.cumsum_strategy)
    _reg.register_shape_func("cumsum", False, elemwise_shape_func)

    # cumprod
    @_reg.register_compute("cumprod")
    def compute_cumprod(attrs, inputs, output_type):
        """Compute definition of cumprod"""
        return [topi.cumprod(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]


    _reg.register_strategy("cumprod", strategy.cumprod_strategy)
    _reg.register_shape_func("cumprod", False, elemwise_shape_func)

The shape functions are used for determining output shape given a dynamically shaped tensor. In this
case we tell TVM the output shape will be the same as the input shape.

6. Creating a Relay Call Node and Exposing a Python Hook
--------------------------------------------------------
We now have a working operation and now just need to properly call it
via a Relay Call Node. This step requires simply writing a function that takes
the arguments to the operator (as Relay expressions) and
returning a call node to the operator (i.e., the node that
should be placed into the Relay AST where the call to the
operator is intended).

At present call attributes and type arguments (the last two fields)
are not supported, so it suffices to use ``Op::Get`` to fetch
the operator's information from the operator registry and pass in
the arguments to the call node, as below. In ``src/relay/op/tensor/transform.cc``:

.. code:: c++

    Expr MakeCumsum(Expr data, Integer axis, DataType dtype, Bool exclusive) {
        auto attrs = make_object<ScanopAttrs>();
        attrs->dtype = dtype;
        attrs->axis = axis;
        attrs->exclusive = exclusive;
        static const Op& op = Op::Get("cumsum");
        return Call(op, {data}, Attrs(attrs), {});
    }

    TVM_REGISTER_GLOBAL("relay.op._make.cumsum").set_body_typed(MakeCumsum);

    Expr MakeCumprod(Expr data, Integer axis, DataType dtype, Bool exclusive) {
        auto attrs = make_object<ScanopAttrs>();
        attrs->dtype = dtype;
        attrs->axis = axis;
        attrs->exclusive = exclusive;
        static const Op& op = Op::Get("cumprod");
        return Call(op, {data}, Attrs(attrs), {});
    }

    TVM_REGISTER_GLOBAL("relay.op._make.cumprod").set_body_typed(MakeCumprod);

Where ``TVM_REGISTER_GLOBAL`` exposes the ``MakeCumsum`` and ``MakeCumprod`` functions
in Python via ``relay.op._make.cumsum(...)`` and ``relay.op._make.cumprod(...)``.

7. Including a Cleaner Python API Hook
--------------------------------------

It is generally the convention in Relay, that functions exported
through ``TVM_REGISTER_GLOBAL`` should be wrapped in a separate
Python function rather than called directly in Python. For our
operators we expose this cleaner interface in ``python/tvm/relay/op/transform.py``

.. code:: python

    def cumsum(data, axis=None, dtype=None, exclusive=None):
        return _make.cumsum(data, axis, dtype, exclusive)

    def cumprod(data, axis=None, dtype=None, exclusive=None):
        return _make.cumprod(data, axis, dtype, exclusive)

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

8. Writing Unit Tests!
----------------------
This is self explanatory! Some example unit tests can be found in
`tests/python/relay/test_op_level3.py`_ for our cumulative sum
and product operators.

.. _tests/python/relay/test_op_level3.py: https://github.com/apache/tvm/blob/main/tests/python/relay/test_op_level3.py


Other Topics
------------

Gradient Operators
~~~~~~~~~~~~~~~~~~

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

First, make sure ``src/relay/transforms/pattern_utils.h`` is included. It provides
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
