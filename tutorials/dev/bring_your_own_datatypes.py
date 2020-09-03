# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Bring Your Own Datatypes to TVM
===============================
**Authors**: `Gus Smith <https://github.com/gussmith23>`_, `Andrew Liu <https://github.com/hypercubestart>`_

In this tutorial, we will show you how to utilize the Bring Your Own Datatypes framework to use your own custom datatypes in TVM.
Note that the Bring Your Own Datatypes framework currently only handles **software emulated versions of datatypes**.
The framework does not support compiling for custom accelerator datatypes out-of-the-box.

Datatype Libraries
------------------

The Bring Your Own Datatypes allows users to register their own datatype implementations alongside TVM's native datatypes (such as ``float``).
In the wild, these datatype implementations often appear as libraries.
For example:

- `libposit <https://github.com/cjdelisle/libposit>`_, a posit library
- `Stillwater Universal <https://github.com/stillwater-sc/universal>`_, a library with posits, fixed-point numbers, and other types
- `SoftFloat <https://github.com/ucb-bar/berkeley-softfloat-3>`_, Berkeley's software implementation of IEEE 754 floating-point

The Bring Your Own Datatypes enables users to plug these datatype implementations into TVM!

In this section, we will use an example library, `Stillwater Universal <https://github.com/stillwater-sc/universal>`_.
**Posits** are a datatype developed to compete with IEEE 754 floating point numbers.
We won't go into much detail about the datatype itself.
If you'd like to learn more, read through John Gustafson's `Beating Floating Point at its Own Game <https://posithub.org/docs/BeatingFloatingPoint.pdf>`_.

The Universal library is included as a 3rd party library in TVM already.
We have built a wrapper around the library, located at 3rdparty/posit/posit-wrapper.cc.
If you would like to try this with your own datatype library, first bring the library's functions into the process space with ``CDLL``:

.. code-block :: python

    ctypes.CDLL('my-datatype-lib.so', ctypes.RTLD_GLOBAL)


A Simple TVM Program
--------------------

We'll begin by writing a simple program in TVM; afterwards, we will re-write it to use custom datatypes.
"""
import tvm
from tvm import relay

# Our basic program: Z = X + Y
x = relay.var('x', shape=(3, ), dtype='float32')
y = relay.var('y', shape=(3, ), dtype='float32')
z = x + y
program = relay.Function([x, y], z)
module = tvm.IRModule.from_expr(program)

######################################################################
# Now, we create random inputs to feed into this program using numpy:

import numpy as np
np.random.seed(23)  # for reproducibility

x_input = np.random.rand(3).astype('float32')
y_input = np.random.rand(3).astype('float32')
print("x: {}".format(x_input))
print("y: {}".format(y_input))

######################################################################
# Finally, we're ready to run the program:

ex = relay.create_executor(mod=module)

z_output = ex.evaluate()(x_input, y_input)
print("z: {}".format(z_output))

######################################################################
# Adding Custom Datatypes
# -----------------------
# Now, we will do the same, but we will use a custom datatype for our intermediate computation.
#
# We use the same input variables ``x`` and ``y`` as above, but before adding ``x + y``, we first cast both ``x`` and ``y`` to a custom datatype via the ``relay.cast(...)`` call.
#
# Note how we specify the custom datatype: we indicate it using the special ` custom[...]`  syntax.
# Additionally, note the "16" after the datatype: this is the bitwidth of the custom datatype. This tells TVM that each instance of ``posit`` is 16 bits wide.

try:
    x_posit = relay.cast(x, dtype='custom[posit]16')
    y_posit = relay.cast(y, dtype='custom[posit]16')
    z_posit = x_posit + y_posit
    z = relay.cast(z_posit, dtype='float32')
except tvm.TVMError as e:
    # Print last line of error
    print(str(e).split('\n')[-1])

######################################################################
# Trying to generate this program throws an error from TVM.
# TVM does not know how to handle any custom datatype out of the box!
# We first have to register the custom type with TVM, giving it a name and a type code:

tvm.target.datatype.register('posit', 150)

######################################################################
# Note that the type code, 150, is currently chosen manually by the user.
# See ``TVMTypeCode::kCustomBegin`` in `include/tvm/runtime/c_runtime_api.h <https://github.com/apache/incubator-tvm/blob/master/include/tvm/runtime/data_type.h>`_.
# Now we can generate our program again:

x_posit = relay.cast(x, dtype='custom[posit]16')
y_posit = relay.cast(y, dtype='custom[posit]16')
z_posit = x_posit + y_posit
z = relay.cast(z_posit, dtype='float32')
program = relay.Function([x, y], z)
module = tvm.IRModule.from_expr(program)

######################################################################
# Now we have a Relay program that uses posits!
print(program)

######################################################################
# Now that we can express our program without errors, let's try running it!
try:
    ex = relay.create_executor(mod=module)
    z_output_posit = ex.evaluate()(x_input, y_input)
    print("z: {}".format(z_output_posit))
except tvm.TVMError as e:
    # Print last line of error
    print(str(e).split('\n')[-1])

######################################################################
# Now, trying to compile this program throws an error.
# Let's dissect this error.
#
# The error is occurring during the process of lowering the custom datatype code to code that TVM can compile and run.
# TVM is telling us that it cannot find a *lowering function* for the ``Cast`` operation, when casting from source type 2 (``float``, in TVM), to destination type 150 (our custom datatype).
# When lowering custom datatypes, if TVM encounters an operation over a custom datatype, it looks for a user-registered **lowering function**, which tells it how to lower the operation to an operation over datatypes it understands.
# We have not told TVM how to lower ``Cast`` operations for our custom datatypes; thus, the source of this error.
#
# To fix this error, we simply need to specify a lowering function:

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func("posit16_fromf"), "Cast", "llvm",
    "float", "posit")

######################################################################
# The ``register_op(...)`` call takes a lowering function, and a number of parameters which specify exactly the operation which should be lowered with the provided lowering function.
# In this case, the arguments we pass specify that this lowering function is for lowering a `Cast` from `float` to `posit` for target `"llvm"`.
#
# The lowering function passed into this call is very general: it should take an operation of the specified type (in this case, `Cast`) and return another operation which only uses datatypes which TVM understands.
#
# In the general case, we expect users to implement operations over their custom datatypes using calls to an external library.
# In our example, our `posit` library implements a `Cast` from `float` to 16-bit `posit` in the function `posit16_fromf`.
# To provide for the general case, we have made a helper function, `create_lower_func(...)`, which does just this: given a function name, it replaces the given operation with a `Call` to the function name provided.
# It additionally removes usages of the custom datatype by storing the custom datatype in an opaque `uint` of the appropriate width; in our case, a `uint16_t`.

# We can now re-try running the program:
try:
    ex = relay.create_executor(mod=module)
    z_output_posit = ex.evaluate()(x_input, y_input)
    print("z: {}".format(z_output_posit))
except tvm.TVMError as e:
    # Print last line of error
    print(str(e).split('\n')[-1])

######################################################################
# This new error tells us that the `Add` lowering function is not found, which is good news, as it's no longer complaining about the `Cast`!
# We know what to do from here: we just need to register the lowering functions for the other operations in our program.

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func("posit16_add"), "Add", "llvm",
    "posit")
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func("posit16_tof"), "Cast", "llvm",
    "posit", "float")

# Now, we can run our program without errors.
compiled = ex.evaluate(program)
z_output_posit = compiled(x_input, y_input)
print("z: {}".format(z_output_posit))

print("x:\t\t{}".format(x_input))
print("y:\t\t{}".format(y_input))
print("z (float32):\t{}".format(z_output))
print("z (posit16):\t{}".format(z_output_posit))

# Perhaps as expected, the `posit16` results are very close to the `float32` results, but with some loss in precision!

######################################################################
# Running Models With Custom Datatypes
# ------------------------------------
#
# We will first choose the model which we would like to run with posits. In this case we use [Mobilenet](https://arxiv.org/abs/1704.04861). We choose Mobilenet due to its small size. In this alpha state of the Bring Your Own Datatypes framework, we have not implemented any software optimizations for running software emulations of custom datatypes; the result is poor performance due to many calls into our datatype emulation library.
#
# Relay has packaged up many models within its [python/tvm/relay/testing](https://github.com/dmlc/tvm/tree/master/python/tvm/relay/testing) directory. We will go ahead and grab Mobilenet:

from tvm.relay.testing.mobilenet import get_workload as get_mobilenet

module, params = get_mobilenet(image_shape=(3, 32, 32), num_classes=10)

######################################################################
# It's easy to execute MobileNet with native TVM:

ex = tvm.relay.create_executor("graph", mod=module)
input = np.random.rand(1, 3, 32, 32).astype("float32")
result = ex.evaluate()(input, **params)
print(result)

######################################################################
# Now, we would like to change the model to use posits internally. To do so, we need to convert the network. To do this, we first define a function which will help us convert tensors:


def convert_ndarray(dst_dtype, array):
    """Converts an NDArray into the specified datatype"""
    ex = relay.create_executor('graph')
    x = relay.var('x', shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    return ex.evaluate(cast)(array)


######################################################################
# Now, to actually convert the entire network, we have written [a pass in Relay](https://github.com/gussmith23/tvm/blob/ea174c01c54a2529e19ca71e125f5884e728da6e/python/tvm/relay/frontend/change_datatype.py#L21) which simply converts all nodes within the model to use the new datatype.

from tvm.relay.frontend.change_datatype import ChangeDatatype

src_dtype = 'float32'
dst_dtype = 'custom[posit]16'

# Currently, custom datatypes only work if you run simplify_inference beforehand
module = tvm.relay.transform.SimplifyInference()(module)

# Run type inference before changing datatype
module = tvm.relay.transform.InferType()(module)

# Change datatype from float to posit and re-infer types
cdtype = ChangeDatatype(src_dtype, dst_dtype)
expr = cdtype.visit(module['main'])
module = tvm.relay.transform.InferType()(tvm.relay.Module.from_expr(expr))

# We also convert the parameters:
params = dict((p, convert_ndarray(dst_dtype, params[p])) for p in params)

# We also need to convert our input:
input = convert_ndarray(dst_dtype, input)

# Finally, we can try to run the converted model:
try:
    # Vectorization is not implemented with custom datatypes.
    with tvm.build_config(disable_vectorize=True):
        result_posit = ex.evaluate(expr)(input, **params)
except tvm.TVMError as e:
    print(str(e).split('\n')[-1])

######################################################################
# When we attempt to run the model, we get a familiar error telling us that more funcions need to be registerd for posits.
#
# Because this is a neural network, many more operations are required.
# Here, we register all the needed functions:

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func("posit16_fromf"), "FloatImm", "llvm",
    "posit")
tvm.target.datatype.register_op(tvm.target.datatype.lower_ite,
                                "Call",
                                "llvm",
                                "posit",
                                intrinsic_name="tvm_if_then_else")
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func("posit16_mul"), "Mul", "llvm",
    "posit")
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func("posit16_div"), "Div", "llvm",
    "posit")
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func("posit16_sqrt"),
    "Call",
    "llvm",
    "posit",
    intrinsic_name="sqrt")
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func("posit16_sub"), "Sub", "llvm",
    "posit")
# //TODO(hypercubestart): add notes on this
tvm.target.datatype.register_min_func(lambda _: -268435456, "posit")
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func("posit16_exp"),
    "Call",
    "llvm",
    "posit",
    intrinsic_name="exp")
# tvm.target.datatype.register_op(tvm.target.datatype.create_lower_func("posit16_max"),
#                         "Max", "llvm", "posit")
# //TODO(hypercubestart): after we figure out third-party library, need to add this back in

######################################################################
# Note that, to implement the `Max` operator, we needed to rewrite our wrapper library with a new function, `posit16_max`.
# This is the only function we needed to implement by hand! All other functions we took straight from the posit library.
#
#
# Now we can finally run the model:

# Vectorization is not implemented with custom datatypes.
with tvm.build_config(disable_vectorize=True):
    result_posit = ex.evaluate(expr)(input, **params)
    result_posit = convert_ndarray(src_dtype, result_posit)
    print(result_posit)

# Again, note that the output using 16-bit posits is understandably different from that of 32-bit floats,
# but is still within a sane distance:
np.testing.assert_allclose(result.asnumpy(),
                           result_posit.asnumpy(),
                           rtol=1e-6,
                           atol=1e-5)
