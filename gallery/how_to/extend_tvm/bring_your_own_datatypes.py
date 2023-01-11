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

In this section, we will use an example library we have already implemented, located at ``3rdparty/byodt/myfloat.cc``.
This datatype, which we dubbed "myfloat", is really just a IEE-754 float under-the-hood, but it serves a useful example
to show that any datatype can be used in the BYODT framework.

Setup
-----

Since we do not use any 3rdparty library, there is no setup needed.

If you would like to try this with your own datatype library, first bring the library's functions into the process space with ``CDLL``:

.. code-block:: python

    ctypes.CDLL('my-datatype-lib.so', ctypes.RTLD_GLOBAL)
"""


######################
# A Simple TVM Program
# --------------------
#
# We'll begin by writing a simple program in TVM; afterwards, we will re-write it to use custom datatypes.
import tvm
from tvm import relay

# Our basic program: Z = X + Y
x = relay.var("x", shape=(3,), dtype="float32")
y = relay.var("y", shape=(3,), dtype="float32")
z = x + y
program = relay.Function([x, y], z)
module = tvm.IRModule.from_expr(program)

######################################################################
# Now, we create random inputs to feed into this program using numpy:

import numpy as np

np.random.seed(23)  # for reproducibility

x_input = np.random.rand(3).astype("float32")
y_input = np.random.rand(3).astype("float32")
print("x: {}".format(x_input))
print("y: {}".format(y_input))

######################################################################
# Finally, we're ready to run the program:

z_output = relay.create_executor(mod=module).evaluate()(x_input, y_input)
print("z: {}".format(z_output))

######################################################################
# Adding Custom Datatypes
# -----------------------
# Now, we will do the same, but we will use a custom datatype for our intermediate computation.
#
# We use the same input variables ``x`` and ``y`` as above, but before adding ``x + y``, we first cast both ``x`` and ``y`` to a custom datatype via the ``relay.cast(...)`` call.
#
# Note how we specify the custom datatype: we indicate it using the special ``custom[...]`` syntax.
# Additionally, note the "32" after the datatype: this is the bitwidth of the custom datatype. This tells TVM that each instance of ``myfloat`` is 32 bits wide.

try:
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        x_myfloat = relay.cast(x, dtype="custom[myfloat]32")
        y_myfloat = relay.cast(y, dtype="custom[myfloat]32")
        z_myfloat = x_myfloat + y_myfloat
        z = relay.cast(z_myfloat, dtype="float32")
except tvm.TVMError as e:
    # Print last line of error
    print(str(e).split("\n")[-1])

######################################################################
# Trying to generate this program throws an error from TVM.
# TVM does not know how to handle any custom datatype out of the box!
# We first have to register the custom type with TVM, giving it a name and a type code:

tvm.target.datatype.register("myfloat", 150)

######################################################################
# Note that the type code, 150, is currently chosen manually by the user.
# See ``TVMTypeCode::kCustomBegin`` in `include/tvm/runtime/c_runtime_api.h <https://github.com/apache/tvm/blob/main/include/tvm/runtime/data_type.h>`_.
# Now we can generate our program again:

x_myfloat = relay.cast(x, dtype="custom[myfloat]32")
y_myfloat = relay.cast(y, dtype="custom[myfloat]32")
z_myfloat = x_myfloat + y_myfloat
z = relay.cast(z_myfloat, dtype="float32")
program = relay.Function([x, y], z)
module = tvm.IRModule.from_expr(program)
module = relay.transform.InferType()(module)

######################################################################
# Now we have a Relay program that uses myfloat!
print(program)

######################################################################
# Now that we can express our program without errors, let's try running it!
try:
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        z_output_myfloat = relay.create_executor("graph", mod=module).evaluate()(x_input, y_input)
        print("z: {}".format(y_myfloat))
except tvm.TVMError as e:
    # Print last line of error
    print(str(e).split("\n")[-1])

######################################################################
# Now, trying to compile this program throws an error.
# Let's dissect this error.
#
# The error is occurring during the process of lowering the custom datatype code to code that TVM can compile and run.
# TVM is telling us that it cannot find a *lowering function* for the ``Cast`` operation, when casting from source type 2 (``float``, in TVM), to destination type 150 (our custom datatype).
# When lowering custom datatypes, if TVM encounters an operation over a custom datatype, it looks for a user-registered *lowering function*, which tells it how to lower the operation to an operation over datatypes it understands.
# We have not told TVM how to lower ``Cast`` operations for our custom datatypes; thus, the source of this error.
#
# To fix this error, we simply need to specify a lowering function:

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func(
        {
            (32, 32): "FloatToCustom32",  # cast from float32 to myfloat32
        }
    ),
    "Cast",
    "llvm",
    "float",
    "myfloat",
)

######################################################################
# The ``register_op(...)`` call takes a lowering function, and a number of parameters which specify exactly the operation which should be lowered with the provided lowering function.
# In this case, the arguments we pass specify that this lowering function is for lowering a ``Cast`` from ``float`` to ``myfloat`` for target ``"llvm"``.
#
# The lowering function passed into this call is very general: it should take an operation of the specified type (in this case, `Cast`) and return another operation which only uses datatypes which TVM understands.
#
# In the general case, we expect users to implement operations over their custom datatypes using calls to an external library.
# In our example, our ``myfloat`` library implements a ``Cast`` from ``float`` to 32-bit ``myfloat`` in the function ``FloatToCustom32``.
# To provide for the general case, we have made a helper function, ``create_lower_func(...)``,
# which does just this: given a dictionary, it replaces the given operation with a ``Call`` to the appropriate function name provided based on the op and the bit widths.
# It additionally removes usages of the custom datatype by storing the custom datatype in an opaque ``uint`` of the appropriate width; in our case, a ``uint32_t``.
# For more information, see `the source code <https://github.com/apache/tvm/blob/main/python/tvm/target/datatype.py>`_.

# We can now re-try running the program:
try:
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        z_output_myfloat = relay.create_executor("graph", mod=module).evaluate()(x_input, y_input)
        print("z: {}".format(z_output_myfloat))
except tvm.TVMError as e:
    # Print last line of error
    print(str(e).split("\n")[-1])

######################################################################
# This new error tells us that the ``Add`` lowering function is not found, which is good news, as it's no longer complaining about the ``Cast``!
# We know what to do from here: we just need to register the lowering functions for the other operations in our program.
#
# Note that for ``Add``, ``create_lower_func`` takes in a dict where the key is an integer.
# For ``Cast`` operations, we require a 2-tuple to specify the ``src_bit_length`` and the ``dest_bit_length``,
# while for all other operations, the bit length is the same between the operands so we only require one integer to specify ``bit_length``.
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Add"}),
    "Add",
    "llvm",
    "myfloat",
)
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({(32, 32): "Custom32ToFloat"}),
    "Cast",
    "llvm",
    "myfloat",
    "float",
)

# Now, we can run our program without errors.
with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
    z_output_myfloat = relay.create_executor(mod=module).evaluate()(x_input, y_input)
print("z: {}".format(z_output_myfloat))

print("x:\t\t{}".format(x_input))
print("y:\t\t{}".format(y_input))
print("z (float32):\t{}".format(z_output))
print("z (myfloat32):\t{}".format(z_output_myfloat))

# Perhaps as expected, the ``myfloat32`` results and ``float32`` are exactly the same!

######################################################################
# Running Models With Custom Datatypes
# ------------------------------------
#
# We will first choose the model which we would like to run with myfloat.
# In this case we use `Mobilenet <https://arxiv.org/abs/1704.04861>`_.
# We choose Mobilenet due to its small size.
# In this alpha state of the Bring Your Own Datatypes framework, we have not implemented any software optimizations for running software emulations of custom datatypes; the result is poor performance due to many calls into our datatype emulation library.
#
# First let us define two helper functions to get the mobilenet model and a cat image.


def get_mobilenet():
    dshape = (1, 3, 224, 224)
    from mxnet.gluon.model_zoo.vision import get_model

    block = get_model("mobilenet0.25", pretrained=True)
    shape_dict = {"data": dshape}
    return relay.frontend.from_mxnet(block, shape_dict)


def get_cat_image():
    from tvm.contrib.download import download_testdata
    from PIL import Image

    url = "https://gist.githubusercontent.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png"
    dst = "cat.png"
    real_dst = download_testdata(url, dst, module="data")
    img = Image.open(real_dst).resize((224, 224))
    # CoreML's standard model image format is BGR
    img_bgr = np.array(img)[:, :, ::-1]
    img = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :]
    return np.asarray(img, dtype="float32")


module, params = get_mobilenet()

######################################################################
# It's easy to execute MobileNet with native TVM:

ex = tvm.relay.create_executor("graph", mod=module, params=params)
input = get_cat_image()
result = ex.evaluate()(input).numpy()
# print first 10 elements
print(result.flatten()[:10])

######################################################################
# Now, we would like to change the model to use myfloat internally. To do so, we need to convert the network. To do this, we first define a function which will help us convert tensors:


def convert_ndarray(dst_dtype, array):
    """Converts an NDArray into the specified datatype"""
    x = relay.var("x", shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        return relay.create_executor("graph").evaluate(cast)(array)


######################################################################
# Now, to actually convert the entire network, we have written `a pass in Relay <https://github.com/gussmith23/tvm/blob/ea174c01c54a2529e19ca71e125f5884e728da6e/python/tvm/relay/frontend/change_datatype.py#L21>`_ which simply converts all nodes within the model to use the new datatype.

from tvm.relay.frontend.change_datatype import ChangeDatatype

src_dtype = "float32"
dst_dtype = "custom[myfloat]32"

module = relay.transform.InferType()(module)

# Currently, custom datatypes only work if you run simplify_inference beforehand
module = tvm.relay.transform.SimplifyInference()(module)

# Run type inference before changing datatype
module = tvm.relay.transform.InferType()(module)

# Change datatype from float to myfloat and re-infer types
cdtype = ChangeDatatype(src_dtype, dst_dtype)
expr = cdtype.visit(module["main"])
module = tvm.relay.transform.InferType()(module)

# We also convert the parameters:
params = {k: convert_ndarray(dst_dtype, v) for k, v in params.items()}

# We also need to convert our input:
input = convert_ndarray(dst_dtype, input)

# Finally, we can try to run the converted model:
try:
    # Vectorization is not implemented with custom datatypes.
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        result_myfloat = tvm.relay.create_executor("graph", mod=module).evaluate(expr)(
            input, **params
        )
except tvm.TVMError as e:
    print(str(e).split("\n")[-1])

######################################################################
# When we attempt to run the model, we get a familiar error telling us that more functions need to be registered for myfloat.
#
# Because this is a neural network, many more operations are required.
# Here, we register all the needed functions:

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "FloatToCustom32"}),
    "FloatImm",
    "llvm",
    "myfloat",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.lower_ite, "Call", "llvm", "myfloat", intrinsic_name="tir.if_then_else"
)

tvm.target.datatype.register_op(
    tvm.target.datatype.lower_call_pure_extern,
    "Call",
    "llvm",
    "myfloat",
    intrinsic_name="tir.call_pure_extern",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Mul"}),
    "Mul",
    "llvm",
    "myfloat",
)
tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Div"}),
    "Div",
    "llvm",
    "myfloat",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Sqrt"}),
    "Call",
    "llvm",
    "myfloat",
    intrinsic_name="tir.sqrt",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Sub"}),
    "Sub",
    "llvm",
    "myfloat",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Exp"}),
    "Call",
    "llvm",
    "myfloat",
    intrinsic_name="tir.exp",
)

tvm.target.datatype.register_op(
    tvm.target.datatype.create_lower_func({32: "Custom32Max"}),
    "Max",
    "llvm",
    "myfloat",
)

tvm.target.datatype.register_min_func(
    tvm.target.datatype.create_min_lower_func({32: "MinCustom32"}, "myfloat"),
    "myfloat",
)

######################################################################
# Note we are making use of two new functions: ``register_min_func`` and ``create_min_lower_func``.
#
# ``register_min_func`` takes in an integer ``num_bits`` for the bit length, and should return an operation
# representing the minimum finite representable value for the custom data type with the specified bit length.
#
# Similar to ``register_op`` and ``create_lower_func``, the ``create_min_lower_func`` handles the general case
# where the minimum representable custom datatype value is implemented using calls to an external library.
#
# Now we can finally run the model:

# Vectorization is not implemented with custom datatypes.
with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
    result_myfloat = relay.create_executor(mod=module).evaluate(expr)(input, **params)
    result_myfloat = convert_ndarray(src_dtype, result_myfloat).numpy()
    # print first 10 elements
    print(result_myfloat.flatten()[:10])

# Again, note that the output using 32-bit myfloat exactly the same as 32-bit floats,
# because myfloat is exactly a float!
np.testing.assert_array_equal(result, result_myfloat)
