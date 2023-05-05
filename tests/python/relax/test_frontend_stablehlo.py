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

# pylint: disable=c-extension-no-member

import functools
from typing import Union, Tuple, List
import pytest
import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.frontend.stablehlo import from_stablehlo


def generate_np_inputs(
    input_shapes: Union[Tuple, List[Tuple]], dtype: str = "float32"
) -> Union[np.ndarray, List[np.ndarray]]:
    """Generate numpy data as the inputs of model

    Parameters
    ----------
    input_shapes: Union[Tuple, List[Tuple]]
        shapes for inputs
    dtype: str
        the data type of inputs

    Results
    -------
    out: List[np.ndarray]
        numpy input data
    """
    if not isinstance(input_shapes[0], (list, tuple)):
        return [np.random.uniform(size=input_shapes).astype(dtype)]
    out = []
    for input_shape in input_shapes:
        out.append(np.random.uniform(size=input_shape).astype(dtype))
    return out


def np2jnp(inputs_np: Union[np.ndarray, List[np.ndarray]]):
    """Convert data from numpy to jax.numpy

    Parameters
    ----------
    inputs_np: Union[np.ndarray, List[np.ndarray]]
        numpy input data

    Results
    -------
    out: Union[jnp.ndarray, List[jnp.ndarray]]
        jax numpy data
    """
    import jax.numpy as jnp

    # Use jnp.asarray to avoid unnecessary memory copies
    inputs_jnp = []
    if isinstance(inputs_np, (tuple, list)):
        for input_np in inputs_np:
            inputs_jnp.append(jnp.asarray(input_np))
        return inputs_jnp
    return jnp.asarray(inputs_np)


def check_correctness(
    jax_jit_mod,
    input_shapes: Union[Tuple, List[Tuple]],
    dtype: str = "float32",
) -> None:
    """Run a jax model and the translated TVM IRModule,
       verify the inference accuracy.

    Parameters
    ----------
    jax_jit_mod: jaxlib.xla_extension.CompiledFunction
        The input jax jitted model
    input_shapes: Union[Tuple, List[Tuple]]
        shapes for inputs
    dtype: str
        the data type of inputs
    """
    # Generate numpy inputs
    inputs_np = generate_np_inputs(input_shapes, dtype)
    # Get the jax numpy data
    inputs_jnp = np2jnp(inputs_np)

    # lower the jitted function to StableHLO
    lowered = jax_jit_mod.lower(*inputs_np)

    # lowered.as_text(dialect="stablehlo") generates text format
    # compiler_ir generates the related jaxlib.mlir.Module
    stablehlo_module = lowered.compiler_ir(dialect="stablehlo")

    # Convert the StableHLO IR to Relax
    ir_mod = from_stablehlo(stablehlo_module)

    # Run the jax jitted model with the input jax numpy data
    jax_output = jax_jit_mod(*inputs_jnp)

    # Legalize the Relax Operators into TensorIR
    # TODO (relax-team): add LegalizeOps in default seq in vm_build
    ir_mod = relax.transform.LegalizeOps()(ir_mod)
    # TODO (yongwww): support multiple targets,
    # "llvm" should be good for this check
    target = tvm.target.Target("llvm", host="llvm")
    # Compile and run
    ex = relax.build(ir_mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    vm.set_input("main", *inputs_np)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")

    # Single ouput
    if isinstance(tvm_output, tvm.nd.NDArray):
        tvm.testing.assert_allclose(tvm_output.numpy(), jax_output, rtol=1e-5, atol=1e-5)
        return

    # Multiple ouputs
    assert len(tvm_output) == len(jax_output), "numbers of outputs mismatch"
    for (tvm_out, jax_out) in zip(tvm_output, jax_output):
        tvm.testing.assert_allclose(tvm_out.numpy(), jax_out, rtol=1e-5, atol=1e-5)


def get_vm_res(
    ir_mod: tvm.IRModule, weights: Union[np.ndarray, List[np.ndarray]]
) -> Union[tvm.nd.NDArray, List[tvm.nd.NDArray]]:
    """Compile and run an ir_module on Relax VM

    Parameters
    ----------
    ir_mod: tvm.IRModule
        input ir module

    weights: Union[np.ndarray, List[np.ndarray]]
         input weights

    Results
    -------
    out: Union[tvm.nd.NDArray, List[tvm.nd.NDArray]]
        inference result
    """
    ir_mod = relax.transform.LegalizeOps()(ir_mod)
    target = tvm.target.Target("llvm", host="llvm")
    # Compile and run
    ex = relax.build(ir_mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    vm.set_input("main", *weights)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    return tvm_output


@tvm.testing.requires_gpu
def test_add_dynamic():
    add_dyn = """
    func.func @test(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
      %1 = stablehlo.add %arg0, %arg1 : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
      func.return %1 : tensor<?x?xf32>
    }
    """

    mod = from_stablehlo(add_dyn)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            arg0: R.Tensor(("n_0", "n_1"), dtype="float32"),
            arg1: R.Tensor(("n_2", "n_3"), dtype="float32"),
        ) -> R.Tensor(dtype="float32", ndim=2):
            n_0 = T.int64()
            n_1 = T.int64()
            n_2 = T.int64()
            n_3 = T.int64()
            with R.dataflow():
                lv: R.Tensor(dtype="float32", ndim=2) = R.add(arg0, arg1)
                gv: R.Tensor(dtype="float32", ndim=2) = lv
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(mod, Expected)


@tvm.testing.requires_gpu
def test_unary():
    import jax

    def _rsqrt(x):
        return jax.lax.rsqrt(x)

    def _sqrt(x):
        return jax.lax.sqrt(x)

    def _sin(x):
        return jax.lax.sin(x)

    def _sinh(x):
        return jax.lax.sinh(x)

    def _cos(x):
        return jax.lax.cos(x)

    def _cosh(x):
        return jax.lax.cos(x)

    def _exp(x):
        return jax.lax.exp(x)

    def _round(x):
        return jax.lax.round(x)

    input_shapes = (2, 3, 4)
    for fn in [_rsqrt, _sqrt, _sin, _cos, _cosh, _exp, _round]:
        check_correctness(jax.jit(fn), input_shapes)


@tvm.testing.requires_gpu
def test_binary():
    import jax

    def fn(x, y):
        r1 = x + y
        r2 = r1 * r1
        r3 = r2 / r1
        r = r2 - r3
        return r

    input_shape = (1, 2, 3)
    input_shapes = (input_shape, input_shape)

    # jit the function
    jit_fn = jax.jit(fn)

    # verify inference accuracy
    check_correctness(jit_fn, input_shapes)


@tvm.testing.requires_gpu
def test_const():
    import jax

    def fn(x):
        return x + 1

    check_correctness(jax.jit(fn), (2,))


@tvm.testing.requires_gpu
def test_maximum():
    import jax
    import jax.numpy as jnp

    def fn(x, y):
        return jnp.maximum(x, y)

    check_correctness(jax.jit(fn), ((2, 3), (2, 3)))


@tvm.testing.requires_gpu
def test_minimum():
    import jax
    import jax.numpy as jnp

    def fn(x, y):
        return jnp.minimum(x, y)

    check_correctness(jax.jit(fn), ((2, 3), (2, 3)))


@tvm.testing.requires_gpu
def test_reduce():
    import jax
    import jax.numpy as jnp

    def fn(x):
        return jnp.mean(x, axis=(1, 2))

    check_correctness(jax.jit(fn), (2, 3, 4, 5))


@tvm.testing.requires_gpu
def test_reduce_window():
    import jax
    from flax import linen as nn

    def fn(x):
        return nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

    check_correctness(jax.jit(fn), (2, 3, 4))


@tvm.testing.requires_gpu
def test_dot_general():
    import jax

    def fn(x, y):
        return jax.lax.dot_general(x, y, (([1], [0]), ([], [])))

    input_shapes = ((1, 512), (512, 2))
    check_correctness(jax.jit(fn), input_shapes)


@tvm.testing.requires_gpu
def test_conv():
    import jax
    from flax import linen as nn
    import jax.random as jrandom

    conv = nn.Conv(64, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init")
    input_shape = (7, 7, 5, 64)

    input_np = generate_np_inputs(input_shape)[0]
    input_jnp = np2jnp(input_np)
    # initialize the conv
    weights = conv.init(jrandom.PRNGKey(0), input_jnp)
    # get jax inference output
    jax_output = conv.apply(weights, input_jnp)

    # assemble numpy data using weights generated above
    kernel_np = np.asarray(weights["params"]["kernel"])
    bias_np = np.asarray(weights["params"]["bias"])
    inputs_np = [bias_np, kernel_np, input_np]

    # jit and lower to StableHLO
    apply = functools.partial(conv.apply)
    stablehlo_module = jax.jit(apply).lower(weights, input_jnp).compiler_ir(dialect="stablehlo")

    # convert in Relax
    ir_mod = from_stablehlo(stablehlo_module)
    # compile and run
    tvm_output = get_vm_res(ir_mod, inputs_np)
    # verify accuracy
    tvm.testing.assert_allclose(tvm_output.numpy(), jax_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
