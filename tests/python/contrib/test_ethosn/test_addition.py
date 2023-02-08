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

"""Arm(R) Ethos(TM)-N integration addition tests"""

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_model(
    lhs_shape,
    rhs_shape,
    lhs_zp,
    lhs_sc,
    rhs_zp,
    rhs_sc,
    out_zp,
    out_sc,
    dtype,
    lhs_is_constant=False,
    rhs_is_constant=False,
    constant_data=None,
):
    """Return a model and any parameters it may have"""

    def create_or_assign_constant(shape, dtype, default_data):
        """Creates new numpy array or assigns default_data if available."""

        iinfo = np.iinfo(dtype)
        data_min = iinfo.min
        data_max = iinfo.max

        nparray = None
        if default_data:
            nparray = np.array(default_data, dtype=dtype).reshape(shape)
        else:
            nparray = np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype)

        return relay.const(nparray, dtype=dtype)

    if lhs_is_constant:
        a = create_or_assign_constant(lhs_shape, dtype, constant_data)
    else:
        a = relay.var("a", shape=lhs_shape, dtype=dtype)

    if rhs_is_constant:
        b = create_or_assign_constant(rhs_shape, dtype, constant_data)
    else:
        b = relay.var("b", shape=rhs_shape, dtype=dtype)

    model = relay.qnn.op.add(
        lhs=a,
        rhs=b,
        lhs_scale=relay.const(lhs_sc, "float32"),
        lhs_zero_point=relay.const(lhs_zp, "int32"),
        rhs_scale=relay.const(rhs_sc, "float32"),
        rhs_zero_point=relay.const(rhs_zp, "int32"),
        output_scale=relay.const(out_sc, "float32"),
        output_zero_point=relay.const(out_zp, "int32"),
    )
    return model


def _get_addition_qnn_params(dtype):
    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    lhs_zp = np.random.randint(data_min, data_max)
    lhs_sc = np.random.random() * 2
    rhs_zp = np.random.randint(data_min, data_max)
    rhs_sc = np.random.random() * 2

    input1_max = lhs_sc * (255 - lhs_zp)
    input1_min = -lhs_sc * lhs_zp
    input2_max = rhs_sc * (255 - rhs_zp)
    input2_min = -rhs_sc * rhs_zp
    output_max = input1_max + input2_max
    output_min = input1_min + input2_min
    output_sc = (output_max - output_min) / 255
    output_zp = -int(output_min / output_sc)
    return lhs_zp, lhs_sc, rhs_zp, rhs_sc, output_zp, output_sc


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize("shape", [(1, 22, 9, 9), (1, 27, 21, 16)])
def test_addition(dtype, shape):
    """Compare Addition output with TVM."""
    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc = _get_addition_qnn_params(dtype)

    outputs = []
    inputs = {
        "a": tvm.nd.array(np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype)),
        "b": tvm.nd.array(np.random.randint(data_min, data_max + 1, size=shape, dtype=dtype)),
    }
    model = _get_model(shape, shape, lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc, dtype)
    for npu in [False, True]:
        mod = tei.make_module(model, [])
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1,
                {},
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "lhs_shape,lhs_is_constant,rhs_shape,rhs_is_constant",
    [
        ((1, 4, 4, 8), True, (1, 1, 1, 8), True),
        ((4,), True, (1, 16, 12, 4), True),
        ((1, 1, 1, 8), True, (1, 4, 4, 8), True),
        ((1, 16, 12, 4), True, (4,), True),
    ],
)
def test_addition_both_inputs_constants(
    dtype, lhs_shape, lhs_is_constant, rhs_shape, rhs_is_constant
):
    """Check if addition is simplified when both inputs are constants."""
    np.random.seed(0)

    lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc = _get_addition_qnn_params(dtype)

    model = _get_model(
        lhs_shape,
        rhs_shape,
        lhs_zp,
        lhs_sc,
        rhs_zp,
        rhs_sc,
        out_zp,
        out_sc,
        dtype,
        lhs_is_constant=lhs_is_constant,
        rhs_is_constant=rhs_is_constant,
    )
    from tvm.relay.op.contrib import partition_for_ethosn  # pylint: disable=import-outside-toplevel

    mod = tei.make_module(model, {})
    assert "qnn.add" in mod.astext(False)
    mod = partition_for_ethosn(mod, {})
    assert "qnn.add" not in mod.astext(False)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "lhs_shape,lhs_is_constant,rhs_shape,rhs_is_constant",
    [
        ((1, 4, 4, 8), False, (1, 4, 4, 8), True),
        ((1, 16, 12, 4), True, (1, 16, 12, 4), False),
    ],
)
def test_addition_with_one_constant(dtype, lhs_shape, lhs_is_constant, rhs_shape, rhs_is_constant):
    """Validate addition with one input as a constant."""
    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc = _get_addition_qnn_params(dtype)

    model = _get_model(
        lhs_shape,
        rhs_shape,
        lhs_zp,
        lhs_sc,
        rhs_zp,
        rhs_sc,
        out_zp,
        out_sc,
        dtype,
        lhs_is_constant=lhs_is_constant,
        rhs_is_constant=rhs_is_constant,
    )
    input_shape = rhs_shape if lhs_is_constant else lhs_shape
    input_name = "b" if lhs_is_constant else "a"
    inputs = {
        input_name: tvm.nd.array(
            np.random.randint(data_min, data_max + 1, size=input_shape, dtype=dtype)
        )
    }

    outputs = []
    for npu in [False, True]:
        mod = tei.make_module(model, {})
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1,
                {},
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )
    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "lhs_shape,lhs_is_constant,rhs_shape,rhs_is_constant",
    [
        ((1, 4, 4, 8), False, (1, 1, 1, 8), True),
        ((4,), True, (1, 16, 12, 4), False),
        ((1, 1, 1, 8), True, (1, 4, 4, 8), False),
        ((1, 16, 12, 4), False, (4,), True),
    ],
)
def test_addition_to_depthwise(dtype, lhs_shape, lhs_is_constant, rhs_shape, rhs_is_constant):
    """Compare addition to depthwise with TVM."""
    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc = _get_addition_qnn_params(dtype)

    model = _get_model(
        lhs_shape,
        rhs_shape,
        lhs_zp,
        lhs_sc,
        rhs_zp,
        rhs_sc,
        out_zp,
        out_sc,
        dtype,
        lhs_is_constant=lhs_is_constant,
        rhs_is_constant=rhs_is_constant,
    )
    input_shape = rhs_shape if lhs_is_constant else lhs_shape
    input_name = "b" if lhs_is_constant else "a"
    inputs = {
        input_name: tvm.nd.array(
            np.random.randint(data_min, data_max + 1, size=input_shape, dtype=dtype)
        )
    }
    outputs = []
    for npu in [False, True]:
        mod = tei.make_module(model, {})
        outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))
    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "lhs_shape,lhs_is_constant,rhs_shape,rhs_is_constant",
    [
        ((1, 2, 8, 4), False, None, True),
        ((1, 5, 6, 7), False, (1, 1, 1, 1), True),
        (None, True, (1, 2, 8, 4), False),
        ((1, 1, 1, 1), True, (1, 5, 6, 7), False),
    ],
)
def test_addition_to_reinterpret_quantize(lhs_shape, lhs_is_constant, rhs_shape, rhs_is_constant):
    """Compare addition to depthwise with TVM."""
    np.random.seed(0)

    dtype = "uint8"
    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max

    # Add can only be offloaded as a reinterpret quantize operation if
    # it is an identity operation. We must choose the quantization and
    # constant data carefully to maske sure that this is the case.
    if lhs_is_constant:
        rhs_zp = 128
        rhs_sc = 0.0078125
        lhs_zp = 0
        lhs_sc = 0.003921568859368563
    else:
        lhs_zp = 128
        lhs_sc = 0.0078125
        rhs_zp = 0
        rhs_sc = 0.003921568859368563
    out_zp = 0
    out_sc = 0.007814894430339336
    constant_data = 255

    model = _get_model(
        lhs_shape,
        rhs_shape,
        lhs_zp,
        lhs_sc,
        rhs_zp,
        rhs_sc,
        out_zp,
        out_sc,
        dtype,
        lhs_is_constant=lhs_is_constant,
        rhs_is_constant=rhs_is_constant,
        constant_data=constant_data,
    )
    input_shape = rhs_shape if lhs_is_constant else lhs_shape
    input_name = "b" if lhs_is_constant else "a"
    inputs = {
        input_name: tvm.nd.array(
            np.random.randint(data_min, data_max + 1, size=input_shape, dtype=dtype)
        )
    }
    outputs = []
    for npu in [False, True]:
        mod = tei.make_module(model, {})
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1,
                {},
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )
    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "dtype,shape,err_msg",
    [
        (
            "uint8",
            (2, 4, 4, 4),
            "batch size=2, batch size must = 1; batch size=2, batch size must = 1",
        ),
        (
            "int16",
            (1, 4, 4, 4),
            "dtype='int16', dtype must be either uint8, int8 or int32; dtype='int16', "
            "dtype must be either uint8, int8 or int32",
        ),
    ],
)
def test_addition_failure(dtype, shape, err_msg):
    """Check addition error messages."""
    np.random.seed(0)

    lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc = _get_addition_qnn_params(dtype)

    model = _get_model(shape, shape, lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc, dtype)
    model = tei.make_ethosn_composite(model, "ethos-n.qnn_add")
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "lhs_shape,lhs_is_constant,rhs_shape,rhs_is_constant",
    [
        ((1, 4, 4, 8), True, (1, 1, 4, 8), False),
        ((1, 4, 4, 8), False, (1, 1, 4, 8), False),
        ((1, 16, 1, 4), True, (1, 1, 12, 4), False),
    ],
)
def test_unsupported_broadcast_addition(
    dtype, lhs_shape, lhs_is_constant, rhs_shape, rhs_is_constant
):
    """Test broadcast compatible addition falls back to TVM."""
    np.random.seed(0)

    lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc = _get_addition_qnn_params(dtype)

    model = _get_model(
        lhs_shape,
        rhs_shape,
        lhs_zp,
        lhs_sc,
        rhs_zp,
        rhs_sc,
        out_zp,
        out_sc,
        dtype,
        lhs_is_constant=lhs_is_constant,
        rhs_is_constant=rhs_is_constant,
    )
    from tvm.relay.op.contrib import partition_for_ethosn  # pylint: disable=import-outside-toplevel

    mod = tei.make_module(model, {})
    assert "qnn.add" in mod.astext(False)
    mod = partition_for_ethosn(mod, {})
    assert "qnn.add" in mod.astext(False)
    assert "ethos-n.qnn_add" not in mod.astext(False)
