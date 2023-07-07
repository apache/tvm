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
import pytest

pytest.importorskip("ethosu.vela")
import numpy as np
from ethosu.vela import api as vapi
from unittest.mock import patch

import tvm
from tvm.script import tir as T
from tvm.tir import stmt_functor
from tvm.relay.backend.contrib.ethosu import vela_api
import tvm.relay.backend.contrib.ethosu.tir_to_cs_translator as tirtocs

ACCEL_TYPES = [
    vapi.NpuAccelerator.Ethos_U55_256,
    vapi.NpuAccelerator.Ethos_U55_128,
    vapi.NpuAccelerator.Ethos_U55_64,
    vapi.NpuAccelerator.Ethos_U55_32,
]


"""Test case 1"""


@tvm.script.ir_module
class Module1:
    @T.prim_func
    def main(
        placeholder: T.handle,
        placeholder_1: T.handle,
        placeholder_2: T.handle,
        ethosu_conv2d: T.handle,
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_3 = T.match_buffer(
            placeholder, [192], dtype="uint8", elem_offset=0, align=64, offset_factor=1
        )
        placeholder_4 = T.match_buffer(
            placeholder_1, [48], dtype="uint8", elem_offset=0, align=64, offset_factor=1
        )
        placeholder_5 = T.match_buffer(
            placeholder_2, [16], dtype="int32", elem_offset=0, align=64, offset_factor=1
        )
        ethosu_conv2d_1 = T.match_buffer(
            ethosu_conv2d, [1024], dtype="uint8", elem_offset=0, align=64, offset_factor=1
        )
        # body
        T.evaluate(
            T.call_extern(
                "ethosu_conv2d",
                "uint8",
                8,
                8,
                3,
                8,
                0,
                8,
                placeholder_3[0],
                0,
                0,
                0,
                T.float32(0.5),
                10,
                "NHWC",
                24,
                3,
                1,
                "uint8",
                8,
                8,
                16,
                8,
                0,
                8,
                ethosu_conv2d_1[0],
                0,
                0,
                0,
                T.float32(0.25),
                14,
                "NHWC",
                128,
                16,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                placeholder_4[0],
                0,
                12,
                placeholder_5[0],
                0,
                0,
                0,
                0,
                0,
                "CLIP",
                0,
                0,
                "TFL",
                "NONE",
                dtype="uint8",
            )
        )

    __tvm_meta__ = None


"""Test case 2 with per-channel quantization"""


@tvm.script.ir_module
class Module2:
    @T.prim_func
    def main(
        placeholder: T.handle,
        placeholder_1: T.handle,
        placeholder_2: T.handle,
        placeholder_6: T.handle,
        ethosu_conv2d: T.handle,
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_3 = T.match_buffer(
            placeholder, [192], dtype="uint8", elem_offset=0, align=64, offset_factor=1
        )
        placeholder_4 = T.match_buffer(
            placeholder_1, [48], dtype="uint8", elem_offset=0, align=64, offset_factor=1
        )
        placeholder_5 = T.match_buffer(
            placeholder_2, [16], dtype="int32", elem_offset=0, align=64, offset_factor=1
        )
        # Per-channel weight scales
        placeholder_7 = T.match_buffer(
            placeholder_6, [16], dtype="float32", elem_offset=0, align=64, offset_factor=1
        )
        ethosu_conv2d_1 = T.match_buffer(
            ethosu_conv2d, [1024], dtype="uint8", elem_offset=0, align=64, offset_factor=1
        )
        # body
        T.evaluate(
            T.call_extern(
                "ethosu_conv2d",
                "uint8",
                8,
                8,
                3,
                8,
                0,
                8,
                placeholder_3[0],
                0,
                0,
                0,
                T.float32(0.5),
                10,
                "NHWC",
                24,
                3,
                1,
                "uint8",
                8,
                8,
                16,
                8,
                0,
                8,
                ethosu_conv2d_1[0],
                0,
                0,
                0,
                T.float32(0.25),
                14,
                "NHWC",
                128,
                16,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                placeholder_4[0],
                0,
                12,
                placeholder_5[0],
                0,
                0,
                0,
                0,
                0,
                "CLIP",
                0,
                0,
                "TFL",
                "NONE",
                dtype="uint8",
            )
        )

    __tvm_meta__ = None


# fmt: off
@tvm.script.ir_module
class Module3:
    @T.prim_func
    def main(ethos_u_0_i0: T.Buffer((1, 299, 299, 2), "int8"), ethosu_write: T.Buffer((1, 299, 299, 3), "int8")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "global_symbol": "main", "tir.noalias": T.bool(True)})
        p2_global = T.allocate([128], "uint8", "global", annotations={"disable_lower_builtin": T.bool(True)})
        ax0_ax1_fused_ax2_fused_ax3_fused = T.int32()
        p2_global_1 = T.Buffer((128,), "uint8", data=p2_global)
        with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused, None, "DataPar", ""), "pragma_compute_cycles_hint", 1056):
            p1_encoded = T.Buffer((128,), "uint8")
            T.call_extern("handle", "ethosu_copy", p1_encoded[0], 128, p2_global_1[0])
        nn = T.int32()
        T.attr(T.iter_var(nn, None, "DataPar", ""), "pragma_compute_cycles_hint", T.int64(179570))
        ethos_u_0_i0_1 = T.Buffer((178802,), "int8", data=ethos_u_0_i0.data)
        ethosu_write_1 = T.Buffer((268203,), "int8", data=ethosu_write.data)
        T.call_extern("handle", "ethosu_conv2d", "int8", 299, 299, 2, 299, 0, 299, ethos_u_0_i0_1[0], 0, 0, 0, T.float32(0.0039215683937072754), -128, "NHWC", 598, 2, 1, "int8", 299, 299, 3, 299, 0, 299, ethosu_write_1[0], 0, 0, 0, T.float32(0.025585981085896492), -128, "NHWC", 897, 3, 1, 2, 3, 1, 1, 1, 2, p2_global_1[0], 96, T.int8(-1), T.int8(-1), 0, p2_global_1[96], 32, T.int8(-1), T.int8(-1), 2, 0, 2, 1, "NONE", 0, 0, "TFL", "NONE", 32, 12, 8)

    __tvm_meta__ = None
# fmt: on


def test_get_optimal_block_config():
    block_configs_cases = [
        {
            "test": [
                vapi.NpuShape3D(10, 20, 8),
                vapi.NpuShape3D(10, 30, 16),
                vapi.NpuShape3D(10, 40, 32),
            ],
            "ref": vapi.NpuShape3D(10, 40, 32),
        },
        {
            "test": [
                vapi.NpuShape3D(10, 20, 8),
                vapi.NpuShape3D(10, 50, 32),
                vapi.NpuShape3D(10, 40, 32),
            ],
            "ref": vapi.NpuShape3D(10, 50, 32),
        },
        {
            "test": [
                vapi.NpuShape3D(50, 50, 8),
                vapi.NpuShape3D(10, 30, 32),
                vapi.NpuShape3D(8, 8, 64),
            ],
            "ref": vapi.NpuShape3D(8, 8, 64),
        },
    ]

    for test_case in block_configs_cases:
        assert vela_api._get_optimal_block_config(test_case["test"]) == test_case["ref"]


@pytest.mark.parametrize(
    "block_config_str, expected_block_config",
    [("4x4x8", vapi.NpuShape3D(4, 4, 8)), ("3x7x16", vapi.NpuShape3D(3, 7, 16))],
)
def test_force_block_config(block_config_str, expected_block_config):
    config = {
        "dev_force_block_config": block_config_str,
    }
    with tvm.transform.PassContext(config={"relay.ext.ethos-u.options": config}):
        block_config = vela_api.get_optimal_block_config(None, vapi.NpuAccelerator.Ethos_U55_128)
        assert block_config == expected_block_config


def test_compress_weights():
    test_vecs = [
        {
            # Stimulus
            "accel": vapi.NpuAccelerator.Ethos_U55_256,
            "block_depth": 8,
            "ifm_dtype": np.uint8,
            "shape": (3, 3, 16, 64),
            "layout": "HWIO",
            "zero_point": np.int64(134),
            "dilation": (1, 1),
            "is_depthwise": False,
            # Reference outputs
            "block_traversal": vapi.NpuBlockTraversal.PART_KERNEL_FIRST,
        },
        {
            # Stimulus
            "accel": vapi.NpuAccelerator.Ethos_U55_256,
            "block_depth": 8,
            "ifm_dtype": np.uint8,
            "shape": (3, 3, 32, 64),
            "layout": "HWIO",
            "zero_point": np.int64(134),
            "dilation": (1, 1),
            "is_depthwise": False,
            # Reference outputs
            "block_traversal": vapi.NpuBlockTraversal.DEPTH_FIRST,
        },
        {
            # Stimulus
            "accel": vapi.NpuAccelerator.Ethos_U55_256,
            "block_depth": 8,
            "ifm_dtype": np.int16,
            "shape": (3, 3, 16, 64),
            "layout": "HWIO",
            "zero_point": np.int64(134),
            "dilation": (1, 1),
            "is_depthwise": False,
            # Reference outputs
            "block_traversal": vapi.NpuBlockTraversal.DEPTH_FIRST,
        },
        # Pass-through value check
        {
            # Stimulus
            "accel": vapi.NpuAccelerator.Ethos_U55_128,
            "block_depth": 16,
            "ifm_dtype": np.uint8,
            "shape": (243, 152, 7, 1),
            "layout": "HWOI",
            "zero_point": np.int64(110),
            "dilation": (2, 2),
            "is_depthwise": True,
            # Reference outputs
            "block_traversal": vapi.NpuBlockTraversal.DEPTH_FIRST,
        },
        {
            # Stimulus
            "accel": vapi.NpuAccelerator.Ethos_U55_128,
            "block_depth": 32,
            "ifm_dtype": np.uint8,
            "shape": (64, 67, 35, 8),
            "layout": "OHWI",
            "zero_point": np.int64(100),
            "dilation": (1, 2),
            "is_depthwise": False,
            # Reference outputs
            "block_traversal": vapi.NpuBlockTraversal.PART_KERNEL_FIRST,
        },
    ]

    def verify(test_vec, mock_obj):
        layout_transform_indices = {
            "HWIO": (3, 0, 1, 2),
            "HWOI": (2, 0, 1, 3),
            "OHWI": (0, 1, 2, 3),
        }

        assert mock_obj
        mock_obj.assert_called_once()
        assert mock_obj.call_args[1]["accelerator"] == test_vec["accel"]
        assert mock_obj.call_args[1]["accelerator"] == test_vec["accel"]
        ishape = test_vec["shape"]
        shape_owhi = (
            ishape[layout_transform_indices[test_vec["layout"]][0]],
            ishape[layout_transform_indices[test_vec["layout"]][1]],
            ishape[layout_transform_indices[test_vec["layout"]][2]],
            ishape[layout_transform_indices[test_vec["layout"]][3]],
        )
        assert mock_obj.call_args[1]["weights_volume"].shape == shape_owhi
        assert mock_obj.call_args[1]["dilation_xy"] == test_vec["dilation"]
        assert mock_obj.call_args[1]["ifm_bitdepth"] == np.iinfo(test_vec["ifm_dtype"]).bits
        assert mock_obj.call_args[1]["ofm_block_depth"] == test_vec["block_depth"]
        assert mock_obj.call_args[1]["is_depthwise"] == test_vec["is_depthwise"]
        assert mock_obj.call_args[1]["block_traversal"] == test_vec["block_traversal"]

    def create_mock(test_vec):
        with patch("ethosu.vela.api.npu_encode_weights") as mock_npu_encode_weights:
            ifm_bitdepth = np.iinfo(test_vec["ifm_dtype"]).bits
            ifm_dtype = test_vec["ifm_dtype"]
            max = np.iinfo(ifm_dtype).max
            min = np.iinfo(ifm_dtype).min
            values = np.random.randint(min, max, test_vec["shape"], ifm_dtype)
            vela_api.compress_weights(
                weights=values,
                weights_zp=test_vec["zero_point"],
                weights_layout=test_vec["layout"],
                ifm_bitdepth=ifm_bitdepth,
                block_depth=test_vec["block_depth"],
                dilation=test_vec["dilation"],
                accel_config=test_vec["accel"],
                is_depthwise=test_vec["is_depthwise"],
            )
            return mock_npu_encode_weights

    for tv in test_vecs:
        mock_obj = create_mock(tv)
        verify(tv, mock_obj)


def test_pack_biases():
    test_vecs = [
        {
            # Stimulus
            "bias_length": 3,
            "ifm_scale": np.single(1.11111111),
            "ifm_dtype": np.uint8,
            "weight_scales": np.array(
                [np.single(0.91111111), np.single(1.01111111), np.single(1.11111111)]
            ),
            "ofm_scale": np.single(1.2),
            "is_activation_tanh_or_sigmoid": False,
            # Reference outputs
            "hw_scales": (1811663288, 2010504240, 1104672703),
            "hw_shifts": (31, 31, 30),
        },
        {
            # Stimulus
            "bias_length": 3,
            "ifm_scale": np.single(1.11111111),
            "ifm_dtype": np.int8,
            "weight_scales": np.array(
                [np.single(0.91111111), np.single(1.01111111), np.single(1.11111111)]
            ),
            "ofm_scale": np.single(1.2),
            "is_activation_tanh_or_sigmoid": False,
            # Reference outputs
            "hw_scales": (1811663185, 2010504312, 1104672720),
            "hw_shifts": (31, 31, 30),
        },
        {
            # Stimulus
            "bias_length": 3,
            "ifm_scale": np.single(1.11111111),
            "ifm_dtype": np.int16,
            "weight_scales": np.array(
                [np.single(0.91111111), np.single(1.01111111), np.single(1.11111111)]
            ),
            "ofm_scale": np.single(1.2),
            "is_activation_tanh_or_sigmoid": False,
            # Reference outputs
            "hw_scales": (27644, 30678, 16856),
            "hw_shifts": (15, 15, 14),
        },
    ]

    def verify(test_vec, mock_obj, packed_biases):
        assert mock_obj
        for idx, val in enumerate(test_vec["bias_values"]):
            assert val == mock_obj.call_args_list[idx][0][0]
            assert test_vec["hw_scales"][idx] == mock_obj.call_args_list[idx][0][1]
            assert test_vec["hw_shifts"][idx] == mock_obj.call_args_list[idx][0][2]

    def create_mock(test_vec):
        with patch("ethosu.vela.api.npu_encode_bias") as mock_npu_encode_bias:
            mock_npu_encode_bias.return_value = bytearray(10)
            ifm_dtype = test_vec["ifm_dtype"]
            max = np.iinfo(ifm_dtype).max
            min = np.iinfo(ifm_dtype).min
            # tvm will always create biases in int32
            biases = np.random.randint(min, max, test_vec["bias_length"], np.int32)
            packed_biases = vela_api.pack_biases(
                biases=biases,
                ifm_scale=test_vec["ifm_scale"],
                ifm_dtype=test_vec["ifm_dtype"],
                weight_scales=test_vec["weight_scales"],
                ofm_scale=test_vec["ofm_scale"],
                is_activation_tanh_or_sigmoid=test_vec["is_activation_tanh_or_sigmoid"],
            )
            test_vec["bias_values"] = biases
            return mock_npu_encode_bias, packed_biases
        return None

    for _test_vec in test_vecs:
        mock_obj, packed_biases = create_mock(_test_vec)
        verify(_test_vec, mock_obj, packed_biases)


def extract_ethosu_conv2d_extern_calls(mod):
    """This function will obtain all ethosu_conv2d
    calls from a NPU TIR module

    Parameters
    ----------
    mod : tvm.IRModule
        This is a NPU TIR Module

    Returns
    -------
    list
        List of tvm.tir.Call objects
        that are tir extern calls
        for ethosu_conv2d
    """
    # There should only be a single function
    assert len(mod.functions.items()) == 1
    primfunc = mod.functions.items()[0][1]

    ethosu_conv2d_calls = list()

    def populate_ethosu_conv2d_calls(stmt):
        if (
            isinstance(stmt, tvm.tir.Call)
            and stmt.op.name == "T.call_extern"
            and stmt.args[0] == "ethosu_conv2d"
        ):
            ethosu_conv2d_calls.append(stmt)

    stmt_functor.post_order_visit(primfunc.body, populate_ethosu_conv2d_calls)
    return ethosu_conv2d_calls


@pytest.mark.parametrize(
    "accel",
    ACCEL_TYPES,
)
def test_encode_weights(accel):
    test_vecs = [
        {
            # Stimulus
            "tir_module": Module1,
            "param_dict": {
                1: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [48], "uint8"),
                2: np.random.randint(np.iinfo("int32").min, np.iinfo("int32").max, [16], "int32"),
            },
            "accel_type": accel,
            # Reference outputs
            "block_traversal": vapi.NpuBlockTraversal.PART_KERNEL_FIRST,
        },
    ]

    def create_mock(test_vec):
        with patch("ethosu.vela.api.npu_encode_weights") as mock_enc_w:
            with patch("ethosu.vela.api.npu_find_block_configs") as mock_blk_cfg:
                mock_blk_cfg.return_value = [vapi.NpuShape3D(8, 8, 8)]
                ethosu_conv2d_calls = extract_ethosu_conv2d_extern_calls(test_vec["tir_module"])
                buffer_info = tirtocs.extract_buffer_info(
                    test_vec["tir_module"], test_vec["param_dict"]
                )
                for ethosu_conv2d_call in ethosu_conv2d_calls:
                    npu_op, _ = tirtocs.translate_ethosu_conv2d(ethosu_conv2d_call)
                    weights = buffer_info[npu_op.weights[0].address.buffer_var][0]
                    vela_api.encode_weights(ethosu_conv2d_call, weights, accel)
                return mock_enc_w

    def verify(test_vec, mock_enc_w):
        ethosu_conv2d_calls = extract_ethosu_conv2d_extern_calls(test_vec["tir_module"])
        buffer_info = tirtocs.extract_buffer_info(test_vec["tir_module"], test_vec["param_dict"])
        for ethosu_conv2d_call in ethosu_conv2d_calls:
            npu_op, w_zero_point = tirtocs.translate_ethosu_conv2d(ethosu_conv2d_call)
            weights = buffer_info[npu_op.weights[0].address.buffer_var][0]

            assert mock_enc_w.call_args[1]["accelerator"] == accel
            assert (
                mock_enc_w.call_args[1]["weights_volume"].flatten()
                == weights.astype(np.int64) - w_zero_point
            ).all()
            assert mock_enc_w.call_args[1]["dilation_xy"] == (
                npu_op.kernel.dilation_x,
                npu_op.kernel.dilation_y,
            )
            assert mock_enc_w.call_args[1]["dilation_xy"] == (
                npu_op.kernel.dilation_x,
                npu_op.kernel.dilation_y,
            )
            assert mock_enc_w.call_args[1]["ifm_bitdepth"] == npu_op.ifm.data_type.size_in_bits()
            assert mock_enc_w.call_args[1]["block_traversal"] == test_vec["block_traversal"]

    for _test_vec in test_vecs:
        _mock_enc_w = create_mock(_test_vec)
        verify(_test_vec, _mock_enc_w)


def test_find_block_config_with_vela():
    block_configs_cases = [
        {
            "accel_type": vapi.NpuAccelerator.Ethos_U55_256,
            "ref": vapi.NpuShape3D(30, 12, 8),
        },
        {
            "accel_type": vapi.NpuAccelerator.Ethos_U55_128,
            "ref": vapi.NpuShape3D(17, 10, 8),
        },
        {
            "accel_type": vapi.NpuAccelerator.Ethos_U55_64,
            "ref": vapi.NpuShape3D(25, 5, 8),
        },
        {
            "accel_type": vapi.NpuAccelerator.Ethos_U55_32,
            "ref": vapi.NpuShape3D(25, 5, 4),
        },
    ]

    mod = Module3
    ethosu_conv2d_call = mod["main"].body.body.seq[1].body.value
    npu_op, _ = tirtocs.translate_ethosu_conv2d(ethosu_conv2d_call)

    for case in block_configs_cases:
        assert vela_api._find_block_config_with_vela(npu_op, case["accel_type"]) == case["ref"]


if __name__ == "__main__":
    tvm.testing.main()
