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

# NOTE: We name this test file to start with test_graph_tuner
# to make it execute after zero_rank tensor test cases. This
# helps avoid topi arithmetic operator overloading issue:
# https://github.com/apache/tvm/issues/3240.
# TODO: restore the file name after this issue is resolved.
import os
import copy
import numpy as np
import tvm
from tvm import te
import tvm.relay.testing

from tvm import autotvm
from tvm import relay
from tvm.autotvm.task import ConfigEntity
from tvm.autotvm.measure import MeasureResult, MeasureInput
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner


def _create_args(dshape, kshape, strides, padding, dilation, layout, out_layout, dtype, out_dtype):
    data = tvm.te.placeholder(dshape, dtype=dtype)
    kernel = tvm.te.placeholder(kshape, dtype=dtype)
    return autotvm.task.serialize_args(
        [data, kernel, strides, padding, dilation, layout, layout, out_dtype]
    )


def _create_data(target, dshape, dtype, layout):
    data = relay.var("data", shape=dshape, dtype=dtype)
    w0 = relay.var("w0_weight")
    conv0 = relay.nn.conv2d(data, w0, channels=16, kernel_size=(3, 3), padding=(1, 1))
    w1 = relay.var("w1_weight")
    conv1 = relay.nn.conv2d(conv0, w1, channels=32, kernel_size=(1, 1))
    w2 = relay.var("w2_weight")
    conv2 = relay.nn.conv2d(conv1, w2, channels=32, kernel_size=(3, 3), padding=(1, 1))
    out = relay.add(conv1, conv2)
    net = relay.Function(relay.analysis.free_vars(out), out)
    mod, params = relay.testing.create_workload(net)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )
    new_args = [
        _create_args(
            (1, 3, 8, 8), (16, 3, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), layout, layout, dtype, dtype
        ),
        _create_args(
            (1, 16, 8, 8),
            (32, 16, 1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            layout,
            layout,
            dtype,
            dtype,
        ),
        _create_args(
            (1, 32, 8, 8),
            (32, 32, 3, 3),
            (1, 1),
            (1, 1, 1, 1),
            (1, 1),
            layout,
            layout,
            dtype,
            dtype,
        ),
    ]

    costs = [0.04, 0.012, 0.03]
    config_list = []
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [3, 1]],
            ["tile_oc", "sp", [4, 4]],
            ["tile_ow", "sp", [4, 2]],
            ["unroll_kw", "ot", True],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [2, 8]],
            ["tile_oc", "sp", [1, 32]],
            ["tile_oh", "ot", 1],
            ["tile_ow", "sp", [4, 2]],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [8, 4]],
            ["tile_oc", "sp", [4, 8]],
            ["tile_ow", "sp", [2, 4]],
            ["unroll_kw", "ot", False],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))

    records = []
    for args, cost, config, task in zip(new_args, costs, config_list, tasks):
        task.args = args
        ms_input = MeasureInput(target=target, task=task, config=config)
        ms_output = MeasureResult(costs=(cost,), error_no=0, all_cost=-1, timestamp=-1)
        records.append((ms_input, ms_output))

    ltf_records = []
    ltf_arg = [te.placeholder((1, 64, 16, 16, 8), dtype=dtype), "NCHW8c", "NCHW512c"]
    ltf_task = autotvm.task.create("layout_transform", ltf_arg, target)
    ms_input = MeasureInput(target=target, task=ltf_task, config=None)
    ms_output = MeasureResult(costs=(1.91224744e-05,), error_no=0, all_cost=-1, timestamp=-1)
    ltf_records.append((ms_input, ms_output))

    ltf_keys = []
    ltf_arg = [te.placeholder((1, 4, 8, 8, 4), dtype=dtype), "NCHW4c", "NCHW8c"]
    ltf_wkl = autotvm.task.args_to_workload(ltf_arg, "layout_transform")
    ltf_keys.append(ltf_wkl)
    ltf_arg = [te.placeholder((1, 1, 8, 8, 32), dtype=dtype), "NCHW32c", "NCHW4c"]
    ltf_wkl = autotvm.task.args_to_workload(ltf_arg, "layout_transform")
    ltf_keys.append(ltf_wkl)
    ltf_arg = [te.placeholder((1, 4, 8, 8, 8), dtype=dtype), "NCHW8c", "NCHW32c"]
    ltf_wkl = autotvm.task.args_to_workload(ltf_arg, "layout_transform")
    ltf_keys.append(ltf_wkl)

    return net, records, ltf_records, ltf_keys, tasks


def test_graph_tuner_layout_transform():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    target = "llvm"
    dshape = (1, 3, 8, 8)
    dtype = "float32"
    layout = "NCHW"
    conv2d = relay.op.get("nn.conv2d")
    target_ops = [conv2d]

    g, records, ltf_records, ltf_keys, _ = _create_data(target, dshape, dtype, layout)
    executor = DPTuner(g, {"data": dshape}, records, target_ops, target=target, log_file=log_file)
    executor.benchmark_layout_transform(layout_records=ltf_records, infer_layout=True)
    out = executor._layout_transform_perf_records

    num_flops = 0
    total_time = 0
    for record in ltf_records:
        ltf_wkl = record[0].task.workload
        input_shape = ltf_wkl[1][1]
        flops = np.prod(input_shape)
        num_flops += flops
        total_time += record[1].costs[0]
    avg_time = total_time / num_flops

    for ltf_workload in out:
        input_shape = ltf_workload[1][1]
        flops = 1
        for i in input_shape:
            flops *= i
        expected_time = flops * avg_time
        out_time = out[ltf_workload][1].costs[0]
        assert (
            expected_time == out_time
        ), "Inferred layout transformation time mismatch for %s: " "expecting %f but got %f" % (
            str(ltf_workload),
            expected_time,
            out_time,
        )


def test_graph_tuner_layout_transform_runner():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    target = "llvm"
    dshape = (1, 3, 8, 8)
    dtype = "float32"
    layout = "NCHW"
    conv2d = relay.op.get("nn.conv2d")
    target_ops = [conv2d]

    g, records, ltf_records, ltf_keys, _ = _create_data(target, dshape, dtype, layout)
    executor = DPTuner(g, {"data": dshape}, records, target_ops, target=target, log_file=log_file)
    runner = autotvm.LocalRunner(number=100, repeat=1, timeout=10)
    executor.benchmark_layout_transform(
        layout_records=ltf_records, infer_layout=True, runner=runner
    )
    out = executor._layout_transform_perf_records

    num_flops = 0
    total_time = 0
    for record in ltf_records:
        ltf_wkl = record[0].task.workload
        input_shape = ltf_wkl[1][1]
        flops = np.prod(input_shape)
        num_flops += flops
        total_time += record[1].costs[0]
    avg_time = total_time / num_flops

    for ltf_workload in out:
        input_shape = ltf_workload[1][1]
        flops = 1
        for i in input_shape:
            flops *= i
        expected_time = flops * avg_time
        out_time = out[ltf_workload][1].costs[0]
        assert (
            expected_time == out_time
        ), "Inferred layout transformation time mismatch for %s: " "expecting %f but got %f" % (
            str(ltf_workload),
            expected_time,
            out_time,
        )


def test_DPTuner_run():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    target = "llvm"
    dtype = "float32"
    layout = "NCHW"
    dshape = (1, 3, 8, 8)
    conv2d = relay.op.get("nn.conv2d")
    target_ops = [conv2d]

    g, records, ltf_records, ltf_keys, tasks = _create_data(target, dshape, dtype, layout)
    mod = tvm.IRModule()
    mod["main"] = g
    costs = [0.02, 0.02, 0.045]
    config_list = []
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [1, 3]],
            ["tile_oc", "sp", [2, 8]],
            ["tile_ow", "sp", [4, 2]],
            ["unroll_kw", "ot", True],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [4, 4]],
            ["tile_oc", "sp", [2, 16]],
            ["tile_oh", "ot", 1],
            ["tile_ow", "sp", [4, 2]],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [16, 2]],
            ["tile_oc", "sp", [8, 4]],
            ["tile_ow", "sp", [2, 4]],
            ["unroll_kw", "ot", False],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    for cost, config, task in zip(costs, config_list, tasks):
        ms_input = MeasureInput(target=target, task=task, config=config)
        ms_output = MeasureResult(costs=(cost,), error_no=0, all_cost=-1, timestamp=-1)
        records.append((ms_input, ms_output))

    executor = DPTuner(mod, {"data": dshape}, records, target_ops, target, log_file=log_file)
    executor.benchmark_layout_transform(layout_records=ltf_records, infer_layout=True)
    executor.run()
    out = [record[0].config for record in executor.get_optimal_records()]
    expected_out = [records[3][0].config, records[1][0].config, records[2][0].config]
    assert expected_out == out, "Output mismatch: expecting %s but got %s" % (
        str(expected_out),
        str(out),
    )
    assert os.path.isfile(log_file), "No log file with name %s exists." % log_file


def test_PBQPTuner_run():
    target = "llvm"
    dtype = "float32"
    layout = "NCHW"
    dshape = (1, 3, 8, 8)
    conv2d = relay.op.get("nn.conv2d")
    target_ops = [conv2d]

    g, records, ltf_records, ltf_keys, tasks = _create_data(target, dshape, dtype, layout)
    costs = [0.02, 0.02, 0.045]
    config_list = []
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [1, 3]],
            ["tile_oc", "sp", [2, 8]],
            ["tile_ow", "sp", [4, 2]],
            ["unroll_kw", "ot", True],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [4, 4]],
            ["tile_oc", "sp", [2, 16]],
            ["tile_oh", "ot", 1],
            ["tile_ow", "sp", [4, 2]],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [16, 2]],
            ["tile_oc", "sp", [8, 4]],
            ["tile_ow", "sp", [2, 4]],
            ["unroll_kw", "ot", False],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    for cost, config, task in zip(costs, config_list, tasks):
        ms_input = MeasureInput(target=target, task=task, config=config)
        ms_output = MeasureResult(costs=(cost,), error_no=0, all_cost=-1, timestamp=-1)
        records.append((ms_input, ms_output))

    executor = PBQPTuner(g, {"data": dshape}, records, target_ops, target)
    executor.benchmark_layout_transform(layout_records=ltf_records, infer_layout=True)
    executor.run()
    out = [record[0].config for record in executor.get_optimal_records()]
    expected_out = [records[3][0].config, records[1][0].config, records[2][0].config]
    assert expected_out == out, "Output mismatch: expecting %s but got %s" % (
        str(expected_out),
        str(out),
    )


def test_many_sub_graphs():
    target = "llvm"
    dtype = "float32"
    dshape = (1, 8, 8, 3)
    layout = "NCHW"
    conv2d = relay.op.get("nn.conv2d")
    target_ops = [conv2d]

    data = relay.var("data", shape=dshape, dtype=dtype)
    t0 = relay.transpose(data, (0, 3, 1, 2))
    w0 = relay.var("w0_weight")
    conv0 = relay.nn.conv2d(t0, w0, channels=16, kernel_size=(3, 3), padding=(1, 1))
    t1 = relay.transpose(conv0, (0, 2, 3, 1))
    w1 = relay.var("w1_weight")
    t2 = relay.transpose(t1, (0, 3, 1, 2))
    conv1 = relay.nn.conv2d(t2, w1, channels=32, kernel_size=(1, 1))
    t3 = relay.transpose(conv1, (0, 2, 3, 1))
    w2 = relay.var("w2_weight")
    t4 = relay.transpose(t3, (0, 3, 1, 2))
    conv2 = relay.nn.conv2d(t4, w2, channels=32, kernel_size=(3, 3), padding=(1, 1))
    t5 = relay.transpose(conv2, (0, 2, 3, 1))
    out = relay.add(t3, t5)
    net = relay.Function(relay.analysis.free_vars(out), out)
    net, params = relay.testing.create_workload(net)

    tasks = autotvm.task.extract_from_program(
        net["main"], target=target, params=params, ops=(conv2d,)
    )
    new_args = [
        _create_args(
            (1, 3, 8, 8), (16, 3, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), layout, layout, dtype, dtype
        ),
        _create_args(
            (1, 16, 8, 8),
            (32, 16, 1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            layout,
            layout,
            dtype,
            dtype,
        ),
        _create_args(
            (1, 32, 8, 8),
            (32, 32, 3, 3),
            (1, 1),
            (1, 1, 1, 1),
            (1, 1),
            layout,
            layout,
            dtype,
            dtype,
        ),
    ]

    costs = [0.04, 0.012, 0.03, 0.02, 0.02, 0.045]
    config_list = []
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [3, 1]],
            ["tile_oc", "sp", [4, 4]],
            ["tile_ow", "sp", [4, 2]],
            ["unroll_kw", "ot", True],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [2, 8]],
            ["tile_oc", "sp", [1, 32]],
            ["tile_oh", "ot", 1],
            ["tile_ow", "sp", [4, 2]],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [8, 4]],
            ["tile_oc", "sp", [4, 8]],
            ["tile_ow", "sp", [2, 4]],
            ["unroll_kw", "ot", False],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [1, 3]],
            ["tile_oc", "sp", [2, 8]],
            ["tile_ow", "sp", [4, 2]],
            ["unroll_kw", "ot", True],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [4, 4]],
            ["tile_oc", "sp", [2, 16]],
            ["tile_oh", "ot", 1],
            ["tile_ow", "sp", [4, 2]],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [16, 2]],
            ["tile_oc", "sp", [8, 4]],
            ["tile_ow", "sp", [2, 4]],
            ["unroll_kw", "ot", False],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))

    records = []
    new_args = new_args + new_args
    tasks = tasks + tasks
    for args, cost, config, task in zip(new_args, costs, config_list, tasks):
        task.args = args
        ms_input = MeasureInput(target=target, task=task, config=config)
        ms_output = MeasureResult(costs=(cost,), error_no=0, all_cost=-1, timestamp=-1)
        records.append((ms_input, ms_output))

    ltf_records = []
    ltf_arg = [te.placeholder((1, 64, 16, 16, 8), dtype=dtype), "NCHW8c", "NCHW512c"]
    ltf_task = autotvm.task.create("layout_transform", ltf_arg, target)
    ms_input = MeasureInput(target=target, task=ltf_task, config=None)
    ms_output = MeasureResult(costs=(1.91224744e-05,), error_no=0, all_cost=-1, timestamp=-1)
    ltf_records.append((ms_input, ms_output))

    executor = DPTuner(net, {"data": dshape}, records, target_ops, target)
    executor.benchmark_layout_transform(layout_records=ltf_records, infer_layout=True)
    executor.run()
    out = [record[0].config for record in executor.get_optimal_records()]
    expected_out = [records[3][0].config, records[1][0].config, records[2][0].config]
    assert expected_out == out, "Output mismatch: expecting %s but got %s" % (
        str(expected_out),
        str(out),
    )

    executor = PBQPTuner(net, {"data": dshape}, records, target_ops, target)
    executor.benchmark_layout_transform(layout_records=ltf_records, infer_layout=True)
    executor.run()
    out = [record[0].config for record in executor.get_optimal_records()]
    expected_out = [records[3][0].config, records[1][0].config, records[2][0].config]
    assert expected_out == out, "Output mismatch: expecting %s but got %s" % (
        str(expected_out),
        str(out),
    )


def test_tuple():
    target = "llvm"
    dtype = "float32"
    dshape = (1, 5, 32, 32)
    layout = "NCHW"
    conv2d = relay.op.get("nn.conv2d")
    target_ops = [conv2d]

    data = relay.var("data", shape=dshape, dtype=dtype)
    w0 = relay.var("w0_weight")
    conv0 = relay.nn.conv2d(data, w0, channels=2, kernel_size=(3, 3), padding=(1, 1))
    w1 = relay.var("w1_weight")
    conv1 = relay.nn.conv2d(data, w1, channels=3, kernel_size=(3, 3), padding=(1, 1))
    out = relay.concatenate([conv0, conv1], axis=1)
    net = relay.Function(relay.analysis.free_vars(out), out)
    net, params = relay.testing.create_workload(net)

    tasks = autotvm.task.extract_from_program(
        net["main"], target=target, params=params, ops=(conv2d,)
    )
    new_args = [
        _create_args(
            (1, 5, 32, 32), (2, 5, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), layout, layout, dtype, dtype
        ),
        _create_args(
            (1, 5, 32, 32), (3, 5, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), layout, layout, dtype, dtype
        ),
    ]
    costs = [0.01, 0.012, 0.03, 0.04]
    config_list = []
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [1, 5]],
            ["tile_oc", "sp", [1, 2]],
            ["tile_ow", "sp", [4, 8]],
            ["unroll_kw", "ot", True],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [1, 5]],
            ["tile_oc", "sp", [1, 3]],
            ["tile_ow", "sp", [2, 16]],
            ["unroll_kw", "ot", False],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [1, 5]],
            ["tile_oc", "sp", [2, 1]],
            ["tile_ow", "sp", [4, 8]],
            ["unroll_kw", "ot", True],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [1, 5]],
            ["tile_oc", "sp", [3, 1]],
            ["tile_ow", "sp", [2, 16]],
            ["unroll_kw", "ot", False],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))

    records = []
    new_args = new_args + new_args
    tasks = tasks + tasks
    for args, cost, config, task in zip(new_args, costs, config_list, tasks):
        task.args = args
        ms_input = MeasureInput(target=target, task=task, config=config)
        ms_output = MeasureResult(costs=(cost,), error_no=0, all_cost=-1, timestamp=-1)
        records.append((ms_input, ms_output))

    ltf_records = []
    ltf_arg = [te.placeholder((1, 64, 16, 16, 8), dtype=dtype), "NCHW8c", "NCHW512c"]
    ltf_task = autotvm.task.create("layout_transform", ltf_arg, target)
    ms_input = MeasureInput(target=target, task=ltf_task, config=None)
    ms_output = MeasureResult(costs=(1.91224744e-05,), error_no=0, all_cost=-1, timestamp=-1)
    ltf_records.append((ms_input, ms_output))

    executor = DPTuner(net, {"data": dshape}, records, target_ops, target)
    executor.benchmark_layout_transform(layout_records=ltf_records, infer_layout=True)
    executor.run()
    out = [record[0].config for record in executor.get_optimal_records()]
    expected_out = [records[2][0].config, records[1][0].config]
    assert expected_out == out, "Output mismatch: expecting %s but got %s" % (
        str(expected_out),
        str(out),
    )

    executor = PBQPTuner(net, {"data": dshape}, records, target_ops, target)
    executor.benchmark_layout_transform(layout_records=ltf_records, infer_layout=True)
    executor.run()
    out = [record[0].config for record in executor.get_optimal_records()]
    expected_out = [records[2][0].config, records[1][0].config]
    assert expected_out == out, "Output mismatch: expecting %s but got %s" % (
        str(expected_out),
        str(out),
    )


def test_triangle_block():
    target = "llvm"
    dtype = "float32"
    dshape = (1, 3, 8, 8)
    layout = "NCHW"
    conv2d = relay.op.get("nn.conv2d")
    target_ops = [conv2d]

    data = relay.var("data", shape=dshape, dtype=dtype)
    w0 = relay.var("w0_weight")
    conv0 = relay.nn.conv2d(data, w0, channels=16, kernel_size=(3, 3), padding=(1, 1))
    w1 = relay.var("w1_weight")
    conv1 = relay.nn.conv2d(conv0, w1, channels=32, kernel_size=(1, 1))
    w2 = relay.var("w2_weight")
    conv2 = relay.nn.conv2d(data, w2, channels=32, kernel_size=(3, 3), padding=(1, 1))
    out = relay.concatenate([conv0, conv1, conv2], axis=1)
    net = relay.Function(relay.analysis.free_vars(out), out)
    net, params = relay.testing.create_workload(net)

    tasks = autotvm.task.extract_from_program(
        net["main"], target=target, params=params, ops=(conv2d,)
    )
    new_args = [
        _create_args(
            (1, 3, 8, 8), (16, 3, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), layout, layout, dtype, dtype
        ),
        _create_args(
            (1, 16, 8, 8),
            (32, 16, 1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            layout,
            layout,
            dtype,
            dtype,
        ),
        _create_args(
            (1, 3, 8, 8), (32, 3, 3, 3), (1, 1), (1, 1, 1, 1), (1, 1), layout, layout, dtype, dtype
        ),
    ]
    costs = [0.04, 0.012, 0.03, 0.02, 0.02, 0.045]
    config_list = []
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [3, 1]],
            ["tile_oc", "sp", [4, 4]],
            ["tile_ow", "sp", [4, 2]],
            ["unroll_kw", "ot", True],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [2, 8]],
            ["tile_oc", "sp", [1, 32]],
            ["tile_oh", "ot", 1],
            ["tile_ow", "sp", [4, 2]],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [8, 4]],
            ["tile_oc", "sp", [4, 8]],
            ["tile_ow", "sp", [2, 4]],
            ["unroll_kw", "ot", False],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [1, 3]],
            ["tile_oc", "sp", [2, 8]],
            ["tile_ow", "sp", [4, 2]],
            ["unroll_kw", "ot", True],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [4, 4]],
            ["tile_oc", "sp", [2, 16]],
            ["tile_oh", "ot", 1],
            ["tile_ow", "sp", [4, 2]],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {
        "index": -1,
        "code_hash": None,
        "entity": [
            ["tile_ic", "sp", [16, 2]],
            ["tile_oc", "sp", [8, 4]],
            ["tile_ow", "sp", [2, 4]],
            ["unroll_kw", "ot", False],
        ],
    }
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))

    records = []
    new_args = new_args + new_args
    tasks = tasks + tasks
    for args, cost, config, task in zip(new_args, costs, config_list, tasks):
        task.args = args
        ms_input = MeasureInput(target=target, task=task, config=config)
        ms_output = MeasureResult(costs=(cost,), error_no=0, all_cost=-1, timestamp=-1)
        records.append((ms_input, ms_output))

    ltf_records = []
    ltf_arg = [te.placeholder((1, 64, 16, 16, 8), dtype=dtype), "NCHW8c", "NCHW512c"]
    ltf_task = autotvm.task.create("layout_transform", ltf_arg, target)
    ms_input = MeasureInput(target=target, task=ltf_task, config=None)
    ms_output = MeasureResult(costs=(1.91224744e-05,), error_no=0, all_cost=-1, timestamp=-1)
    ltf_records.append((ms_input, ms_output))

    executor = DPTuner(net, {"data": dshape}, records, target_ops, target)
    executor.benchmark_layout_transform(layout_records=ltf_records, infer_layout=True)
    executor.run()
    out = [record[0].config for record in executor.get_optimal_records()]
    expected_out = [records[3][0].config, records[1][0].config, records[2][0].config]
    assert expected_out == out, "Output mismatch: expecting %s but got %s" % (
        str(expected_out),
        str(out),
    )

    executor = PBQPTuner(net, {"data": dshape}, records, target_ops, target)
    executor.benchmark_layout_transform(layout_records=ltf_records, infer_layout=True)
    executor.run()
    out = [record[0].config for record in executor.get_optimal_records()]
    expected_out = [records[3][0].config, records[1][0].config, records[2][0].config]
    assert expected_out == out, "Output mismatch: expecting %s but got %s" % (
        str(expected_out),
        str(out),
    )


if __name__ == "__main__":
    test_graph_tuner_layout_transform()
    test_DPTuner_run()
    test_PBQPTuner_run()
    test_many_sub_graphs()
    test_tuple()
    test_triangle_block()
