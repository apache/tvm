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
# https://github.com/dmlc/tvm/issues/3240.
# TODO: restore the file name after this issue is resolved.
import os
import copy
import numpy as np
import tvm
import tvm.relay.testing

from tvm import autotvm
from tvm import relay
from tvm.autotvm.task import ConfigEntity
from tvm.autotvm.measure import MeasureResult, MeasureInput
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from test_graph_tuner_utils import create_workload


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
    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              params=params,
                                              ops=(relay.op.nn.conv2d,))
    wkl_list = [
        create_workload((1, 3, 8, 8), (16, 3, 3, 3), (1, 1), (1, 1), (1, 1), layout, layout, dtype, dtype),
        create_workload((1, 16, 8, 8), (32, 16, 1, 1), (1, 1), (0, 0), (1, 1), layout, layout, dtype, dtype),
        create_workload((1, 32, 8, 8), (32, 32, 3, 3), (1, 1), (1, 1), (1, 1), layout, layout, dtype, dtype),
    ]
    costs = [0.04, 0.012, 0.03]
    config_list = []
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [3, 1]],
                      ["tile_oc", "sp", [4, 4]],
                      ["tile_ow", "sp", [4, 2]],
                      ["unroll_kw", "ot", True]],
                "t": ""}
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [2, 8]],
                      ["tile_oc", "sp", [1, 32]],
                      ["tile_oh", "ot", 1],
                      ["tile_ow", "sp", [4, 2]]],
                "t": ""}
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [8, 4]],
                      ["tile_oc", "sp", [4, 8]],
                      ["tile_ow", "sp", [2, 4]],
                      ["unroll_kw", "ot", False]],
                "t": ""}
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))

    records = []
    for wkl, cost, config, task in zip(wkl_list, costs, config_list, tasks):
        task.workload = wkl
        ms_input = MeasureInput(target=target, task=task, config=config)
        ms_output = MeasureResult(costs=(cost,), error_no=0, all_cost=-1, timestamp=-1)
        records.append((ms_input, ms_output))

    ltf_records = []
    ltf_arg = [tvm.placeholder((1, 64, 16, 16, 8), dtype=dtype), "NCHW8c", "NCHW512c"]
    ltf_arg = autotvm.task.topi_integration.serialize_args(ltf_arg)
    ltf_wkl = ('layout_transform',) + autotvm.task.args_to_workload(ltf_arg)
    ltf_task = copy.deepcopy(tasks[0])
    ltf_task.workload = ltf_wkl
    ms_input = MeasureInput(target=target, task=ltf_task, config=None)
    ms_output =  MeasureResult(costs=(1.91224744e-05,), error_no=0, all_cost=-1, timestamp=-1)
    ltf_records.append((ms_input, ms_output))

    ltf_keys = []
    ltf_arg = [tvm.placeholder((1, 4, 8, 8, 4), dtype=dtype), "NCHW4c", "NCHW8c"]
    ltf_arg = autotvm.task.topi_integration.serialize_args(ltf_arg)
    ltf_wkl = ('layout_transform',) + autotvm.task.args_to_workload(ltf_arg)
    ltf_keys.append(ltf_wkl)
    ltf_arg = [tvm.placeholder((1, 1, 8, 8, 32), dtype=dtype), "NCHW32c", "NCHW4c"]
    ltf_arg = autotvm.task.topi_integration.serialize_args(ltf_arg)
    ltf_wkl = ('layout_transform',) + autotvm.task.args_to_workload(ltf_arg)
    ltf_keys.append(ltf_wkl)
    ltf_arg = [tvm.placeholder((1, 4, 8, 8, 8), dtype=dtype), "NCHW8c", "NCHW32c"]
    ltf_arg = autotvm.task.topi_integration.serialize_args(ltf_arg)
    ltf_wkl = ('layout_transform',) + autotvm.task.args_to_workload(ltf_arg)
    ltf_keys.append(ltf_wkl)

    return net, records, ltf_records, ltf_keys, tasks


def test_graph_tuner_layout_transform():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    target = "llvm"
    dshape = (1, 3, 8, 8)
    dtype = "float32"
    layout = "NCHW"
    target_ops = [relay.nn.conv2d]

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
        assert expected_time == out_time, "Inferred layout transformation time mismatch for %s: " \
                                          "expecting %f but got %f" % (str(ltf_workload), expected_time,
                                                                       out_time)


def test_DPTuner_run():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    target = "llvm"
    dtype = "float32"
    layout = "NCHW"
    dshape = (1, 3, 8, 8)
    target_ops = [relay.nn.conv2d]

    g, records, ltf_records, ltf_keys, tasks = _create_data(target, dshape, dtype, layout)
    mod = relay.module.Module()
    mod["main"] = g
    costs = [0.02, 0.02, 0.045]
    config_list = []
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [1, 3]],
                      ["tile_oc", "sp", [2, 8]],
                      ["tile_ow", "sp", [4, 2]],
                      ["unroll_kw", "ot", True]],
                "t": ""}
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [4, 4]],
                      ["tile_oc", "sp", [2, 16]],
                      ["tile_oh", "ot", 1],
                      ["tile_ow", "sp", [4, 2]]],
                "t": ""}
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [16, 2]],
                      ["tile_oc", "sp", [8, 4]],
                      ["tile_ow", "sp", [2, 4]],
                      ["unroll_kw", "ot", False]],
                "t": ""}
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
    assert expected_out == out, "Output mismatch: expecting %s but got %s" \
                                % (str(expected_out), str(out))
    assert os.path.isfile(log_file), "No log file with name %s exists." % log_file


def test_PBQPTuner_run():
    target = "llvm"
    dtype = "float32"
    layout = "NCHW"
    dshape = (1, 3, 8, 8)
    target_ops = [relay.nn.conv2d]

    g, records, ltf_records, ltf_keys, tasks = _create_data(target, dshape, dtype, layout)
    costs = [0.02, 0.02, 0.045]
    config_list = []
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [1, 3]],
                      ["tile_oc", "sp", [2, 8]],
                      ["tile_ow", "sp", [4, 2]],
                      ["unroll_kw", "ot", True]],
                "t": ""}
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [4, 4]],
                      ["tile_oc", "sp", [2, 16]],
                      ["tile_oh", "ot", 1],
                      ["tile_ow", "sp", [4, 2]]],
                "t": ""}
    config_list.append(ConfigEntity.from_json_dict(cfg_dict))
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [16, 2]],
                      ["tile_oc", "sp", [8, 4]],
                      ["tile_ow", "sp", [2, 4]],
                      ["unroll_kw", "ot", False]],
                "t": ""}
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
    assert expected_out == out, "Output mismatch: expecting %s but got %s" \
                           % (str(expected_out), str(out))


if __name__=="__main__":
    test_graph_tuner_layout_transform()
    test_DPTuner_run()
    test_PBQPTuner_run()
