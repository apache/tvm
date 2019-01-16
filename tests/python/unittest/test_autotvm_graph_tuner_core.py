import os
import copy
import nnvm
import tvm

from nnvm import symbol as sym
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity
from tvm.autotvm.measure import MeasureResult, MeasureInput
from tvm.autotvm.graph_tuner import DPTuner
from tvm.autotvm.graph_tuner.utils import nnvm_get_conv2d_NCHWc_AVX_workload, infer_conv2d_layout_shape_avx
from test_autotvm_graph_tuner_utils import create_workload


def _create_data(target, dshape, dtype, layout):
    data = sym.Variable("data")
    conv0 = sym.conv2d(data, channels=16, kernel_size=(3, 3), padding=(1, 1))
    conv1 = sym.conv2d(conv0, channels=32, kernel_size=(1, 1))
    conv2 = sym.conv2d(conv1, channels=32, kernel_size=(3, 3), padding=(1, 1))
    out = sym.elemwise_add(conv1, conv2)
    tasks = autotvm.task.extract_from_graph(out, target=target,
                                            shape={'data': dshape}, dtype=dtype,
                                            symbols=(sym.conv2d,))
    g = nnvm.graph.create(out)
    wkl_list = [
        create_workload((1, 3, 8, 8), (16, 3, 3, 3), (1, 1), (1, 1), (1, 1),layout, layout, dtype, dtype),
        create_workload((1, 16, 8, 8), (32, 16, 1, 1), (1, 1), (0, 0), (1, 1),layout, layout, dtype, dtype),
        create_workload((1, 32, 8, 8), (32, 32, 3, 3), (1, 1), (1, 1), (1, 1),layout, layout, dtype, dtype),
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
    ltf_arg = [tvm.placeholder((1, 64, 16, 16, 8), dtype=dtype), "NCHW8c", "NCHW512c", (1, 1, 16, 16, 512),
               "layout_transform", "injective"]
    ltf_arg = autotvm.task.topi_integration.serialize_args(ltf_arg)
    ltf_wkl = ('layout_transform',) + autotvm.task.args_to_workload(ltf_arg)
    ltf_task = copy.deepcopy(tasks[0])
    ltf_task.workload = ltf_wkl
    ms_input = MeasureInput(target=target, task=ltf_task, config=None)
    ms_output =  MeasureResult(costs=(1.91224744e-05,), error_no=0, all_cost=-1, timestamp=-1)
    ltf_records.append((ms_input, ms_output))

    ltf_keys = []
    ltf_arg = [tvm.placeholder((1, 4, 8, 8, 4), dtype=dtype), "NCHW4c", "NCHW8c", (1, 2, 8, 8, 8),
               "layout_transform", "injective"]
    ltf_arg = autotvm.task.topi_integration.serialize_args(ltf_arg)
    ltf_wkl = ('layout_transform',) + autotvm.task.args_to_workload(ltf_arg)
    ltf_keys.append(ltf_wkl)
    ltf_arg = [tvm.placeholder((1, 1, 8, 8, 32), dtype=dtype), "NCHW32c", "NCHW4c", (1, 8, 8, 8, 4),
               "layout_transform", "injective"]
    ltf_arg = autotvm.task.topi_integration.serialize_args(ltf_arg)
    ltf_wkl = ('layout_transform',) + autotvm.task.args_to_workload(ltf_arg)
    ltf_keys.append(ltf_wkl)
    ltf_arg = [tvm.placeholder((1, 4, 8, 8, 8), dtype=dtype), "NCHW8c", "NCHW32c", (1, 1, 8, 8, 32),
               "layout_transform", "injective"]
    ltf_arg = autotvm.task.topi_integration.serialize_args(ltf_arg)
    ltf_wkl = ('layout_transform',) + autotvm.task.args_to_workload(ltf_arg)
    ltf_keys.append(ltf_wkl)

    return g, records, ltf_records, ltf_keys, tasks


def test_graph_tuner_layout_transform():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    target = "llvm"
    dshape = (1, 3, 8, 8)
    dtype = "float32"
    layout = "NCHW"
    target_op = "conv2d"
    data_layout = "NCHWc"

    g, records, ltf_records, ltf_keys, _ = _create_data(target, dshape, dtype, layout)
    graph_wkl_list = nnvm_get_conv2d_NCHWc_AVX_workload(g, {"data": dshape}, unique_wkl=False)
    executor = DPTuner(g, {"data": dshape}, records, graph_wkl_list, target_op, data_layout,
                       ("tile_ic", "tile_oc"), infer_conv2d_layout_shape_avx, log_file=log_file)
    executor.benchmark_layout_transform(target, records=ltf_records, infer_layout=True)
    out = executor._layout_transform_dict

    for key in ltf_keys:
        if key not in out:
            raise RuntimeError("%s not in output dictionary." % str(key))
    if not os.path.isfile(log_file):
        raise RuntimeError("No log file with name %s exists." % log_file)


def test_DPTuner_run():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    target = "llvm"
    dtype = "float32"
    layout = "NCHW"
    dshape = (1, 3, 8, 8)
    target_op = "conv2d"
    data_layout = "NCHWc"

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

    graph_wkl_list = nnvm_get_conv2d_NCHWc_AVX_workload(g, {"data": dshape}, unique_wkl=False)
    executor = DPTuner(g, {"data": dshape}, records, graph_wkl_list, target_op, data_layout,
                       ("tile_ic", "tile_oc"), infer_conv2d_layout_shape_avx, log_file=log_file)
    executor.benchmark_layout_transform(target, records=ltf_records, infer_layout=True)
    executor.run()
    out = executor.get_optimal_schedules()
    expected_out = [records[3][0].config, records[1][0].config, records[2][0].config]
    if expected_out != out:
        raise RuntimeError("Output mismatch: expecting %s but got %s"
                           % (str(expected_out), str(out)))
    if not os.path.isfile(log_file):
        raise RuntimeError("No log file with name %s exists." % log_file)
    os.remove(log_file)


if __name__=="__main__":
    test_graph_tuner_layout_transform()
    test_DPTuner_run()