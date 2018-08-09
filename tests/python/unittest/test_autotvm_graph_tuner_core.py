import os
import nnvm

from nnvm import symbol as sym
from topi.nn.conv2d import Workload
from topi.x86.conv2d_avx_common import AVXConvCommonFwd
from topi.x86.conv2d_avx_1x1 import AVXConv1x1Fwd
from tvm import rpc
from tvm.autotvm.graph_tuner import tensor_executor
from tvm.autotvm.graph_tuner import graph_executor
from tvm.autotvm.graph_tuner.utils import get_conv2d_workload, infer_layout_shape_avx


def test_Conv2dAVXExecutor_parameter_exec():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    target = "llvm"
    param_list = [{"channels": 3, "kernel_size": (3, 3), "layout": "NCHW3c",
                   "out_layout": "NCHW3c", "kernel_layout": "OIHW3i3o"},
                  {"channels": 64, "kernel_size": (5, 2), "layout": "NCHW8c",
                   "out_layout": "NCHW8c", "kernel_layout": "OIHW8i8o"}]
    input_shape_list = [{"data": (1, 1, 224, 224, 3)}, {"data": (1, 8, 56, 56, 8)}]
    executor = tensor_executor.Conv2dAVXExecutor(target=target, log_file=log_file, verbose=False)
    exec_time_list = executor.parameter_execute(param_list, input_shape_list)
    if len(exec_time_list) != len(param_list):
        raise RuntimeError("Length of returned execution time list is not equal to "
                           "length of parameter: %d vs %d." % (len(exec_time_list), len(param_list)))
    if not os.path.isfile(log_file):
        raise RuntimeError("No log file with name %s exists." % log_file)
    os.remove(log_file)


def test_Conv2dAVXExecutor_workload_exec():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    target = "llvm"
    workload_list = [Workload('float32', 'float32', 4, 4, 3, 4, 2, 2, 0, 0, 1, 1),
                     Workload('float32', 'float32', 2, 2, 3, 4, 2, 2, 0, 0, 1, 1)]
    expected_num_schedules = [2 * 3, 2 * 3]
    executor = tensor_executor.Conv2dAVXExecutor(target=target, log_file=log_file, verbose=False)
    executor.workload_execute(workload_list)
    sch_dict = executor.schedule_dict
    for wkl, num in zip(workload_list, expected_num_schedules):
        if len(sch_dict[wkl]) != num:
            raise RuntimeError("Number of schedules for workload %s incorrect. Expecting %d but got %d."
                               % (str(wkl), num, len(sch_dict[wkl])))
    if not os.path.isfile(log_file):
        raise RuntimeError("No log file with name %s exists." % log_file)
    os.remove(log_file)

    rpc_hosts = ["localhost", "localhost"]
    rpc_ports = [9090, 9091]
    servers = [rpc.Server(host, port) for host, port in zip(rpc_hosts, rpc_ports)]
    executor = tensor_executor.Conv2dAVXExecutor(target=target, log_file=log_file, verbose=False,
                                                 rpc_mode=1, rpc_hosts=rpc_hosts, rpc_ports=rpc_ports)
    executor.workload_execute(workload_list)
    sch_dict = executor.schedule_dict
    for server in servers:
        server.terminate()
    for wkl, num in zip(workload_list, expected_num_schedules):
        if len(sch_dict[wkl]) != num:
            raise RuntimeError("Number of schedules for workload %s incorrect. Expecting %d but got %d."
                               % (str(wkl), num, len(sch_dict[wkl])))
    if not os.path.isfile(log_file):
        raise RuntimeError("No log file with name %s exists." % log_file)
    os.remove(log_file)


def test_graph_executor_layout_transform():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    data = sym.Variable("data")
    conv1 = sym.conv2d(data, channels=16, kernel_size=(3, 3), padding=(1, 1))
    conv2 = sym.conv2d(conv1, channels=32, kernel_size=(1, 1))
    out = sym.conv2d(conv2, channels=8, kernel_size=(3, 3), padding=(1, 1))
    g = nnvm.graph.create(out)
    schedule_dict = {
        Workload('float32', 'float32', 8, 8, 3, 16, 3, 3, 1, 1, 1, 1):
            [{"schedule": AVXConvCommonFwd(1, 4, 2, True), "time": 0.04}],
        Workload('float32', 'float32', 8, 8, 16, 32, 1, 1, 0, 0, 1, 1):
            [{"schedule": AVXConv1x1Fwd(8, 32, 1, 2), "time": 0.012}],
        Workload('float32', 'float32', 8, 8, 32, 8, 3, 3, 1, 1, 1, 1):
            [{"schedule": AVXConvCommonFwd(4, 8, 4, False), "time": 0.03}]
    }
    dshape = (1, 3, 8, 8)
    executor = graph_executor.DPExecutor(g, {"data": dshape}, schedule_dict, target_op="conv2d",
                                         get_wkl_func=get_conv2d_workload,
                                         infer_layout_shape_func=infer_layout_shape_avx,
                                         data_layout="NCHWc", verbose=False, log_file=log_file)
    executor.benchmark_layout_transform()
    out = executor._global_data_dict["layout_time_dict"]
    str_key = ["((1, 4, 8, 8, 4), (1, 2, 8, 8, 8))", "((1, 1, 8, 8, 32), (1, 8, 8, 8, 4))"]
    for key in str_key:
        if key not in out:
            raise RuntimeError("%s not in output dictionary." % key)
    if not os.path.isfile(log_file):
        raise RuntimeError("No log file with name %s exists." % log_file)
    os.remove(log_file)


def test_DPExecutor_run():
    log_file = "%s/test_tuner.log" % (os.getcwd())
    data = sym.Variable("data")
    conv1 = sym.conv2d(data, channels=16, kernel_size=(3, 3), padding=(1, 1))
    conv2 = sym.conv2d(conv1, channels=32, kernel_size=(3, 3), padding=(1, 1))
    conv3 = sym.conv2d(conv1, channels=32, kernel_size=(1, 1))
    net = sym.elemwise_add(conv2, conv3)
    g = nnvm.graph.create(net)
    dshape = (1, 3, 8, 8)
    sch_dict = {
        Workload('float32', 'float32', 8, 8, 3, 16, 3, 3, 1, 1, 1, 1):
            [{"schedule": AVXConvCommonFwd(3, 2, 1, True), "time": 1.3},
             {"schedule": AVXConvCommonFwd(1, 4, 2, True), "time": 1.5}],
        Workload('float32', 'float32', 8, 8, 16, 32, 3, 3, 1, 1, 1, 1):
            [{"schedule": AVXConvCommonFwd(2, 8, 4, False), "time": 2.4},
             {"schedule": AVXConvCommonFwd(16, 2, 8, False), "time": 2.9}],
        Workload('float32', 'float32', 8, 8, 16, 32, 1, 1, 0, 0, 1, 1):
            [{"schedule": AVXConv1x1Fwd(4, 16, 2, 4), "time": 0.2},
             {"schedule": AVXConv1x1Fwd(8, 32, 1, 2), "time": 0.3}]
    }
    layout_time_dict = {
        "((1, 4, 8, 8, 4), (1, 2, 8, 8, 8))": 0.5,
        "((1, 4, 8, 8, 4), (1, 4, 8, 8, 4))": 0.25,
        "((1, 8, 8, 8, 2), (1, 2, 8, 8, 8))": 0.6,
        "((1, 8, 8, 8, 2), (1, 4, 8, 8, 4))": 0.75,
        "((1, 4, 8, 8, 4), (1, 8, 8, 8, 2))": 0.45,
        "((1, 4, 8, 8, 4), (1, 1, 8, 8, 16))": 0.15,
        "((1, 8, 8, 8, 2), (1, 8, 8, 8, 2))": 0.05,
        "((1, 8, 8, 8, 2), (1, 1, 8, 8, 16))": 0.25,
        "((1, 1, 8, 8, 32), (1, 4, 8, 8, 8))": 0.95,
        "((1, 1, 8, 8, 32), (1, 16, 8, 8, 2))": 0.75,
        "((1, 2, 8, 8, 16), (1, 4, 8, 8, 8))": 0.65,
        "((1, 2, 8, 8, 16), (1, 16, 8, 8, 2))": 0.55,
    }
    expected_out = [
        AVXConvCommonFwd(3, 2, 1, True),
        AVXConvCommonFwd(2, 8, 4, False),
        AVXConv1x1Fwd(4, 16, 2, 4),
    ]

    executor = graph_executor.DPExecutor(g, {"data": dshape}, sch_dict, target_op="conv2d",
                                         get_wkl_func=get_conv2d_workload,
                                         infer_layout_shape_func=infer_layout_shape_avx,
                                         data_layout="NCHWc", verbose=False, log_file=log_file)
    executor._global_data_dict["layout_time_dict"] = layout_time_dict
    executor.run()
    out = executor.get_optimal_schedules()
    if expected_out != out:
        raise RuntimeError("Output mismatch: expecting %s but got %s"
                           % (str(expected_out), str(out)))
    if not os.path.isfile(log_file):
        raise RuntimeError("No log file with name %s exists." % log_file)
    os.remove(log_file)


if __name__=="__main__":
    test_Conv2dAVXExecutor_parameter_exec()
    test_Conv2dAVXExecutor_workload_exec()
    test_graph_executor_layout_transform()
    test_DPExecutor_run()
