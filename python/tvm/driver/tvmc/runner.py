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
Provides support to run compiled networks both locally and remotely.
"""
import json
import logging
import os
import tarfile
import tempfile

import numpy as np
from tvm import rpc
from tvm.autotvm.measure import request_remote
from tvm.contrib import graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime
from tvm.relay import load_param_dict

from . import common
from .common import TVMCException
from .main import register_parser


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_run_parser(subparsers):
    """ Include parser for 'run' subcommand """

    parser = subparsers.add_parser("run", help="run a compiled module")
    parser.set_defaults(func=drive_run)

    # TODO --device needs to be extended and tested to support other targets,
    #      like 'webgpu', etc (@leandron)
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "cl"],
        default="cpu",
        help="target device to run the compiled module. Defaults to 'cpu'",
    )
    parser.add_argument(
        "--fill-mode",
        choices=["zeros", "ones", "random"],
        default="random",
        help="fill all input tensors with values. In case --inputs/-i is provided, "
        "they will take precedence over --fill-mode. Any remaining inputs will be "
        "filled using the chosen fill mode. Defaults to 'random'",
    )
    parser.add_argument("-i", "--inputs", help="path to the .npz input file")
    parser.add_argument("-o", "--outputs", help="path to the .npz output file")
    parser.add_argument(
        "--print-time", action="store_true", help="record and print the execution time(s)"
    )
    parser.add_argument(
        "--print-top",
        metavar="N",
        type=int,
        help="print the top n values and indices of the output tensor",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="generate profiling data from the runtime execution. "
        "Using --profile requires the Graph Runtime Debug enabled on TVM. "
        "Profiling may also have an impact on inference time, "
        "making it take longer to be generated.",
    )
    parser.add_argument(
        "--repeat", metavar="N", type=int, default=1, help="repeat the run n times. Defaults to '1'"
    )
    parser.add_argument(
        "--rpc-key",
        help="the RPC tracker key of the target device",
    )
    parser.add_argument(
        "--rpc-tracker",
        help="hostname (required) and port (optional, defaults to 9090) of the RPC tracker, "
        "e.g. '192.168.0.100:9999'",
    )
    parser.add_argument("FILE", help="path to the compiled module file")


def drive_run(args):
    """Invoke runner module with command line arguments

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.
    """

    rpc_hostname, rpc_port = common.tracker_host_port_from_cli(args.rpc_tracker)

    outputs, times = run_module(
        args.FILE,
        rpc_hostname,
        rpc_port,
        args.rpc_key,
        inputs_file=args.inputs,
        device=args.device,
        fill_mode=args.fill_mode,
        repeat=args.repeat,
        profile=args.profile,
    )

    if args.print_time:
        stat_table = format_times(times)
        # print here is intentional
        print(stat_table)

    if args.print_top:
        top_results = get_top_results(outputs, args.print_top)
        # print here is intentional
        print(top_results)

    if args.outputs:
        # Save the outputs
        np.savez(args.outputs, **outputs)


def get_input_info(graph_str, params):
    """Return the 'shape' and 'dtype' dictionaries for the input
    tensors of a compiled module.

    .. note::
        We can't simply get the input tensors from a TVM graph
        because weight tensors are treated equivalently. Therefore, to
        find the input tensors we look at the 'arg_nodes' in the graph
        (which are either weights or inputs) and check which ones don't
        appear in the params (where the weights are stored). These nodes
        are therefore inferred to be input tensors.

    Parameters
    ----------
    graph_str : str
        JSON graph of the module serialized as a string.
    params : bytearray
        Params serialized as a bytearray.

    Returns
    -------
    shape_dict : dict
        Shape dictionary - {input_name: tuple}.
    dtype_dict : dict
        dtype dictionary - {input_name: dtype}.
    """

    shape_dict = {}
    dtype_dict = {}
    params_dict = load_param_dict(params)
    param_names = [k for (k, v) in params_dict.items()]
    graph = json.loads(graph_str)
    for node_id in graph["arg_nodes"]:
        node = graph["nodes"][node_id]
        # If a node is not in the params, infer it to be an input node
        name = node["name"]
        if name not in param_names:
            shape_dict[name] = graph["attrs"]["shape"][1][node_id]
            dtype_dict[name] = graph["attrs"]["dltype"][1][node_id]

    logger.debug("collecting graph input shape and type:")
    logger.debug("graph input shape: %s", shape_dict)
    logger.debug("graph input type: %s", dtype_dict)

    return shape_dict, dtype_dict


def generate_tensor_data(shape, dtype, fill_mode):
    """Generate data to produce a tensor of given shape and dtype.

    Random data generation depends on the dtype. For int8 types,
    random integers in the range 0->255 are generated. For all other
    types, random floats are generated in the range -1->1 and then
    cast to the appropriate dtype.

    This is used to quickly generate some data to input the models, as
    a way to check that compiled module is sane for running.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor.
    dtype : str
        The dtype of the tensor.
    fill_mode : str
        The fill-mode to use, either "zeros", "ones" or "random".

    Returns
    -------
    tensor : np.array
        The generated tensor as a np.array.
    """
    if fill_mode == "zeros":
        tensor = np.zeros(shape=shape, dtype=dtype)
    elif fill_mode == "ones":
        tensor = np.ones(shape=shape, dtype=dtype)
    elif fill_mode == "random":
        if "int8" in dtype:
            tensor = np.random.randint(128, size=shape, dtype=dtype)
        else:
            tensor = np.random.uniform(-1, 1, size=shape).astype(dtype)
    else:
        raise TVMCException("unknown fill-mode: {}".format(fill_mode))

    return tensor


def make_inputs_dict(inputs_file, shape_dict, dtype_dict, fill_mode):
    """Make the inputs dictionary for a graph.

    Use data from 'inputs' where specified. For input tensors
    where no data has been given, generate data according to the
    chosen fill-mode.

    Parameters
    ----------
    inputs_file : str
        Path to a .npz file containing the inputs.
    shape_dict : dict
        Shape dictionary - {input_name: tuple}.
    dtype_dict : dict
        dtype dictionary - {input_name: dtype}.
    fill_mode : str
        The fill-mode to use when generating tensor data.
        Can be either "zeros", "ones" or "random".

    Returns
    -------
    inputs_dict : dict
        Complete inputs dictionary - {input_name: np.array}.
    """
    logger.debug("creating inputs dict")

    try:
        inputs = np.load(inputs_file) if inputs_file else {}
    except IOError as ex:
        raise TVMCException("Error loading inputs file: %s" % ex)

    # First check all the keys in inputs exist in the graph
    for input_name in inputs:
        if input_name not in shape_dict.keys():
            raise TVMCException(
                "the input tensor '{}' is not in the graph. Expected inputs: '{}'".format(
                    input_name, shape_dict.keys()
                )
            )

    # Now construct the input dict, generating tensors where no
    # data already exists in 'inputs'
    inputs_dict = {}
    for input_name in shape_dict:
        if input_name in inputs.keys():
            logger.debug("setting input '%s' with user input data", input_name)
            inputs_dict[input_name] = inputs[input_name]
        else:
            shape = shape_dict[input_name]
            dtype = dtype_dict[input_name]

            logger.debug(
                "generating data for input '%s' (shape: %s, dtype: %s), using fill-mode '%s'",
                input_name,
                shape,
                dtype,
                fill_mode,
            )
            data = generate_tensor_data(shape, dtype, fill_mode)
            inputs_dict[input_name] = data

    return inputs_dict


def run_module(
    module_file,
    hostname,
    port=9090,
    rpc_key=None,
    device=None,
    inputs_file=None,
    fill_mode="random",
    repeat=1,
    profile=False,
):
    """Run a compiled graph runtime module locally or remotely with
    optional input values.

    If input tensors are not specified explicitly, they can be filled
    with zeroes, ones or random data.

    Parameters
    ----------
    module_file : str
        The path to the module file (a .tar file).
    hostname : str
        The hostname of the target device on which to run.
    port : int, optional
        The port of the target device on which to run.
    rpc_key : str, optional
        The tracker key of the target device. If this is set, it
        will be assumed that remote points to a tracker.
    device: str, optional
        the device (e.g. "cpu" or "gpu") to be targeted by the RPC
        session, local or remote).
    inputs_file : str, optional
        Path to an .npz file containing the inputs.
    fill_mode : str, optional
        The fill-mode to use when generating data for input tensors.
        Valid options are "zeros", "ones" and "random".
        Defaults to "random".
    repeat : int, optional
        How many times to repeat the run.
    profile : bool
        Whether to profile the run with the debug runtime.

    Returns
    -------
    outputs : dict
        a dictionary with output tensors, generated by the module
    times : list of str
        execution times generated by the time evaluator
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.debug("extracting module file %s", module_file)
        t = tarfile.open(module_file)
        t.extractall(tmp_dir)
        graph = open(os.path.join(tmp_dir, "mod.json")).read()
        params = bytearray(open(os.path.join(tmp_dir, "mod.params"), "rb").read())

        if hostname:
            # Remote RPC
            if rpc_key:
                logger.debug("running on remote RPC tracker with key %s", rpc_key)
                session = request_remote(rpc_key, hostname, port, timeout=1000)
            else:
                logger.debug("running on remote RPC with no key")
                session = rpc.connect(hostname, port)
        else:
            # Local
            logger.debug("running a local session")
            session = rpc.LocalSession()

        session.upload(os.path.join(tmp_dir, "mod.so"))
        lib = session.load_module("mod.so")

        # TODO expand to other supported devices, as listed in tvm.rpc.client (@leandron)
        logger.debug("device is %s", device)
        if device == "gpu":
            ctx = session.gpu()
        elif device == "cl":
            ctx = session.cl()
        else:
            assert device == "cpu"
            ctx = session.cpu()

        if profile:
            logger.debug("creating runtime with profiling enabled")
            module = debug_runtime.create(graph, lib, ctx, dump_root="./prof")
        else:
            logger.debug("creating runtime with profiling disabled")
            module = runtime.create(graph, lib, ctx)

        logger.debug("load params into the runtime module")
        module.load_params(params)

        shape_dict, dtype_dict = get_input_info(graph, params)
        inputs_dict = make_inputs_dict(inputs_file, shape_dict, dtype_dict, fill_mode)

        logger.debug("setting inputs to the module")
        module.set_input(**inputs_dict)

        # Run must be called explicitly if profiling
        if profile:
            logger.debug("running the module with profiling enabled")
            module.run()

        # create the module time evaluator (returns a function)
        timer = module.module.time_evaluator("run", ctx, 1, repeat=repeat)
        # call the evaluator function to invoke the module and save execution times
        prof_result = timer()
        # collect a list of execution times from the profiling results
        times = prof_result.results

        logger.debug("collecting the output tensors")
        num_outputs = module.get_num_outputs()
        outputs = {}
        for i in range(num_outputs):
            output_name = "output_{}".format(i)
            outputs[output_name] = module.get_output(i).asnumpy()

        return outputs, times


def get_top_results(outputs, max_results):
    """Return the top n results from the output tensor.

    This function is primarily for image classification and will
    not necessarily generalise.

    Parameters
    ----------
    outputs : dict
        Outputs dictionary - {output_name: np.array}.
    max_results : int
        Number of results to return

    Returns
    -------
    top_results : np.array
        Results array of shape (2, n).
        The first row is the indices and the second is the values.

    """
    output = np.copy(outputs["output_0"])
    sorted_labels = output.argsort()[0][-max_results:][::-1]
    output.sort()
    sorted_values = output[0][-max_results:][::-1]
    top_results = np.array([sorted_labels, sorted_values])
    return top_results


def format_times(times):
    """Format the mean, max, min and std of the execution times.

    This has the effect of producing a small table that looks like:

        Execution time summary:
        mean (s)   max (s)    min (s)    std (s)
        0.14310    0.16161    0.12933    0.01004

    Parameters
    ----------
    times : list
        A list of execution times (in seconds).

    Returns
    -------
    str
        A formatted string containing the statistics.
    """

    # timestamps
    mean_ts = np.mean(times)
    std_ts = np.std(times)
    max_ts = np.max(times)
    min_ts = np.min(times)

    header = "Execution time summary:\n{0:^10} {1:^10} {2:^10} {3:^10}".format(
        "mean (s)", "max (s)", "min (s)", "std (s)"
    )
    stats = "{0:^10.5f} {1:^10.5f} {2:^10.5f} {3:^10.5f}".format(mean_ts, max_ts, min_ts, std_ts)
    return "%s\n%s\n" % (header, stats)
