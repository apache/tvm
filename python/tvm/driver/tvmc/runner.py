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
from typing import Dict, List, Optional, Union
from tarfile import ReadError

import numpy as np
import tvm
from tvm import rpc
from tvm.autotvm.measure import request_remote
from tvm.contrib import graph_executor as runtime
from tvm.contrib.debugger import debug_executor
from tvm.relay.param_dict import load_param_dict

from . import common
from .common import TVMCException
from .main import register_parser
from .model import TVMCPackage, TVMCResult
from .result_utils import get_top_results

# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_run_parser(subparsers):
    """Include parser for 'run' subcommand"""

    parser = subparsers.add_parser("run", help="run a compiled module")
    parser.set_defaults(func=drive_run)

    # TODO --device needs to be extended and tested to support other targets,
    #      like 'webgpu', etc (@leandron)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "cl", "metal", "vulkan", "rocm"],
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
        "Using --profile requires the Graph Executor Debug enabled on TVM. "
        "Profiling may also have an impact on inference time, "
        "making it take longer to be generated.",
    )
    parser.add_argument(
        "--repeat", metavar="N", type=int, default=1, help="run the model n times. Defaults to '1'"
    )
    parser.add_argument(
        "--number", metavar="N", type=int, default=1, help="repeat the run n times. Defaults to '1'"
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

    try:
        inputs = np.load(args.inputs) if args.inputs else {}
    except IOError as ex:
        raise TVMCException("Error loading inputs file: %s" % ex)

    try:
        tvmc_package = TVMCPackage(package_path=args.FILE)
    except IsADirectoryError:
        raise TVMCException(f"File {args.FILE} must be an archive, not a directory.")
    except FileNotFoundError:
        raise TVMCException(f"File {args.FILE} does not exist.")
    except ReadError:
        raise TVMCException(f"Could not read model from archive {args.FILE}!")

    result = run_module(
        tvmc_package,
        args.device,
        hostname=rpc_hostname,
        port=rpc_port,
        rpc_key=args.rpc_key,
        inputs=inputs,
        fill_mode=args.fill_mode,
        repeat=args.repeat,
        number=args.number,
        profile=args.profile,
    )

    if args.print_time:
        stat_table = result.format_times()
        # print here is intentional
        print(stat_table)

    if args.print_top:
        top_results = get_top_results(result, args.print_top)
        # print here is intentional
        print(top_results)

    if args.outputs:
        # Save the outputs
        result.save(args.outputs)


def get_input_info(graph_str: str, params: Dict[str, tvm.nd.NDArray]):
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
    params : dict
        Parameter dictionary mapping name to value.

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

    logger.debug("Collecting graph input shape and type:")
    logger.debug("Graph input shape: %s", shape_dict)
    logger.debug("Graph input type: %s", dtype_dict)

    return shape_dict, dtype_dict


def generate_tensor_data(shape: tuple, dtype: str, fill_mode: str):
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


def make_inputs_dict(
    shape_dict: Dict[str, List[int]],
    dtype_dict: Dict[str, str],
    inputs: Optional[Dict[str, np.ndarray]] = None,
    fill_mode: str = "random",
):
    """Make the inputs dictionary for a graph.

    Use data from 'inputs' where specified. For input tensors
    where no data has been given, generate data according to the
    chosen fill-mode.

    Parameters
    ----------
    shape_dict : dict
        Shape dictionary - {input_name: tuple}.
    dtype_dict : dict
        dtype dictionary - {input_name: dtype}.
    inputs : dict, optional
        A dictionary that maps input names to numpy values.
    fill_mode : str, optional
        The fill-mode to use when generating tensor data.
        Can be either "zeros", "ones" or "random".

    Returns
    -------
    inputs_dict : dict
        Complete inputs dictionary - {input_name: np.array}.
    """
    logger.debug("creating inputs dict")

    if inputs is None:
        inputs = {}

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
    tvmc_package: TVMCPackage,
    device: str,
    hostname: Optional[str] = None,
    port: Union[int, str] = 9090,
    rpc_key: Optional[str] = None,
    inputs: Optional[Dict[str, np.ndarray]] = None,
    fill_mode: str = "random",
    repeat: int = 10,
    number: int = 10,
    profile: bool = False,
):
    """Run a compiled graph executor module locally or remotely with
    optional input values.

    If input tensors are not specified explicitly, they can be filled
    with zeroes, ones or random data.

    Parameters
    ----------
    tvmc_package: TVMCPackage
        The compiled model package object that will be run.
    device: str,
        the device (e.g. "cpu" or "cuda") to be targeted by the RPC
        session, local or remote).
    hostname : str, optional
        The hostname of the target device on which to run.
    port : int, optional
        The port of the target device on which to run.
    rpc_key : str, optional
        The tracker key of the target device. If this is set, it
        will be assumed that remote points to a tracker.
    inputs : dict, optional
        A dictionary that maps input names to numpy values. If not provided,
        inputs will be generated using the fill_mode argument.
    fill_mode : str, optional
        The fill-mode to use when generating data for input tensors.
        Valid options are "zeros", "ones" and "random".
        Defaults to "random".
    repeat : int, optional
        How many times to repeat the run.
    number : int, optional
        The number of runs to measure within each repeat.
    profile : bool
        Whether to profile the run with the debug runtime.

    Returns
    -------
    outputs : dict
        a dictionary with output tensors, generated by the module
    times : list of str
        execution times generated by the time evaluator
    """
    if not isinstance(tvmc_package, TVMCPackage):
        raise TVMCException(
            "This model doesn't seem to have been compiled yet. "
            "Try calling tvmc.compile on the model before running it."
        )

    # Currently only two package formats are supported: "classic" and
    # "mlf". The later can only be used for micro targets, i.e. with microTVM.
    if tvmc_package.type == "mlf":
        raise TVMCException(
            "You're trying to run a model saved using the Model Library Format (MLF)."
            "MLF can only be used to run micro targets (microTVM)."
        )

    if hostname:
        if isinstance(port, str):
            port = int(port)
        # Remote RPC
        if rpc_key:
            logger.debug("Running on remote RPC tracker with key %s.", rpc_key)
            session = request_remote(rpc_key, hostname, port, timeout=1000)
        else:
            logger.debug("Running on remote RPC with no key.")
            session = rpc.connect(hostname, port)
    else:
        # Local
        logger.debug("Running a local session.")
        session = rpc.LocalSession()

    session.upload(tvmc_package.lib_path)
    lib = session.load_module(tvmc_package.lib_name)

    # TODO expand to other supported devices, as listed in tvm.rpc.client (@leandron)
    logger.debug("Device is %s.", device)
    if device == "cuda":
        dev = session.cuda()
    elif device == "cl":
        dev = session.cl()
    elif device == "metal":
        dev = session.metal()
    elif device == "vulkan":
        dev = session.vulkan()
    elif device == "rocm":
        dev = session.rocm()
    else:
        assert device == "cpu"
        dev = session.cpu()

    if profile:
        logger.debug("Creating runtime with profiling enabled.")
        module = debug_executor.create(tvmc_package.graph, lib, dev, dump_root="./prof")
    else:
        logger.debug("Creating runtime with profiling disabled.")
        module = runtime.create(tvmc_package.graph, lib, dev)

    logger.debug("Loading params into the runtime module.")
    module.load_params(tvmc_package.params)

    shape_dict, dtype_dict = get_input_info(tvmc_package.graph, tvmc_package.params)
    inputs_dict = make_inputs_dict(shape_dict, dtype_dict, inputs, fill_mode)

    logger.debug("Setting inputs to the module.")
    module.set_input(**inputs_dict)

    # Run must be called explicitly if profiling
    if profile:
        logger.info("Running the module with profiling enabled.")
        report = module.profile()
        # This print is intentional
        print(report)

    # call the benchmarking function of the executor
    times = module.benchmark(dev, number=number, repeat=repeat)

    logger.debug("Collecting the output tensors.")
    num_outputs = module.get_num_outputs()
    outputs = {}
    for i in range(num_outputs):
        output_name = "output_{}".format(i)
        outputs[output_name] = module.get_output(i).numpy()

    return TVMCResult(outputs, times)
