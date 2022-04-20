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
from contextlib import ExitStack
import logging
import pathlib
from typing import Dict, Optional, Union
from tarfile import ReadError
import argparse
import sys
import numpy as np

import tvm
from tvm import rpc
from tvm.runtime import vm
from tvm.autotvm.measure import request_remote
from tvm.contrib import graph_executor as executor
from tvm.contrib.debugger import debug_executor
from tvm.runtime import profiler_vm
from . import TVMCException
from .arguments import TVMCSuppressedArgumentParser
from .project import (
    get_project_options,
    get_and_check_options,
    get_project_dir,
)

from .main import register_parser
from .model import TVMCPackage, TVMCResult
from .result_utils import get_top_results
from .tracker import tracker_host_port_from_cli

try:
    import tvm.micro.project as project
    from tvm.micro.project import TemplateProjectError
    from tvm.micro.project_api.client import ProjectAPIServerNotFoundError

    SUPPORT_MICRO = True
except (ImportError, AttributeError) as exception:
    SUPPORT_MICRO = False

# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_run_parser(subparsers, main_parser, json_params):
    """Include parser for 'run' subcommand"""

    # Use conflict_handler='resolve' to allow '--list-options' option to be properly overriden when
    # augmenting the parser with the micro device options (i.e. when '--device micro').
    parser = subparsers.add_parser("run", help="run a compiled module", conflict_handler="resolve")
    parser.set_defaults(func=drive_run)

    # TODO --device needs to be extended and tested to support other targets,
    #      like 'webgpu', etc (@leandron)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "cl", "metal", "vulkan", "rocm", "micro"],
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
        "--print-time",
        action="store_true",
        help="record and print the execution time(s). (non-micro devices only)",
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
        "making it take longer to be generated. (non-micro devices only)",
    )
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="Measure data transfers as well as model execution. This can provide a "
        "more realistic performance measurement in many cases.",
    )
    parser.add_argument(
        "--repeat", metavar="N", type=int, default=1, help="run the model n times. Defaults to '1'"
    )
    parser.add_argument(
        "--number", metavar="N", type=int, default=1, help="repeat the run n times. Defaults to '1'"
    )
    parser.add_argument(
        "--rpc-key",
        help="the RPC tracker key of the target device. (non-micro devices only)",
    )
    parser.add_argument(
        "--rpc-tracker",
        help="hostname (required) and port (optional, defaults to 9090) of the RPC tracker, "
        "e.g. '192.168.0.100:9999'. (non-micro devices only)",
    )
    parser.add_argument(
        "PATH",
        help="path to the compiled module file or to the project directory if '--device micro' "
        "is selected.",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="show all run options and option choices when '--device micro' is selected. "
        "(micro devices only)",
    )

    disposable_parser = TVMCSuppressedArgumentParser(main_parser)
    try:
        known_args, _ = disposable_parser.parse_known_args()
    except TVMCException:
        return

    if vars(known_args).get("device") != "micro":
        # No need to augment the parser for micro targets.
        return

    if SUPPORT_MICRO is False:
        sys.exit(
            "'--device micro' is not supported. "
            "Please build TVM with micro support (USE_MICRO ON)!"
        )

    project_dir = get_project_dir(known_args.PATH)

    try:
        project_ = project.GeneratedProject.from_directory(project_dir, None)
    except ProjectAPIServerNotFoundError:
        sys.exit(f"Error: Project API server not found in {project_dir}!")
    except TemplateProjectError:
        sys.exit(
            f"Error: Project directory error. That usually happens when model.tar is not found."
        )

    project_info = project_.info()
    options_by_method = get_project_options(project_info)
    mlf_path = project_info["model_library_format_path"]

    parser.formatter_class = (
        argparse.RawTextHelpFormatter
    )  # Set raw help text so customized help_text format works

    parser.set_defaults(valid_options=options_by_method["open_transport"], mlf_path=mlf_path)

    required = any([opt["required"] for opt in options_by_method["open_transport"]])
    nargs = "+" if required else "*"

    help_text_by_option = [opt["help_text"] for opt in options_by_method["open_transport"]]
    help_text = "\n\n".join(help_text_by_option) + "\n\n"

    parser.add_argument(
        "--project-option", required=required, metavar="OPTION=VALUE", nargs=nargs, help=help_text
    )

    parser.add_argument(
        "--list-options",
        action="help",
        help="show this help message with platform-specific options and exit.",
    )

    for one_entry in json_params:
        parser.set_defaults(**one_entry)


def drive_run(args):
    """Invoke runner module with command line arguments

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.
    """

    path = pathlib.Path(args.PATH)
    options = None
    project_dir = None
    if args.device == "micro":
        # If it's a micro device, then grab the model.tar path from Project API instead.
        # args.PATH will be used too since it points to the project directory. N.B.: there is no
        # way to determine the model.tar path from the project dir or vice-verse (each platform
        # is free to put model.tar whereever it's convenient).
        project_dir = path
        path = pathlib.Path(args.mlf_path)

        # Check for options unavailable for micro targets.

        if args.rpc_key or args.rpc_tracker:
            raise TVMCException(
                "--rpc-key and/or --rpc-tracker can't be specified for micro targets."
            )

        if args.device != "micro":
            raise TVMCException(
                f"Device '{args.device}' not supported. "
                "Only device 'micro' is supported to run a model in MLF, "
                "i.e. when '--device micro'."
            )

        if args.profile:
            raise TVMCException("--profile is not currently supported for micro devices.")

        if args.print_time:
            raise TVMCException("--print-time is not currently supported for micro devices.")

        # Get and check options for micro targets.
        options = get_and_check_options(args.project_option, args.valid_options)

    else:
        # Check for options only availabe for micro targets.

        if args.list_options:
            raise TVMCException(
                "--list-options is only availabe on micro targets, i.e. when '--device micro'."
            )

    try:
        tvmc_package = TVMCPackage(package_path=path, project_dir=project_dir)
    except IsADirectoryError:
        raise TVMCException(f"File {path} must be an archive, not a directory.")
    except FileNotFoundError:
        raise TVMCException(f"File {path} does not exist.")
    except ReadError:
        raise TVMCException(f"Could not read model from archive {path}!")

    rpc_hostname, rpc_port = tracker_host_port_from_cli(args.rpc_tracker)

    try:
        inputs = np.load(args.inputs) if args.inputs else {}
    except IOError as ex:
        raise TVMCException("Error loading inputs file: %s" % ex)

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
        end_to_end=args.end_to_end,
        options=options,
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
    shape_dict: tvm.container.Map,
    dtype_dict: tvm.container.Map,
    inputs: Optional[Dict[str, np.ndarray]] = None,
    fill_mode: str = "random",
):
    """Make the inputs dictionary for a graph.

    Use data from 'inputs' where specified. For input tensors
    where no data has been given, generate data according to the
    chosen fill-mode.

    Parameters
    ----------
    shape_dict : Map
        Shape dictionary - {input_name: tuple}.
    dtype_dict : Map
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
                    input_name, list(shape_dict.keys())
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
            # container.ShapleTuple -> tuple
            shape = tuple(shape_dict[input_name])
            # container.String -> str
            dtype = str(dtype_dict[input_name])

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
    end_to_end: bool = False,
    options: dict = None,
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
        Whether to profile the run with the debug executor.
    end_to_end : bool
        Whether to measure the time of memory copies as well as model
        execution. Turning this on can provide a more realistic estimate
        of how long running the model in production would take.

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

    with ExitStack() as stack:
        # Currently only two package formats are supported: "classic" and
        # "mlf". The later can only be used for micro targets, i.e. with microTVM.
        if device == "micro":
            if tvmc_package.type != "mlf":
                raise TVMCException(f"Model {tvmc_package.package_path} is not a MLF archive.")

            project_dir = get_project_dir(tvmc_package.project_dir)

            # This is guaranteed to work since project_dir was already checked when
            # building the dynamic parser to accommodate the project options, so no
            # checks are in place when calling GeneratedProject.
            project_ = project.GeneratedProject.from_directory(project_dir, options)
        else:
            if tvmc_package.type == "mlf":
                raise TVMCException(
                    "You're trying to run a model saved using the Model Library Format (MLF). "
                    "MLF can only be used to run micro device ('--device micro')."
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
        elif device == "micro":
            # Remote RPC (running on a micro target)
            logger.debug("Running on remote RPC (micro target).")
            try:
                session = tvm.micro.Session(project_.transport())
                stack.enter_context(session)
            except:
                raise TVMCException("Could not open a session with the micro target.")
        else:
            # Local
            logger.debug("Running a local session.")
            session = rpc.LocalSession()

        # Micro targets don't support uploading a model. The model to be run
        # must be already flashed into the micro target before one tries
        # to run it. Hence skip model upload for micro targets.
        if device != "micro":
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
        elif device == "micro":
            dev = session.device
            lib = session.get_system_lib()
        else:
            assert device == "cpu"
            dev = session.cpu()

        if tvmc_package.type == "vm":
            assert inputs is not None, "vm runner requires inputs to be provided as a dict"

            input_tensor = {}
            for e, i in inputs.items():
                input_tensor[e] = tvm.nd.array(i, dev)

            if profile:
                logger.debug("Creating vm with profile enabled.")
                exe = profiler_vm.VirtualMachineProfiler(lib, dev)
                res = exe.profile(**input_tensor, func_name="main")
                # This print is intentional
                print(res)
            else:
                exe = vm.VirtualMachine(lib, dev)

            exe_outputs = exe.invoke("main", **input_tensor)
            times = exe.benchmark(
                dev,
                **input_tensor,
                func_name="main",
                repeat=repeat,
                number=number,
                end_to_end=end_to_end,
            )

            # Special handling if the output only has a single value
            if not isinstance(exe_outputs, list):
                exe_outputs = [exe_outputs]

            outputs = {}
            for i, val in enumerate(exe_outputs):
                output_name = "output_{}".format(i)
                outputs[output_name] = val.numpy()
        else:
            # TODO(gromero): Adjust for micro targets.
            if profile:
                logger.debug("Creating runtime with profiling enabled.")
                module = debug_executor.create(tvmc_package.graph, lib, dev, dump_root="./prof")
            else:
                if device == "micro":
                    logger.debug("Creating runtime (micro) with profiling disabled.")
                    module = tvm.micro.create_local_graph_executor(tvmc_package.graph, lib, dev)
                else:
                    logger.debug("Creating runtime with profiling disabled.")
                    module = executor.create(tvmc_package.graph, lib, dev)

            logger.debug("Loading params into the runtime module.")
            module.load_params(tvmc_package.params)

            logger.debug("Collecting graph input shape and type:")
            shape_dict, dtype_dict = module.get_input_info()
            logger.debug("Graph input shape: %s", shape_dict)
            logger.debug("Graph input type: %s", dtype_dict)

            inputs_dict = make_inputs_dict(shape_dict, dtype_dict, inputs, fill_mode)

            logger.debug("Setting inputs to the module.")
            module.set_input(**inputs_dict)

            # Run must be called explicitly if profiling
            if profile:
                logger.info("Running the module with profiling enabled.")
                report = module.profile()
                # This print is intentional
                print(report)

            if device == "micro":
                # TODO(gromero): Fix time_evaluator() for micro targets. Once it's
                # fixed module.benchmark() can be used instead and this if/else can
                # be removed.
                module.run()
                times = []
            else:
                # Call the benchmarking function of the executor.
                # Optionally measure e2e data transfers from the
                # CPU to device memory overheads (e.g. PCIE
                # overheads if the device is a discrete GPU).
                if end_to_end:
                    dev = session.cpu()
                times = module.benchmark(dev, number=number, repeat=repeat, end_to_end=end_to_end)

            logger.debug("Collecting the output tensors.")
            num_outputs = module.get_num_outputs()
            outputs = {}
            for i in range(num_outputs):
                output_name = "output_{}".format(i)
                outputs[output_name] = module.get_output(i).numpy()

        return TVMCResult(outputs, times)
