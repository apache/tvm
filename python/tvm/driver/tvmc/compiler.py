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
Provides support to compile networks both AOT and JIT.
"""
import logging
import os.path
from typing import Any, Optional, Dict, List, Union, Callable
from pathlib import Path

import tvm
from tvm import autotvm, auto_scheduler
from tvm import relay
from tvm.target import Target

from . import common, composite_target, frontends
from .model import TVMCModel, TVMCPackage
from .main import register_parser
from .target import generate_target_args, reconstruct_target_args


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_compile_parser(subparsers):
    """Include parser for 'compile' subcommand"""

    parser = subparsers.add_parser("compile", help="compile a model.")
    parser.set_defaults(func=drive_compile)
    parser.add_argument(
        "--cross-compiler",
        default="",
        help="the cross compiler to generate target libraries, e.g. 'aarch64-linux-gnu-gcc'.",
    )
    parser.add_argument(
        "--cross-compiler-options",
        default="",
        help="the cross compiler options to generate target libraries, e.g. '-mfpu=neon-vfpv4'.",
    )
    parser.add_argument(
        "--desired-layout",
        choices=["NCHW", "NHWC"],
        default=None,
        help="change the data layout of the whole graph.",
    )
    parser.add_argument(
        "--dump-code",
        metavar="FORMAT",
        default="",
        help="comma separated list of formats to export the input model, e.g. 'asm,ll,relay'.",
    )
    parser.add_argument(
        "--model-format",
        choices=frontends.get_frontend_names(),
        help="specify input model format.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="module.tar",
        help="output the compiled module to a specifed archive. Defaults to 'module.tar'.",
    )
    parser.add_argument(
        "-f",
        "--output-format",
        choices=["so", "mlf"],
        default="so",
        help="output format. Use 'so' for shared object or 'mlf' for Model Library Format "
        "(only for microTVM targets). Defaults to 'so'.",
    )
    parser.add_argument(
        "--pass-config",
        action="append",
        metavar=("name=value"),
        help="configurations to be used at compile time. This option can be provided multiple "
        "times, each one to set one configuration value, "
        "e.g. '--pass-config relay.backend.use_auto_scheduler=0'.",
    )
    generate_target_args(parser)
    parser.add_argument(
        "--tuning-records",
        metavar="PATH",
        default="",
        help="path to an auto-tuning log file by AutoTVM. If not presented, "
        "the fallback/tophub configs will be used.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity.")
    # TODO (@leandron) This is a path to a physical file, but
    #     can be improved in future to add integration with a modelzoo
    #     or URL, for example.
    parser.add_argument("FILE", help="path to the input model file.")
    parser.add_argument(
        "--input-shapes",
        help="specify non-generic shapes for model to run, format is "
        '"input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]".',
        type=common.parse_shape_string,
        default=None,
    )
    parser.add_argument(
        "--disabled-pass",
        help="disable specific passes, comma-separated list of pass names.",
        type=common.parse_pass_list_str,
        default="",
    )


def drive_compile(args):
    """Invoke tvmc.compiler module with command line arguments

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.

    Returns
    -------
    int
        Zero if successfully completed

    """
    tvmc_model = frontends.load_model(args.FILE, args.model_format, args.input_shapes)

    dump_code = [x.strip() for x in args.dump_code.split(",")] if args.dump_code else None

    compile_model(
        tvmc_model,
        args.target,
        tuning_records=args.tuning_records,
        package_path=args.output,
        cross=args.cross_compiler,
        cross_options=args.cross_compiler_options,
        output_format=args.output_format,
        dump_code=dump_code,
        target_host=None,
        desired_layout=args.desired_layout,
        disabled_pass=args.disabled_pass,
        pass_context_configs=args.pass_config,
        additional_target_options=reconstruct_target_args(args),
    )

    return 0


def compile_model(
    tvmc_model: TVMCModel,
    target: str,
    tuning_records: Optional[str] = None,
    package_path: Optional[str] = None,
    cross: Optional[Union[str, Callable]] = None,
    cross_options: Optional[str] = None,
    output_format: str = "so",
    dump_code: Optional[List[str]] = None,
    target_host: Optional[str] = None,
    desired_layout: Optional[str] = None,
    disabled_pass: Optional[str] = None,
    pass_context_configs: Optional[List[str]] = None,
    additional_target_options: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Compile a model from a supported framework into a TVM module.

    This function takes a union of the arguments of both frontends.load_model
    and compiler.compile_relay. The resulting TVM module can be executed using
    the graph executor.

    Parameters
    ----------
    tvmc_model : TVMCModel
        The model object that should be compiled.
    target : str
        The target for which to compile. Can be a plain string or
        a path.
    tuning_records : str
        A path to tuning records produced using tvmc.tune. When provided,
        compilation will use more optimized kernels leading to better results.
    package_path : str, optional
        The path to export the compiled model to. If not provided it will
        be saved in a temporary directory.
    cross : str or callable object, optional
        Function that performs the actual compilation
    cross_options : str, optional
        Command line options to be passed to the cross compiler.
    output_format : str
        What format to use when saving the function library. Must be one of "so" or "tar".
        When compiling for a remote device without a cross compiler, "tar" will likely work better.
    dump_code : list, optional
        Dump the generated code for the specified source types, on
        the requested target.
    target_host : str, optional
        The target of the host machine if host-side code
        needs to be generated.
    desired_layout: str, optional
        The layout to convert the graph to. Note, the convert layout
        pass doesn't currently guarantee the whole of the graph will
        be converted to the chosen layout.
    disabled_pass: str, optional
        Comma-separated list of passes which needs to be disabled
        during compilation
    pass_context_configs: list[str], optional
        List of strings containing a set of configurations to be passed to the
        PassContext.
    additional_target_options: Optional[Dict[str, Dict[str, Any]]]
        Additional target options in a dictionary to combine with initial Target arguments


    Returns
    -------
    compiled_model : TVMCPackage
        The compiled TVMCModel ready to be run.

    """
    mod, params = tvmc_model.mod, tvmc_model.params

    config = common.parse_configs(pass_context_configs)

    if desired_layout:
        mod = common.convert_graph_layout(mod, desired_layout)

    tvm_target, extra_targets = common.target_from_cli(target, additional_target_options)
    tvm_target, target_host = Target.check_and_update_host_consist(tvm_target, target_host)

    for codegen_from_cli in extra_targets:
        codegen = composite_target.get_codegen_by_target(codegen_from_cli["name"])
        partition_function = codegen["pass_pipeline"]
        mod = partition_function(mod, params, **codegen_from_cli["opts"])
        if codegen["config_key"] is not None:
            config[codegen["config_key"]] = codegen_from_cli["opts"]

    if tuning_records and os.path.exists(tuning_records):
        logger.debug("tuning records file provided: %s", tuning_records)

        use_autoscheduler = True
        try:
            auto_scheduler.load_records(tuning_records)
        except tvm._ffi.base.TVMError:
            use_autoscheduler = False

        if use_autoscheduler:
            with auto_scheduler.ApplyHistoryBest(tuning_records):
                config["relay.backend.use_auto_scheduler"] = True
                with tvm.transform.PassContext(
                    opt_level=3, config=config, disabled_pass=disabled_pass
                ):
                    logger.debug("building relay graph with autoscheduler")
                    graph_module = relay.build(mod, target=tvm_target, params=params)
        else:
            with autotvm.apply_history_best(tuning_records):
                with tvm.transform.PassContext(
                    opt_level=3, config=config, disabled_pass=disabled_pass
                ):
                    logger.debug("building relay graph with tuning records")
                    graph_module = relay.build(mod, target=tvm_target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3, config=config, disabled_pass=disabled_pass):
            logger.debug("building relay graph (no tuning records provided)")
            graph_module = relay.build(mod, target=tvm_target, params=params)

    # Generate output dump files with sources
    if dump_code is None:
        dump_code = []
    if not isinstance(dump_code, list):
        dump_code = [dump_code]
    dumps = {}
    for source_type in dump_code:
        lib = graph_module.get_lib()
        # TODO lib.get_source call have inconsistent behavior for unsupported
        #      formats (@leandron).
        source = str(mod) if source_type == "relay" else lib.get_source(source_type)
        dumps[source_type] = source

    # Create a new tvmc model package object from the graph definition.
    package_path = tvmc_model.export_package(
        graph_module,
        package_path,
        cross,
        cross_options,
        output_format,
    )

    # Write dumps to file.
    if dumps:
        save_dumps(package_path, dumps)

    return TVMCPackage(package_path)


def save_dumps(module_name: str, dumps: Dict[str, str], dump_root: str = "."):
    """
    Serialize dump files to the disk.

    Parameters
    ----------
    module_name : str
        File name, referring to the module that generated
        the dump contents
    dumps : dict
        The output contents to be saved into the files
    dump_root : str, optional
        Path in which dump files will be created
    """

    for dump_format in dumps:
        dump_name = module_name + "." + dump_format
        with open(Path(dump_root, dump_name), "w") as f:
            f.write(dumps[dump_format])
