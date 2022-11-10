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
from typing import Any, Optional, Dict, List, Union, Callable, Sequence
from pathlib import Path

import tvm
from tvm import autotvm, auto_scheduler
from tvm import relay
from tvm.driver.tvmc.registry import generate_registry_args, reconstruct_registry_entity
from tvm.ir.instrument import PassInstrument
from tvm.ir.memory_pools import WorkspaceMemoryPools
from tvm.target import Target
from tvm.relay.backend import Executor, Runtime

from . import composite_target, frontends, TVMCException
from .model import TVMCModel, TVMCPackage
from .main import register_parser
from .target import target_from_cli, generate_target_args, reconstruct_target_args
from .pass_config import parse_configs
from .pass_list import parse_pass_list_str
from .transform import convert_graph_layout
from .shape_parser import parse_shape_string
from .workspace_pools import generate_workspace_pools_args, workspace_pools_recombobulate

# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_compile_parser(subparsers, _, json_params):
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
        help="output the compiled module to a specified archive. Defaults to 'module.tar'.",
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
        "e.g. '--pass-config relay.backend.use_auto_scheduler=0', "
        "e.g. '--pass-config tir.add_lower_pass=opt_level1,pass1,opt_level2,pass2'.",
    )

    generate_target_args(parser)
    parser.add_argument(
        "--tuning-records",
        metavar="PATH",
        default="",
        help="path to an auto-tuning log file by AutoTVM. If not presented, "
        "the fallback/tophub configs will be used.",
    )
    generate_registry_args(parser, Executor, "graph")
    generate_registry_args(parser, Runtime, "cpp")

    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity.")
    # TODO (@leandron) This is a path to a physical file, but
    #     can be improved in future to add integration with a modelzoo
    #     or URL, for example.
    parser.add_argument("FILE", help="path to the input model file.")
    parser.add_argument(
        "-O",
        "--opt-level",
        default=3,
        type=int,
        choices=range(0, 4),
        metavar="[0-3]",
        help="specify which optimization level to use. Defaults to '3'.",
    )
    parser.add_argument(
        "--input-shapes",
        help="specify non-generic shapes for model to run, format is "
        '"input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]".',
        type=parse_shape_string,
        default=None,
    )
    parser.add_argument(
        "--disabled-pass",
        help="disable specific passes, comma-separated list of pass names.",
        type=parse_pass_list_str,
        default="",
    )
    parser.add_argument(
        "--module-name",
        default="default",
        help="The output module name. Defaults to 'default'.",
    )
    for one_entry in json_params:
        parser.set_defaults(**one_entry)

    generate_workspace_pools_args(parser)


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

    if not os.path.isfile(args.FILE):
        raise TVMCException(
            f"Input file '{args.FILE}' doesn't exist, is a broken symbolic link, or a directory."
        )

    tvmc_model = frontends.load_model(args.FILE, args.model_format, args.input_shapes)

    dump_code = [x.strip() for x in args.dump_code.split(",")] if args.dump_code else None

    additional_targets = reconstruct_target_args(args)
    workspace_pools_target, extra_targets = target_from_cli(args.target, additional_targets)

    compile_model(
        tvmc_model,
        args.target,
        opt_level=args.opt_level,
        executor=reconstruct_registry_entity(args, Executor),
        runtime=reconstruct_registry_entity(args, Runtime),
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
        mod_name=args.module_name,
        additional_target_options=additional_targets,
        workspace_pools=(
            workspace_pools_recombobulate(args, [workspace_pools_target], extra_targets)
        ),
    )

    return 0


def compile_model(
    tvmc_model: TVMCModel,
    target: str,
    opt_level: int = 3,
    executor: Optional[Executor] = Executor("graph"),
    runtime: Optional[Runtime] = Runtime("cpp"),
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
    use_vm: bool = False,
    mod_name: Optional[str] = "default",
    workspace_pools: Optional[WorkspaceMemoryPools] = None,
    instruments: Optional[Sequence[PassInstrument]] = None,
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
    opt_level : int
        The option that controls various sorts of optimizations.
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
    use_vm: bool
        Whether to use the VM to compile the model as opposed to the graph executor
    mod_name: str, optional
        The module name
    workspace_pools: WorkspaceMemoryPools, optional
        Specification of WorkspacePoolInfo objects to be used as workspace memory in the
        compilation.
    instruments: Optional[Sequence[PassInstrument]]
        The list of pass instrument implementations.

    Returns
    -------
    compiled_model : TVMCPackage
        The compiled TVMCModel ready to be run.

    """
    mod, params = tvmc_model.mod, tvmc_model.params

    config = parse_configs(pass_context_configs)

    tvm_target, extra_targets = target_from_cli(target, additional_target_options)
    tvm_target, target_host = Target.canon_target_and_host(tvm_target, target_host)

    partition_functions = []
    partition_opts = []
    for codegen_from_cli in extra_targets:
        codegen = composite_target.get_codegen_by_target(codegen_from_cli["name"])
        partition_functions.append(codegen["pass_pipeline"])
        partition_opts.append(codegen_from_cli["opts"])
        if codegen["config_key"] is not None:
            config[codegen["config_key"]] = codegen_from_cli["opts"]

    with tvm.transform.PassContext(
        opt_level=opt_level,
        config=config,
        disabled_pass=disabled_pass,
        instruments=instruments,
    ):
        if desired_layout:
            mod = convert_graph_layout(mod, desired_layout)

        for partition_function, opts in zip(partition_functions, partition_opts):
            mod = partition_function(mod, params, mod_name=mod_name, **opts)

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
                    logger.debug("building relay graph with autoscheduler")
                    graph_module = build(
                        mod,
                        tvm_target=tvm_target,
                        executor=executor,
                        runtime=runtime,
                        params=params,
                        use_vm=use_vm,
                        mod_name=mod_name,
                        workspace_pools=workspace_pools,
                    )
            else:
                with autotvm.apply_history_best(tuning_records):
                    logger.debug("building relay graph with tuning records")
                    graph_module = build(
                        mod,
                        tvm_target=tvm_target,
                        executor=executor,
                        runtime=runtime,
                        params=params,
                        use_vm=use_vm,
                        mod_name=mod_name,
                        workspace_pools=workspace_pools,
                    )
        else:
            logger.debug("building relay graph (no tuning records provided)")
            graph_module = build(
                mod,
                tvm_target=tvm_target,
                executor=executor,
                runtime=runtime,
                params=params,
                use_vm=use_vm,
                mod_name=mod_name,
                workspace_pools=workspace_pools,
            )

        # Generate output dump files with sources
        if dump_code is None:
            dump_code = []
        if not isinstance(dump_code, list):
            dump_code = [dump_code]
        dumps = {}
        for source_type in dump_code:
            if use_vm:
                lib = graph_module.lib
            else:
                lib = graph_module.get_lib()
            # TODO lib.get_source call have inconsistent behavior for unsupported
            #      formats (@leandron).
            source = str(mod) if source_type == "relay" else lib.get_source(source_type)
            dumps[source_type] = source

        # Create a new tvmc model package object from the graph definition.
        package_path = tvmc_model.export_package(
            graph_module, package_path, cross, cross_options, output_format
        )

        # Write dumps to file.
        if dumps:
            save_dumps(package_path, dumps)

        return TVMCPackage(package_path)


def build(
    mod: tvm.IRModule,
    tvm_target: str,
    executor: Executor,
    runtime: Runtime,
    params: Dict[str, tvm.nd.NDArray],
    use_vm: bool,
    mod_name: str,
    workspace_pools: Optional[WorkspaceMemoryPools],
):
    """
    Builds the model with the provided executor.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module corresponding to this model.
    tvm_target : str
        The target for which to compile. Can be a plain string or
        a path.
    executor : Executor
        The graph executor to build the model if use_vm is not True
    runtime : Runtime
        The runtime configuration.
    params : dict
        A parameter dictionary for the model.
    use_vm: bool
        Whether to use the VM to compile the model as opposed to the graph executor
    mod_name: str
        The module name

    """
    if use_vm:
        logger.debug("building with vm compile")
        return relay.vm.compile(mod, target=tvm_target, params=params)
    logger.debug("building with relay build")
    return relay.build(
        mod,
        target=tvm_target,
        executor=executor,
        runtime=runtime,
        params=params,
        mod_name=mod_name,
        workspace_memory_pools=workspace_pools,
    )


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
