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
# pylint: disable=unused-argument
"""
Provides support to compile networks both AOT and JIT.
"""
import logging
import os.path
import re
import itertools
from copy import deepcopy
from typing import Any, Optional, Dict, List, Union, Callable, Sequence
from pathlib import Path
from collections import defaultdict

import tvm
from tvm import autotvm, auto_scheduler
from tvm import relay
from tvm.driver.tvmc.registry import generate_registry_args, reconstruct_registry_entity
from tvm.ir.instrument import PassInstrument, PassTimingInstrument, PassPrintingInstrument
from tvm.ir.memory_pools import WorkspaceMemoryPools
from tvm.target import Target
from tvm.relay.backend import Executor, Runtime
from tvm.relay.analysis.operations_distribution import analyze_operations_distribution
from tvm.relay.transform.suffixes import tag_suffixes

from . import composite_target, frontends, TVMCException
from .model import TVMCModel, TVMCPackage
from .main import register_parser
from .target import target_from_cli, generate_target_args, reconstruct_target_args
from .pass_config import parse_configs
from .pass_list import parse_pass_list_str
from .transform import generate_transform_args, parse_graph_transform_args, apply_graph_transforms
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
    generate_transform_args(parser)
    parser.add_argument(
        "--dump-code",
        metavar="FORMAT",
        default="",
        help="comma separated list of formats to export the input model, e.g. 'asm,ll,tir,relay'.",
    )
    parser.add_argument(
        "--dump-offloads",
        default="",
        help="output a mapping of which operations of the initial Relay "
        "will be transferred to which backend, indicating the composite "
        "that includes those operations, "
        "e.g. '--dump-offloads -' to dump to the console, "
        "e.g. '--dump-offloads <path_to_file>' to dump to the file. "
        "If not presented, no output is done. ",
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
        choices=["so"],
        default="so",
        help="output format. Use 'so' for shared object. Defaults to 'so'.",
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
    parser.add_argument(
        "--print-pass-times",
        action="store_true",
        help="print compilation time per pass",
    )
    parser.add_argument(
        "--print-ir-before",
        help="print IR before each named pass of a comma-separated list of pass names."
        "e.g. '--print-ir-before [tir.SplitHostDevice,tir.ConvertSSA]' ",
        default="",
    )
    parser.add_argument(
        "--print-ir-after",
        help="print IR after each named pass of a comma-separated list of pass names."
        "e.g. '--print-ir-after [tir.SplitHostDevice,tir.ConvertSSA]' ",
        default="",
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

    dump_offloads = args.dump_offloads if args.dump_offloads else ""

    additional_targets = reconstruct_target_args(args)
    workspace_pools_target, extra_targets = target_from_cli(args.target, additional_targets)
    transform_args = parse_graph_transform_args(args)

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
        dump_offloads=dump_offloads,
        target_host=None,
        disabled_pass=args.disabled_pass,
        pass_context_configs=args.pass_config,
        mod_name=args.module_name,
        additional_target_options=additional_targets,
        workspace_pools=(
            workspace_pools_recombobulate(args, [workspace_pools_target], extra_targets)
        ),
        print_pass_times=args.print_pass_times,
        print_ir_before=args.print_ir_before,
        print_ir_after=args.print_ir_after,
        **transform_args,
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
    dump_offloads: str = "",
    target_host: Optional[str] = None,
    disabled_pass: Optional[str] = None,
    pass_context_configs: Optional[List[str]] = None,
    additional_target_options: Optional[Dict[str, Dict[str, Any]]] = None,
    use_vm: bool = False,
    mod_name: Optional[str] = "default",
    workspace_pools: Optional[WorkspaceMemoryPools] = None,
    print_pass_times: bool = False,
    print_ir_before: Optional[List[str]] = None,
    print_ir_after: Optional[List[str]] = None,
    instruments: Optional[Sequence[PassInstrument]] = None,
    desired_layout: Optional[str] = None,
    desired_layout_ops: Optional[List[str]] = None,
    mixed_precision: bool = False,
    mixed_precision_ops: Optional[List[str]] = None,
    mixed_precision_calculation_type: Optional[str] = None,
    mixed_precision_acc_type: Optional[str] = None,
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
    dump_code : list[str], optional
        Dump the generated code for the specified source types, on
        the requested target. Choose from: ["asm", "ll", "tir", "relay"].
    dump_offloads : str
        Dump the information about the partition of input model's layers by external codegen.
        Can be '' to not dump at all, '-' to dump to the console
        or '<path_to_file>' to dump to the specified file.
    target_host : str, optional
        The target of the host machine if host-side code
        needs to be generated.
    disabled_pass: str, optional
        Comma-separated list of passes which needs to be disabled
        during compilation.
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
    print_pass_times: bool
        To enable printing a breakdown of compilation times by pass. Disabled by default.
    print_ir_before: list[str], optional
        To print IR before each named pass of a comma-separated list of passes.
    print_ir_after: list[str], optional
        To print IR after each named pass of a comma-separated list of passes.
    instruments: Optional[Sequence[PassInstrument]]
        The list of pass instrument implementations.
    desired_layout: str, optional
        Can be one of "NCHW" or "NHWC". When specified, compatible operations in the graph
        will have their layout set to this format. Tasks will then be tuned using this
        specified layout.
    desired_layout_ops: list[str], optional
        The list of operators to be transformed with desired layout.
    mixed_precision: bool
        To enable mixed precision transformation. Disabled by default.
    mixed_precision_ops: list[str], optional
        The list of operators to be converted to mixed precision.
        Set to ["nn.conv2d", "nn.dense"] by default
    mixed_precision_calculation_type: str
        The calculation dtype to be used while mixed precision. Set to "float16" by default.
    mixed_precision_acc_type: str
        The accumulation data type to be used while mixed precision. Set to "float16" by default.

    Returns
    -------
    compiled_model : TVMCPackage
        The compiled TVMCModel ready to be run.

    """
    mod, params = tvmc_model.mod, tvmc_model.params

    if dump_code is None:
        dump_code = []
    if not isinstance(dump_code, list):
        dump_code = [dump_code]
    dumps = {}

    config = parse_configs(pass_context_configs)
    if "tir" in dump_code:
        config, dumps = add_tir_to_dumps(config, dumps)

    initial_relay = None
    if dump_offloads != "":
        # add suffixes to the span field for calls in Relay
        mod = tag_suffixes(mod)
        # remember initial Relay
        initial_relay = deepcopy(mod)

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

    if print_pass_times:
        timing_inst = PassTimingInstrument()
        instruments = [timing_inst] if instruments is None else [timing_inst] + instruments

    if print_ir_before or print_ir_after:
        print_ir_instr = PassPrintingInstrument(
            print_before_pass_names=print_ir_before, print_after_pass_names=print_ir_after
        )
        instruments = [print_ir_instr] if instruments is None else [print_ir_instr] + instruments

    with tvm.transform.PassContext(
        opt_level=opt_level,
        config=config,
        disabled_pass=disabled_pass,
        instruments=instruments,
    ):
        transform_args = parse_graph_transform_args(locals())
        mod = apply_graph_transforms(mod, transform_args, params)

        for partition_function, opts in zip(partition_functions, partition_opts):
            mod = partition_function(mod, params, mod_name=mod_name, **opts)

        if initial_relay:
            # dump which operations are offloaded to which backend
            dump_operation_offloads(mod, initial_relay, dump_offloads)

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
        for source_type in dump_code:
            if source_type == "relay":
                dumps[source_type] = str(mod)
            elif source_type == "tir":
                dumps[source_type] = "\n".join(dumps[source_type])
            else:
                lib = graph_module.lib if use_vm else graph_module.get_lib()
                # TODO lib.get_source call have inconsistent behavior for unsupported
                #      formats (@leandron).
                try:
                    dumps[source_type] = lib.get_source(source_type)
                except tvm.TVMError:
                    pass
                for smod in lib.imported_modules:
                    try:
                        if smod.type_key not in dumps:
                            dumps[smod.type_key] = ""
                        else:
                            dumps[smod.type_key] += "\n"
                        dumps[smod.type_key] += smod.get_source()
                    except tvm.TVMError:
                        print(f"Imported module {smod.type_key} doesn't support source dump")

        # Create a new tvmc model package object from the graph definition.
        package_path = tvmc_model.export_package(
            graph_module, package_path, cross, cross_options, output_format
        )

        # Write dumps to file.
        if dumps:
            save_dumps(package_path, dumps)

        # Print compilation times per pass
        if print_pass_times:
            print("Compilation time breakdown by pass:")
            print(timing_inst.render())

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


def add_tir_to_dumps(config, dumps):
    """
    Creates a debug pass that dumps TIR functions as a list of strings.
    """
    key = "tir"
    phase = 3  # final TIR phase before codegen
    dumps[key] = []

    @tvm.tir.transform.prim_func_pass(opt_level=0)
    def _dump_tir_pass(tir_func, _, __):
        dumps[key].append(str(tir_func))
        return tir_func

    tir_lower_passes = config.get("tir.add_lower_pass", [])
    tir_lower_passes.append((phase, _dump_tir_pass))
    config["tir.add_lower_pass"] = tir_lower_passes

    return config, dumps


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


def dump_operation_offloads(mod: tvm.ir.IRModule, initial_mod: tvm.ir.IRModule, dump_path: str):
    """This helper function forms a line-by-line output of the initial Relay lines,
    indicating which operations are ported to which target,
    and indicating the composite that includes those operations;
    the 'generic' target refers to operations uploaded to the host, e.g
    'target1        <-     target1.qnn_conv2d'
    'target1        <-          %0 = qnn.conv2d(%tfl.quantize, %v_param_1, ...'
    'target1        <-          %1 = nn.bias_add(%0, %v_param_2, axis=3);'
    'target1        <-          %2 = qnn.requantize(%1, meta[relay.Constant]...'
    'target2        <-     target2.reshape'
    'target2        <-          %3 = reshape(%2, newshape=[1, 1001]);'
    'generic        <-     %4 = nn.pad(%3, -128f, pad_width=[[0, 0], [1, 1]...'

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The partitioned IRModule with external global functions.
    initial_mod : tvm.ir.IRModule
        The initial IRModule that gets generated from a relay frontend.
    dump_path: str
        Value of the "dump_offloads" compiler atribute.
        Could be dash ("-") or file path or empty string for
        printing to console, file or doing nothing respectively.
    """
    print_to_console = dump_path == "-"
    save_to_file = all([dump_path != "-", dump_path != ""])

    if print_to_console or save_to_file:
        operations_distribution = analyze_operations_distribution(mod)

        def annotate_f(x):
            ret = ""
            if isinstance(x, relay.Call):
                # if there is no x.span.source_name.name in operations_distribution,
                # this could mean that the span was not copied during the application of passes
                # to the Relay, in which case we can not associate the initial Relay string
                # with the resulting Relay call
                source_name = x.span.source_name.name
                suffix = tvm.relay.transform.suffixes.SUFFIX_STRING
                result = re.search(r"(.*)(" + suffix + r")(.*)", source_name)
                func_id = result.group(1)
                if func_id in operations_distribution:
                    compiler_name, op_name = operations_distribution[func_id]
                    ret = (
                        f", compiler_name: {compiler_name}, op_name: {op_name}, "
                        f"func_id: {func_id}"
                    )
                else:
                    ret = ", compiler_name: unknown, op_name: unknown, func_id: unknown"
            elif isinstance(x, (relay.Tuple, relay.TupleGetItem)):
                ret = ", compiler_name: none, op_name: none, func_id: none"

            return ret

        initial_relay_astext = initial_mod.astext(show_meta_data=False, annotate=annotate_f).split(
            "\n"
        )

        # funcs_list is a list of internal composite/function IDs.
        # funcs_list helps keep the order of lines from the initial Relay.
        funcs_list = []

        # target_statistic is a mapping of the target name to the
        # number of initial Relay calls offloaded on the target
        target_statistic = defaultdict(int)

        # funcs_dict is a mapping of the generated analyze_operations_distribution
        # internal composite/function IDs to a list, where:
        # 1st element is
        #   (1a): "generic"|"unknown"|"none"* or
        #   (1b): specific target name, like "ethos-u" or "cmsis-nn"
        # 2nd element is
        #   (2a): corresponding initial Relay line for the case (1a) or
        #   (2b): the name of the target composite functon in the other case (1b)
        # 3rd element or subsequent ones are presented only for the case (2b)
        # and are the initial Relay's lines included in the corresponding
        # target composite functon
        #
        # *Description of what is meant by "generic"|"unknown"|"none":
        # "generic" means that operation will be run on a host
        # "unknown" means that unique identifier of this Relay line not found in the partitioned
        #           Relay and therefore not present in the operations_distribution dictionary
        # "none" means that this Relay line is not relay.Call
        funcs_dict = {}

        # Here we group together initial Relay lines from the one composite
        counter = itertools.count()
        for s in initial_relay_astext:
            result = re.search(
                r"(compiler_name: )(.*)(, op_name: )(.*)(, func_id: )((.*)(?=;)|(.*))", s
            )
            if result:
                target_name = result.group(2)
                op_name = result.group(4)
                func_id = result.group(6)
                if target_name != "none":
                    target_statistic[target_name] += 1

                # create an identifier for each "unknown" or "none" case to keep the lines order
                if func_id == "unknown" or func_id == "none" or target_name == "generic":
                    func_id = str(next(counter) * -1)

                if func_id not in funcs_dict:
                    funcs_list.append(func_id)
                    funcs_dict[func_id] = [target_name]
                    if target_name not in ["unknown", "generic", "none"]:
                        funcs_dict[func_id].append(op_name)

                s = re.sub(r", compiler_name: (.*)", "", s).lstrip()
                funcs_dict[func_id].append(s)

        # Here we prepare the output for printing.
        # The output in most cases keeps the original order of the Relay lines
        # but some lines are moved to be in the corresponding composite group
        output = []
        total = 0
        output.append("Total number of operators and distribution by targets")
        output.append("Total:")
        for target, statistic in target_statistic.items():
            total += statistic
            output.append(f"{target}: {statistic}")
        output[1] += f" {total}"
        output[len(target_statistic) + 1] += "\n"

        for func_id in funcs_list:
            _list = funcs_dict[func_id]

            if _list[0] != "none":
                output.append(f"{_list[0]:<15}<-{' ':5}{_list[1]}")
            else:
                output.append(f"{' ':>22}{_list[1]}")

            if _list[0] == "unknown":
                output.append(
                    "Warning: The above line means that some pass(es) \
                              in Relay partitioning"
                )
                output.append("do not copy the span when the call is recreated")
                output.append(
                    "and a line from initial Relay could not be associated \
                              with the resulting Relay"
                )
            for el in _list[2:]:
                output.append(f"{_list[0]:<15}<-{' ':10}{el}")

        if print_to_console:
            print("\n" + "\n".join(output))
        if save_to_file:
            file_path = os.path.abspath(dump_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write("\n".join(output))
                f.write("\n")
