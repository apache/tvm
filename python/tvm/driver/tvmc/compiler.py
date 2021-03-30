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
import tarfile
from pathlib import Path

import tvm
from tvm import autotvm, auto_scheduler
from tvm import relay, runtime
from tvm.contrib import cc
from tvm.contrib import utils

from . import common, composite_target, frontends
from .main import register_parser


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_compile_parser(subparsers):
    """ Include parser for 'compile' subcommand """

    parser = subparsers.add_parser("compile", help="compile a model")
    parser.set_defaults(func=drive_compile)
    parser.add_argument(
        "--cross-compiler",
        default="",
        help="the cross compiler to generate target libraries, e.g. 'aarch64-linux-gnu-gcc'",
    )
    parser.add_argument(
        "--desired-layout",
        choices=["NCHW", "NHWC"],
        default=None,
        help="change the data layout of the whole graph",
    )
    parser.add_argument(
        "--dump-code",
        metavar="FORMAT",
        default="",
        help="comma separarated list of formats to export, e.g. 'asm,ll,relay' ",
    )
    parser.add_argument(
        "--model-format",
        choices=frontends.get_frontend_names(),
        help="specify input model format",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="module.tar",
        help="output the compiled module to an archive",
    )
    parser.add_argument(
        "--target",
        help="compilation targets as comma separated string, inline JSON or path to a JSON file.",
        required=True,
    )
    parser.add_argument(
        "--tuning-records",
        metavar="PATH",
        default="",
        help="path to an auto-tuning log file by AutoTVM. If not presented, "
        "the fallback/tophub configs will be used",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="increase verbosity")
    # TODO (@leandron) This is a path to a physical file, but
    #     can be improved in future to add integration with a modelzoo
    #     or URL, for example.
    parser.add_argument("FILE", help="path to the input model file")
    parser.add_argument(
        "--input-shapes",
        help="specify non-generic shapes for model to run, format is "
        '"input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]"',
        type=common.parse_shape_string,
        default=None,
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

    graph, lib, params, dumps = compile_model(
        args.FILE,
        args.target,
        args.dump_code,
        None,
        args.model_format,
        args.tuning_records,
        args.desired_layout,
        args.input_shapes,
    )

    if dumps:
        save_dumps(args.output, dumps)

    save_module(args.output, graph, lib, params, args.cross_compiler)
    return 0


def compile_model(
    path,
    target,
    dump_code=None,
    target_host=None,
    model_format=None,
    tuning_records=None,
    alter_layout=None,
    shape_dict=None,
):
    """Compile a model from a supported framework into a TVM module.

    This function takes a union of the arguments of both frontends.load_model
    and compiler.compile_relay. The resulting TVM module can be executed using
    the graph executor.

    Parameters
    ----------
    path: str
        Path to a file
    target : str
        The target for which to compile. Can be a plain string or
        a path.
    dump_code : list, optional
        Dump the generated code for the specified source types, on
        the requested target.
    target_host : str, optional
        The target of the host machine if host-side code
        needs to be generated.
    model_format: str, optional
        A string representing a name of a frontend to be used
    tuning_records: str, optional
        Path to the file produced by the tuning to be used during
        compilation.
    alter_layout: str, optional
        The layout to convert the graph to. Note, the convert layout
        pass doesn't currently guarantee the whole of the graph will
        be converted to the chosen layout.
    shape_dict: dict, optional
        A mapping from input names to their shape. When present,
        the default shapes in the model will be overwritten.

    Returns
    -------
    graph : str
        A JSON-serialized TVM execution graph.
    lib : tvm.module.Module
        A TVM module containing the compiled functions.
    params : dict
        The parameters (weights) for the TVM module.
    dumps : dict
        Dictionary containing the dumps specified.

    """
    dump_code = [x.strip() for x in dump_code.split(",")] if dump_code else None
    mod, params = frontends.load_model(path, model_format, shape_dict)
    config = {}

    if alter_layout:
        mod = common.convert_graph_layout(mod, alter_layout)

    tvm_target, extra_targets = common.target_from_cli(target)
    target_host = tvm_target if not target_host else target_host

    for codegen_from_cli in extra_targets:
        codegen = composite_target.get_codegen_by_target(codegen_from_cli["name"])
        partition_function = codegen["pass_pipeline"]
        mod = partition_function(mod, params)
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
                with tvm.transform.PassContext(opt_level=3, config=config):
                    logger.debug("building relay graph with autoscheduler")
                    graph_module = relay.build(
                        mod, target=target, params=params, target_host=target_host
                    )
        else:
            with autotvm.apply_history_best(tuning_records):
                with tvm.transform.PassContext(opt_level=3, config=config):
                    logger.debug("building relay graph with tuning records")
                    graph_module = relay.build(
                        mod, tvm_target, params=params, target_host=target_host
                    )
    else:
        with tvm.transform.PassContext(opt_level=3, config=config):
            logger.debug("building relay graph (no tuning records provided)")
            graph_module = relay.build(mod, tvm_target, params=params, target_host=target_host)

    # Generate output dump files with sources
    dump_code = dump_code or []
    dumps = {}
    for source_type in dump_code:
        lib = graph_module.get_lib()
        # TODO lib.get_source call have inconsistent behavior for unsupported
        #      formats (@leandron).
        source = str(mod) if source_type == "relay" else lib.get_source(source_type)
        dumps[source_type] = source

    # TODO we need to update this return to use the updated graph module APIs
    #      as these getter functions will be deprecated in the next release (@leandron)
    return graph_module.get_json(), graph_module.get_lib(), graph_module.get_params(), dumps


def save_module(module_path, graph, lib, params, cross=None):
    """
    Create a tarball containing the generated TVM graph,
    exported library and parameters

    Parameters
    ----------
    module_path : str
        path to the target tar.gz file to be created,
        including the file name
    graph : str
        A JSON-serialized TVM execution graph.
    lib : tvm.module.Module
        A TVM module containing the compiled functions.
    params : dict
        The parameters (weights) for the TVM module.
    cross : str or callable object, optional
        Function that performs the actual compilation

    """
    lib_name = "mod.so"
    graph_name = "mod.json"
    param_name = "mod.params"
    temp = utils.tempdir()
    path_lib = temp.relpath(lib_name)
    if not cross:
        logger.debug("exporting library to %s", path_lib)
        lib.export_library(path_lib)
    else:
        logger.debug("exporting library to %s , using cross compiler %s", path_lib, cross)
        lib.export_library(path_lib, cc.cross_compiler(cross))

    with open(temp.relpath(graph_name), "w") as graph_file:
        logger.debug("writing graph to file to %s", graph_file.name)
        graph_file.write(graph)

    with open(temp.relpath(param_name), "wb") as params_file:
        logger.debug("writing params to file to %s", params_file.name)
        params_file.write(runtime.save_param_dict(params))

    logger.debug("saving module as tar file to %s", module_path)
    with tarfile.open(module_path, "w") as tar:
        tar.add(path_lib, lib_name)
        tar.add(temp.relpath(graph_name), graph_name)
        tar.add(temp.relpath(param_name), param_name)


def save_dumps(module_name, dumps, dump_root="."):
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
