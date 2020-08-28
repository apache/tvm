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
import argparse
import json
import logging
import os.path
import tarfile
from pathlib import Path

import tvm
from tvm import autotvm
from tvm import relay
from tvm._ffi.runtime_ctypes import TVMContext
from tvm.contrib import cc
from tvm.contrib import util
from tvm.relay.op.contrib import get_pattern_table

from . import common, frontends
from .main import register_parser


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
        "--dump-code",
        metavar="FORMAT",
        default="",
        help="comma separarated list of formats to export, e.g. 'asm,ll,relay' "
    )
    parser.add_argument(
        "--model-format",
        choices=frontends.get_frontends(),
        help="specify input model format",
    )
    parser.add_argument(
        "--input-shape",
        type=common.parse_input_shapes,
        metavar="INPUT_SHAPE,[INPUT_SHAPE]...",
        help="for pytorch, e.g. '(1,3,224,224)'",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="module.tar",
        help="output the compiled module to an archive",
    )
    parser.add_argument(
        "--target",
        help="compilation target as plain string, inline JSON or path to a JSON file",
        required=True
    )
    parser.add_argument(
        "--tuning-records",
        metavar="PATH",
        default="",
        help="path to an auto-tuning log file from AutoTVM"
    )
    parser.add_argument(
        "--desired-layout",
        choices=["NCHW", "NHWC"],
        default=None,
        help="change the data layout of the whole graph",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase verbosity"
    )
    parser.add_argument("FILE")


def drive_compile(args):
    """ Invoke tvmc.compiler module with command line arguments """

    graph, lib, params, dumps = compile_model(
        args.FILE,
        args.target,
        args.dump_code,
        "",
        args.model_format,
        args.input_shape,
        args.tuning_records,
        args.tensor_layout,
    )

    if dumps:
        save_dumps(args.output, dumps)

    save_module(args.output, graph, lib, params, args.cross_compiler)
    return 0


def compile_model(
        path,
        target,
        dump_sources=None,
        target_host=None,
        model_format=None,
        shapes=None,
        tuning_records=None,
        alter_layout=None,
):
    """Compile a model from a supported framework into a TVM module.

    This function takes a union of the arguments of both frontends.load_model
    and compiler.compile_relay. The resulting TVM module can be executed using
    the graph runtime.

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
    dump_sources = [x.strip() for x in  dump_sources.split(',')] if dump_sources else None
    mod, params = frontends.load_model(path, model_format, shapes)

    return compile_relay(
        mod,
        params,
        target,
        dump_sources=dump_sources,
        target_host=target_host,
        tuning_records=tuning_records,
        alter_layout=alter_layout,
    )


def compile_relay(
        mod,
        params,
        target,
        dump_sources=None,
        target_host=None,
        tuning_records=None,
        alter_layout=None,
):
    """Compile a relay module to a TVM module for the graph runtime.

    Parameters
    ----------
    mod : tvm.relay.Module
        The relay module to compile.
    params : dict
        The parameters (weights) for the relay module.
    target : str
        The target for which to compile. Can be a plain string or
        a path.
    dump_sources : list, optional
        Dump the generated code for the specified source types, on
        the requested target.
    target_host : Union[str, tvm.target.Target], optional
        The target of the host machine if host-side code
        needs to be generated.
    tuning_records: str, optional
        Name of the file produced by the tuning to be used during
        compilation.
    alter_layout: str, optional
        The layout to convert the graph to. Note, the convert layout
        pass doesn't currently guarantee the whole of the graph will
        be converted to the chosen layout.

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

    if alter_layout:
        mod = common.convert_graph_layout(mod, alter_layout)

    if os.path.exists(str(target)):
        with open(target) as target_file:
            logging.info(f"using target input from file: {target}")
            target = "".join(target_file.readlines())

    logging.debug(f"creating target from input: {target}")
    tvm_target = tvm.target.create(target)
    target_host = ""

    if tuning_records:
        logging.debug(f"tuning records file provided: {tuning_records}")
        with autotvm.apply_history_best(tuning_records):
            with tvm.transform.PassContext(opt_level=3):
                logging.debug("building relay graph with tuning records")
                graph_module = relay.build(mod, tvm_target, params=params, target_host=tvm_target)
    else:
        with tvm.transform.PassContext(opt_level=3):
            logging.debug("building relay graph (no tuning records provided)")
            graph_module = relay.build(mod, tvm_target, params=params, target_host=tvm_target)

    # Generate output dump files with sources
    dump_sources = dump_sources or []
    dumps = {}
    for source_type in dump_sources:
        lib = graph_module.get_lib()
        # TODO lib.get_source call here have inconsistent behavior for unsupported
        #      formats. This is an open discussion (@leandron).
        source = str(mod) if source_type == "relay" else lib.get_source(source_type)
        dumps[source_type] = source

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
    cross : Union[str, Callable[[str, str, Optional[str]], None]]
        Function that performs the actual compilation

    """
    lib_name = "mod.so"
    graph_name = "mod.json"
    param_name = "mod.params"
    temp = util.tempdir()
    path_lib = temp.relpath(lib_name)
    if not cross:
        logging.debug(f"exporting library to {path_lib}")
        lib.export_library(path_lib)
    else:
        logging.debug(f"exporting library to {path_lib}, using cross compiler {cross}")
        lib.export_library(path_lib, cc.cross_compiler(cross))

    with open(temp.relpath(graph_name), "w") as graph_file:
        logging.debug(f"writing graph to file to {graph_file.name}")
        graph_file.write(graph)

    with open(temp.relpath(param_name), "wb") as params_file:
        logging.debug(f"writing params to file to {params_file.name}")
        params_file.write(relay.save_param_dict(params))

    logging.debug(f"saving module as tar file to {module_path}")
    with tarfile.open(module_path, "w") as tar:
        tar.add(path_lib, lib_name)
        tar.add(temp.relpath(graph_name), graph_name)
        tar.add(temp.relpath(param_name), param_name)


def save_dumps(module_name, dumps, dump_root="."):
    """
    Serialize dump files to the disk.

    Parameters
    ----------
    module_name : list(Union[str, tvm.target.Target])
        file name, referring to the module that generated
        the dump contents
    dumps : dict
        the output contents to be saved into the files
    dump_root : str
        path in which dump files will be created
    """

    for dump_format in dumps:
        dump_name = module_name + "." + dump_format
        with open(Path(dump_root, dump_name), "w") as f:
            f.write(dumps[dump_format])
