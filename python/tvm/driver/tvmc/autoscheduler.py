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
Provides support to auto-tuning networks using AutoScheduler.
"""
import logging

from urllib.parse import urlparse

from tvm import auto_scheduler
from tvm.auto_scheduler.auto_schedule import HardwareParams

from . import common, frontends
from .common import add_tuning_options
from .main import register_parser


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_autoscheduler_parser(subparsers):
    """ Include parser for 'autoschedule' subcommand """
    parser = subparsers.add_parser("autoschedule", help="auto-schedule a model")
    parser.set_defaults(func=drive_autoschedule)
    add_tuning_options(parser)

    parser.add_argument(
        "--cache-line-bytes",
        default=64,
        help="the size of cache line in bytes",
    )
    parser.add_argument(
        "--num-cores",
        default=4,
        help="the number of device cores",
    )
    parser.add_argument(
        "--vector-unit-bytes",
        default=16,
        help="the width of vector units in bytes",
    )
    parser.add_argument(
        "--model-format",
        choices=frontends.get_frontend_names(),
        help="specify input model format",
    )


def drive_autoschedule(args):
    """Invoke auto-scheduling with command line arguments

    Parameters
    ----------
    args: argparse.Namespace
        Arguments from command line parser.
    """

    # extra arguments validation before importing the model, so that obvious errors
    # are pointed in advance.
    if args.rpc_tracker:
        parsed_url = urlparse("//%s" % args.rpc_tracker)
        rpc_hostname = parsed_url.hostname
        rpc_port = parsed_url.port or 9090
        logger.info("RPC tracker hostname: %s", rpc_hostname)
        logger.info("RPC tracker port: %s", rpc_port)

        if not args.rpc_key:
            raise common.TVMCException(
                "need to provide an RPC tracker key (--rpc-key) for remote tuning"
            )

    target = common.target_from_cli(args.target)
    mod, params = frontends.load_model(args.FILE, args.model_format)

    # min_repeat_ms should be:
    # a. the value provided by the user, if any, or
    # b. 0ms in case target is "cpu"; otherwise 1000ms
    if args.min_repeat_ms is not None:
        min_repeat_ms = args.min_repeat_ms
    else:
        min_repeat_ms = 0 if target.keys[0] == "cpu" else 1000
        logger.debug("Default --min-repeat-ms for this target is %s", min_repeat_ms)

    if args.rpc_tracker:

        runner = auto_scheduler.RPCRunner(
            key=args.rpc_key,
            host=rpc_hostname,
            port=rpc_port,
            number=args.number,
            repeat=args.repeat,
            n_parallel=args.parallel,
            timeout=args.timeout,
            min_repeat_ms=min_repeat_ms,
        )
    else:
        logger.info("starting localhost tuning")
        runner = auto_scheduler.LocalRunner(
            number=args.number,
            repeat=args.repeat,
            timeout=args.timeout,
            min_repeat_ms=min_repeat_ms,
        )

    # Create the autoscheduler tuning options
    tuning_options = auto_scheduler.TuningOptions(
        num_measure_trials=args.trials,
        measure_callbacks=[auto_scheduler.RecordToFile(args.output)],
        runner=runner,
        builder="local",
        early_stopping=args.early_stopping,
    )

    # Specify hardware parameters
    hardware_params = HardwareParams(
        args.num_cores, args.vector_unit_bytes, args.cache_line_bytes, None, None, None, None, None
    )

    # Extract the tasks from the model
    tasks, weights = get_tuning_tasks(
        mod, params, target, target_host, args.desired_layout, hardware_params
    )

    # Schedule the tasks (i.e., produce a schedule for each task)
    schedule_tasks(
        tasks,
        weights,
        tuning_options,
        args.tuning_records,
    )


def get_tuning_tasks(
    mod, params, target, target_host=None, alter_layout=None, hardware_params=None
):
    """Get the tuning tasks for a given relay module.

    Parameters
    ----------
    mod : tvm.relay.Module
        The relay module from which to extract tuning tasks.
    params : dict
        The params for the relay module.
    target : tvm.target.Target
        The compilation target.
    target_host : str, optional
        The compilation target for the host.
    alter_layout : str, optional
        The layout to convert the graph to. Note, the convert layout
        pass doesn't currently guarantee the whole of the graph will
        be converted to the chosen layout.

    Returns
    -------
    tasks : list of autotvm.Tasks
        list of tasks to be tuned
    weights : List[int]
        the weight (i.e. the number of appearance) of extracted tasks
    """
    if alter_layout:
        mod = common.convert_graph_layout(mod, alter_layout)

    # Extract the tasks
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"], params, target=target, target_host=target_host, hardware_params=hardware_params
    )

    return tasks, task_weights


def schedule_tasks(
    tasks,
    task_weights,
    tuning_options,
    tuning_records=None,
):
    """Generate the schedules for the different tasks (i.e., subgraphs) contained in the module.
    Store the schedules in a json file that will be used later by the compiler.

    Parameters
    ----------
    tasks : list
        A list of autotvm.Tasks to tune.
    task_weights : list
        The weight (i.e. the number of appearance) of extracted tasks
    tuning_records : str, optional
        The json file used to preload the autoscheduler
    tuning_options:
        The options of tuning
    """

    # Create the scheduler
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=tuning_records)

    # Tune the tasks
    tuner.tune(tuning_options)
