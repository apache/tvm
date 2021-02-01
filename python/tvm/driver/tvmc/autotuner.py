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
Provides support to auto-tuning networks using AutoTVM.
"""
import os.path
import logging
import time

from urllib.parse import urlparse

from tvm import autotvm, auto_scheduler
from tvm.autotvm.tuner import GATuner
from tvm.autotvm.tuner import GridSearchTuner
from tvm.autotvm.tuner import RandomTuner
from tvm.autotvm.tuner import XGBTuner

from . import common, frontends
from .common import TVMCException
from .main import register_parser


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


@register_parser
def add_tune_parser(subparsers):
    """ Include parser for 'tune' subcommand """

    parser = subparsers.add_parser("tune", help="auto-tune a model")
    parser.set_defaults(func=drive_tune)
    parser.add_argument(
        "--early-stopping",
        type=int,
        help="minimum number of trials before early stopping",
    )

    # There is some extra processing required to define the actual default value
    # for --min-repeat-ms. This is done in `drive_tune`.
    parser.add_argument(
        "--min-repeat-ms",
        default=None,
        type=int,
        help="minimum time to run each trial, in milliseconds. "
        "Defaults to 0 on x86 and 1000 on all other targets",
    )
    parser.add_argument(
        "--model-format",
        choices=frontends.get_frontend_names(),
        help="specify input model format",
    )
    parser.add_argument(
        "--number",
        default=10,
        type=int,
        help="number of runs a single repeat is made of. "
        "The final number of tuning executions is: "
        "(1 + number * repeat)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output file to store the tuning records for the tuning process",
    )
    parser.add_argument(
        "--parallel",
        default=4,
        type=int,
        help="the maximum number of parallel devices to use when tuning",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="how many times to repeat each measurement",
    )
    parser.add_argument(
        "--rpc-key",
        help="the RPC tracker key of the target device. Required when --rpc-tracker is provided.",
    )
    parser.add_argument(
        "--rpc-tracker",
        help="hostname (required) and port (optional, defaults to 9090) of the RPC tracker, "
        "e.g. '192.168.0.100:9999'",
    )
    parser.add_argument(
        "--target",
        help="compilation target as plain string, inline JSON or path to a JSON file",
        required=True,
    )
    parser.add_argument(
        "--target-host",
        help="the host compilation target, defaults to 'llvm'",
        default="llvm",
    )
    parser.add_argument("--timeout", default=10, help="compilation timeout, in seconds")
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="the maximum number of tuning trials to perform",
    )
    parser.add_argument(
        "--tuning-records",
        metavar="PATH",
        help="path to an auto-tuning log file by AutoTVM.",
    )
    parser.add_argument(
        "--desired-layout",
        choices=["NCHW", "NHWC"],
        default=None,
        help="change the data layout of the whole graph",
    )
    parser.add_argument(
        "--enable-autoscheduler",
        help="enable tuning the graph through the autoscheduler",
        action="store_true",
    )

    auto_scheduler_group = parser.add_argument_group(
        "Autoscheduler options",
        "Autoscheduler options, used when --enabled-auto-scheduler is provided",
    )

    auto_scheduler_group.add_argument(
        "--cache-line-bytes",
        type=int,
        default=64,
        help="the size of cache line in bytes",
    )
    auto_scheduler_group.add_argument(
        "--num-cores",
        type=int,
        default=4,
        help="the number of device cores",
    )
    auto_scheduler_group.add_argument(
        "--vector-unit-bytes",
        type=int,
        default=16,
        help="the width of vector units in bytes",
    )
    auto_scheduler_group.add_argument(
        "--max-shared-memory-per-block",
        type=int,
        default=0,
        help="the max shared memory per block in bytes",
    )
    auto_scheduler_group.add_argument(
        "--max-local-memory-per-block",
        type=int,
        default=0,
        help="the max local memory per block in bytes",
    )
    auto_scheduler_group.add_argument(
        "--max-threads-per-block",
        type=int,
        default=0,
        help="the max number of threads per block",
    )
    auto_scheduler_group.add_argument(
        "--max-vthread-extent",
        type=int,
        default=0,
        help="the max vthread extent",
    )
    auto_scheduler_group.add_argument(
        "--warp-size",
        type=int,
        default=0,
        help="the thread numbers of a warp",
    )
    auto_scheduler_group.add_argument(
        "--include-simple-tasks",
        help="whether to extract simple tasks that do not include complicated ops",
        action="store_true",
    )
    auto_scheduler_group.add_argument(
        "--log-estimated-latency",
        help="whether to log the estimated latency to the file after tuning a task",
        action="store_true",
    )
    autotvm_group = parser.add_argument_group(
        "autotvm options",
        "autotvm options, used when the autoscheduler is not enabled",
    )
    autotvm_group.add_argument(
        "--tuner",
        choices=["ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb-rank"],
        default="xgb",
        help="type of tuner to use when tuning with autotvm.",
    )
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


def drive_tune(args):
    """Invoke auto-tuning with command line arguments

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
    mod, params = frontends.load_model(args.FILE, args.model_format, shape_dict=args.input_shapes)

    # min_repeat_ms should be:
    # a. the value provided by the user, if any, or
    # b. 0ms in case target is "cpu"; otherwise 1000ms
    if args.min_repeat_ms is not None:
        min_repeat_ms = args.min_repeat_ms
    else:
        min_repeat_ms = 0 if target.keys[0] == "cpu" else 1000
        logger.debug("Default --min-repeat-ms for this target is %s", min_repeat_ms)

    if args.rpc_tracker:
        runner_ctor = auto_scheduler.RPCRunner if args.enable_autoscheduler else autotvm.RPCRunner
        runner = runner_ctor(
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
        runner_ctor = (
            auto_scheduler.LocalRunner if args.enable_autoscheduler else autotvm.LocalRunner
        )
        runner = runner_ctor(
            number=args.number,
            repeat=args.repeat,
            timeout=args.timeout,
            min_repeat_ms=min_repeat_ms,
        )

    if args.enable_autoscheduler:
        # Specify hardware parameters
        hardware_params = auto_scheduler.HardwareParams(
            args.num_cores,
            args.vector_unit_bytes,
            args.cache_line_bytes,
            args.max_shared_memory_per_block,
            args.max_local_memory_per_block,
            args.max_threads_per_block,
            args.max_vthread_extent,
            args.warp_size,
        )
        tasks, weights = autoscheduler_get_tuning_tasks(
            mod=mod,
            params=params,
            target=target,
            target_host=args.target_host,
            alter_layout=args.desired_layout,
            hardware_params=hardware_params,
            include_simple_tasks=args.include_simple_tasks,
        )

        # Create the autoscheduler tuning options
        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=args.trials,
            measure_callbacks=[auto_scheduler.RecordToFile(args.output)],
            runner=runner,
            early_stopping=args.early_stopping,
        )

        # Schedule the tasks (i.e., produce a schedule for each task)
        schedule_tasks(
            tasks, weights, tuning_options, args.tuning_records, args.log_estimated_latency
        )
    else:
        tasks = autotvm_get_tuning_tasks(
            mod=mod,
            params=params,
            target=target,
            target_host=args.target_host,
            alter_layout=args.desired_layout,
        )

        tuning_option = {
            "tuner": args.tuner,
            "trials": args.trials,
            "early_stopping": args.early_stopping,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="default"), runner=runner
            ),
            "tuning_records": args.tuning_records,
        }
        logger.debug(" tuning options: %s", tuning_option)

        tune_tasks(tasks, args.output, **tuning_option)


def autotvm_get_tuning_tasks(mod, params, target, target_host=None, alter_layout=None):
    """Get the autotvm tuning tasks for a given relay module.

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
    """
    if alter_layout:
        mod = common.convert_graph_layout(mod, alter_layout)

    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        target_host=target_host,
        params=params,
    )

    return tasks


def autoscheduler_get_tuning_tasks(
    mod,
    params,
    target,
    target_host=None,
    alter_layout=None,
    hardware_params=None,
    include_simple_tasks=False,
):
    """Get the autoscheduler tuning tasks for a given relay module.

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
    hardware_params : Optional[HardwareParams]
        Hardware parameters used for the search tasks

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
        mod["main"],
        params,
        target=target,
        target_host=target_host,
        hardware_params=hardware_params,
        include_simple_tasks=include_simple_tasks,
    )

    return tasks, task_weights


def schedule_tasks(
    tasks, task_weights, tuning_options, tuning_records=None, log_estimated_latency=False
):
    """Generate the schedules for the different tasks (i.e., subgraphs) contained in the module.
    Store the schedules in a json file that will be used later by the compiler.

    Parameters
    ----------
    tasks : list
        A list of auto_scheduler.SearchTask to tune.
    task_weights : list
        The weight (i.e. the number of appearance) of extracted tasks
    tuning_options: dict
        The options of tuning
    tuning_records : str, optional
        The json file used to preload the autoscheduler
    """
    if not log_estimated_latency:
        callbacks = [auto_scheduler.task_scheduler.PrintTableInfo()]
    else:
        callbacks = [
            auto_scheduler.task_scheduler.PrintTableInfo(),
            auto_scheduler.task_scheduler.LogEstimatedLatency(("total_latency.tsv")),
        ]

    # Create the scheduler
    tuner = auto_scheduler.TaskScheduler(
        tasks, task_weights, load_log_file=tuning_records, callbacks=callbacks
    )

    # Tune the tasks
    tuner.tune(tuning_options)


def tune_tasks(
    tasks,
    log_file,
    measure_option,
    tuner,
    trials,
    early_stopping=None,
    tuning_records=None,
):
    """Tune a list of tasks and output the history to a log file.

    Parameters
    ----------
    tasks : list
        A list of autotvm.Tasks to tune.
    log_file : str
        A file to output the tuning history, in JSON.
    measure_option : autotvm.measure_option
        Options to build and run a tuning task.
    tuner : str
        Which tuner to use.
    trials : int
        The maximum number of tuning trials to perform.
    early_stopping : int, optional
        The minimum number of tuning trials to perform.
        This will be equal to 'trials' if not specified.
    tuning_records: str, optional
        Path to the file produced by the tuning, to be used during
        tuning.
    """
    if not tasks:
        logger.warning("there were no tasks found to be tuned")
        return

    if not early_stopping:
        early_stopping = trials

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # Create a tuner
        if tuner in ("xgb", "xgb-rank"):
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise TVMCException("invalid tuner: %s " % tuner)

        # If transfer learning is being used, load the existing results
        if tuning_records and os.path.exists(tuning_records):
            logger.info("loading tuning records from %s", tuning_records)
            start_time = time.time()
            tuner_obj.load_history(autotvm.record.load_from_file(tuning_records))
            logging.info("loaded history in %.2f sec(s)", time.time() - start_time)

        tuner_obj.tune(
            n_trial=min(trials, len(tsk.config_space)),
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(trials, prefix=prefix),
                autotvm.callback.log_to_file(log_file),
            ],
        )
