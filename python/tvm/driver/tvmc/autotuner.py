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

from tvm import autotvm
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
        "--tuner",
        choices=["ga", "gridsearch", "random", "xgb", "xgb_knob", "xgb-rank"],
        default="xgb",
        help="type of tuner to use",
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
    # TODO (@leandron) This is a path to a physical file, but
    #     can be improved in future to add integration with a modelzoo
    #     or URL, for example.
    parser.add_argument("FILE", help="path to the input model file")


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
    mod, params = frontends.load_model(args.FILE, args.model_format)

    # min_repeat_ms should be:
    # a. the value provided by the user, if any, or
    # b. 0ms in case target is "cpu"; otherwise 1000ms
    if args.min_repeat_ms is not None:
        min_repeat_ms = args.min_repeat_ms
    else:
        min_repeat_ms = 0 if target.keys[0] == "cpu" else 1000
        logger.debug("Default --min-repeat-ms for this target is %s", min_repeat_ms)

    tasks = get_tuning_tasks(
        mod=mod,
        params=params,
        target=target,
        target_host=args.target_host,
        alter_layout=args.desired_layout,
    )

    if args.rpc_tracker:

        runner = autotvm.RPCRunner(
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
        runner = autotvm.LocalRunner(
            number=args.number,
            repeat=args.repeat,
            timeout=args.timeout,
            min_repeat_ms=min_repeat_ms,
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


def get_tuning_tasks(mod, params, target, target_host=None, alter_layout=None):
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
