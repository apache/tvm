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
# pylint: disable=consider-using-enumerate,invalid-name
"""Namespace of callback utilities of AutoTVM"""
import sys
import time
import logging

import numpy as np

from .. import record
from ..utils import format_si_prefix

logger = logging.getLogger("autotvm")


def log_to_file(file_out, protocol="json"):
    """Log the tuning records into file.
    The rows of the log are stored in the format of autotvm.record.encode.

    Parameters
    ----------
    file_out : File or str
        The file to log to.
    protocol: str, optional
        The log protocol. Can be 'json' or 'pickle'

    Returns
    -------
    callback : callable
        Callback function to do the logging.
    """

    def _callback(_, inputs, results):
        """Callback implementation"""
        if isinstance(file_out, str):
            with open(file_out, "a") as f:
                for inp, result in zip(inputs, results):
                    f.write(record.encode(inp, result, protocol) + "\n")
        else:
            for inp, result in zip(inputs, results):
                file_out.write(record.encode(inp, result, protocol) + "\n")

    # pylint: disable=import-outside-toplevel
    from pathlib import Path

    if isinstance(file_out, Path):
        file_out = str(file_out)

    return _callback


def log_to_database(db):
    """Save the tuning records to a database object.

    Parameters
    ----------
    db: Database
        The database
    """

    def _callback(_, inputs, results):
        """Callback implementation"""
        for inp, result in zip(inputs, results):
            db.save(inp, result)

    return _callback


class Monitor(object):
    """A monitor to collect statistic during tuning"""

    def __init__(self):
        self.scores = []
        self.timestamps = []

    def __call__(self, tuner, inputs, results):
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                flops = inp.task.flop / np.mean(res.costs)
                self.scores.append(flops)
            else:
                self.scores.append(0)

            self.timestamps.append(res.timestamp)

    def reset(self):
        self.scores = []
        self.timestamps = []

    def trial_scores(self):
        """get scores (currently is flops) of all trials"""
        return np.array(self.scores)

    def trial_timestamps(self):
        """get wall clock time stamp of all trials"""
        return np.array(self.timestamps)


def progress_bar(total, prefix="", si_prefix="G"):
    """Display progress bar for tuning

    Parameters
    ----------
    total: int
        The total number of trials
    prefix: str
        The prefix of output message
    si_prefix: str
        SI prefix for flops
    """

    class _Context(object):
        """Context to store local variables"""

        def __init__(self):
            self.best_flops = 0
            self.cur_flops = 0
            self.ct = 0
            self.total = total

        def __del__(self):
            if logger.level < logging.DEBUG:  # only print progress bar in non-debug mode
                sys.stdout.write(" Done.\n")

    ctx = _Context()
    tic = time.time()

    # Validate si_prefix argument
    format_si_prefix(0, si_prefix)

    if logger.level < logging.DEBUG:  # only print progress bar in non-debug mode
        sys.stdout.write(
            "\r%s Current/Best: %7.2f/%7.2f %sFLOPS | Progress: (%d/%d) "
            "| %.2f s" % (prefix, 0, 0, si_prefix, 0, total, time.time() - tic)
        )
        sys.stdout.flush()

    def _callback(tuner, inputs, results):
        ctx.ct += len(inputs)

        flops = 0
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                flops = inp.task.flop / np.mean(res.costs)

        if not logger.isEnabledFor(logging.DEBUG):  # only print progress bar in non-debug mode
            ctx.cur_flops = flops
            ctx.best_flops = tuner.best_flops

            sys.stdout.write(
                "\r%s Current/Best: %7.2f/%7.2f %sFLOPS | Progress: (%d/%d) "
                "| %.2f s"
                % (
                    prefix,
                    format_si_prefix(ctx.cur_flops, si_prefix),
                    format_si_prefix(ctx.best_flops, si_prefix),
                    si_prefix,
                    ctx.ct,
                    ctx.total,
                    time.time() - tic,
                )
            )
            sys.stdout.flush()

    return _callback


def visualize_progress(idx, title="AutoTVM Progress", multi=False,  si_prefix="G"):
    """Display tuning progress in graph

    Parameters
    ----------
    idx: int
        Index of the current task.
    title: str
        Specify the title of the matplotlib figure.
    multi: bool
        Add traces for alls tuned tasks into a single plot.
        If False, one plot is generated for each task.
    si_prefix: str
        SI prefix for flops
    """
    import matplotlib.pyplot as plt

    class _Context(object):
        """Context to store local variables"""

        def __init__(self):
            self.best_flops = [0]
            self.all_flops = []
            if multi and idx > 0:
                plt.figure(title)
            else:
                plt.figure(title).clear()
            self.color = plt.cm.tab10(idx)
            (self.p,) = plt.plot([0], [0], color=self.color, label=f"Task {idx}")
            plt.xlabel("Iterations")
            plt.ylabel(f"{si_prefix}FLOPS")
            plt.legend(loc="upper left")
            plt.pause(0.05)

    ctx = _Context()

    def _callback(_, inputs, results):

        flops = 0
        for inp, res in zip(inputs, results):
            m = "x"
            if res.error_no == 0:
                flops = inp.task.flop / np.mean(res.costs)
                m = "."

            flops = format_si_prefix(flops, si_prefix)
            ctx.all_flops.append(flops)
            best = max(flops, ctx.best_flops[-1])
            ctx.best_flops.append(best)

            axes = plt.gca()
            _, ymax = axes.get_ylim()
            _, xmax = axes.get_xlim()
            plt.axis([0, max(len(ctx.all_flops) + 1, xmax), 0, max(ctx.best_flops[-1] * 1.1, ymax)])
            plt.scatter(len(ctx.all_flops), flops, color=ctx.color, marker=m, s=15)
            ctx.p.set_data(list(range(0, len(ctx.all_flops) + 1)), ctx.best_flops)
            plt.pause(0.05)

    return _callback
