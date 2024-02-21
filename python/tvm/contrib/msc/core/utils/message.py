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
"""tvm.contrib.msc.core.utils.message"""

import datetime
import logging
from typing import List

from .arguments import dump_dict
from .log import get_global_logger
from .namespace import MSCMap, MSCKey


class MSCStage(object):
    """Enum all msc stage names"""

    SETUP = "setup"
    PREPARE = "prepare"
    PARSE = "parse"
    BASELINE = "baseline"
    PRUNE = "prune"
    QUANTIZE = "quantize"
    DISTILL = "distill"
    OPTIMIZE = "optimize"
    COMPILE = "compile"
    SUMMARY = "summary"
    ALL = [SETUP, PREPARE, PARSE, BASELINE, PRUNE, QUANTIZE, DISTILL, OPTIMIZE, COMPILE, SUMMARY]

    @classmethod
    def all_stages(cls) -> List[str]:
        """Get all stage names"""
        return cls.ALL


def time_stamp(stage: str, log_stage: bool = True, logger: logging.Logger = None):
    """Mark the stamp and record time.

    Parameters
    ----------
    stage: str
        The stage name.
    log_stage: bool
        Whether to log the stage
    logger: logging.Logger
        The logger.
    """

    logger = logger or get_global_logger()
    time_stamps = MSCMap.get(MSCKey.TIME_STAMPS, [])
    time_stamps.append((stage, datetime.datetime.now()))
    MSCMap.set(MSCKey.TIME_STAMPS, time_stamps)
    if stage in MSCStage.all_stages():
        if log_stage:
            last_stage = MSCMap.get(MSCKey.MSC_STAGE)
            if last_stage:
                end_msg = "[MSC] End {}".format(last_stage.upper())
                logger.info("\n{0} {1} {0}\n".format("#" * 20, end_msg.center(40)))
            start_msg = "[MSC] Start {}".format(stage.upper())
            logger.info("\n{0} {1} {0}".format("#" * 20, start_msg.center(40)))
        MSCMap.set(MSCKey.MSC_STAGE, stage.upper())
    elif log_stage:
        logger.debug("Start {}".format(stage))


def get_duration() -> dict:
    """Get duration of the whole process.

    Returns
    -------
    duration: dict
        The duration of the process.
    """

    time_stamps = MSCMap.get(MSCKey.TIME_STAMPS, [])
    if not time_stamps:
        return {}

    def _get_duration(start_idx, end_idx):
        return (time_stamps[end_idx][1] - time_stamps[start_idx][1]).total_seconds()

    total = _get_duration(0, -1)
    duration = {"total": total}
    for idx in range(len(time_stamps) - 1):
        duration[time_stamps[idx][0]] = _get_duration(idx, idx + 1)
    sub_durations = {}
    for stage, _ in time_stamps:
        if stage not in duration:
            continue
        if "." in stage:
            main_stage = stage.split(".")[0]
            if main_stage not in sub_durations:
                sub_durations[main_stage] = {"total": 0}
            if main_stage in duration and "init" not in sub_durations[main_stage]:
                sub_durations[main_stage]["init"] = duration[main_stage]
                sub_durations[main_stage]["total"] += duration[main_stage]
            sub_duration = duration.pop(stage)
            sub_durations[main_stage][stage.replace(main_stage + ".", "")] = sub_duration
            sub_durations[main_stage]["total"] += sub_duration

    # change to report format
    def _to_str(dur):
        return "{:.2f} s({:.2f}%)".format(dur, dur * 100 / total)

    for sub_dur in sub_durations.values():
        for stage in sub_dur:
            sub_dur[stage] = _to_str(sub_dur[stage])
    for stage in duration:
        duration[stage] = _to_str(duration[stage])
    duration.update(sub_durations)
    return duration


def msg_table(title: str, msg: str, width: int = 100):
    """Log message in table format

    Parameters
    ----------
    title: str
        The title of the block
    msg: str
        The message to log.
    width: int
        The max width of block message

    Returns
    -------
    msg: str
        The block message.
    """

    if isinstance(msg, dict):
        msg = dump_dict(msg, "table:" + str(width))
    return "\n{0} {1} {0}\n{2}\n".format("-" * 20, title.center(40), msg)


def msg_block(title: str, msg: str, width: int = 100):
    """Log message in block format

    Parameters
    ----------
    title: str
        The title of the block
    msg: str
        The message to log.
    width: int
        The max width of block message

    Returns
    -------
    msg: str
        The block message.
    """

    if isinstance(msg, dict):
        msg = dump_dict(msg, "table:" + str(width))
    return "\n{0} {1} {0}\n{2}\n{3} {1} {3}".format(">" * 20, title.center(40), msg, "<" * 20)


def current_stage():
    """Get the current stage"""

    return MSCMap.get(MSCKey.MSC_STAGE, "Unknown")
