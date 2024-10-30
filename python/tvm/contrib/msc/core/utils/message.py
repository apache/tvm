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
from typing import List, Tuple

from .arguments import dump_dict, map_dict
from .log import get_global_logger, split_line
from .namespace import MSCMap, MSCKey


class MSCStage(object):
    """Enum all msc stage names"""

    SETUP = "setup"
    PREPARE = "prepare"
    PARSE = "parse"
    PRUNE = "prune"
    QUANTIZE = "quantize"
    DISTILL = "distill"
    TRACK = "track"
    BASELINE = "baseline"
    OPTIMIZE = "optimize"
    COMPILE = "compile"
    SUMMARY = "summary"
    EXPORT = "export"
    ALL = [
        SETUP,
        PREPARE,
        PARSE,
        PRUNE,
        QUANTIZE,
        DISTILL,
        TRACK,
        BASELINE,
        OPTIMIZE,
        COMPILE,
        SUMMARY,
        EXPORT,
    ]

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
        Whether to log the stage.
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
                end_msg = "End {}".format(last_stage.upper())
                logger.info("%s\n", split_line(end_msg))
            start_msg = "Start {}".format(stage.upper())
            logger.info(split_line(start_msg))
        MSCMap.set(MSCKey.MSC_STAGE, stage.upper())
    elif log_stage:
        start_msg = "Start {}".format(stage)
        logger.debug(split_line(start_msg, "+"))


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

    def _get_duration(idx):
        return (time_stamps[idx + 1][1] - time_stamps[idx][1]).total_seconds()

    def _set_stage(stage: str, info: Tuple[float, dict], collect: dict):
        if "." in stage:
            main_stage, sub_stage = stage.split(".", 1)
            _set_stage(sub_stage, info, collect.setdefault(main_stage, {}))
        else:
            collect[stage] = info

    def _set_total(collect: dict):
        collect["total"] = 0
        for dur in collect.values():
            collect["total"] += _set_total(dur) if isinstance(dur, dict) else dur
        return collect["total"]

    duration, depth = {}, 1
    left_durs = {time_stamps[i][0]: _get_duration(i) for i in range(len(time_stamps) - 1)}
    while left_durs:
        current_durs = {s: dur for s, dur in left_durs.items() if len(s.split(".")) == depth}
        left_durs = {k: v for k, v in left_durs.items() if k not in current_durs}
        for stage, dur in current_durs.items():
            info = {"init": dur} if any(s.startswith(stage + ".") for s in left_durs) else dur
            _set_stage(stage, info, duration)
        depth += 1

    _set_total(duration)

    def _to_str(dur):
        if not isinstance(dur, float):
            return dur
        return "{:.2f} s({:.2f}%)".format(dur, dur * 100 / duration["total"])

    return map_dict(duration, _to_str)


def msg_block(title: str, msg: str, width: int = 100, symbol: str = "-"):
    """Log message in block format

    Parameters
    ----------
    title: str
        The title of the block
    msg: str
        The message to log.
    width: int
        The max width of block message
    symbol: str
        The split symbol.

    Returns
    -------
    msg: str
        The block message.
    """

    if isinstance(msg, dict):
        msg = dump_dict(msg, "table:" + str(width))
    return "{}\n{}".format(split_line(title, symbol), msg)


def current_stage():
    """Get the current stage"""

    return MSCMap.get(MSCKey.MSC_STAGE, "Unknown")
