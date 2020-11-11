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

"""Utilities for binary file manipulation"""
import logging
import os
import subprocess
import tvm._ffi
from . import utils

_LOG = logging.getLogger(__name__)


def run_cmd(cmd):
    """Runs `cmd` in a subprocess and awaits its completion.

    Parameters
    ----------
    cmd : List[str]
        list of command-line arguments

    Returns
    -------
    output : str
        resulting stdout capture from the subprocess
    """
    _LOG.debug('execute: %s', ' '.join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output, _) = proc.communicate()
    output = output.decode("utf-8")
    if proc.returncode != 0:
        cmd_str = " ".join(cmd)
        msg = f'error while running command "{cmd_str}":\n{output}'
        raise RuntimeError(msg)
    return output
