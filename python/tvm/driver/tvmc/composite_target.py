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
Provides support to composite target on TVMC.
"""
import logging

from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
from tvm.relay.op.contrib.ethosn import partition_for_ethosn

from .common import TVMCException


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")

# Global dictionary to map targets with the configuration key
# to be used in the PassContext (if any), and a function
# responsible for partitioning to that target.
REGISTERED_CODEGEN = {
    "compute-library": {
        "config_key": None,
        "pass_pipeline": partition_for_arm_compute_lib,
    },
    "ethos-n77": {
        "config_key": "relay.ext.ethos-n.options",
        "pass_pipeline": partition_for_ethosn,
    },
}


def get_codegen_names():
    """Return a list of all registered codegens.

    Returns
    -------
    list of str
        all registered targets
    """
    return list(REGISTERED_CODEGEN.keys())


def get_codegen_by_target(name):
    """Return a codegen entry by name.

    Returns
    -------
    dict
        requested target information
    """
    try:
        return REGISTERED_CODEGEN[name]
    except KeyError:
        raise TVMCException("Composite target %s is not defined in TVMC." % name)
