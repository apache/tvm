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

# Make sure Vitis AI codegen is registered
import tvm.contrib.target.vitis_ai  # pylint: disable=unused-import

from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
from tvm.relay.op.contrib.ethosn import partition_for_ethosn
from tvm.relay.op.contrib.cmsisnn import partition_for_cmsisnn
from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tvm.relay.op.contrib.bnns import partition_for_bnns
from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai


from tvm.driver.tvmc import TVMCException


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


# Global dictionary to map targets
#
# Options
# -------
# config_key : str
#   The configuration key to be used in the PassContext (if any).
# pass_pipeline : Callable
#   A function to transform a Module before compilation, mainly used
#   for partitioning for the target currently.
REGISTERED_CODEGEN = {
    "compute-library": {
        "config_key": None,
        "pass_pipeline": partition_for_arm_compute_lib,
    },
    "cmsis-nn": {
        "config_key": "relay.ext.cmsisnn.options",
        "pass_pipeline": partition_for_cmsisnn,
    },
    "ethos-n": {
        "config_key": "relay.ext.ethos-n.options",
        "pass_pipeline": partition_for_ethosn,
    },
    "ethos-u": {
        "config_key": "relay.ext.ethos-u.options",
        "pass_pipeline": partition_for_ethosu,
    },
    "bnns": {
        "config_key": None,
        "pass_pipeline": partition_for_bnns,
    },
    "vitis-ai": {
        "config_key": "relay.ext.vitis_ai.options",
        "pass_pipeline": partition_for_vitis_ai,
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

    Parameters
    ----------
    name : str
        The name of the target for which the codegen info should be retrieved.

    Returns
    -------
    dict
        requested target codegen information
    """
    try:
        return REGISTERED_CODEGEN[name]
    except KeyError:
        raise TVMCException("Composite target %s is not defined in TVMC." % name)
