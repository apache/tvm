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
Common utility functions shared by TVMC modules.
"""
import logging
import os.path

import tvm

from tvm import relay
from tvm import transform


# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


class TVMCException(Exception):
    """TVMC Exception"""


def convert_graph_layout(mod, desired_layout):
    """Alter the layout of the input graph.

    Parameters
    ----------
    mod : tvm.relay.Module
        The relay module to convert.
    desired_layout : str
        The layout to convert to.

    Returns
    -------
    mod : tvm.relay.Module
        The converted module.
    """

    # Assume for the time being that graphs only have
    # conv2d as heavily-sensitive operators.
    desired_layouts = {
        "nn.conv2d": [desired_layout, "default"],
        "qnn.conv2d": [desired_layout, "default"],
    }

    # Convert the layout of the graph where possible.
    seq = transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )

    with transform.PassContext(opt_level=3):
        try:
            return seq(mod)
        except Exception as err:
            raise TVMCException(
                "Error converting layout to {0}: {1}".format(desired_layout, str(err))
            )


# TODO In a separate PR, eliminate the duplicated code here and in compiler.py (@leandron)
def target_from_cli(target):
    """
    Create a tvm.target.Target instance from a
    command line interface (CLI) string.

    Parameters
    ----------
    target : str
        compilation target as plain string,
        inline JSON or path to a JSON file

    Returns
    -------
    tvm.target.Target
        an instance of target device information
    """

    if os.path.exists(target):
        with open(target) as target_file:
            logger.info("using target input from file: %s", target)
            target = "".join(target_file.readlines())

    # TODO(@leandron) We don't have an API to collect a list of supported
    #       targets yet
    logger.debug("creating target from input: %s", target)

    return tvm.target.Target(target)
