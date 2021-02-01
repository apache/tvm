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
import re
import logging
import os.path
import argparse

from urllib.parse import urlparse

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


def tracker_host_port_from_cli(rpc_tracker_str):
    """Extract hostname and (optional) port from strings
    like "1.2.3.4:9090" or "4.3.2.1".

    Used as a helper function to cover --rpc-tracker
    command line argument, in different subcommands.

    Parameters
    ----------
    rpc_tracker_str : str
        hostname (or IP address) and port of the RPC tracker,
        in the format 'hostname[:port]'.

    Returns
    -------
    rpc_hostname : str or None
        hostname or IP address, extracted from input.
    rpc_port : int or None
        port number extracted from input (9090 default).
    """

    rpc_hostname = rpc_port = None

    if rpc_tracker_str:
        parsed_url = urlparse("//%s" % rpc_tracker_str)
        rpc_hostname = parsed_url.hostname
        rpc_port = parsed_url.port or 9090
        logger.info("RPC tracker hostname: %s", rpc_hostname)
        logger.info("RPC tracker port: %s", rpc_port)

    return rpc_hostname, rpc_port


def parse_shape_string(inputs_string):
    """Parse an input shape dictionary string to a usable dictionary.

    Parameters
    ----------
    inputs_string: str
        A string of the form "name:num1xnum2x...xnumN,name2:num1xnum2xnum3" that indicates
        the desired shape for specific model inputs.

    Returns
    -------
    shape_dict: dict
        A dictionary mapping input names to their shape for use in relay frontend converters.
    """
    # Simplify passed input string by removing spaces.
    inputs_string = inputs_string.replace(" ", "")
    # Check if the passed input is in the proper format.
    valid_pattern = re.compile(
        r"(\w+:(\-{0,1}\d+(x|X))*\-{0,1}(\d)+)(,(\w+:(\-{0,1}\d+(x|X))*\-{0,1}(\d)+))*"
    )
    result = re.fullmatch(valid_pattern, inputs_string)
    if result is None:
        raise argparse.ArgumentTypeError(
            "--input-shapes argument must be of the form "
            "input_name:dim1xdim2x...xdimN,input_name2:dim1xdim2"
        )
    shape_dict = {}
    # Break apart each specific input string
    inputs_list = inputs_string.split(",")
    for shape_mapping in inputs_list:
        # Split name from shape string.
        input_name, input_shape_string = shape_mapping.split(":")
        # Separate each dimension in the shape.
        input_shape_chars = input_shape_string.lower().split("x")
        # Parse each dimension into an integer.
        input_shape = []
        for x in input_shape_chars:
            x = int(x)
            # Negative numbers are converted to dynamic axes.
            input_shape.append(x if x >= 0 else relay.Any())
        # Assign dictionary key value pair.
        shape_dict[input_name] = input_shape
    return shape_dict
