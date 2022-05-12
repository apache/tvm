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
TVMC Shape Parsing
"""

import argparse
import re

from tvm import relay


def parse_shape_string(inputs_string):
    """Parse an input shape dictionary string to a usable dictionary.

    Parameters
    ----------
    inputs_string: str
        A string of the form "input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]" that
        indicates the desired shape for specific model inputs. Colons, forward slashes and dots
        within input_names are supported. Spaces are supported inside of dimension arrays.

    Returns
    -------
    shape_dict: dict
        A dictionary mapping input names to their shape for use in relay frontend converters.
    """

    # Create a regex pattern that extracts each separate input mapping.
    # We want to be able to handle:
    # * Spaces inside arrays
    # * forward slashes inside names (but not at the beginning or end)
    # * colons inside names (but not at the beginning or end)
    # * dots inside names
    pattern = r"(?:\w+\/)?[:\w.]+\:\s*\[\-?\d+(?:\,\s*\-?\d+)*\]"
    input_mappings = re.findall(pattern, inputs_string)
    if not input_mappings:
        raise argparse.ArgumentTypeError(
            "--input-shapes argument must be of the form "
            '"input_name:[dim1,dim2,...,dimn] input_name2:[dim1,dim2]"'
        )
    shape_dict = {}
    for mapping in input_mappings:
        # Remove whitespace.
        mapping = mapping.replace(" ", "")
        # Split mapping into name and shape.
        name, shape_string = mapping.rsplit(":", 1)
        # Convert shape string into a list of integers or Anys if negative.
        shape = [int(x) if int(x) > 0 else relay.Any() for x in shape_string.strip("][").split(",")]
        # Add parsed mapping to shape dictionary.
        shape_dict[name] = shape

    return shape_dict
