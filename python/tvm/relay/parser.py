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
"""A parser for Relay's text format."""
from __future__ import absolute_import
from .. import register_func

def enabled():
    """Checks whether the parser is enabled, this allows users to
       optionally support building the parser.

       We use this check before importing the parser.
    """
    try:
        # pylint: disable=unused-variable
        from tvm.relay import _parser
        return True
    # pylint: disable=broad-except
    except Exception:
        return False

@register_func("relay.fromtext")
def fromtext(data, source_name=None):
    """Parse a Relay program."""
    from tvm.relay import _parser
    return _parser.fromtext(data, source_name)
