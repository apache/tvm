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
# pylint: disable=invalid-name
"""The legacy TVM parser """
from .ir.base import deprecated

# pylint: disable=import-outside-toplevel


@deprecated("tvm.parser.parse", "tvm.relay.parse")
def parse(*args, **kwargs):
    """Deprecated, use `tvm.relay.parse` instead"""
    from tvm.relay import parse as _impl

    return _impl(*args, **kwargs)


@deprecated("tvm.parser.parse_expr", "tvm.relay.parse_expr")
def parse_expr(*args, **kwargs):
    """Deprecated, use `tvm.relay.parse_expr` instead"""
    from tvm.relay import parse_expr as _impl

    return _impl(*args, **kwargs)


@deprecated("tvm.parser.fromtext", "tvm.relay.fromtext")
def fromtext(*args, **kwargs):
    """Deprecated, use `tvm.relay.fromtext` instead"""
    from tvm.relay import fromtext as _impl

    return _impl(*args, **kwargs)


@deprecated("tvm.parser.SpanCheck", "tvm.relay.SpanCheck")
def SpanCheck(*args, **kwargs):
    """Deprecated, use `tvm.relay.SpanCheck` instead"""
    from tvm.relay import SpanCheck as _impl

    return _impl(*args, **kwargs)
