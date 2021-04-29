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
"""Produce codegen result from intermediate results
"""


def finalize(lines, export_obj, options):  # pylint: disable=unused-argument
    """Produce codegen result from intermediate results"""
    lines["result"] = "\n".join(lines["tmp"]["wrapper_class"])
    lines["result"] = "\n".join([s for s in lines["result"].split("\n") if s.strip()])
    return lines, export_obj
