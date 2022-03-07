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
import argparse
import re

default_team_id = "3FR42MXLK9"
default_tvm_build_dir = "path-to-tvm-ios-build-folder"

parser = argparse.ArgumentParser(
    description="Update tvmrpc.xcodeproj\
 developer information"
)
parser.add_argument(
    "--team_id",
    type=str,
    required=True,
    help="Apple Developer Team ID.\n\
                    Can be found here:\n\
                    \n\
                    https://developer.apple.com/account/#/membership\n\
                    (example: {})".format(
        default_team_id
    ),
)

parser.add_argument(
    "--tvm_build_dir",
    type=str,
    required=True,
    help="Path to directory with libtvm_runtime.dylib",
)

args = parser.parse_args()
team_id = args.team_id
tvm_build_dir = args.tvm_build_dir

fi = open("tvmrpc.xcodeproj/project.pbxproj")
proj_config = fi.read()
fi.close()

proj_config = proj_config.replace(default_team_id, team_id)
proj_config = proj_config.replace(default_tvm_build_dir, tvm_build_dir)
fo = open("tvmrpc.xcodeproj/project.pbxproj", "w")
fo.write(proj_config)
fo.close()
