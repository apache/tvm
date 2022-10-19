#!/bin/bash
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

set -e
set -u
set -o pipefail

# caffe2.python.module.download generates a progress bar. in non
# interactive use this results in huge progress debris in the log
# files.  There is no option to disable the progress bar so work
# around it by stripping the progress bar output

filter_progress_bar()
{
  # Progress bars are the 'goto start of line' escape sequence
  # ESC[1000D[ repeated, the end of the progress bar is the end of
  # line.  We can selectively remove progress bars by dropping lines
  # that beging with the escape sequence.
  sed "/^\x1b\[1000D/d"
}

python3 -m caffe2.python.models.download -i -f squeezenet | filter_progress_bar
python3 -m caffe2.python.models.download -i -f resnet50 | filter_progress_bar
python3 -m caffe2.python.models.download -i -f vgg19 | filter_progress_bar
