<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Documents manual TE schedule to illustrate Hexagon operator slicing.

High Level Notes:
* Using float32 (for now) so that tests will pass on CPU
* Using global storage scope (for now) which means "cache" reads and writes from global, to global
* TIR is pending changes from the work-in-progress layout RFC
  (https://github.com/apache/tvm-rfcs/pull/39)
* TIR has been hand-edited for context and clarity
  * Added C-style comments
  * Changed variable names
  * Added spacing and line breaks
* Naming conventions
  * Using input (instead of activation)
  * Using filter (instead of weight, kernel)
  * Using `k` to denote channel-out and `c` or `rc` (reduction channel) to denote channel-in
  * Using `rh` and `rw` (reduction height / width) to denote filter height and width

[Conv2d](test_conv2d_blocked.md)

[Conv2d -> Conv2d](test_conv2d_conv2d.md)
