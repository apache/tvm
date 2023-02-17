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

# Core Cross Platform Regression Tests

CI Unit test cases that will run on all platforms.
To reduce the CI burden, we only put in test-cases that are platform sensitive.
Please use the following guideline:

- Always consider add tests to the unittest folder first.
- If a problems that passes the Linux pipeline but fails in Windows or MacOS,
  we should isolate the problem, write a minimal regression test case
  and add it to this folder.
- A test case in this folder should be minimal and finish in a reasonable amount of time.
- Document about why it should be in the all-platform-minimal-test.
