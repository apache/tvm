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

VTA in Chisel 
===================================================
For contributors who wants to test a chisel module:
	
 - You can add your test files in  `src/test/scala/unitttest`
 - Add your test name and tests to the `test` object in `src/test/scala/unitttest/Launcher.scala`
 - Check out the provided sample test `mvm` which tests the MatrixVectorComputation module
    in `src/main/scala/core/TensorGemm.scala`

- Running unit tests: `make test test_name=your_own test_name`
	


