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

#Combine multi model to one system-lib

## Background
Deploy models on Android & iOS platform must use system-lib. Avoid usage of `dlopen`.  
[bundle_deploy](https://github.com/apache/tvm/tree/main/apps/bundle_deploy) demonstrated how to deploy a model which build a model which target system-lib

BUT we need more than one

`local_test.py` show how to build multi models result in one system-lib and separate graph-json and params for each model

