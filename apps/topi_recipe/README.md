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

# TOPI Recipe: TVM Operator Optimization Recipes

TOPI is the operator collection library for TVM intended at sharing the effort of crafting
and optimizing tvm generated kernels. The goal:

- Provide sugars for operator declaration
- Give common primitives for fused op creation.
- Provide commonly used schedules under each architectures

## Guidelines
- Use numpy-style naming convention for known ops
- Separate operator declaration from schedule when possible.
  - This can be inconvenient but enables more general scheduling across ops.
  - We can always recover the tensors from its outputs by traversing the tree.
- Deliberately assert the requirements
  - Some kernels have requirements on shape and data layout, assert them
- Data layout aware, if not specified in argument or in function, assume NCHW by default.


## Performance Tuning Workflow
Since TVM is work in progress, some optimization might not be perfect.
One quick way I find useful is to do codegen plus manual modification.
The workflow is:

- Generate the GPU kernels, write them into a file, say ```perf/matexp_generated.cu```
- Copy the generated file into another one, say ```perf/matexp_manual.cu```,
  do modifications according to your intuition.
- Set use_manual flag in the script to continue the codegen workflow as normal, but piggy back the manual written code instead.
- Observe the performance difference.
- If the performance improves, mark the manual code and think of optimization pass
  to generate the desired target code.
