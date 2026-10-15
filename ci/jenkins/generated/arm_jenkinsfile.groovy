// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Reject untrusted helper edits before allocating nodes or running pipeline steps.
ci = evaluate(readTrusted('ci/jenkins/generated/common.groovy'))
ci.start()

ci.build('ARM-GRAVITON3', 'BUILD: arm', 'tvm/build-arm', images.ci_arm, 'arm') {
  sh(
    script: "${docker_run} ${images.ci_arm} ./tests/scripts/task_config_build_arm.sh build",
    label: 'Create ARM CMake config',
  )
  ci.cmake_build(images.ci_arm, 'build')
  ci.make_cpp_tests(images.ci_arm, 'build')
  ci.upload('arm', '--bundle tvm_lib --bundle cpptest')
}
