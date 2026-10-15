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

ci.build('CPU', 'BUILD: CPU', 'tvm/build-cpu', images.ci_cpu, 'cpu') {
  sh(
    script: "${docker_run} ${images.ci_cpu} ./tests/scripts/task_config_build_cpu.sh build",
    label: 'Create CPU CMake config',
  )
  ci.cmake_build(images.ci_cpu, 'build')
  ci.make_cpp_tests(images.ci_cpu, 'build')
  ci.upload('cpu', '--bundle tvm_lib --bundle cpptest')
}

stage('Test') {
  parallel('unittest: CPU': {
    ci.test('CPU-SMALL', 'unittest: CPU', 'tvm/ut-python-cpu', images.ci_cpu, 'cpu') {
      ci.download('cpu')
      ci.cpp_unittest(images.ci_cpu)
      ci.python_unittest(images.ci_cpu)
    }
  })
}
