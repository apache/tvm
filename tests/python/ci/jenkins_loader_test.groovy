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

import org.codehaus.groovy.control.CompilerConfiguration

class LoaderPipeline extends Script {
  Object run() {}
  Object evaluate(String source) {
    state.evaluated << source
    if (state.evaluationFailure) throw state.evaluationFailure
    return 'loaded helper'
  }
  def readTrusted(String path) {
    assert path == 'ci/jenkins/generated/common.groovy'
    if (state.readFailure) throw state.readFailure
    return 'selected revision helper'
  }
  def node(String label, Closure body) {
    throw new AssertionError('loader allocated a node')
  }
  def sh(Map options) {
    throw new AssertionError('loader ran a shell command')
  }

}

def loader = new GroovyClassLoader(getClass().classLoader)
def abort = loader.parseClass('package hudson; class AbortException extends IOException { AbortException(String message) { super(message) } }')
def config = new CompilerConfiguration(scriptBaseClass: 'LoaderPipeline')
def helper = 'ci/jenkins/generated/common.groovy'
def modified = { abort.newInstance("‘${helper}’ has been modified in an untrusted revision".toString()) }
def missing = { new FileNotFoundException(helper) }
def cases = [
  [name: 'unchanged PR'],
  [name: 'trusted named branch', branch: true],
  [name: 'modified PR', readFailure: modified()],
  [name: 'deleted PR lightweight', readFailure: missing()],
  [name: 'deleted PR heavyweight', readFailure: abort.newInstance("/workspace/trusted/${helper} not found".toString())],
  [name: 'release target PR', readFailure: modified(), target: 'release'],
  [name: 'missing named branch', branch: true, readFailure: missing()],
  [name: 'rejected named branch', branch: true, readFailure: modified()],
  [name: 'unrelated abort', readFailure: abort.newInstance('controller offline')],
  [name: 'unrelated missing file', readFailure: new FileNotFoundException('other/file')],
  [name: 'read network failure', readFailure: new IOException('network')],
  [name: 'interrupted read', readFailure: new InterruptedException('cancelled')],
  [name: 'selected evaluation error', evaluationFailure: new IllegalStateException('bad helper')],
]
int count = 0
args.each { filename ->
  def fullSource = new File(filename).text
  def source = fullSource.substring(0, fullSource.indexOf('\nci.start()'))
  cases.each { scenario ->
    def state = [nodes: 0, fetches: 0, evaluated: []] + scenario
    def binding = new Binding([state: state, env: [CHANGE_ID: scenario.branch ? null : '123', CHANGE_TARGET: scenario.target ?: 'main']])
    Throwable failure
    try {
      new GroovyShell(loader, binding, config).evaluate(source)
    } catch (Throwable caught) {
      failure = caught
    }
    def expected = scenario.readFailure ?: scenario.evaluationFailure
    assert failure.is(expected): "${filename}: ${scenario.name}: ${failure} != ${expected}"
    assert state.nodes == 0: scenario.name
    assert state.fetches == 0: scenario.name
    def shouldEvaluate = !scenario.readFailure
    assert state.evaluated == (shouldEvaluate ? ['selected revision helper'] : []): scenario.name
    count++
  }
}
println("${count} native loader cases passed")
