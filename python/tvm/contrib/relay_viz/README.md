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


# IR Visualization

This tool target to visualize Relay IR.

# Table of Contents
1. [Requirement](#Requirement)
2. [Usage](#Usage)
3. [Credits](#Credits)
3. [TODO](#TODO)

## Requirement

1. TVM
2. graphviz
2. pydot
3. bokeh >= 2.3.1

```
# To install TVM, please refer to https://tvm.apache.org/docs/install/from_source.html

# requirements of pydot
apt-get install graphviz

# pydot and bokeh
pip install pydot bokeh==2.3.1
```

## Usage

```
from tvm.contrib import relay_viz
mod, params = tvm.relay.frontend.from_onnx(net, shape_dict)
vizer = relay_viz.RelayVisualizer(mod, relay_param=params)
vizer.render("output.html")
```

## Credits

1. https://github.com/apache/tvm/pull/4370

2. https://tvm.apache.org/2020/07/14/bert-pytorch-tvm

3. https://discuss.tvm.apache.org/t/rfc-visualizing-relay-program-as-graph/4825/17