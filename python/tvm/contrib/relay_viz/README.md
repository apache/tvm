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
4. [Design and Customization](#Design-and-Customization)

## Requirement

### Terminal Backend
1. TVM

### Bokeh Backend
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
vizer = relay_viz.RelayVisualizer(mod, relay_param=params, backend=PlotterBackend.BOKEH)
vizer.render("output.html")
```

## Credits

1. https://github.com/apache/tvm/pull/4370

2. https://tvm.apache.org/2020/07/14/bert-pytorch-tvm

3. https://discuss.tvm.apache.org/t/rfc-visualizing-relay-program-as-graph/4825/17

## Design and Customization

This utility is composed of two parts: `node_edge_gen.py` and `plotter.py`.

`plotter.py` define interfaces of `Graph` and `Plotter`. `Plotter` is responsible to render a collection of `Graph`.

`node_edge_gen.py` define interfaces of converting Relay IR modules to nodes and edges. Further, this python module provide a default implementation for common relay types.

If customization is wanted for a certain relay type, we can implement the `NodeEdgeGenerator` interface, handling that relay type accordingly, and delegate other types to the default implementation. See `_terminal.py` for an example usage.

These two interfaces are glued by the top level class `RelayVisualizer`, which passes a relay module to `NodeEdgeGenerator` and add nodes and edges to `Graph`.
Then, it render the plot by calling `Plotter.render()`.
