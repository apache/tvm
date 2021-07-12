
# IR Visualization

This tool target to visualize Relay IR.

# Table of Contents
1. [Requirement](#Requirement)
2. [Usage](#Usage)
3. [Credits](#Credits)
3. [TODO](#TODO)

## Requirement

1. TVM
2. graphviz and graphviz-dev
2. bokeh==2.3.1
3. pygraphviz==1.6
4. networkx==2.5.1

```
# To install TVM, please refer to https://tvm.apache.org/docs/install/from_source.html

# requirements of pygraphviz
apt-get install graphviz graphviz-dev
# pygraphviz
pip install pygraphviz==1.6

# networkx
pip install networkx==2.5.1

# bokeh
pip install bokeh==2.3.1
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

