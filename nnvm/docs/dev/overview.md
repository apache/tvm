
# NNVM Design Overview

NNVM is a reusable graph IR stack for deep learning systems. It provides useful API to construct, represent and transform computation graphs to get most high-level optimization needed in deep learning.
As a part of TVM stack for deep learning, NNVM also provides a shared compiler for deep learning frameworks to optimize, compile and deploy into different hardware backends via [TVM](https://github.com/dmlc/tvm)

## Key Requirements and Design Choices

- Have minimum dependency in the deployment module.
- Being able to add new operators to the IR, in a decentralized fashion.
- Being able to add new optimization passes to the IR and applies to existing graphs.

The item2 and 3 are particularly interesting if we compare it to a typical compiler IR. Compiler IR usually contains a fixed set of primitives(instructions), and use them as a contract between optimization pass designers. This design enables easy addition of new optimization passes, but not new operator(instruction). Because every time we add a new instruction, we need to modify the passes to accommodate these changes.

Deep learning frameworks usually have a fixed operator interface(schema). These interfaces can contain properties like shape inference function, whether in-place computation can happen.  The operator interface is an again contract that makes it easy to add new an operator. But it is hard to add new passes in decentralized fashion a new optimization pass usually requires additional information, and this results in frequent changes of the centralized operator interface when we are exploring new optimizations. There is also a drawback of modularization. For example, a graph compiler for FPGA devices may not need the GPU device specific attributes.

During our explorations in graph optimization and compilation, we find that it is important to quickly add both operators and passes to the framework without changing the core library.

Here is a list of key elements in NNVM's design

-  Operator registry system to register and add new operators
-  Operator attribute system provide property of operator in decentralized fashion
-  A reusable IR data structure for optimization passes.

The above list is more like the generic language part of NNVM, besides of that, we also provide a collection of core operator primitives, and graph optimization passes.   The core tensor operator primitives and optimizations already cover commonly deep learning workloads. This design allows the NNVM compiler to be directly used as optimization and compilation stack for frameworks. The extendible nature of NNVM makes new adjustment easy without constraining the backend providers.

## Minimum Registration for a Symbolic Front-End
To use NNVM to build language front end, a developer only needs to register minimum information about each operator.

```c++
NNVM_REGISTER_OP(add)
.describe("add two data together")
.set_num_inputs(2);

NNVM_REGISTER_OP(conv2d)
.describe("take 2d convolution of input")
.set_num_inputs(2);

NNVM_REGISTER_OP(assign)
.describe("assign second input argument to the first one")
.set_num_inputs(2);
```

Compiling the code with NNVM library. User can use the following interface to compose the computation graph in python, like the following code.

```python
import nnvm.symbol as nn

# symbolic variable
x = nn.Variable('x')
y = nn.Variable('y')
w = nn.Variable('w')

z = nn.conv2d(nn.elemwise_add(x, y), w, kernel_size=(2,2), name='conv1')
```

The graph structure is interchangeable between the frontend and the backend.  Python interface is supported currently. More language support can be easily
moved in the future.

## Operator Attribute for More Extensions

The minimum information provided by the operator is enough to get a front-end. However,   we need more knowledge about each operator to do transformations and executing the graph.
A typical difference between neural nets' computation graph and traditional compiler IR is that there are a lot more high-level operators. We cannot fix the set of operators in the IR.

NNVM allow developers to register attributes of each operator. The attributes can include shape inference function, whether the operator can perform in-place calculation etc.

This design to having an operator attribute registry is not uncommon in deep learning systems.
For example, MXNet has a ```OpProperty``` class, Tensorflow has a ```OpDef``` and Caffe2 have a ```OperatorSchema``` class.
However, the operator attribute interface listed in these frameworks only support a fixed number of defined attributes of interest to the system. If we want to extend the framework to add a new attribute in each operator, we need to change the operator registry.
Eventually, the operator interface grows into to be very big and have to evolve in the centralized repo.

In NNVM, we decided to change the design and support arbitrary type of operator attributes, without changing the interface registry. The minimum interface also makes it easier to share across multiple projects

User can register new attribute, such as inplace property checking function as follows.
```c++
using FInplaceOption = std::function<
  std::vector<std::pair<int, int> > (const NodeAttrs& attrs)>;

// we can register attributes from multiple places.
NNVM_REGISTER_OP(elemwise_add)
.set_num_inputs(2);

// register to tell first input can be calculate inplace with first output
NNVM_REGISTER_OP(add)
.set_attr<FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
 });

NNVM_REGISTER_OP(exp)
.set_num_inputs(1)
.set_attr<FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
 });
```

We can query these attributes at arbitrary parts of the code, like the following parts. Under the hood, each attribute is stored in a columnar store, that can easily be retrieved table and do quick lookups.

```c++
void MyFunction() {
  const Op* add = Op::Get("add");
  // if we need quick query, we can use static variable
  // attribute map contains attributes of all operators.
  static auto& finplace_option_map = Op::GetAttr<FInplaceOption>("FInplaceOption");

  // quick look up attribute of add, O(1) time, vector index lookup internally.
  auto add_inplace = finplace_option_map[add];
}
```
Besides making the code minimum, this attribute store enables decentralization of projects.
Before, all the attributes of operator have to sit on a centralized interface class.
Now, everyone can register attributes of their own, take some other attributes they need from another project without changing the operator interface and core library


## Graph and Pass

We can use the additional information on attribute registry to do optimizations and get more information about the graph. Graph is the unit we manipulate in these steps. A Graph in NNVM contains
two parts:
- The computation graph structure
- A attribute map from string to any type ```map<string, shared_ptr<any> >```

The second attribute map is quite important, as we may need different kinds
of information about the graph during the transformation process. Let it be
shapes of each tensor, types of each tensor or the storage allocation plans.

A ```Pass``` can take a graph with existing attribute information,
and transform it to the same graph structure with more graph attributes or another graph.
