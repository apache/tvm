..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at
..    http://www.apache.org/licenses/LICENSE-2.0
..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

===================
Convert Layout Pass
===================
**Author**: `Animesh Jain <https://github.com/anijain2305>`_

*************
1. Background
*************

Data layout format describes how the data is laid out in the memory. For example, Tensorflow framework default data layout for convolution operator is NHWC, i.e, the data is 4-dimensions and is laid out in row-major format with N being the first dimension and C being the last dimension. Data layout has a major role in model performance, significantly affecting spatial and temporal locality. For example, Intel x86 backend in TVM prefers layout as NCHWc where the C dimension is tiled in 2 dimensions to exploit data locality efficiently. Similarly, CUDA backend prefers the data layout to be in NCHW format.

Essentially, TVM has to deal with data layouts throughout the compiler toolchain - Framework parsers, Relay layout transformations, and TOPI schedules. As we move towards third-party codegen integration, which might have their own data layout restrictions, handling layouts at all levels in TVM toolchain is going to become even more challenging. Therefore, we developed a new Relay pass - **ConvertLayout** -- to reduce some of the complications that arise due to layout handling.

If you directly want to understand the usage of ConvertLayout Pass, directly jump to Section 4 - Usage.

*************
2. Motivation
*************

Lets look at a simple scenario to understand the complications that arise due to different layouts - Suppose we want to compile a Tensorflow NHWC graph for an ARM edge device. But, suppose we currently support only NCHW schedules in TOPI for ARM. So, there is a mismatch between framework layout and TOPI-supported layout. One way to deal with this mismatch is to insert layout transforms before each and after convolution, such that resulting convolution has NCHW input data layout and can use TOPI schedules. However, this can lead to performance degradation because of the presence of too many layout transforms.

We encountered similar problems in other use cases as well

- No way to run TFLite graphs on Nvidia GPUs. TOPI has NCHW-only schedules for GPUs.
- Ever-complicating logic in AlterOpLayout for convolution to support different pairs of layout transformations.
- Sub-optimal performance for TF graphs due to extra layout transforms.
- Complication in third-party codegen integrations like TRT that prefers data layout to be in one format.

To solve these problems, we introduced *ConvertLayout* pass that sets up the infrastructure to change the data layout of the whole graph with minimal number of data layout transforms. In ideal cases, we will have only 2 layout transforms, one at the start and one at the end. An example to show the transformation is below


.. code-block:: python

	# Original graph - 2 convolutions in NHWC format.
	fn (%x: Tensor[(1, 56, 56, 64), float32], %weight1: Tensor[(3, 3, 64, 32), float32], %weight2: Tensor[(3, 3, 32, 32), float32]) {
	  %0 = nn.conv2d(%x, %weight1, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
	  %1 = nn.relu(%0);
	  %2 = nn.conv2d(%1, %weight2, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
	  nn.relu(%2)
	}

	# After ConvertLayout - For data, there is a transform at the start and at the end.
	# For weights, there are transforms to adapt to NCHW layout. These will be removed with FoldConstant pass.
	fn (%x: Tensor[(1, 56, 56, 64), float32], %weight1: Tensor[(3, 3, 64, 32), float32], %weight2: Tensor[(3, 3, 32, 32), float32]) {
	  %0 = layout_transform(%x, src_layout="NHWC", dst_layout="NCHW") /* ty=Tensor[(1, 64, 56, 56), float32] */;
	  %1 = layout_transform(%weight1, src_layout="HWIO", dst_layout="OIHW") /* ty=Tensor[(32, 64, 3, 3), float32] */;
	  %2 = nn.conv2d(%0, %1, padding=[1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 56, 56), float32] */;
	  %3 = nn.relu(%2) /* ty=Tensor[(1, 32, 56, 56), float32] */;
	  %4 = layout_transform(%weight2, src_layout="HWIO", dst_layout="OIHW") /* ty=Tensor[(32, 32, 3, 3), float32] */;
	  %5 = nn.conv2d(%3, %4, padding=[1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 56, 56), float32] */;
	  %6 = nn.relu(%5) /* ty=Tensor[(1, 32, 56, 56), float32] */;
	  layout_transform(%6, src_layout="NCHW", dst_layout="NHWC") /* ty=Tensor[(1, 56, 56, 32), float32] */
	}


*********
3. Design
*********

ConvertLayout pass is heavily built upon Relay layout rewriter infrastructure. To understand the design, lets break the operators into 3 categories

- **Layout agnostic** - Relu, Add etc. Do not get affected, neither functionality nor performance, by layouts.
- **Lightly-layout sensitive** - Pad, Concatenate, Reduce ops like sum etc - Basically these operators have some attributes that are functionally affected if we do a layout transformation before them. But, there is not much difference in performance.
- **Heavily-layout sensitive** - Convolution, Conv2D transpose etc - Highly affected, both functionally and performance-wise, by data layout. They also have data layout as the op attribute.


We use Relay layout rewriter infrastructure to handle layouts. This pass traverses the graph operator-by-operator. For each operator, it goes through 3 components - 1) A Python callback, allowing developers to transform the operator into a new Relay expr with new layouts, 2) Layout inference - using both original layouts, and transformed expr layouts (from previous operator or from the Python callback), and 3) Automatic layout transform insertion if needed. Now, let's connect these components with the operator categories.

**Python callback for layout alteration** - This is used for *heavily-layout sensitive* operators. For example, one can return a new convolution operator with new data and kernel layout. The other 2 components will infer layout and insert layout transforms if needed. One example for convolution operator is follows where we converting to NCHW layout.

.. code-block:: python

    @reg.register_convert_op_layout("nn.conv2d")
    def convert_conv2d(attrs, inputs, tinfos, desired_layout):
        """Convert Layout pass registration for conv2d op.

        Parameters
        ----------
        attrs : tvm.attrs.Attrs
            Attributes of current convolution
        inputs : list of tvm.relay.Expr
            The args of the Relay expr to be legalized
        tinfos : list of types
            List of input and output types
        desired_layout : str
            The desired layout

        Returns
        -------
        result : tvm.relay.Expr
            The transformed expr
        """

        from tvm import relay
        data_layout = attrs['data_layout']
        kernel_layout = attrs['kernel_layout']
        data, weight = inputs
        assert desired_layout == 'NCHW', \
                "Currently only transformation to NCHW layout is supported."
        if desired_layout == 'NCHW':
            new_attrs = dict(attrs)
            new_attrs['data_layout'] = desired_layout
            new_attrs['kernel_layout'] = 'OIHW'

            if data_layout == 'NHWC' and kernel_layout == 'HWIO':
                # Convert (NHWC, HWIO) to (NCHW, OIHW)
                return relay.nn.conv2d(data, weight, **new_attrs)
            if data_layout == 'NHWC' and kernel_layout == 'HWOI':
                # Convert (NHWC, HWOI) to (NCHW, OIHW). Depthwise conv2d.
                return relay.nn.conv2d(data, weight, **new_attrs)
        return None


**Layout inference** - Relay op has an attribute - *FInferCorrectLayout* - that developers can implement to handle data layouts. Currently, this attribute is only exposed in C++. This function takes original input layouts and the new input layouts (passed from the previous operator or from the python callback for layout alteration). A TVM developer can use this function to infer the final data layout and also modify the op attributes if needed.

This component is used for *lightly-layout sensitive* operators. We try to accept the new input layout, and modify the current operator attributes (like axis for concatenate, pad_width for pad) to adapt to the new data layout. By accepting the new input data layout, we prevent the insertion of a layout transform. In absence of this function, Layout rewrite might have to insert a layout transform, if the previous operator has a different output data layout than the original one. One example to adapt to NCHW data layout is presented here for Batch Norm operator.

.. code-block:: c++

    Array<Array<Layout>> BatchNormInferCorrectLayout(const Attrs& attrs,
                                                     const Array<Layout>& new_in_layouts,
                                                     const Array<Layout>& old_in_layouts,
                                                     const Array<Array<IndexExpr>>& old_in_shapes) {
      BatchNormAttrs* param = const_cast<BatchNormAttrs*>(attrs.as<BatchNormAttrs>());

      size_t axis =
          param->axis < 0 ? param->axis + old_in_shapes[0].size() : static_cast<size_t>(param->axis);

      Layout ret = Layout::Undef();

      // If new_in_layouts are defined, this code tries to modify the layout.
      if (new_in_layouts.defined() && old_in_layouts.defined()) {
        // Get the new C axis. Extract the dim in old layout. Find the index of that dim in next layout.
        const auto& bn_dim = old_in_layouts[0][axis];
        auto new_index = new_in_layouts[0].IndexOf(bn_dim);
        param->axis = new_index;
        ret = new_in_layouts[0];
      } else if (old_in_layouts.defined()) {
        ret = old_in_layouts[0];
      }
      // BN has 5 inputs, 3 outputs. The last 4 inputs and last 2 outputs have "C" layout.
      Layout c_layout = Layout("C");

      return Array<Array<Layout>>{{ret, c_layout, c_layout, c_layout, c_layout},
                                  {ret, c_layout, c_layout}};
    }




**Automatic insertion of layout transforms** - Depending on inferred layouts, this component automatically inserts layout transforms at the input expr of the operator. This happens for *layout-agnostic* operators.


********
4. Usage
********

ConvertLayout pass is extremely easy to use. The pass is not a part of default relay.build pipeline. The intended usage is to call it between the framework-to-relay parser and relay.build module call.

.. code-block:: python

    # TFlite framework to Relay parser - Default layout is NHWC
    mod, params = relay.frontend.from_tflite(tflite_model,
                                             shape_dict=shape_dict,
                                             dtype_dict=dtype_dict)

    # Convert the layout to NCHW
    mod = relay.transform.ConvertLayout('NCHW')(mod)

    # Call relay compilation
    with relay.build_config(opt_level=3):
         graph, lib, params = relay.build(mod, target, params=params)

Current implementation has support for almost all the operators commonly used in image classification models. However, if one encounters too many data layout transforms in the graph, it is highly likely that there is an operator whose layouts need special handling as described in Section 3. Some pull requests that can help in such a situation are

- Layout inference for `Batch Norm <https://github.com/apache/incubator-tvm/pull/4600>`_ - Batch normalization falls into the category of lightly-sensitive operator. The PR shows how to handle the layout inference for batch norm.
- Python Callback for `Convolution <https://github.com/apache/incubator-tvm/pull/4335>`_- For highly-sensitive operators, one might have to do python callback as well. The PR shows how to define a python callback function for Convolution operator.
