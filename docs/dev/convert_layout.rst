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

**************************
2. Motivation and Overview
**************************

Let's look at a simple scenario to understand the complications that arise due to different layouts - Suppose we want to compile a Tensorflow NHWC graph for an ARM edge device. But, suppose we currently support only NCHW schedules in TOPI for ARM. So, there is a mismatch between framework layout and TOPI-supported layout. One way to deal with this mismatch is to insert layout transforms before each and after convolution, such that resulting convolution has NCHW input data layout and can use TOPI schedules. However, this can lead to performance degradation because of the presence of too many layout transforms.

We encountered similar problems in other use cases as well

- No way to run TFLite graphs on Nvidia GPUs. TOPI has NCHW-only schedules for GPUs.
- Ever-complicating logic in AlterOpLayout for convolution to support different pairs of layout transformations.
- Sub-optimal performance for TF graphs due to extra layout transforms.
- Complication in third-party codegen integrations like TensorRT that prefers data layout to be in one format.

To solve these problems, we introduced *ConvertLayout* pass that sets up the infrastructure to change the data layout of the whole graph with minimal number of data layout transforms. In ideal cases, we will have only 2 layout transforms for data, one at the start and one at the end. An example to show the transformation is below


.. code-block:: python

	# Original graph - 2 convolutions in NHWC format.
	fn (%x: Tensor[(1, 56, 56, 64), float32], %weight1: Tensor[(3, 3, 64, 32), float32], %weight2: Tensor[(3, 3, 32, 32), float32]) {
	  %0 = nn.conv2d(%x, %weight1, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
	  %1 = nn.relu(%0);
	  %2 = nn.conv2d(%1, %weight2, padding=[1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO");
	  nn.relu(%2)
	}

	# After ConvertLayout - For data, there is a transform at the start and at the end.
	# For weights, there are transforms to adapt to NCHW layout. These will be removed by FoldConstant pass.
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

Before delving into ConvertLayout pass, let's categorize the operators into 3 categories based on their sensitivity to data layouts. This categorization will be useful later to understand Convertlayout pass details.

- **Layout agnostic** - Relu, pow etc. These operators are not affected, neither functionality nor performance, by data layouts.
- **Lightly-layout sensitive** - pad, concatenate, reduce ops like sum etc. These operators have some attributes that are functionally affected if we do a layout transformation before them. However, performance-wise, the difference is not significant. For these operators, it is beneficial to just adapt to the previous operator output data layout.
- **Heavily-layout sensitive** - Convolution, conv2d_transpose etc. These operators are heavily affected, both functionally and performance-wise, by data layouts. They also have data layout as the op attribute. Typically, it is beneficial to modify the input data layouts for these operators (if its not a performant data layout), while the rest of *layout agnostic* and *lightly-layout sensitive* operators adapt to the layout governed by the output of these *heavliy-layout sensitive* operators.


Let us now look at two relevant Relay operator properties. Each relay operator has properties, like InferType, that can be defined by a TVM developer. Typically, a Relay pass traverses the graph operator-by-operator and reads these operator properties. For example, InferType pass looks at the InferType property of on operator, determines its output shape and type, and then passes it to the next operator InferType property. Similarly, in our context, we have 2 such properties - *FTVMConvertLayout* and *FInferCorrectLayout*. ConvertLayout pass traverses the graph and looks at these 2 properties along with an automatic layout transform insertion module to handle data layouts. So, the whole process can be broken down into 3 steps:

- Run FTVMConvertLayout property - This allows the developers to transform the original Relay expr into a new Relay expr with new layouts, allowing user-defined layout alteration. There is a python callback for developer's ease. This is used only for heavily-layout sensitive operators.
- Run FTVMInferCorretLayout property - We can view this as layout inference. It looks at the original input layout and the new input layouts, which are either coming from previous operator or from the FTVMConvertLayout modified expr (if it was used). This can be used by lightly-layout sensitive operators to adapt its attributes to new data layouts. Layout inference happens for each operator.
- Automatic insertion of layout transforms - The previous step - layout inference - sets the new layout for the input exprs. If these layouts are different from the original layouts, then this component automatically inserts a layout transform. Therefore, a developer does not need to do anything for this component.

These steps happen for each operator in sequence, where ConvertLayout pass keeps on passing the new layouts to the next operator properties, finally resulting in modifying the whole graph operator-by-operator. Now, let's look at a couple of examples of how to define the two properties.

**FTVMConvertLayout - Python callback for layout alteration** - This is used for *heavily-layout sensitive* operators. For example, one can return a new convolution operator with new data and kernel layout. The other 2 components will infer layout and insert layout transforms if needed. One example for convolution operator is as follows where we are converting to NCHW layout.

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
            # Actual insertion of layout transforms is taken care internally
            # by ConvertLayout pass.
            return relay.nn.conv2d(data, weight, **new_attrs)
        return None


**FInferCorrectLayout - Layout inference** - Currently, this attribute is exposed only in C++. This function takes original input layouts and the new input layouts (passed from the previous operator or from the python callback for layout alteration), and infers the final data layouts. Layout inference is called for each operator. The usage might vary for different operator categories. For layout agnostic operators, we just want to return the new data layouts in this function. For lightly-layout and heavily-layout sensitive operators, we can change the operator attributes (like axis for concatenate, pad_width for pad) so that we can adapt to the new data layout, preventing insertion of layout transforms. Let's look at a couple of examples to understand this better.

First example is for layout agnostic operators. These operators do not have any operator attributes that are affected by data layouts, so we just adapt to new layouts.

.. code-block:: c++

    // For operator set its attributes like following
    // 		.set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout);

    // Take arbitrary input layouts and copy to outputs.
    inline Array<Array<Layout> > ElemwiseArbitraryLayout(const Attrs& attrs,
                                                         const Array<Layout>& new_in_layouts,
                                                         const Array<Layout>& old_in_layouts,
                                                         const Array<Array<IndexExpr>> &old_in_shapes) {
      Layout ret;

      if (new_in_layouts.defined()) {
        CHECK_GE(new_in_layouts.size(), 1);
        ret = new_in_layouts[0];
      } else {
        for (size_t i = 0; i < old_in_layouts.size(); ++i) {
          if (old_in_layouts[i].defined()) {
            ret = old_in_layouts[i];
            break;
          }
        }
      }

      return Array<Array<Layout> >{Array<Layout>(old_in_layouts.size(), ret), {ret}};
    }


Second example is for a lightly-layout sensitive operator - batch normalization. BatchNorm has an axis operator that has to change when we go from NHWC to NCHW data layout. (Similar handling also needs to be for heavily-layout sensitive operators)


.. code-block:: c++

    Array<Array<Layout>> BatchNormInferCorrectLayout(const Attrs& attrs,
                                                     const Array<Layout>& new_in_layouts,
                                                     const Array<Layout>& old_in_layouts,
                                                     const Array<Array<IndexExpr>>& old_in_shapes) {
      BatchNormAttrs* param = const_cast<BatchNormAttrs*>(attrs.as<BatchNormAttrs>());

      size_t axis =
          param->axis < 0 ? param->axis + old_in_shapes[0].size() : static_cast<size_t>(param->axis);

      Layout ret = Layout::Undef();

      // For example, consider old_layout = NHWC, and new_layout = NCHW, and param->axis = 3

      if (new_in_layouts.defined() && old_in_layouts.defined()) {
        // Get the new C axis. Extract the dim in old layout. Find the index of that dim in next layout.

        // Following line gives bn_dim = C as old_layout = NHWC, axis = 3
        const auto& bn_dim = old_in_layouts[0][axis];

        // The new_index is 1 because new_layout = NCHW and bn_dim is C
        auto new_index = new_in_layouts[0].IndexOf(bn_dim);

        // We modify the layout-dependent attribute here - axis to 1.
        param->axis = new_index;

        // Finally, we adapt to the new layout.
        ret = new_in_layouts[0];

      } else if (old_in_layouts.defined()) {
        ret = old_in_layouts[0];
      }

      // In case both new and old layouts are undefined, then there is no need of a change.
      // ConvertLayout pass skips the automatic insertion of layout transforms in this case.

      // Following line is not important to tutorial. But, layout inference needs to define
      // the layout for all input and output data layouts. For batch norm, the other inputs
      // and outputs are vector having length of C dim in the input. So, we set the other
      // layouts as C. BN has 5 inputs, 3 outputs. The last 4 inputs and last 2 outputs
      // have "C" layout.
      Layout c_layout = Layout("C");

      return Array<Array<Layout>>{{ret, c_layout, c_layout, c_layout, c_layout},
                                  {ret, c_layout, c_layout}};
    }


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
    # RemoveUnunsedFunctions is used to clean up the graph.
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                      relay.transform.ConvertLayout('NCHW')])
    with relay.transform.PassContext(opt_level=3):
        mod = seq(mod)

    # Call relay compilation
    with relay.build_config(opt_level=3):
         graph, lib, params = relay.build(mod, target, params=params)

Current implementation has support for almost all the operators commonly used in image classification models. However, if one encounters too many data layout transforms in the graph, it is highly likely that there is an operator whose layouts need special handling as described in Section 3. Some pull requests that can help in such a situation are

- Layout inference for `Batch Norm <https://github.com/apache/incubator-tvm/pull/4600>`_ - Batch normalization falls into the category of lightly-sensitive operator. The PR shows how to handle the layout inference for batch norm.
- Python Callback for `Convolution <https://github.com/apache/incubator-tvm/pull/4335>`_- For highly-sensitive operators, one might have to do python callback as well. The PR shows how to define a python callback function for Convolution operator.
