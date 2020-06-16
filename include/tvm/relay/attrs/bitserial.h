/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relay/attrs/bitserial.h
 * \brief Auxiliary attributes for bitserial operators.
 */

#ifndef TVM_RELAY_ATTRS_BITSERIAL_H_
#define TVM_RELAY_ATTRS_BITSERIAL_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>

#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in bitpack operators */
struct BitPackAttrs : public tvm::AttrsNode<BitPackAttrs> {
  int bits;
  int pack_axis;
  int bit_axis;
  DataType pack_type;
  std::string name;

  TVM_DECLARE_ATTRS(BitPackAttrs, "relay.attrs.BitPackAttrs") {
    TVM_ATTR_FIELD(bits).set_default(1).describe("Number of bits to quantize with.");
    TVM_ATTR_FIELD(pack_axis).set_default(1).describe(
        "Axis that should be compressed, typically channels.");
    TVM_ATTR_FIELD(bit_axis).set_default(-1).describe("New axis for packed bits.");
    TVM_ATTR_FIELD(pack_type)
        .set_default(NullValue<DataType>())
        .describe("Type of int to pack bits into.");
    TVM_ATTR_FIELD(name).set_default("BitPack").describe("Name of operation.");
  }
};

/*! \brief Attribues used in bitserial convolution operators */
struct BinaryConv2DAttrs : public tvm::AttrsNode<BinaryConv2DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  IndexExpr channels;
  Array<IndexExpr> kernel_size;
  int activation_bits;
  int weight_bits;
  std::string data_layout;
  std::string kernel_layout;
  DataType pack_dtype;
  DataType out_dtype;
  bool unipolar;

  TVM_DECLARE_ATTRS(BinaryConv2DAttrs, "relay.attrs.BinaryConv2DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<IndexExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<IndexExpr>({0, 0}))
        .describe(
            "If padding is non-zero the input is implicitly zero-padded"
            "on both sides for padding number of points.");
    TVM_ATTR_FIELD(kernel_size)
        .set_default(Array<IndexExpr>({3, 3}))
        .describe("Specifies the dimensions of the convolution window.");
    TVM_ATTR_FIELD(channels)
        .set_default(NullValue<IndexExpr>())
        .describe("Number of output channels, needed for shape inference.");
    TVM_ATTR_FIELD(activation_bits)
        .set_default(1)
        .describe("Number of bits activation should be packed with.");
    TVM_ATTR_FIELD(weight_bits)
        .set_default(1)
        .describe("Number of bits kernel should be packed with.");
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe("Dimension ordering of input data, can be 'NCHW' or NHWC'.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIHW")
        .describe("Dimension ordering of kernel data, can be 'OIHW' or HWIO'.");
    TVM_ATTR_FIELD(pack_dtype)
        .set_default(NullValue<DataType>())
        .describe("Datatype to pack bits into.");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output datatype.");
    TVM_ATTR_FIELD(unipolar).set_default(true).describe(
        "Whether to use unipolar or bipolar quantization.");
  }
};

/*~ \brief Attributes for bitserial dense operator */
struct BinaryDenseAttrs : public tvm::AttrsNode<BinaryDenseAttrs> {
  IndexExpr units;
  int data_bits;
  int weight_bits;
  DataType pack_dtype;
  DataType out_dtype;
  bool unipolar;

  TVM_DECLARE_ATTRS(BinaryDenseAttrs, "relay.attrs.BinaryDenseAttrs") {
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");
    TVM_ATTR_FIELD(data_bits).set_default(1).describe(
        "Number of bits to pack for incoming tensor.");
    TVM_ATTR_FIELD(weight_bits)
        .set_default(1)
        .describe("Number of bits to pack for weight tensor.");
    TVM_ATTR_FIELD(pack_dtype)
        .set_default(NullValue<DataType>())
        .describe("Datatype to pack bits into before computation.");
    TVM_ATTR_FIELD(out_dtype).set_default(NullValue<DataType>()).describe("Output data type.");
    TVM_ATTR_FIELD(unipolar).set_default(true).describe(
        "Whether to use unipolar or bipolar quantization for inputs.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_BITSERIAL_H_
