# ToMixedPrecision 混合精度策略表格

## 策略说明

- **kAlways**: 总是将输入转换为 FP16，累加器使用 out_dtype（默认 FP32），输出存储为 FP16
- **kFollow**: 如果任何输入是 FP32 则全部用 FP32，否则用 FP16
- **kNever**: 总是保持 FP32，确保数值稳定性（默认策略，未显式注册时使用）

---

## kAlways 策略运算符（共8个）

这些是计算密集型运算，适合使用 FP16 加速（如利用 Tensor Core）

| 类别 | 运算符 | 文件位置 |
|------|--------|----------|
| **线性代数** | `relax.matmul` | `tvm/src/relax/op/tensor/linear_algebra.cc:172` |
| **线性代数** | `relax.outer` | `tvm/src/relax/op/tensor/linear_algebra.cc:302` |
| **卷积运算** | `relax.nn.conv1d` | `tvm/src/relax/op/nn/convolution.cc:199` |
| **卷积运算** | `relax.nn.conv2d` | `tvm/src/relax/op/nn/convolution.cc:402` |
| **卷积运算** | `relax.nn.conv3d` | `tvm/src/relax/op/nn/convolution.cc:582` |
| **注意力机制** | `relax.nn.attention` | `tvm/src/relax/op/nn/attention.cc:157` |
| **注意力机制** | `relax.nn.attention_bias` | `tvm/src/relax/op/nn/attention.cc:169` |
| **注意力机制** | `relax.nn.attention_var_len` | `tvm/src/relax/op/nn/attention.cc:184` |

---

## kFollow 策略运算符（共70+个）

这些运算根据输入精度动态选择，保持精度一致性

### 1. 二元运算 (Binary Operations) - 20个

通过宏 `RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL` 注册

| 运算符 | 说明 | 文件位置 |
|--------|------|----------|
| `relax.add` | 加法 | `tvm/src/relax/op/tensor/binary.cc:193` |
| `relax.subtract` | 减法 | `tvm/src/relax/op/tensor/binary.cc:199` |
| `relax.multiply` | 乘法 | `tvm/src/relax/op/tensor/binary.cc:197` |
| `relax.divide` | 除法 | `tvm/src/relax/op/tensor/binary.cc:194` |
| `relax.floor_divide` | 向下取整除法 | `tvm/src/relax/op/tensor/binary.cc:195` |
| `relax.power` | 幂运算 | `tvm/src/relax/op/tensor/binary.cc:198` |
| `relax.mod` | 取模 | `tvm/src/relax/op/tensor/binary.cc:200` |
| `relax.floor_mod` | 向下取整取模 | `tvm/src/relax/op/tensor/binary.cc:201` |
| `relax.log_add_exp` | Log-Sum-Exp | `tvm/src/relax/op/tensor/binary.cc:196` |
| `relax.minimum` | 最小值 | `tvm/src/relax/op/tensor/binary.cc:214` |
| `relax.maximum` | 最大值 | `tvm/src/relax/op/tensor/binary.cc:215` |
| `relax.logical_and` | 逻辑与 | `tvm/src/relax/op/tensor/binary.cc:219` |
| `relax.logical_or` | 逻辑或 | `tvm/src/relax/op/tensor/binary.cc:220` |
| `relax.logical_xor` | 逻辑异或 | `tvm/src/relax/op/tensor/binary.cc:221` |
| `relax.bitwise_and` | 位与 | `tvm/src/relax/op/tensor/binary.cc:225` |
| `relax.bitwise_or` | 位或 | `tvm/src/relax/op/tensor/binary.cc:226` |
| `relax.bitwise_xor` | 位异或 | `tvm/src/relax/op/tensor/binary.cc:227` |
| `relax.left_shift` | 左移 | `tvm/src/relax/op/tensor/binary.cc:228` |
| `relax.right_shift` | 右移 | `tvm/src/relax/op/tensor/binary.cc:229` |

### 2. 激活函数 (Activation Functions) - 5个

通过宏 `RELAX_REGISTER_UNARY_NN_OP_AND_IMPL` 注册

| 运算符 | 说明 | 文件位置 |
|--------|------|----------|
| `relax.nn.relu` | ReLU 激活 | `tvm/src/relax/op/nn/nn.cc:47` |
| `relax.nn.gelu` | GELU 激活 | `tvm/src/relax/op/nn/nn.cc:50` |
| `relax.nn.gelu_tanh` | GELU Tanh 激活 | `tvm/src/relax/op/nn/nn.cc:53` |
| `relax.nn.selu` | SELU 激活 | `tvm/src/relax/op/nn/nn.cc:56` |
| `relax.nn.silu` | SiLU 激活 | `tvm/src/relax/op/nn/nn.cc:59` |

### 3. 归一化层 (Normalization) - 5个

| 运算符 | 说明 | 文件位置 |
|--------|------|----------|
| `relax.nn.layer_norm` | Layer Normalization | `tvm/src/relax/op/nn/nn.cc:590` |
| `relax.nn.group_norm` | Group Normalization | `tvm/src/relax/op/nn/nn.cc:704` |
| `relax.nn.instance_norm` | Instance Normalization | `tvm/src/relax/op/nn/nn.cc:807` |
| `relax.nn.rms_norm` | RMS Normalization | `tvm/src/relax/op/nn/nn.cc:867` |
| `relax.nn.dropout` | Dropout | `tvm/src/relax/op/nn/nn.cc:896` |

### 4. 池化运算 (Pooling) - 9个

| 运算符 | 说明 | 文件位置 |
|--------|------|----------|
| `relax.nn.max_pool1d` | 1D 最大池化 | `tvm/src/relax/op/nn/pooling.cc:143` |
| `relax.nn.max_pool2d` | 2D 最大池化 | `tvm/src/relax/op/nn/pooling.cc:283` |
| `relax.nn.avg_pool1d` | 1D 平均池化 | `tvm/src/relax/op/nn/pooling.cc:414` |
| `relax.nn.avg_pool2d` | 2D 平均池化 | `tvm/src/relax/op/nn/pooling.cc:436` |
| `relax.nn.adaptive_avg_pool1d` | 1D 自适应平均池化 | `tvm/src/relax/op/nn/pooling.cc:458` |
| `relax.nn.adaptive_avg_pool2d` | 2D 自适应平均池化 | `tvm/src/relax/op/nn/pooling.cc:480` |
| `relax.nn.adaptive_max_pool1d` | 1D 自适应最大池化 | `tvm/src/relax/op/nn/pooling.cc:562` |
| `relax.nn.adaptive_max_pool2d` | 2D 自适应最大池化 | `tvm/src/relax/op/nn/pooling.cc:664` |
| `relax.nn.adaptive_avg_pool3d` | 3D 自适应平均池化 | `tvm/src/relax/op/nn/pooling.cc:751` |

### 5. 张量创建 (Tensor Creation) - 7个

| 运算符 | 说明 | 文件位置 |
|--------|------|----------|
| `relax.full` | 创建填充张量 | `tvm/src/relax/op/tensor/create.cc:99` |
| `relax.full_like` | 创建同形状填充张量 | `tvm/src/relax/op/tensor/create.cc:141` |
| `relax.ones` | 创建全1张量 | `tvm/src/relax/op/tensor/create.cc:201` |
| `relax.ones_like` | 创建同形状全1张量 | `tvm/src/relax/op/tensor/create.cc:238` |
| `relax.zeros` | 创建全0张量 | `tvm/src/relax/op/tensor/create.cc:323` |
| `relax.zeros_like` | 创建同形状全0张量 | `tvm/src/relax/op/tensor/create.cc:386` |
| `relax.tril` | 下三角矩阵 | `tvm/src/relax/op/tensor/create.cc:440` |

### 6. 张量操作 (Tensor Manipulation) - 10个

| 运算符 | 说明 | 文件位置 |
|--------|------|----------|
| `relax.broadcast_to` | 广播到指定形状 | `tvm/src/relax/op/tensor/manipulate.cc:139` |
| `relax.concat` | 拼接张量 | `tvm/src/relax/op/tensor/manipulate.cc:359` |
| `relax.expand_dims` | 扩展维度 | `tvm/src/relax/op/tensor/manipulate.cc:463` |
| `relax.flatten` | 展平张量 | `tvm/src/relax/op/tensor/manipulate.cc:509` |
| `relax.permute_dims` | 维度置换 | `tvm/src/relax/op/tensor/manipulate.cc:732` |
| `relax.reshape` | 重塑形状 | `tvm/src/relax/op/tensor/manipulate.cc:853` |
| `relax.split` | 分割张量 | `tvm/src/relax/op/tensor/manipulate.cc:1013` |
| `relax.squeeze` | 压缩维度 | `tvm/src/relax/op/tensor/manipulate.cc:1189` |
| `relax.strided_slice` | 步长切片 | `tvm/src/relax/op/tensor/manipulate.cc:1606` |
| `relax.tile` | 复制张量 | `tvm/src/relax/op/tensor/manipulate.cc:2277` |

### 7. 一元数学运算 (Unary Math Operations) - 28个

通过宏 `RELAX_REGISTER_UNARY_ARITH_OP_AND_IMPL` 注册，所有都是 **kFollow** 策略

| 运算符 | 说明 | 文件位置 |
|--------|------|----------|
| `relax.abs` | 绝对值 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.acos` | 反余弦 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.acosh` | 反双曲余弦 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.asin` | 反正弦 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.asinh` | 反双曲正弦 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.atan` | 反正切 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.atanh` | 反双曲正切 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.bitwise_not` | 位非 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.ceil` | 向上取整 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.cos` | 余弦 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.cosh` | 双曲余弦 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.exp` | 指数 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.floor` | 向下取整 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.log` | 自然对数 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.logical_not` | 逻辑非 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.negative` | 取负 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.round` | 四舍五入 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.rsqrt` | 平方根倒数 | `tvm/src/relax/op/tensor/unary.cc:58` |
| `relax.sigmoid` | Sigmoid | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.sign` | 符号函数 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.sin` | 正弦 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.sinh` | 双曲正弦 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.square` | 平方 | `tvm/src/relax/op/tensor/unary.cc` |
| **`relax.sqrt`** | **平方根** | **`tvm/src/relax/op/tensor/unary.cc:64`** |
| `relax.tan` | 正切 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.tanh` | 双曲正切 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.trunc` | 截断 | `tvm/src/relax/op/tensor/unary.cc` |
| `relax.erf` | 误差函数 | `tvm/src/relax/op/tensor/unary.cc` |

> **注意**: 所有一元数学运算通过宏 `RELAX_REGISTER_UNARY_OP` 自动设置为 `kFollow` 策略  
> 源码位置: `tvm/src/relax/op/op_common.h:170`

### 8. 其他运算

| 运算符 | 说明 | 文件位置 |
|--------|------|----------|
| `relax.astype` | 类型转换 | `tvm/src/relax/op/tensor/datatype.cc:68` |
| `relax.where` | 条件选择 | `tvm/src/relax/op/tensor/ternary.cc:140` |
| `relax.take` | 索引选择 | `tvm/src/relax/op/tensor/index.cc:481` |
| `relax.image.resize2d` | 图像大小调整 | `tvm/src/relax/op/image/resize.cc:148` |
| `relax.layout_transform` | 布局转换 | `tvm/src/relax/op/tensor/manipulate.cc:1354` |

---

## kNever 策略运算符

这些运算未显式注册 `TMixedPrecisionPolicy` 属性，默认使用 kNever 策略以确保数值稳定性

| 运算符 | 说明 | 文件位置 | 备注 |
|--------|------|----------|------|
| `relax.nn.softmax` | Softmax | `tvm/src/relax/op/nn/nn.cc:224` | 数值敏感，保持 FP32 |
| `relax.nn.log_softmax` | Log Softmax | `tvm/src/relax/op/nn/nn.cc:245` | 数值敏感，保持 FP32 |
| 其他未注册的运算 | - | - | 默认 kNever |

---

## 混合精度策略总结

### GPT-2 模型中的应用

在您的 `Compile_GPT2.py` 中使用 `transform.ToMixedPrecision()` 时：

```python
mod_mixed = compile_model(
    f"{base_path}/model.onnx",
    dtype_converter=lambda m: transform.ToMixedPrecision()(m)["main"],
    use_vectorize=True
)
```

**效果**：
- ✅ **MatMul、Attention** → 输入用 FP16，累加器用 FP32，输出存为 FP16
- ✅ **卷积** → 同上（虽然 GPT-2 不用卷积）
- ✅ **GELU、Layer Norm、Add 等** → 跟随输入精度（主要是 FP16）
- ✅ **Softmax** → 保持 FP32，确保数值稳定性
- 📊 **结果**：计算密集型运算加速，数值敏感运算保持精度

### 与 Posit 的区别

| 特性 | ToMixedPrecision | ChangeDatatype (Posit) |
|------|------------------|------------------------|
| **策略依赖** | ✅ 使用 TMixedPrecisionPolicy | ❌ 不使用 |
| **转换方式** | 选择性（按运算符类型） | 全局性（所有匹配类型） |
| **精度控制** | 运算符级别 | 全模型级别 |
| **适用场景** | FP32/FP16 混合精度 | 自定义数据类型 |

---

## 参考文件

- **策略定义**: `tvm/src/relax/transform/infer_amp_utils.h:44`
- **核心实现**: `tvm/src/relax/transform/to_mixed_precision.cc`
- **测试用例**: `tvm/tests/python/relax/test_transform_to_mixed_precision.py`
