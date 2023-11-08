use crate::runtime::array::Array;
use crate::runtime::function::Result;
use crate::runtime::string::String as TVMString;
use crate::runtime::{external, IsObjectRef, Object, ObjectRef};
use crate::ir::PrimExpr;
use crate::ir::relay::Expr;
use crate::function::ffi::DLDataType;

external! {
    #[name("relay.op.nn._make.conv1d")]
    pub fn conv1d(data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;

    #[name("relay.op.nn._make.conv2d")]
    pub fn conv2d(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;

    #[name("relay.op.nn._make.conv3d")]
    pub fn conv3d(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;

    #[name("relay.op.nn._make.contrib_conv3d_winograd_without_weight_transform")]
    pub fn contrib_conv3d_winograd_without_weight_transform(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;

    #[name("relay.op.nn._make.conv3d_transpose")]
    pub fn conv3d_transpose(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;

    #[name("relay.op.nn._make.conv2d_transpose")]
    pub fn conv2d_transpose(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;

    #[name("relay.op.nn._make.conv1d_transpose")]
    pub fn conv1d_transpose(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;
    
    #[name("relay.op.nn._make.softmax")]
    pub fn softmax(data: Expr, axis: i32) -> Expr;

    #[name("relay.op.nn._make.fast_softmax")]
    pub fn fast_softmax(data: Expr, axis: i32) -> Expr;

    #[name("relay.op.nn._make.log_softmax")]
    pub fn log_softmax(data: Expr, axis: i32) -> Expr;

    #[name("relay.op.nn._make.max_pool1d")]
    pub fn max_pool1d(
        data: Expr, 
        pool_size: Array<PrimExpr>, 
        strides: Array<PrimExpr>, 
        dilation: Array<PrimExpr>,
        padding: Array<PrimExpr>, 
        layout: TVMString,
        ceil_mode: bool,
    ) -> Expr;

    #[name("relay.op.nn._make.max_pool2d")]
    pub fn max_pool2d(
        data: Expr, 
        pool_size: Array<PrimExpr>, 
        strides: Array<PrimExpr>, 
        dilation: Array<PrimExpr>,
        padding: Array<PrimExpr>, 
        layout: TVMString,
        out_out_layout: TVMString,
        ceil_mode: bool,
    ) -> Expr;

    #[name("relay.op.nn._make.max_pool3d")]
    pub fn max_pool3d(
        data: Expr, 
        pool_size: Array<PrimExpr>, 
        strides: Array<PrimExpr>, 
        dilation: Array<PrimExpr>,
        padding: Array<PrimExpr>, 
        layout: TVMString,
        out_layout: TVMString,
        ceil_mode: bool,
    ) -> Expr;

    #[name("relay.op.nn._make.avg_pool1d")]
    pub fn avg_pool1d(
        data: Expr, 
        pool_size: Array<PrimExpr>, 
        strides: Array<PrimExpr>, 
        dilation: Array<PrimExpr>,
        padding: Array<PrimExpr>, 
        layout: TVMString,
        out_layout: TVMString,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Expr;

    #[name("relay.op.nn._make.avg_pool2d")]
    pub fn avg_pool2d(
        data: Expr, 
        pool_size: Array<PrimExpr>, 
        strides: Array<PrimExpr>, 
        dilation: Array<PrimExpr>,
        padding: Array<PrimExpr>, 
        layout: TVMString,
        out_layout: TVMString,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Expr;

    #[name("relay.op.nn._make.avg_pool3d")]
    pub fn avg_pool3d(
        data: Expr, 
        pool_size: Array<PrimExpr>, 
        strides: Array<PrimExpr>, 
        dilation: Array<PrimExpr>,
        padding: Array<PrimExpr>, 
        layout: TVMString,
        out_layout: TVMString,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Expr;

    #[name("relay.op.nn._make.upsampling")]
    pub fn upsampling(
        data: Expr, 
        scale_h: i32,
        scale_w: i32, 
        layout: TVMString,
        method: TVMString,
        align_corners: bool,
    ) -> Expr;

    #[name("relay.op.nn._make.upsampling3d")]
    pub fn upsampling3d(
        data: Expr, 
        scale_d: i32,
        scale_h: i32,
        scale_w: i32, 
        layout: TVMString,
        method: TVMString,
        coordinate_transformation_mode: TVMString,
    ) -> Expr;

    #[name("relay.op.nn._make.batch_flatten")]
    pub fn batch_flatten(data: Expr) -> Expr;

    #[name("relay.op.nn._make.bias_add")]
    pub fn bias_add(data: Expr, bias: Expr, axis: i32) -> Expr;

    #[name("relay.op.nn._make.matmul")]
    pub fn matmul(lhs: Expr, rhs: Expr, units: i32, out_dtype: DLDataType, transpose_a: bool, transpose_b: bool) -> Expr;

    #[name("relay.op.nn._make.dense")]
    pub fn dense(data: Expr, weight: Expr, units: i32, out_dtype: DLDataType) -> Expr;

    //FIXME: contrib_dense_pack, fifo_buffer

    #[name("relay.op.nn._make.relu")]
    pub fn relu(data: Expr) -> Expr;

    #[name("relay.op.nn._make.leaky_relu")]
    pub fn leaky_relu(data: Expr, alpha: f32) -> Expr;

    #[name("relay.op.nn._make.prelu")]
    pub fn prelu(data: Expr, alpha: Expr, axis: i32) -> Expr;

    #[name("relay.op.nn._make.pad")]
    pub fn pad(data: Expr, pad_width: Array<Array<PrimExpr>>, pad_value: Expr, pad_mode: TVMString) -> Expr;

    #[name("relay.op.nn._make.dilate")]
    pub fn dilate(data: Expr, strides: Array<PrimExpr>, dilation_value: Array<PrimExpr>) -> Expr;

    #[name("relay.op.nn._make.mirror_pad")]
    pub fn mirror_pad(data: Expr, pad_width: Array<PrimExpr>, pad_mode: TVMString) -> Expr;

    #[name("relay.op.nn._make.lrn")]
    pub fn lrn(data: Expr, size: i32, axis: i32, alpha: f32, beta: f32) -> Expr;

    #[name("relay.op.nn._make.l2_normalize")]
    pub fn l2_normalize(data: Expr, eps: f32, axis: i32) -> Expr;

    // FIXME: dropout, dropout_raw

    #[name("relay.op.nn._make.batch_norm")]
    pub fn batch_norm(data: Expr, gamma: Expr, beta: Expr, moving_mean: Expr, moving_var: Expr, axis: i32, epsilon: f32, center: bool, scale: bool) -> Expr;

    #[name("relay.op.nn._make.instance_norm")]
    pub fn instance_norm(data: Expr, gamma: Expr, beta: Expr, epsilon: f32, center: bool, scale: bool) -> Expr;

    #[name("relay.op.nn._make.layer_norm")]
    pub fn layer_norm(data: Expr, gamma: Expr, beta: Expr, axis: i32, epsilon: f32, center: bool, scale: bool) -> Expr;

    #[name("relay.op.nn._make.group_norm")]
    pub fn group_norm(data: Expr, gamma: Expr, beta: Expr, num_groups: i32, epsilon: f32, center: bool, scale: bool) -> Expr;

    #[name("relay.op.nn._make.batch_matmul")]
    pub fn batch_matmul(lhs: Expr, rhs: Expr, out_dtype: TVMString, transpose_a: bool, transpose_b: bool) -> Expr;

    // FIXME: sparse_add, sparse_dense

    #[name("relay.op.nn._make.contrib_conv2d_winograd_without_weight_transform")]
    pub fn conv2d_winograd_without_weight_transform(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;


    #[name("relay.op.nn._make.contrib_conv2d_gemm_without_weight_transform")]
    pub fn conv2d_gemm_without_weight_transform(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
    ) -> Expr;

    // FIXME: contrib_conv2d_nchwc, contrib_depthwise_conv2d

    #[name("relay.op.nn._make.contrib_conv2d_winograd_weight_transform")]
    pub fn conv2d_winograd_weight_transform(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        data_layout: TVMString,
        kernel_layout: TVMString, 
        out_layout: TVMString,  
        out_dtype: DLDataType,
        tile_size: i32,
    ) -> Expr;

    #[name("relay.op.nn._make.contrib_gemm_weight_transform")]
    pub fn gemm_weight_transform(
        data: Expr, 
        weight: Expr, 
        transpose_a: bool, 
        transpose_b: bool, 
        out_dtype: DLDataType,
    ) -> Expr;

    #[name("relay.op.nn._make.contrib_conv3d_winograd_weight_transform")]
    pub fn conv3d_winograd_weight_transform(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        dilation: Array<PrimExpr>, 
        groups: i32, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        tile_size: i32,
    ) -> Expr;

    #[name("relay.op.nn._make.contrib_conv2d_winograd_nnpack_weight_transform")]
    pub fn conv2d_winograd_nnpack_weight_transform(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        out_dtype: DLDataType,
    ) -> Expr;

    #[name("relay.op.nn._make.bitpack")]
    pub fn bitpack(data: Expr, bits: i32, pack_axis: i32, bit_axis: i32) -> Expr;

    #[name("relay.op.nn._make.bitserial_conv2d")]
    pub fn bitserial_conv2d(
        data: Expr, 
        weight: Expr, 
        strides: Array<PrimExpr>, 
        padding: Array<PrimExpr>, 
        channels: PrimExpr, 
        kernel_size: Array<PrimExpr>,
        pack_dtype: DLDataType,
        out_dtype: DLDataType,
        pack_axis: i32,
        bit_axis: i32,
        unipolar: bool,
    ) -> Expr;


    #[name("relay.op.nn._make.bitserial_dense")]
    pub fn bitserial_dense(
        data: Expr, 
        weight: Expr, 
        units: i32, 
        pack_dtype: DLDataType,
        out_dtype: DLDataType,
        pack_axis: i32,
        bit_axis: i32,
        unipolar: bool,
    ) -> Expr;

    #[name("relay.op.nn._make.depth_to_space")]
    pub fn depth_to_space(data: Expr, block_size: i32, layout: TVMString, mode: TVMString) -> Expr;

    #[name("relay.op.nn._make.space_to_depth")]
    pub fn space_to_depth(data: Expr, block_size: i32, layout: TVMString) -> Expr;

    #[name("relay.op.nn._make.global_avg_pool2d")]
    pub fn global_avg_pool2d_(data: Expr, layout: TVMString, out_layout: TVMString) -> Expr; //FIXME: global_avg_pool2d without underscore creates error

    #[name("relay.op.nn._make.adaptive_avg_pool1d")]
    pub fn adaptive_avg_pool1d(data: Expr, output_size: Array<PrimExpr>, layout: TVMString, out_layout: TVMString) -> Expr;
    
    #[name("relay.op.nn._make.adaptive_avg_pool2d")]
    pub fn adaptive_avg_pool2d(data: Expr, output_size: Array<PrimExpr>, layout: TVMString, out_layout: TVMString) -> Expr;

    #[name("relay.op.nn._make.adaptive_avg_pool3d")]
    pub fn adaptive_avg_pool3d(data: Expr, output_size: Array<PrimExpr>, layout: TVMString, out_layout: TVMString) -> Expr;

    #[name("relay.op.nn._make.space_to_batch_nd")]
    pub fn space_to_batch_nd(data: Expr, block_shape: Array<PrimExpr>, pads: Array<PrimExpr>, layout: TVMString) -> Expr;

    #[name("relay.op.nn._make.batch_to_space_nd")]
    pub fn batch_to_space_nd(data: Expr, block_shape: Array<PrimExpr>, crops: Array<PrimExpr>, layout: TVMString) -> Expr;
}


// Test Cases
#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::ir::as_text;
    use crate::ir::relay::Var;
    use crate::ir::ty::TensorType;
    use anyhow::Result;
    use tvm_rt::DataType;

    #[test]
    fn test_conv2d() -> Result<()> {
        let data = Var::static_tensor("data".to_string(), vec![1,1,1,1], DataType::float32());
        let weight = Var::static_tensor("data".to_string(), vec![1,1,1,1], DataType::float32());
        let strides = Array::from_vec(vec![PrimExpr::from(1), PrimExpr::from(1)]).unwrap();
        let padding = Array::from_vec(vec![PrimExpr::from(1), PrimExpr::from(1)]).unwrap();
        let dilation = Array::from_vec(vec![PrimExpr::from(1), PrimExpr::from(1)]).unwrap();
        let groups = 1.into();
        let channels = 1.into();
        let kernel_size = Array::from_vec(vec![PrimExpr::from(1), PrimExpr::from(1)]).unwrap();
        let data_layout = "NCHW".to_string().into();
        let kernel_layout = "OIHW".to_string().into();
        let out_layout = "".to_string().into();
        let out_dtype = DLDataType::from(DataType::float32());
        let conv2d = conv2d(
            data.upcast::<Expr>(), 
            weight.upcast::<Expr>(), 
            strides, 
            padding, 
            dilation, 
            groups, 
            channels, 
            kernel_size,
            data_layout,
            kernel_layout, 
            out_layout,  
            out_dtype,
        ).unwrap();
        assert!(as_text(conv2d).contains("conv2d"));
        Ok(())
    }

    #[test]
    fn test_conv1d() -> Result<()> {
        let data = Var::static_tensor("data".to_string(), vec![1,1,1], DataType::float32());
        let weight = Var::static_tensor("data".to_string(), vec![1,1,1], DataType::float32());
        let strides = Array::from_vec(vec![PrimExpr::from(1)]).unwrap();
        let padding = Array::from_vec(vec![PrimExpr::from(1)]).unwrap();
        let dilation = Array::from_vec(vec![PrimExpr::from(1)]).unwrap();
        let groups = 1.into();
        let channels = 1.into();
        let kernel_size = Array::from_vec(vec![PrimExpr::from(1)]).unwrap();
        let data_layout = "NCW".to_string().into();
        let kernel_layout = "OIW".to_string().into();
        let out_layout = "".to_string().into();
        let out_dtype = DLDataType::from(DataType::float32());
        let conv1d = conv1d(
            data.upcast::<Expr>(), 
            weight.upcast::<Expr>(), 
            strides, 
            padding, 
            dilation, 
            groups, 
            channels, 
            kernel_size,
            data_layout,
            kernel_layout, 
            out_layout,  
            out_dtype,
        ).unwrap();
        assert!(as_text(conv1d).contains("conv1d"));
        Ok(())
    }

    #[test]
    fn test_conv3d() -> Result<()> {
        let data = Var::static_tensor("data".to_string(), vec![1,1,1,1,1], DataType::float32());
        let weight = Var::static_tensor("data".to_string(), vec![1,1,1,1,1], DataType::float32());
        let strides = Array::from_vec(vec![PrimExpr::from(1), PrimExpr::from(1), PrimExpr::from(1)]).unwrap();
        let padding = Array::from_vec(vec![PrimExpr::from(1), PrimExpr::from(1), PrimExpr::from(1)]).unwrap();
        let dilation = Array::from_vec(vec![PrimExpr::from(1), PrimExpr::from(1), PrimExpr::from(1)]).unwrap();
        let groups = 1.into();
        let channels = 1.into();
        let kernel_size = Array::from_vec(vec![PrimExpr::from(1), PrimExpr::from(1), PrimExpr::from(1)]).unwrap();
        let data_layout = "NCDHW".to_string().into();
        let kernel_layout = "OIDHW".to_string().into();
        let out_layout = "".to_string().into();
        let out_dtype = DLDataType::from(DataType::float32());
        let conv3d = conv3d(
            data.upcast::<Expr>(), 
            weight.upcast::<Expr>(), 
            strides, 
            padding, 
            dilation, 
            groups, 
            channels, 
            kernel_size,
            data_layout,
            kernel_layout, 
            out_layout,  
            out_dtype,
        ).unwrap();
        assert!(as_text(conv3d).contains("conv3d"));
        Ok(())
    }

}