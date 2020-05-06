use tvm_sys::ffi;
pub use tvm_sys::context::*;

use crate::function;

use std::os::raw::c_void;
use std::convert::TryInto;
use std::ptr;

trait ContextExt {
    /// Checks whether the context exists or not.
    fn exist(&self) -> bool;
    fn sync(&self) -> anyhow::Result<()>;
    fn max_threads_per_block(&self) -> isize;
    fn warp_size(&self) -> isize;
    fn max_shared_memory_per_block(&self) -> isize;
    fn compute_version(&self) -> isize;
    fn device_name(&self) -> isize;
    fn max_clock_rate(&self) -> isize;
    fn multi_processor_count(&self) -> isize;
    fn max_thread_dimensions(&self) -> isize;
}

macro_rules! impl_device_attrs {
    ($(($attr_name:ident, $attr_kind:expr));+) => {
        $(
                fn $attr_name(&self) -> isize {
                    let func = function::Function::get("runtime.GetDeviceAttr")
                        .expect("TVM FFI functions must always be registered.");
                    let dt = self.device_type.0 as isize;
                    // TODO(@jroesch): these functions CAN and WILL return NULL
                    // we should make these optional or somesuch to handle this.
                    // `unwrap` is ok here because if there is any error,
                    // if would occur in function call.
                    function::Builder::from(func)
                        .arg(dt)
                        .arg(self.device_id as isize)
                        .arg($attr_kind)
                        .invoke()
                        .unwrap()
                        .try_into()
                        .unwrap()
                }

        )+
    };
}

impl ContextExt for Context {
    fn exist(&self) -> bool {
        let func = function::Function::get("runtime.GetDeviceAttr")
            .expect("TVM FFI functions must always be registered.");
        let dt = self.device_type.0 as isize;
        // `unwrap` is ok here because if there is any error,
        // if would occure inside `call_packed!`
        let ret: i64 = call_packed!(func, dt, self.device_id, 0)
            .unwrap()
            .try_into()
            .unwrap();
        ret != 0
    }

    /// Synchronize the context stream.
    fn sync(&self) -> anyhow::Result<()> {
        check_call!(ffi::TVMSynchronize(
            self.device_type.0 as i32,
            self.device_id as i32,
            ptr::null_mut() as *mut c_void
        ));
        Ok(())
    }

    impl_device_attrs!((max_threads_per_block, 1);
        (warp_size, 2);
        (max_shared_memory_per_block, 3);
        (compute_version, 4);
        (device_name, 5);
        (max_clock_rate, 6);
        (multi_processor_count, 7);
        (max_thread_dimensions, 8));
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync() {
        let ctx = Context::cpu(0);
        assert!(ctx.sync().is_ok())
    }
}
