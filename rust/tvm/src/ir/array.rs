use crate::runtime::object::{ObjectPtr, Object, ObjectRef, ToObjectRef};
use tvm_sys::{TVMArgValue, TVMRetValue};
use crate::runtime::external_func;
use std::convert::{TryFrom, TryInto};
use crate::runtime::function::Builder;
use std::marker::PhantomData;

use anyhow::Result;

pub struct Array<T: ToObjectRef> {
    object: ObjectRef,
    _data: PhantomData<T>
}

impl<T: ToObjectRef> Array<T> {
    pub fn from_vec(data: Vec<T>) -> Result<Array<T>> {
        let mut args: Vec<TVMArgValue> = vec![];
        let iter = data.iter().map(|element| element.to_object_ref());

        let array_data =
            Builder::default()
                .get_function("node.Array")
                .args(iter)
                .invoke()?
                .try_into()?;

        Ok(Array { object: array_data, _data: PhantomData })
    }

    pub fn get(&self, index: isize) -> Result<T>
    where T: TryFrom<TVMRetValue, Error=anyhow::Error> {
        // TODO(@jroesch): why do we used a signed index here?
        let element: T =
            Builder::default()
                .get_function("node.ArrayGetItem")
                .arg(self.object.clone())
                .arg(index)
                .invoke()?
                .try_into()?;

        Ok(element)
    }
}
// mod array_api {
//     extern_fn! {
//         fn _create_array(
//     }
// }

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use crate::ir::relay::Var;
    use crate::runtime::object::ObjectRef;
    use super::Array;

    #[test]
    fn create_array_and_get() -> Result<()> {
        let vec = vec![
            Var::new("foo".into(), ObjectRef::null()),
            Var::new("bar".into(), ObjectRef::null()),
        ];
        let array = Array::from_vec(vec)?;
        assert_eq!(array.get(0)?.name_hint.into(), "foo");
        assert_eq!(array.get(1)?.name_hint.into(), "bar");
        Ok(())
    }
}
