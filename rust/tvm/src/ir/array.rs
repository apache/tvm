use std::convert::TryFrom;
use std::marker::PhantomData;

use crate::runtime::object::{ObjectRef, ToObjectRef};

use tvm_rt::external;
use tvm_rt::RetValue;

use anyhow::Result;

#[derive(Clone)]
pub struct Array<T: ToObjectRef> {
    object: ObjectRef,
    _data: PhantomData<T>,
}

external! {
    #[name("node.ArrayGetItem")]
    fn array_get_item(array: ObjectRef, index: isize) -> ObjectRef;
}

impl<T: ToObjectRef> Array<T> {
    pub fn from_vec(data: Vec<T>) -> Result<Array<T>> {
        unimplemented!()
        // let iter = data.iter().map(|element| element.to_object_ref());

        // let array_data = Builder::default()
        //     .get_function("node.Array")
        //     .args(iter)
        //     .invoke()?
        //     .try_into()?;

        // Ok(Array {
        //     object: array_data,
        //     _data: PhantomData,
        // })
    }

    pub fn get(&self, index: isize) -> Result<T>
    where
        T: TryFrom<RetValue, Error = anyhow::Error>,
    {
        unimplemented!()
        // // TODO(@jroesch): why do we used a signed index here?
        // let element: T = Builder::default()
        //     .get_function("node.ArrayGetItem")
        //     .arg(self.object.clone())
        //     .arg(index)
        //     .invoke()?
        //     .try_into()?;

        // Ok(element)
    }
}

#[cfg(test)]
mod tests {
    use super::Array;
    use crate::ir::relay::Var;
    use crate::runtime::object::ObjectRef;
    use anyhow::Result;

    #[test]
    fn create_array_and_get() -> Result<()> {
        let vec = vec![
            Var::new("foo".into(), ObjectRef::null()),
            Var::new("bar".into(), ObjectRef::null()),
        ];
        let array = Array::from_vec(vec)?;
        assert_eq!(array.get(0)?.name_hint().to_string()?, "foo");
        assert_eq!(array.get(1)?.name_hint().to_string()?, "bar");
        Ok(())
    }
}
