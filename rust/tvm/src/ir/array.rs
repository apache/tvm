use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;

use crate::runtime::object::{ObjectRef, IsObjectRef};

use tvm_rt::{external, RetValue, function::{Function, Result}};
use tvm_rt::errors::Error;

#[derive(Clone)]
pub struct Array<T: IsObjectRef> {
    object: ObjectRef,
    _data: PhantomData<T>,
}

// TODO(@jroesch): convert to use generics instead of casting inside
// the implementation.
external! {
    #[name("node.ArrayGetItem")]
    fn array_get_item(array: ObjectRef, index: isize) -> ObjectRef;
}

impl<T: IsObjectRef> Array<T> {
    pub fn from_vec(data: Vec<T>) -> Result<Array<T>> {
        let iter = data.iter().map(|element| element.to_object_ref().into()).collect();

        let func = Function::get("node.Array")
            .expect("node.Array function is not registered, this is most likely a build or linking error");

        let array_data = func.invoke(iter)?.try_into()?;

        Ok(Array {
            object: array_data,
            _data: PhantomData,
        })
    }

    pub fn get(&self, index: isize) -> Result<T>
    where
        T: TryFrom<RetValue, Error = Error>,
    {
       let oref: ObjectRef = array_get_item(self.object.clone(), index)?;
       oref.downcast()
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
