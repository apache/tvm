use std::{cell::RefCell, collections::HashMap};

use Function;

// access TVM internal API
thread_local! {
    pub(crate) static API: RefCell<HashMap<String, Function>> = RefCell::new(HashMap::new());
}

pub(crate) fn get(name: String) -> Option<Function> {
    API.with(|hm| hm.borrow().get(&name).map(|f| f.clone()))
}

pub(crate) fn set(name: String, func: Function) {
    API.with(|hm| {
        (*hm.borrow_mut()).insert(name, func);
    })
}

pub(crate) fn get_api(name: String) -> Function {
    let mut func = get(name.clone());
    if func.is_none() {
        func = Function::get_function(&name, true);
        set(
            name,
            func.clone().expect("access to `internal_api` never panics"),
        );
    }
    func.expect("access to `internal_api` never panics")
}
