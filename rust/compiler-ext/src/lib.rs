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

 use std::os::raw::c_int;
 use tvm::initialize;
 use tvm::ir::{tir, PrimExpr};
 use tvm::runtime::function::register_override;
 use tvm::runtime::map::Map;
 use tvm::runtime::object::{IsObject, IsObjectRef};
 
 use ordered_float::NotNan;
 
 mod interval;
 mod math;
 
 use math::{BoundsMap, Expr, RecExpr};
 use tvm::ir::arith::ConstIntBound;
 use tvm_rt::{ObjectRef, array::Array};
 
 macro_rules! downcast_match {
     ($id:ident; { $($t:ty => $arm:expr $(,)? )+ , else => $default:expr }) => {
         $( if let Ok($id) = $id.downcast_clone::<$t>() { $arm } else )+
         { $default }
     }
 }
 
 #[derive(Default)]
 struct VarMap {
     vars: Vec<(tvm::ir::tir::Var, egg::Symbol)>,
     objs: Vec<ObjectRef>,
 }
 
 impl VarMap {
     // FIXME this should eventually do the right thing for TVM variables
     // right now it depends on them having unique names
     fn make_symbol(&mut self, var: tvm::ir::tir::Var) -> egg::Symbol {
         let sym = egg::Symbol::from(var.name_hint.as_str().unwrap());
         for (_, sym2) in &self.vars {
             if sym == *sym2 {
                 return sym;
             }
         }
 
         self.vars.push((var, sym));
         sym
     }
 
     fn get_symbol(&self, sym: egg::Symbol) -> tvm::ir::tir::Var {
         for (v, sym2) in &self.vars {
             if sym == *sym2 {
                 return v.clone();
             }
         }
         panic!("Should have found a var")
     }
 
     fn push_obj(&mut self, obj: impl IsObjectRef) -> usize {
         let i = self.objs.len();
         self.objs.push(obj.upcast());
         i
     }
 
     fn get_obj<T: IsObjectRef>(&self, i: usize) -> T {
         self.objs[i].clone().downcast().expect("bad downcast")
     }
 }
 
 fn to_egg(vars: &mut VarMap, prim: &PrimExpr) -> RecExpr {
     fn build(vars: &mut VarMap, p: &PrimExpr, recexpr: &mut RecExpr) -> egg::Id {
         macro_rules! r {
             ($e:expr) => {
                 build(vars, &$e, recexpr)
             };
         }
 
         let dt = recexpr.add(Expr::DataType(p.datatype));
         let e = downcast_match!(p; {
             tir::Add => Expr::Add([dt, r!(p.a), r!(p.b)]),
             tir::Sub => Expr::Sub([dt, r!(p.a), r!(p.b)]),
             tir::Mul => Expr::Mul([dt, r!(p.a), r!(p.b)]),
 
             tir::Div => Expr::Div([dt, r!(p.a), r!(p.b)]),
             tir::Mod => Expr::Mod([dt, r!(p.a), r!(p.b)]),
             tir::FloorDiv => Expr::FloorDiv([dt, r!(p.a), r!(p.b)]),
             tir::FloorMod => Expr::FloorMod([dt, r!(p.a), r!(p.b)]),
 
             tir::Min => Expr::Min([dt, r!(p.a), r!(p.b)]),
             tir::Max => Expr::Max([dt, r!(p.a), r!(p.b)]),
 
             tir::Ramp => Expr::Ramp([dt, r!(p.start), r!(p.stride), recexpr.add(Expr::Int(p.lanes.into()))]),
             tir::Select => Expr::Select([dt, r!(p.condition), r!(p.true_value), r!(p.false_value)]),
 
             tir::Eq => Expr::Equal([dt, r!(p.a), r!(p.b)]),
             tir::Ne => Expr::NotEqual([dt, r!(p.a), r!(p.b)]),
             tir::Lt => Expr::Less([dt, r!(p.a), r!(p.b)]),
             tir::Le => Expr::LessEqual([dt, r!(p.a), r!(p.b)]),
             tir::Gt => Expr::Greater([dt, r!(p.a), r!(p.b)]),
             tir::Ge => Expr::GreaterEqual([dt, r!(p.a), r!(p.b)]),
 
             tir::And => Expr::And([dt, r!(p.a), r!(p.b)]),
             tir::Or => Expr::Or([dt, r!(p.a), r!(p.b)]),
             tir::Not => Expr::Not([dt, r!(p.value)]),
 
             tir::Broadcast => Expr::Broadcast([dt, r!(p.value), recexpr.add(Expr::Int(p.lanes.into()))]),
 
             tir::Let => {
                 let sym = recexpr.add(Expr::Symbol(vars.make_symbol(p.var.clone())));
                 Expr::Let([dt, sym, r!(p.value), r!(p.body)])
             }
             tir::Var => {
                 let sym = recexpr.add(Expr::Symbol(vars.make_symbol(p)));
                 Expr::Var([dt, sym])
             }
             tir::IntImm => {
                 let int = recexpr.add(Expr::Int(p.value));
                 Expr::IntImm([dt, int])
             }
             tir::FloatImm => {
                 let float = recexpr.add(Expr::Float(NotNan::new(p.value).unwrap()));
                 Expr::FloatImm([dt, float])
             }
             tir::Cast => Expr::Cast([dt, r!(p.value)]),
 
             tir::Call => {
                 let op = vars.push_obj(p.op.clone());
                 let mut arg_ids = vec![dt];
                 for i in 0..p.args.len() {
                     let arg: PrimExpr = p.args.get(i as isize).expect("array get fail");
                     arg_ids.push(r!(arg));
                 }
                 Expr::Call(op, arg_ids)
             },
             tir::Load => {
                 let sym = recexpr.add(Expr::Symbol(vars.make_symbol(p.var.clone())));
                 Expr::Load([dt, sym, r!(p.index), r!(p.predicate)])
             },
             else => {
                 println!("Failed to downcast type '{}': {}", p.type_key(), tvm::runtime::debug_print(p.clone().upcast()).unwrap().to_str().unwrap());
                 Expr::Object(vars.push_obj(p.clone()))
             }
         });
 
         recexpr.add(e)
     }
 
     let mut recexpr = Default::default();
     build(vars, prim, &mut recexpr);
     recexpr
 }
 
 fn from_egg(vars: &VarMap, recexpr: &RecExpr) -> PrimExpr {
     fn build(vars: &VarMap, nodes: &[Expr]) -> PrimExpr {
         let go = |i: &egg::Id| build(vars, &nodes[..usize::from(*i) + 1]);
         let get_dt = |i: &egg::Id| nodes[usize::from(*i)].to_dtype().unwrap();
         let prim: PrimExpr = match nodes.last().expect("cannot be empty") {
             Expr::Var([_dt, s]) => match &nodes[usize::from(*s)] {
                 Expr::Symbol(sym) => vars.get_symbol(*sym).upcast(),
                 n => panic!("Expected a symbol, got {:?}", n),
             },
             Expr::IntImm([dt, v]) => {
                 let value = nodes[usize::from(*v)].to_int().unwrap();
                 tir::IntImm::new(get_dt(dt), value).upcast()
             }
             Expr::FloatImm([dt, v]) => {
                 let value = nodes[usize::from(*v)].to_float().unwrap();
                 tir::FloatImm::new(get_dt(dt), value).upcast()
             }
             Expr::Let([dt, s, value, body]) => {
                 let var = match &nodes[usize::from(*s)] {
                     Expr::Symbol(sym) => vars.get_symbol(*sym).upcast(),
                     n => panic!("Expected a symbol, got {:?}", n),
                 };
                 tir::Let::new(get_dt(dt), var, go(value), go(body)).upcast()
             }
             Expr::Load([dt, s, value, body]) => {
                 let var = match &nodes[usize::from(*s)] {
                     Expr::Symbol(sym) => vars.get_symbol(*sym).upcast(),
                     n => panic!("Expected a symbol, got {:?}", n),
                 };
                 tir::Load::new(get_dt(dt), var, go(value), go(body)).upcast()
             }
 
             Expr::Add([dt, a, b]) => tir::Add::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::Sub([dt, a, b]) => tir::Sub::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::Mul([dt, a, b]) => tir::Mul::new(get_dt(dt), go(a), go(b)).upcast(),
 
             Expr::Div([dt, a, b]) => tir::Div::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::Mod([dt, a, b]) => tir::Mod::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::FloorDiv([dt, a, b]) => tir::FloorDiv::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::FloorMod([dt, a, b]) => tir::FloorMod::new(get_dt(dt), go(a), go(b)).upcast(),
 
             Expr::Min([dt, a, b]) => tir::Min::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::Max([dt, a, b]) => tir::Max::new(get_dt(dt), go(a), go(b)).upcast(),
 
             Expr::Equal([dt, a, b]) => tir::Eq::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::NotEqual([dt, a, b]) => tir::Ne::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::Less([dt, a, b]) => tir::Lt::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::LessEqual([dt, a, b]) => tir::Le::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::Greater([dt, a, b]) => tir::Gt::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::GreaterEqual([dt, a, b]) => tir::Ge::new(get_dt(dt), go(a), go(b)).upcast(),
 
             Expr::And([dt, a, b]) => tir::And::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::Or([dt, a, b]) => tir::Or::new(get_dt(dt), go(a), go(b)).upcast(),
             Expr::Not([dt, a]) => tir::Not::new(get_dt(dt), go(a)).upcast(),
 
             Expr::Ramp([dt, a, b, c]) => {
                 let len = &nodes[usize::from(*c)];
                 let i = len
                     .to_int()
                     .unwrap_or_else(|| panic!("Ramp lanes must be an int, got {:?}", len));
                 tir::Ramp::new(get_dt(dt), go(a), go(b), i as i32).upcast()
             }
             Expr::Broadcast([dt, val, lanes]) => {
                 let lanes = &nodes[usize::from(*lanes)];
                 let lanes = lanes
                     .to_int()
                     .unwrap_or_else(|| panic!("Ramp lanes must be an int, got {:?}", lanes));
                 println!("dt: {}", get_dt(dt));
                 tir::Broadcast::new(get_dt(dt), go(val), lanes as i32).upcast()
             }
 
             Expr::Select([dt, a, b, c]) => tir::Select::new(get_dt(dt), go(a), go(b), go(c)).upcast(),
             Expr::Cast([dt, a]) => tir::Cast::new(get_dt(dt), go(a)).upcast(),
             Expr::Call(expr, args) => {
                 let arg_exprs: Vec<PrimExpr> = args[1..].iter().map(go).collect();
                 let arg_exprs = Array::from_vec(arg_exprs).expect("failed to convert args");
                 tir::Call::new(get_dt(&args[0]), vars.get_obj(*expr), arg_exprs).upcast()
             }
 
             Expr::Object(i) => vars.get_obj(*i),
             node => panic!("I don't know how to extract {:?}", node),
         };
         assert_ne!(prim.datatype.bits(), 0);
         assert_ne!(prim.datatype.lanes(), 0);
         prim
     }
     build(vars, recexpr.as_ref())
 }
 
 fn run(
     input: PrimExpr,
     expected: Option<PrimExpr>,
     map: Map<PrimExpr, ConstIntBound>,
 ) -> Result<PrimExpr, String> {
     use egg::{CostFunction, Extractor};
 
     let mut bounds = BoundsMap::default();
     for (k, v) in map {
         if let Ok(var) = k.downcast_clone::<tir::Var>() {
             let sym: egg::Symbol = var.name_hint.as_str().unwrap().into();
             bounds.insert(sym, (v.min_value, v.max_value));
         } else {
             println!("Non var in bounds map: {}", tvm::ir::as_text(k));
         }
     }
 
     let mut vars = VarMap::default();
     let expr = to_egg(&mut vars, &input);
     let mut runner = math::default_runner();
     runner.egraph.analysis.bounds = bounds;
 
     let mut runner = runner.with_expr(&expr).run(&math::rules());
     // runner.print_report();
     let mut extractor = Extractor::new(&runner.egraph, math::CostFn);
     let root = runner.egraph.find(runner.roots[0]);
     let (cost, best) = extractor.find_best(root);
     if let Some(expected) = expected {
         let mut expected_vars = VarMap::default();
         let expected_expr = to_egg(&mut expected_vars, &expected);
         let expected_root = runner.egraph.add_expr(&expected_expr);
         if expected_root != root {
             return Err(format!(
                 "\n\nFailed to prove them equal!\nExpected:\n{}\nFound:\n{}\n",
                 expected_expr.pretty(40),
                 best.pretty(40)
             ));
         }
         let expected_cost = math::CostFn.cost_rec(&expected_expr);
         if expected_cost != cost {
             let msg = format!(
                 "\n\nCosts not equal: Expected {}:\n{}\nFound {}:\n{}\n",
                 expected_cost,
                 expected_expr.pretty(40),
                 cost,
                 best.pretty(40)
             );
             if cost < expected_cost {
                 println!("egg wins: {}", msg)
             } else {
                 return Err(msg);
             }
         }
     }
     log::info!("  returning... {}", best.pretty(60));
     Ok(from_egg(&vars, &best))
 }
 
 fn simplify(prim: PrimExpr, map: Map<PrimExpr, ConstIntBound>) -> Result<PrimExpr, tvm::Error> {
     log::debug!("map: {:?}", map);
     run(prim, None, map).map_err(tvm::Error::CallFailed)
 }
 
 fn simplify_and_check(
     prim: PrimExpr,
     check: PrimExpr,
     map: Map<PrimExpr, ConstIntBound>,
 ) -> Result<PrimExpr, tvm::Error> {
     log::debug!("check map: {:?}", map);
     run(prim, Some(check), map).map_err(tvm::Error::CallFailed)
 }
 
 initialize!({
     let _ = env_logger::try_init();
     // NOTE this print prevents a segfault (on Linux) for now...
     println!("Initializing simplifier... ");
     register_override(simplify, "egg.simplify", true).expect("failed to initialize simplifier");
     register_override(simplify_and_check, "egg.simplify_and_check", true)
         .expect("failed to initialize simplifier");
     log::debug!("done!");
 });
 